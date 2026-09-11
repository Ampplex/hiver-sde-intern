import argparse
import json
import os
import random
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import boto3
from botocore.config import Config
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from src.agent.response_generator import ResponseGenerator
from src.retrieval.hybrid_retriever import HybridRetriever

load_dotenv()
PREDICTIONS = Path("data/processed/eval/system_predictions.parquet")
CORPUS = Path("data/processed/retriever/retrieval_corpus.parquet")
OUTPUT = Path("data/processed/eval/llm_judge_scores.parquet")
METRICS = ["correctness", "groundedness", "resolution_appropriateness", "completeness", "communication_quality", "overall"]

MIN_JUDGE_CASES = 40


def clean_json(text):
    text = str(text).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"): lines = lines[1:]
        if lines and lines[-1].strip() == "```": lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def judge(client, model_id, customer_problem, response_text, evidence, max_retries=6, base_delay=2.0, max_delay=60.0):
    prompt = f"""
Evaluate this customer-support response using ONLY the supplied customer problem and AmazonHelp evidence.
Do not infer policy from general knowledge. Historical customer messages are evidence about the problem only; only AmazonHelp messages are evidence of how AmazonHelp responded.

CUSTOMER PROBLEM:
{customer_problem}

AMAZONHELP EVIDENCE:
{json.dumps(evidence, ensure_ascii=False, indent=2)}

RESPONSE:
{response_text}

Score each metric 1-5 using these anchors:
1 = wrong/unsupported/unusable; 3 = partially useful but has a meaningful defect; 5 = fully correct, grounded, appropriate, complete, and sendable.

correctness: Does it address the actual problem?
groundedness: Are substantive support claims supported by AmazonHelp evidence?
resolution_appropriateness: Does it follow the demonstrated AmazonHelp resolution pattern rather than inventing one?
completeness: Does it provide the appropriate supported next step?
communication_quality: Is it concise, clear, professional, and natural for Twitter?
overall: Would this be sendable with no meaningful revision?

Return ONLY JSON with integer scores.
"""
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"temperature": 0.0},
            )
            result = json.loads(clean_json(response["output"]["message"]["content"][0]["text"]))
            for metric in METRICS:
                value = int(result[metric])
                if value < 1 or value > 5:
                    raise ValueError(f"Invalid {metric}: {value}")
            time.sleep(1.0 + random.uniform(0.1, 0.3))
            return result
        except Exception as exc:
            last_error = exc
            delay = min(max_delay, base_delay * (2 ** attempt)) * random.uniform(0.8, 1.2)
            print(f"\n[Judge retry {attempt + 1}/{max_retries}] {type(exc).__name__}: {exc}. Backing off {delay:.1f}s...")
            time.sleep(delay)
    raise last_error


def evidence_rows(df, ids):
    sub = df[df["conversation_id"].isin(ids)]
    rows = []
    for r in sub.itertuples():
        raw_resps = getattr(r, "amazonhelp_responses", None)
        if raw_resps is None:
            resps = []
        elif hasattr(raw_resps, "tolist"):
            resps = [str(x) for x in raw_resps.tolist()]
        elif isinstance(raw_resps, (list, tuple)):
            resps = [str(x) for x in raw_resps]
        else:
            resps = [str(raw_resps)]

        rows.append({
            "conversation_id": r.conversation_id,
            "customer_problem": str(r.customer_problem),
            "amazonhelp_responses": resps,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cases", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    predictions = pd.read_parquet(PREDICTIONS)
    corpus = pd.read_parquet(CORPUS)

    # Benchmark cohort: current audited gold auto_handle cases.
    # Methodological framing: conditional response-generation quality with the final confidence gate ablated.
    # Evaluates whether retrieval + LLM generation produce high-quality grounded replies when a human
    # confirms the inquiry is safe to automate, independent of the deployed system's conservative safety gate.
    eligible = predictions[
        predictions["gold_action"].eq("auto_handle")
        & predictions["bm25_reply"].notna()
    ].copy()
    if len(eligible) > args.max_cases:
        eligible = eligible.sample(args.max_cases, random_state=args.seed)
    eligible = eligible.sort_values("conversation_id")

    if len(eligible) < MIN_JUDGE_CASES:
        raise RuntimeError(
            f"Need at least {MIN_JUDGE_CASES} gold auto_handle benchmark cases "
            f"for response-quality evaluation; found {len(eligible)}."
        )

    cfg = Config(retries={"max_attempts": 10, "mode": "adaptive"})
    client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-west-2"), config=cfg)
    model_id = os.getenv("BEDROCK_MODEL_ID", "mistral.mistral-large-2407-v1:0")
    generator = ResponseGenerator(region=os.getenv("AWS_REGION", "us-west-2"), model_id=model_id)
    retriever = HybridRetriever()

    # Load existing checkpoint if resuming, keeping ONLY currently eligible cases
    existing_records = []
    judged_ids = set()
    eligible_ids = set(eligible["conversation_id"])
    if OUTPUT.exists():
        try:
            existing_df = pd.read_parquet(OUTPUT)
            if "conversation_id" in existing_df.columns:
                all_records = existing_df.to_dict(orient="records")
                # Filter out stale records that are no longer in the eligible cohort
                existing_records = [r for r in all_records if r.get("conversation_id") in eligible_ids]
                judged_ids = {r["conversation_id"] for r in existing_records}
                print(f"Resuming from existing checkpoint: {len(judged_ids)} currently eligible cases already judged.")
        except Exception:
            existing_records = []
            judged_ids = set()

    rows = list(existing_records)
    generation_failed_count = 0

    for row in tqdm(eligible.itertuples(), total=len(eligible), desc="LLM judging"):
        if row.conversation_id in judged_ids:
            continue

        raw_retrieved = getattr(row, "retrieved_conversation_ids", None)
        if raw_retrieved is None:
            retrieved_ids = []
        elif hasattr(raw_retrieved, "tolist"):
            retrieved_ids = raw_retrieved.tolist()
        else:
            retrieved_ids = list(raw_retrieved)

        pred_intent = getattr(row, "predicted_intent_id", None)

        # If confidence gate stopped retrieval during end-to-end evaluation, use the classifier's existing
        # predicted_intent_id (top-1 candidate) to perform intent-conditioned retrieval for this benchmark.
        if len(retrieved_ids) == 0:
            if pred_intent:
                search_res = retriever.search(row.customer_problem, intent_id=pred_intent, top_k=3)
                retrieved_ids = [c["conversation_id"] for c in search_res]
            else:
                retrieved_ids = []

        if len(retrieved_ids) == 0:
            generation_failed_count += 1
            continue

        system_evidence = evidence_rows(corpus, retrieved_ids)
        bm25_evidence = evidence_rows(corpus, [row.bm25_source_conversation_id])

        # If draft_reply already exists from end-to-end evaluation, use it; otherwise generate using ResponseGenerator
        draft_reply = getattr(row, "draft_reply", None)
        if not isinstance(draft_reply, str) or not draft_reply.strip():
            gen_res = generator.generate_response(row.customer_problem, pred_intent, system_evidence)
            draft_reply = gen_res.get("draft_reply")

        # Do not fabricate a fallback reply: exclude if generation failed or returned null
        if not isinstance(draft_reply, str) or not draft_reply.strip():
            generation_failed_count += 1
            continue

        system_scores = judge(client, model_id, row.customer_problem, draft_reply, system_evidence)
        bm25_scores = judge(client, model_id, row.customer_problem, row.bm25_reply, bm25_evidence)

        record = {
            "conversation_id": row.conversation_id,
            "predicted_intent_id": pred_intent,
            "gold_intent": row.gold_intent,
            "gold_action": row.gold_action,
            "system_action": row.system_action,
            "judge_subset": "gold_auto_handle_response_quality",
            "draft_reply": draft_reply,
            "bm25_reply": row.bm25_reply,
        }
        for metric in METRICS:
            record[f"system_{metric}"] = system_scores[metric]
            record[f"bm25_{metric}"] = bm25_scores[metric]
        rows.append(record)
        judged_ids.add(row.conversation_id)

        # Checkpoint every 5 rows
        if len(rows) % 5 == 0:
            pd.DataFrame(rows).to_parquet(OUTPUT, index=False)

    if len(rows) < MIN_JUDGE_CASES:
        raise RuntimeError(
            f"Need at least {MIN_JUDGE_CASES} successfully generated cases "
            f"for response-quality evaluation; scored {len(rows)}/{len(eligible)} "
            f"({generation_failed_count} excluded due to unavailable generation/evidence)."
        )

    result = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT, index=False)

    print(f"\n============================================================")
    print(f"RESPONSE-QUALITY BENCHMARK RESULTS (Ablated Confidence Gate)")
    print(f"============================================================")
    print(f"Total gold auto_handle cases:        {len(eligible)}")
    print(f"Successfully generated/judged cases: {len(result)}")
    print(f"Excluded generation failures:        {generation_failed_count}")
    print(f"------------------------------------------------------------")
    for metric in METRICS:
        print(f"{metric:<30} Proposed={result[f'system_{metric}'].mean():.2f}/5  BM25={result[f'bm25_{metric}'].mean():.2f}/5") if len(result) else None
    print(f"============================================================")
    print(f"Results saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
