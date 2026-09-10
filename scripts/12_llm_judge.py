import argparse
import json
import os
from pathlib import Path

import boto3
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

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


def judge(client, model_id, customer_problem, response_text, evidence):
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
    return result


def evidence_rows(df, ids):
    sub = df[df["conversation_id"].isin(ids)]
    return [
        {
            "conversation_id": r.conversation_id,
            "customer_problem": str(r.customer_problem),
            "amazonhelp_responses": [str(x) for x in (r.amazonhelp_responses or [])],
        }
        for r in sub.itertuples()
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cases", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    predictions = pd.read_parquet(PREDICTIONS)
    corpus = pd.read_parquet(CORPUS)
    # Conditional comparison: response quality is evaluated only where the system actually produced an auto-handle response.
    eligible = predictions[
        predictions["system_action"].eq("auto_handle")
        & predictions["draft_reply"].notna()
        & predictions["bm25_reply"].notna()
    ].copy()
    if len(eligible) > args.max_cases:
        eligible = eligible.sample(args.max_cases, random_state=args.seed)
    eligible = eligible.sort_values("conversation_id")

    if len(eligible) < MIN_JUDGE_CASES:
        raise RuntimeError(
            f"Need at least {MIN_JUDGE_CASES} matched auto-handled cases "
            f"for response-quality/human-agreement evaluation; "
            f"found {len(eligible)}."
        )

    client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-west-2"))
    model_id = os.getenv("BEDROCK_MODEL_ID", "mistral.mistral-large-2407-v1:0")
    rows = []

    for row in tqdm(eligible.itertuples(), total=len(eligible), desc="LLM judging"):
        system_evidence = evidence_rows(corpus, row.retrieved_conversation_ids)
        bm25_evidence = evidence_rows(corpus, [row.bm25_source_conversation_id])
        system_scores = judge(client, model_id, row.customer_problem, row.draft_reply, system_evidence)
        bm25_scores = judge(client, model_id, row.customer_problem, row.bm25_reply, bm25_evidence)
        record = {"conversation_id": row.conversation_id, "system_action": row.system_action, "judge_subset": "matched_auto_handled_system_vs_bm25"}
        for metric in METRICS:
            record[f"system_{metric}"] = system_scores[metric]
            record[f"bm25_{metric}"] = bm25_scores[metric]
        rows.append(record)

    result = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT, index=False)
    print(f"Judged matched auto-handled cases: {len(result)}")
    for metric in METRICS:
        print(f"{metric:<30} System={result[f'system_{metric}'].mean():.2f}/5  BM25={result[f'bm25_{metric}'].mean():.2f}/5") if len(result) else None


if __name__ == "__main__":
    main()
