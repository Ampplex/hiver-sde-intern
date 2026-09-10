import json
import os
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from tqdm import tqdm

from src.agent.capability_guard import CapabilityGuard
from src.agent.policy import apply_safety_policy
from src.agent.response_generator import ResponseGenerator
from src.classification.intent_classifier import IntentClassifier
from src.retrieval.hybrid_retriever import HybridRetriever

GOLD_PATH = Path("data/processed/eval/golden_set_labeled.parquet")
TRAIN_PATH = Path("data/processed/support_cases_train.parquet")
ASSIGNMENTS_PATH = Path("data/processed/intent_assignments.parquet")
TAXONOMY_PATH = Path("data/processed/intent_taxonomy.json")
OUTPUT_PATH = Path("data/processed/eval/system_predictions.parquet")
METRICS_PATH = Path("data/processed/eval/evaluation_metrics.json")


def validate_gold(df, taxonomy):
    required = {"conversation_id", "customer_problem", "gold_intent", "gold_action", "gold_reason"}
    missing = required - set(df.columns)
    if missing or len(df) != 200 or df["conversation_id"].duplicated().any():
        raise ValueError(f"Invalid golden set: missing={sorted(missing)}")
    invalid = set(df["gold_intent"]) - set(taxonomy)
    if invalid:
        raise ValueError(f"Golden labels contain intents outside frozen taxonomy: {sorted(invalid)}")
    valid_actions = {"auto_handle", "escalate"}
    invalid_actions = set(df["gold_action"]) - valid_actions
    if invalid_actions:
        raise ValueError(f"Golden labels contain invalid actions: {sorted(invalid_actions)}")
    for col in ["gold_intent", "gold_action", "gold_reason"]:
        if df[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"Golden labels contain empty values in {col}.")


def action_metrics(gold, pred):
    gold_escalate = [x == "escalate" for x in gold]
    pred_escalate = [x == "escalate" for x in pred]
    gold_auto = [not x for x in gold_escalate]
    pred_auto = [not x for x in pred_escalate]
    unsafe_auto = sum(g and not p for g, p in zip(gold_escalate, pred_escalate))
    predicted_auto = sum(pred_auto)
    return {
        "auto_handle_coverage": float(sum(pred_auto) / len(pred_auto)),
        "unsafe_auto_handle_rate": float(unsafe_auto / len(gold)),
        "unsafe_auto_handle_rate_given_auto": float(unsafe_auto / predicted_auto) if predicted_auto else 0.0,
        "escalation_recall": float(recall_score(gold_escalate, pred_escalate, zero_division=0)),
        "auto_handle_precision": float(precision_score(gold_auto, pred_auto, zero_division=0)),
    }


def main():
    with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)
    gold = pd.read_parquet(GOLD_PATH)
    validate_gold(gold, taxonomy)

    assignments = pd.read_parquet(ASSIGNMENTS_PATH)
    overlap = set(gold["conversation_id"].astype(str)) & set(assignments["conversation_id"].astype(str))
    if overlap:
        raise RuntimeError(f"Golden/training leakage detected: {len(overlap)} IDs overlap.")

    classifier = IntentClassifier()
    retriever = HybridRetriever()
    generator = ResponseGenerator()
    capability_guard = CapabilityGuard()
    predictions = []

    for _, row in tqdm(gold.iterrows(), total=len(gold), desc="Evaluating agent"):
        message = str(row["customer_problem"])
        classification = classifier.predict(message)
        intent = classification["intent_id"]
        historical = []
        if intent != "uncertain":
            historical = retriever.search(message, intent_id=intent, top_k=3)

        llm_output = {"draft_reply": None, "decision": "escalate", "reason": "Classifier uncertainty." if intent == "uncertain" else "No historical evidence retrieved."}
        capability_result = None
        if intent != "uncertain" and historical:
            llm_output = generator.generate_response(message, intent, historical)
            if str(llm_output.get("decision", "")).lower() == "auto_handle" and llm_output.get("draft_reply"):
                capability_result = capability_guard.assess(
                    customer_message=message,
                    predicted_intent=intent,
                    historical_cases=historical,
                    draft_reply=llm_output.get("draft_reply"),
                )

        final = apply_safety_policy(
            customer_message=message,
            classification=classification,
            historical_cases=historical,
            llm_output=llm_output,
            capability_result=capability_result,
        )

        bm25 = retriever.search_bm25_only(message, top_k=1)
        predictions.append({
            "conversation_id": row["conversation_id"],
            "customer_problem": message,
            "gold_intent": row["gold_intent"],
            "gold_action": row["gold_action"],
            "system_intent": intent,
            "predicted_intent_id": classification["predicted_intent_id"],
            "classifier_similarity": classification["similarity_score"],
            "classifier_margin": classification["margin"],
            "retrieved_conversation_ids": [c["conversation_id"] for c in historical],
            "top_dense_score": historical[0]["dense_score"] if historical else 0.0,
            "system_action": final["decision"],
            "system_reason": final["reason"],
            "draft_reply": final["draft_reply"],
            "llm_reason": llm_output.get("reason"),
            "capability_guard_decision": capability_result.get("decision") if capability_result else None,
            "capability_guard_reason": capability_result.get("reason") if capability_result else None,
            "bm25_reply": bm25[0]["first_amazon_response"] if bm25 else None,
            "bm25_source_conversation_id": bm25[0]["conversation_id"] if bm25 else None,
        })

    result = pd.DataFrame(predictions)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_PATH, index=False)

    labels = sorted(set(result["gold_intent"]) | set(result["system_intent"]))
    non_abstained = result["system_intent"].ne("uncertain")
    conditional = result[non_abstained]
    conditional_labels = sorted(set(conditional["gold_intent"]) | set(conditional["system_intent"]))
    intent_metrics = {
        "accuracy_including_abstentions": float(accuracy_score(result["gold_intent"], result["system_intent"])),
        "macro_f1_including_abstentions": float(f1_score(result["gold_intent"], result["system_intent"], labels=labels, average="macro", zero_division=0)),
        "abstention_rate": float((~non_abstained).mean()),
        "conditional_accuracy_non_abstained": float(accuracy_score(conditional["gold_intent"], conditional["system_intent"])) if len(conditional) else 0.0,
        "conditional_macro_f1_non_abstained": float(f1_score(conditional["gold_intent"], conditional["system_intent"], labels=conditional_labels, average="macro", zero_division=0)) if len(conditional) else 0.0,
        "labels": labels,
        "confusion_matrix": confusion_matrix(result["gold_intent"], result["system_intent"], labels=labels).tolist(),
    }

    train_assigned = assignments["intent_id"].astype(str)
    majority_intent = Counter(train_assigned).most_common(1)[0][0]
    majority_acc = accuracy_score(result["gold_intent"], [majority_intent] * len(result))
    metrics = {
        "n_golden_examples": int(len(result)),
        "system_intent": intent_metrics,
        "majority_intent_baseline": {"accuracy": float(majority_acc), "majority_intent": majority_intent, "baseline_training_examples": int(len(assignments))},
        "system_action": action_metrics(result["gold_action"].tolist(), result["system_action"].tolist()),
        "always_escalate_baseline": action_metrics(result["gold_action"].tolist(), ["escalate"] * len(result)),
        "simple_baseline": {"name": "BM25 top-1 historical response", "response_available_rate": float(result["bm25_reply"].notna().mean())},
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
