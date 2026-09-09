import os
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from src.agent.policy import apply_safety_policy
from src.agent.response_generator import ResponseGenerator
from src.classification.intent_classifier import IntentClassifier
from src.retrieval.hybrid_retriever import HybridRetriever


GOLD_PATH = Path(
    "data/processed/eval/golden_set_labeled.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/eval/system_predictions.parquet"
)

METRICS_PATH = Path(
    "data/processed/eval/evaluation_metrics.json"
)


def validate_gold(df):

    required = {
        "conversation_id",
        "customer_problem",
        "gold_intent",
        "gold_action",
        "gold_reason",
    }

    missing = (
        required - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing gold columns: {sorted(missing)}"
        )

    if len(df) != 200:
        raise ValueError(
            f"Expected 200 golden examples, found {len(df)}."
        )

    if df["conversation_id"].duplicated().any():
        raise ValueError(
            "Duplicate conversation_id in golden set."
        )

    if df["gold_intent"].isna().any():
        raise ValueError(
            "Gold intent contains missing labels."
        )

    if df["gold_action"].isna().any():
        raise ValueError(
            "Gold action contains missing labels."
        )

    invalid_actions = (
        set(df["gold_action"])
        - {"auto_handle", "escalate"}
    )

    if invalid_actions:
        raise ValueError(
            f"Invalid gold actions: {invalid_actions}"
        )


def calculate_intent_metrics(df):

    gold = df[
        "gold_intent"
    ].tolist()

    pred = df[
        "system_intent"
    ].tolist()

    labels = sorted(
        set(gold)
        | set(pred)
    )

    return {
        "accuracy": float(
            accuracy_score(
                gold,
                pred,
            )
        ),
        "macro_f1": float(
            f1_score(
                gold,
                pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "labels": labels,
        "confusion_matrix": (
            confusion_matrix(
                gold,
                pred,
                labels=labels,
            ).tolist()
        ),
    }


def calculate_action_metrics(
    gold,
    pred,
):

    gold_binary = [
        1 if x == "escalate" else 0
        for x in gold
    ]

    pred_binary = [
        1 if x == "escalate" else 0
        for x in pred
    ]

    auto_gold = [
        1 - x
        for x in gold_binary
    ]

    auto_pred = [
        1 - x
        for x in pred_binary
    ]

    unsafe_auto = sum(
        gold_value == 1
        and pred_value == 0
        for gold_value, pred_value
        in zip(
            gold_binary,
            pred_binary,
        )
    )

    predicted_auto_count = sum(
        x == 0
        for x in pred_binary
    )

    unsafe_auto_given_auto = (
        unsafe_auto / predicted_auto_count
        if predicted_auto_count > 0
        else 0.0
    )

    return {
        "auto_handle_coverage": (
            sum(
                x == 0
                for x in pred_binary
            )
            / len(pred_binary)
        ),
        "unsafe_auto_handle_rate": (
            unsafe_auto
            / len(gold_binary)
        ),
        "unsafe_auto_handle_rate_given_auto": (
            unsafe_auto_given_auto
        ),
        "escalation_recall": float(
            recall_score(
                gold_binary,
                pred_binary,
                zero_division=0,
            )
        ),
        "auto_handle_precision": float(
            precision_score(
                auto_gold,
                auto_pred,
                zero_division=0,
            )
        ),
    }


def main():

    df = pd.read_parquet(
        GOLD_PATH
    )

    validate_gold(df)

    print(
        f"Evaluating {len(df)} golden examples."
    )

    classifier = IntentClassifier()
    retriever = HybridRetriever()
    generator = ResponseGenerator()

    retrieval_threshold = float(
        os.getenv(
            "RETRIEVAL_EVIDENCE_THRESHOLD",
            "0.55",
        )
    )

    min_evidence_cases = int(
        os.getenv(
            "MIN_STRONG_EVIDENCE_CASES",
            "2",
        )
    )

    predictions = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Evaluating agent",
    ):

        message = str(
            row["customer_problem"]
        )

        # ====================================================
        # SYSTEM: CLASSIFICATION
        # ====================================================

        classification = classifier.predict(
            message
        )

        intent = classification[
            "intent_id"
        ]

        # ====================================================
        # SYSTEM: HYBRID RETRIEVAL
        # ====================================================

        historical_cases = []

        if intent != "uncertain":

            historical_cases = retriever.search(
                query=message,
                intent_id=intent,
                top_k=3,
            )

        # ====================================================
        # SYSTEM: GENERATION
        # ====================================================

        llm_output = {
            "draft_reply": None,
            "decision": "escalate",
            "reason": "Generation not reached.",
        }

        if (
            intent != "uncertain"
            and historical_cases
        ):

            llm_output = (
                generator.generate_response(
                    customer_message=message,
                    predicted_intent=intent,
                    historical_cases=historical_cases,
                )
            )

        # ====================================================
        # SYSTEM: SAFETY POLICY
        # ====================================================

        final = apply_safety_policy(
            customer_message=message,
            classification=classification,
            historical_cases=historical_cases,
            llm_output=llm_output,
            retrieval_threshold=retrieval_threshold,
            min_evidence_cases=min_evidence_cases,
        )

        # ====================================================
        # SIMPLE BASELINE: BM25 TOP-1
        # ====================================================

        bm25_results = (
            retriever.search_bm25_only(
                query=message,
                top_k=1,
            )
        )

        bm25_reply = None
        bm25_source_id = None

        if bm25_results:

            bm25_reply = bm25_results[0][
                "first_amazon_response"
            ]

            bm25_source_id = (
                bm25_results[0][
                    "conversation_id"
                ]
            )

        predictions.append(
            {
                "conversation_id": row[
                    "conversation_id"
                ],

                "customer_problem": message,

                "gold_intent": row[
                    "gold_intent"
                ],

                "gold_action": row[
                    "gold_action"
                ],

                # -------------------------
                # System
                # -------------------------

                "system_intent": intent,

                "classifier_similarity": (
                    classification[
                        "similarity_score"
                    ]
                ),

                "classifier_margin": (
                    classification[
                        "margin"
                    ]
                ),

                "retrieved_conversation_ids": [
                    case[
                        "conversation_id"
                    ]
                    for case
                    in historical_cases
                ],

                "top_dense_score": (
                    historical_cases[0][
                        "dense_score"
                    ]
                    if historical_cases
                    else 0.0
                ),

                "system_action": final[
                    "decision"
                ],

                "system_reason": final[
                    "reason"
                ],

                "draft_reply": final[
                    "draft_reply"
                ],

                "llm_reason": llm_output.get(
                    "reason"
                ),

                # -------------------------
                # BM25 baseline
                # -------------------------

                "bm25_reply": bm25_reply,

                "bm25_source_conversation_id": (
                    bm25_source_id
                ),
            }
        )

    result = pd.DataFrame(
        predictions
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    # ========================================================
    # SYSTEM INTENT
    # ========================================================

    intent_metrics = (
        calculate_intent_metrics(
            result
        )
    )

    majority_intent = (
        Counter(
            result["gold_intent"]
        )
        .most_common(1)[0][0]
    )

    majority_predictions = [
        majority_intent
    ] * len(result)

    majority_accuracy = (
        accuracy_score(
            result["gold_intent"],
            majority_predictions,
        )
    )

    # ========================================================
    # SYSTEM ACTION
    # ========================================================

    action_metrics = (
        calculate_action_metrics(
            result["gold_action"],
            result["system_action"],
        )
    )

    # ========================================================
    # TRIVIAL BASELINE: ALWAYS ESCALATE
    # ========================================================

    always_escalate = [
        "escalate"
    ] * len(result)

    always_escalate_metrics = (
        calculate_action_metrics(
            result["gold_action"],
            always_escalate,
        )
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics = {
        "n_golden_examples": len(result),

        "system_intent": intent_metrics,

        "majority_intent_baseline": {
            "accuracy": float(
                majority_accuracy
            ),
            "majority_intent": majority_intent,
        },

        "system_action": action_metrics,

        "always_escalate_baseline": (
            always_escalate_metrics
        ),

        "simple_baseline": {
            "name": "BM25 top-1 historical response",
            "response_available_rate": float(
                result["bm25_reply"]
                .notna()
                .mean()
            ),
        },
    }

    import json

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
        )

    # ========================================================
    # PRINT
    # ========================================================

    print("\n" + "=" * 60)
    print("INTENT CLASSIFICATION")
    print("=" * 60)

    print(
        f"Majority baseline accuracy: "
        f"{majority_accuracy:.3f}"
    )

    print(
        f"System accuracy: "
        f"{intent_metrics['accuracy']:.3f}"
    )

    print(
        f"System macro F1: "
        f"{intent_metrics['macro_f1']:.3f}"
    )

    print("\n" + "=" * 60)
    print("ESCALATION / SAFETY")
    print("=" * 60)

    for key, value in action_metrics.items():

        print(
            f"{key}: {value:.3f}"
        )

    print("\nAlways-escalate baseline:")

    for key, value in (
        always_escalate_metrics.items()
    ):

        print(
            f"{key}: {value:.3f}"
        )

    print("\nSimple response baseline:")
    print(
        "BM25 top-1 historical response"
    )

    print(
        f"\nPredictions saved to:\n"
        f"{OUTPUT_PATH}"
    )

    print(
        f"Metrics saved to:\n"
        f"{METRICS_PATH}"
    )


if __name__ == "__main__":
    main()
