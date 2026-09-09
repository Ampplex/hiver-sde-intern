import argparse
import json
from pathlib import Path

import pandas as pd


EVAL_DIR = Path(
    "data/processed/eval"
)

GOLDEN_INPUT = (
    EVAL_DIR / "golden_set.parquet"
)

MANUAL_LABELS = (
    EVAL_DIR / "golden_labels_manual.csv"
)

LABELED_OUTPUT = (
    EVAL_DIR / "golden_set_labeled.parquet"
)

TAXONOMY_PATH = Path(
    "data/processed/intent_taxonomy.json"
)

VALID_ACTIONS = {
    "auto_handle",
    "escalate",
}


def load_taxonomy():
    if not TAXONOMY_PATH.exists():
        raise FileNotFoundError(
            f"Taxonomy not found: {TAXONOMY_PATH}\n"
            "Run scripts/04_discover_intents.py first "
            "and freeze the taxonomy before labeling."
        )

    with open(
        TAXONOMY_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        taxonomy = json.load(f)

    if not isinstance(taxonomy, dict):
        raise ValueError(
            "Intent taxonomy must be a JSON object."
        )

    if len(taxonomy) < 2:
        raise ValueError(
            "Taxonomy must contain at least two intents."
        )

    return taxonomy


def prepare():
    if not GOLDEN_INPUT.exists():
        raise FileNotFoundError(
            f"Golden set not found: {GOLDEN_INPUT}"
        )

    taxonomy = load_taxonomy()

    df = pd.read_parquet(
        GOLDEN_INPUT
    )

    required = {
        "conversation_id",
        "customer_problem",
        "first_amazon_response",
        "full_transcript",
    }

    missing = (
        required - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Golden set missing columns: {sorted(missing)}"
        )

    if len(df) != 200:
        raise ValueError(
            f"Expected exactly 200 golden examples, "
            f"found {len(df)}."
        )

    output = df[
        [
            "conversation_id",
            "customer_problem",
            "first_amazon_response",
            "full_transcript",
        ]
    ].copy()

    output["gold_intent"] = ""
    output["gold_action"] = ""
    output["gold_reason"] = ""
    output["labeler_notes"] = ""

    EVAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        MANUAL_LABELS,
        index=False,
    )

    print(
        f"Created manual labeling file:\n"
        f"{MANUAL_LABELS}"
    )

    print("\nFrozen intent taxonomy:")

    for intent_id, metadata in taxonomy.items():
        print(
            f"  {intent_id}: "
            f"{metadata.get('description', '')}"
        )

    print(
        "\nAllowed gold_action values:"
    )
    print("  auto_handle")
    print("  escalate")

    print(
        "\nLabel each example using the frozen taxonomy."
    )


def finalize():
    taxonomy = load_taxonomy()

    if not MANUAL_LABELS.exists():
        raise FileNotFoundError(
            f"Manual labels not found: {MANUAL_LABELS}"
        )

    labels = pd.read_csv(
        MANUAL_LABELS
    )

    required = {
        "conversation_id",
        "gold_intent",
        "gold_action",
        "gold_reason",
    }

    missing = (
        required - set(labels.columns)
    )

    if missing:
        raise ValueError(
            f"Manual labels missing columns: {sorted(missing)}"
        )

    if len(labels) != 200:
        raise ValueError(
            f"Expected 200 manual labels, found {len(labels)}."
        )

    if labels["conversation_id"].duplicated().any():
        raise ValueError(
            "Duplicate conversation_id in manual labels."
        )

    if labels["gold_intent"].isna().any():
        raise ValueError(
            "Missing gold_intent labels."
        )

    if labels["gold_action"].isna().any():
        raise ValueError(
            "Missing gold_action labels."
        )

    if labels["gold_reason"].isna().any():
        raise ValueError(
            "Missing gold_reason labels."
        )

    labels["gold_intent"] = (
        labels["gold_intent"]
        .astype(str)
        .str.strip()
    )

    labels["gold_action"] = (
        labels["gold_action"]
        .astype(str)
        .str.strip()
    )

    labels["gold_reason"] = (
        labels["gold_reason"]
        .astype(str)
        .str.strip()
    )

    valid_intents = set(
        taxonomy.keys()
    )

    invalid_intents = (
        set(labels["gold_intent"])
        - valid_intents
    )

    if invalid_intents:
        raise ValueError(
            "Invalid gold_intent values:\n"
            f"{sorted(invalid_intents)}\n\n"
            "These do not exist in the frozen taxonomy."
        )

    invalid_actions = (
        set(labels["gold_action"])
        - VALID_ACTIONS
    )

    if invalid_actions:
        raise ValueError(
            f"Invalid gold_action values: "
            f"{sorted(invalid_actions)}"
        )

    if (
        labels["gold_reason"]
        .str.len()
        .eq(0)
        .any()
    ):
        raise ValueError(
            "gold_reason cannot be empty."
        )

    golden = pd.read_parquet(
        GOLDEN_INPUT
    )

    merged = golden.merge(
        labels[
            [
                "conversation_id",
                "gold_intent",
                "gold_action",
                "gold_reason",
                "labeler_notes",
            ]
        ],
        on="conversation_id",
        how="left",
        validate="one_to_one",
    )

    if len(merged) != 200:
        raise RuntimeError(
            "Label merge changed golden-set size."
        )

    if merged["gold_intent"].isna().any():
        raise RuntimeError(
            "Some golden examples did not receive labels."
        )

    merged.to_parquet(
        LABELED_OUTPUT,
        index=False,
    )

    print(
        f"Final labeled evaluation set written to:\n"
        f"{LABELED_OUTPUT}"
    )

    print(
        "\nLabel distribution:"
    )

    print(
        merged["gold_intent"]
        .value_counts()
        .to_string()
    )

    print(
        "\nAction distribution:"
    )

    print(
        merged["gold_action"]
        .value_counts()
        .to_string()
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "mode",
        choices=[
            "prepare",
            "finalize",
        ],
    )

    args = parser.parse_args()

    if args.mode == "prepare":
        prepare()
    else:
        finalize()


if __name__ == "__main__":
    main()
