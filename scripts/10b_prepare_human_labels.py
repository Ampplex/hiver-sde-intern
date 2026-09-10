import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

EVAL_DIR = Path("data/processed/eval")
GOLDEN_INPUT = EVAL_DIR / "golden_set.parquet"
MANUAL_LABELS = EVAL_DIR / "golden_labels_manual.csv"
LABELED_OUTPUT = EVAL_DIR / "golden_set_labeled.parquet"
TAXONOMY_PATH = Path("data/processed/intent_taxonomy.json")
VALID_ACTIONS = {"auto_handle", "escalate"}


def load_taxonomy():
    if not TAXONOMY_PATH.exists():
        raise FileNotFoundError("Run scripts/04_discover_intents.py and freeze the taxonomy first.")
    with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)
    if not isinstance(taxonomy, dict) or len(taxonomy) < 2:
        raise ValueError("Frozen taxonomy must contain at least two intents.")
    return taxonomy


def taxonomy_hash(taxonomy):
    return hashlib.sha256(json.dumps(taxonomy, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def prepare():
    if MANUAL_LABELS.exists():
        raise FileExistsError(f"Refusing to overwrite existing labels: {MANUAL_LABELS}")
    taxonomy = load_taxonomy()
    golden = pd.read_parquet(GOLDEN_INPUT)
    if len(golden) != 200 or golden["conversation_id"].duplicated().any():
        raise ValueError("Golden set must contain exactly 200 unique conversation IDs.")

    out = golden[["conversation_id", "customer_problem", "first_amazon_response", "full_transcript"]].copy()
    out["gold_intent"] = ""
    out["gold_action"] = ""
    out["gold_reason"] = ""
    out["labeler_notes"] = ""
    out["taxonomy_hash"] = taxonomy_hash(taxonomy)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(MANUAL_LABELS, index=False)
    print(f"Created {MANUAL_LABELS}")
    print("Frozen intents:")
    for k, v in taxonomy.items():
        print(f"  {k}: {v['description']}")


def finalize():
    taxonomy = load_taxonomy()
    if not MANUAL_LABELS.exists():
        raise FileNotFoundError(f"Missing {MANUAL_LABELS}")
    labels = pd.read_csv(MANUAL_LABELS)
    required = {"conversation_id", "gold_intent", "gold_action", "gold_reason", "taxonomy_hash"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Manual labels missing columns: {sorted(missing)}")
    if len(labels) != 200 or labels["conversation_id"].duplicated().any():
        raise ValueError("Manual labels must contain exactly 200 unique conversation IDs.")

    expected_hash = taxonomy_hash(taxonomy)
    hashes = labels["taxonomy_hash"].astype(str).str.strip()
    if hashes.nunique() != 1 or hashes.iloc[0] != expected_hash:
        raise ValueError("Taxonomy hash mismatch: taxonomy changed after labels were prepared.")

    for col in ["gold_intent", "gold_action", "gold_reason"]:
        labels[col] = labels[col].astype(str).str.strip()
        if labels[col].eq("").any():
            raise ValueError(f"{col} contains empty labels.")

    invalid_intents = set(labels["gold_intent"]) - set(taxonomy)
    if invalid_intents:
        raise ValueError(f"Invalid gold intents: {sorted(invalid_intents)}")
    invalid_actions = set(labels["gold_action"]) - VALID_ACTIONS
    if invalid_actions:
        raise ValueError(f"Invalid gold actions: {sorted(invalid_actions)}")

    golden = pd.read_parquet(GOLDEN_INPUT)
    merged = golden.merge(labels[["conversation_id", "gold_intent", "gold_action", "gold_reason", "labeler_notes"]], on="conversation_id", how="left", validate="one_to_one")
    if len(merged) != 200 or merged[["gold_intent", "gold_action", "gold_reason"]].isna().any().any():
        raise RuntimeError("Golden labeling merge is incomplete.")
    merged.to_parquet(LABELED_OUTPUT, index=False)
    print(f"Wrote {LABELED_OUTPUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "finalize"])
    args = parser.parse_args()
    prepare() if args.mode == "prepare" else finalize()
