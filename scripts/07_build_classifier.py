import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ASSIGNMENTS_PATH = Path("data/processed/intent_assignments.parquet")
TRAIN_PATH = Path("data/processed/support_cases_train.parquet")
EMBEDDINGS_PATH = Path("data/processed/intent_assignment_embeddings.npy")
EMBEDDING_IDS_PATH = Path("data/processed/intent_assignment_ids.json")
TAXONOMY_PATH = Path("data/processed/intent_taxonomy.json")
OUTPUT_DIR = Path("data/processed/classifier")


def normalize(v):
    v = np.asarray(v, dtype=np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def main():
    df = pd.read_parquet(ASSIGNMENTS_PATH)
    train = pd.read_parquet(TRAIN_PATH)
    if df.empty:
        raise ValueError("Intent assignments are empty.")

    required = {"conversation_id", "intent_id", "customer_problem"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing assignment columns: {sorted(missing)}")
    if df["conversation_id"].duplicated().any():
        raise ValueError("Duplicate conversation_id in assignments.")

    train_ids = set(train["conversation_id"].astype(str))
    assignment_ids = set(df["conversation_id"].astype(str))
    if assignment_ids != train_ids:
        missing = train_ids - assignment_ids
        extra = assignment_ids - train_ids
        raise RuntimeError(
            f"Classifier requires complete training coverage; "
            f"missing={len(missing)}, extra={len(extra)}. "
            "Fix the taxonomy/assignments before building the classifier."
        )

    # All training cases must have valid frozen-taxonomy assignments.
    covered_rate = len(assignment_ids) / len(train_ids)
    print(f"Classifier training coverage: {covered_rate:.1%} ({len(df):,}/{len(train_ids):,})")

    embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)
    with open(EMBEDDING_IDS_PATH, "r", encoding="utf-8") as f:
        embedding_ids = [str(x) for x in json.load(f)]
    current_ids = df["conversation_id"].astype(str).tolist()
    if len(embeddings) != len(df) or embedding_ids != current_ids:
        raise ValueError("Classifier embedding cache does not match assignment order.")

    with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)
    valid_intents = set(taxonomy)
    assigned_intents = set(df["intent_id"])

    if not assigned_intents.issubset(valid_intents):
        raise RuntimeError("Assignments contain intent IDs absent from frozen taxonomy.")

    missing_taxonomy_intents = valid_intents - assigned_intents
    if missing_taxonomy_intents:
        raise RuntimeError(
            "Frozen taxonomy contains intents with no training assignments: "
            f"{sorted(missing_taxonomy_intents)}"
        )

    intents = sorted(assigned_intents)
    prototypes = []
    counts = {}
    for intent_id in intents:
        mask = df["intent_id"].eq(intent_id).to_numpy()
        count = int(mask.sum())
        if count == 0:
            raise RuntimeError(f"Intent {intent_id} has no assigned examples.")
        counts[intent_id] = count
        prototypes.append(normalize(embeddings[mask].mean(axis=0)))

    prototypes = np.vstack(prototypes).astype(np.float32)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "intent_prototypes.npy", prototypes)
    with open(OUTPUT_DIR / "intent_list.json", "w", encoding="utf-8") as f:
        json.dump(intents, f, indent=2)

    meta = {
        "taxonomy_hash": hashlib.sha256(
            json.dumps(taxonomy, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "training_cases": int(len(train)),
        "assigned_cases": int(len(df)),
        "coverage": float(covered_rate),
        "intent_counts": counts,
    }
    with open(OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Built classifier with {len(intents)} intents.")
    for intent_id in intents:
        print(f"  {intent_id}: {counts[intent_id]}")


if __name__ == "__main__":
    main()
