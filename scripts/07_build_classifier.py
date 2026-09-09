import json
from pathlib import Path

import numpy as np
import pandas as pd


ASSIGNMENTS_PATH = Path(
    "data/processed/intent_assignments.parquet"
)

EMBEDDINGS_PATH = Path(
    "data/processed/intent_assignment_embeddings.npy"
)

EMBEDDING_IDS_PATH = Path(
    "data/processed/intent_assignment_ids.json"
)

OUTPUT_DIR = Path(
    "data/processed/classifier"
)


def normalize(vector):
    vector = np.asarray(
        vector,
        dtype=np.float32,
    )

    norm = np.linalg.norm(vector)

    return vector / max(
        float(norm),
        1e-12,
    )


def main():

    print(
        "Loading intent assignments..."
    )

    df = pd.read_parquet(
        ASSIGNMENTS_PATH
    )

    if df.empty:
        raise ValueError(
            "Intent assignments are empty."
        )

    required = {
        "conversation_id",
        "intent_id",
        "customer_problem",
    }

    missing = (
        required - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    df = df.dropna(
        subset=["intent_id"]
    ).copy()

    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Missing embedding cache: "
            f"{EMBEDDINGS_PATH}\n"
            "Run scripts/04_discover_intents.py again."
        )

    if not EMBEDDING_IDS_PATH.exists():
        raise FileNotFoundError(
            f"Missing embedding ID file: "
            f"{EMBEDDING_IDS_PATH}\n"
            "Run scripts/04_discover_intents.py again."
        )

    embeddings = np.load(
        EMBEDDINGS_PATH
    ).astype(np.float32)

    with open(
        EMBEDDING_IDS_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        embedding_ids = json.load(f)

    if len(embeddings) != len(df):
        raise ValueError(
            "Embedding count does not match "
            "intent assignment count."
        )

    if len(embedding_ids) != len(df):
        raise ValueError(
            "Embedding ID count does not match "
            "intent assignment count."
        )

    assignment_ids = (
        df["conversation_id"]
        .astype(str)
        .tolist()
    )

    if assignment_ids != embedding_ids:
        raise ValueError(
            "Embedding IDs do not match the "
            "current assignment order."
        )

    print(
        f"Assigned cases: {len(df):,}"
    )

    intents = sorted(
        df["intent_id"].unique()
    )

    if len(intents) < 2:
        raise ValueError(
            "At least two intents are required."
        )

    prototypes = []

    for intent_id in intents:

        mask = (
            df["intent_id"]
            == intent_id
        ).to_numpy()

        intent_embeddings = embeddings[
            mask
        ]

        if len(intent_embeddings) < 3:
            print(
                f"WARNING: {intent_id} has "
                f"only {len(intent_embeddings)} "
                "training examples."
            )

        prototype = intent_embeddings.mean(
            axis=0
        )

        prototype = normalize(
            prototype
        )

        prototypes.append(
            prototype
        )

        print(
            f"{intent_id}: "
            f"{len(intent_embeddings)} examples"
        )

    prototypes = np.vstack(
        prototypes
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    prototype_path = (
        OUTPUT_DIR
        / "intent_prototypes.npy"
    )

    intent_list_path = (
        OUTPUT_DIR
        / "intent_list.json"
    )

    np.save(
        prototype_path,
        prototypes,
    )

    with open(
        intent_list_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            intents,
            f,
            indent=2,
        )

    print(
        "\nClassifier build complete."
    )

    print(
        f"Intents: {len(intents)}"
    )

    print(
        f"Prototype shape: "
        f"{prototypes.shape}"
    )

    print(
        f"Saved:\n{prototype_path}"
    )

    print(
        f"Saved:\n{intent_list_path}"
    )


if __name__ == "__main__":
    main()
