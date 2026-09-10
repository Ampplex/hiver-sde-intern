import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()

INPUT_PATH = Path("data/processed/support_cases_train.parquet")
ASSIGNMENTS_PATH = Path("data/processed/intent_assignments.parquet")
OUTPUT_DIR = Path("data/processed/retriever")
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
EMBEDDING_MODEL = os.getenv("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")


def tokenize(text):
    return str(text).lower().split()








def fingerprint(df):
    values = [
        f"{row.conversation_id}|{row.intent_id}|{row.customer_problem}"
        for row in df.itertuples()
    ]
    return hashlib.sha256("|".join(values).encode("utf-8")).hexdigest()


def main():
    cases = pd.read_parquet(INPUT_PATH)
    assignments = pd.read_parquet(ASSIGNMENTS_PATH)

    if assignments["conversation_id"].duplicated().any():
        raise ValueError("Duplicate conversation_id in intent assignments.")

    required = {"conversation_id", "intent_id"}
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(f"Missing assignment columns: {sorted(missing)}")

    memory = cases.merge(
        assignments[["conversation_id", "intent_id"]],
        on="conversation_id",
        how="inner",
        validate="one_to_one",
    )
    case_ids = set(cases["conversation_id"].astype(str))
    assignment_ids = set(assignments["conversation_id"].astype(str))

    if case_ids != assignment_ids:
        missing_assignments = case_ids - assignment_ids
        extra_assignments = assignment_ids - case_ids
        raise RuntimeError(
            f"Retriever requires complete training coverage; "
            f"missing={len(missing_assignments)}, "
            f"extra={len(extra_assignments)}."
        )

    if len(memory) != len(assignments):
        raise RuntimeError("Every assigned training case must exist in the retrieval corpus.")
    if memory["conversation_id"].duplicated().any():
        raise RuntimeError("Retrieval corpus contains duplicate conversation IDs.")

    memory = memory.sort_values("conversation_id").reset_index(drop=True)
    corpus_fingerprint = fingerprint(memory)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    corpus_path = OUTPUT_DIR / "retrieval_corpus.parquet"
    bm25_path = OUTPUT_DIR / "bm25.pkl"
    embeddings_path = OUTPUT_DIR / "embeddings_normalized.npy"
    metadata_path = OUTPUT_DIR / "metadata.json"

    memory.to_parquet(corpus_path, index=False)

    with open(bm25_path, "wb") as f:
        pickle.dump(BM25Okapi([tokenize(x) for x in memory["customer_problem"]]), f)

    embeddings = None
    if embeddings_path.exists() and metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if (
            meta.get("corpus_fingerprint") == corpus_fingerprint
            and meta.get("model") == EMBEDDING_MODEL
            and meta.get("row_count") == len(memory)
        ):
            cached = np.load(embeddings_path)
            if (
                cached.ndim == 2
                and cached.shape[0] == len(memory)
                and cached.shape[1] == 1024
            ):
                embeddings = cached

    if embeddings is None:
        training_embs_path = Path("data/processed/intent_assignment_embeddings.npy")
        training_ids_path = Path("data/processed/intent_assignment_ids.json")
        
        if not training_embs_path.exists() or not training_ids_path.exists():
            raise FileNotFoundError("Missing training embeddings. Run scripts/04_discover_intents.py first.")
            
        print("Loading embeddings from 04_discover_intents.py...")
        assignment_embs = np.load(training_embs_path)
        with open(training_ids_path, "r", encoding="utf-8") as f:
            assignment_ids_list = [str(x) for x in json.load(f)]
            
        id_to_idx = {cid: idx for idx, cid in enumerate(assignment_ids_list)}
        
        ordered_embs = []
        for cid in memory["conversation_id"]:
            cid_str = str(cid)
            if cid_str not in id_to_idx:
                raise RuntimeError(f"Missing embedding for conversation {cid_str}")
            ordered_embs.append(assignment_embs[id_to_idx[cid_str]])
            
        embeddings = np.vstack(ordered_embs).astype(np.float32)
        np.save(embeddings_path, embeddings)

    metadata = {
        "model": EMBEDDING_MODEL,
        "row_count": int(len(memory)),
        "corpus_fingerprint": corpus_fingerprint,
        "retrieval_text": "customer_problem",
        "intent_conditioning": "hard_filter_before_BM25_dense_RRF",
        "stored_fields": [
            "conversation_id", "intent_id", "tweet_ids", "customer_problem",
            "first_amazon_response", "amazonhelp_responses", "full_transcript",
        ],
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nRETRIEVER BUILD COMPLETE")
    print(f"Indexed assigned training cases: {len(memory):,}")
    print(f"Corpus fingerprint: {corpus_fingerprint}")
    print(f"Corpus: {corpus_path}")
    print(f"BM25: {bm25_path}")
    print(f"Embeddings: {embeddings_path}")


if __name__ == "__main__":
    main()
