import hashlib
import json
import pickle
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# Configuration
# ============================================================

import os

INPUT_PATH = Path("data/processed/support_cases_train.parquet")
OUTPUT_DIR = Path("data/processed/retriever")

SAMPLE_SIZE = 5000
RANDOM_STATE = 42

AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
EMBEDDING_MODEL = os.getenv("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")


# ============================================================
# Utilities
# ============================================================

def tokenize(text):
    """
    Simple tokenization for BM25.

    Lowercasing keeps lexical matching case-insensitive while
    preserving useful product/entity words.
    """
    return str(text).lower().split()


def create_corpus_fingerprint(df):
    """
    Create a fingerprint for the exact retrieval corpus.

    This prevents accidentally reusing embeddings generated
    for a different sample/corpus.
    """
    values = (
        df["conversation_id"].astype(str)
        + "|"
        + df["customer_problem"].astype(str)
    ).tolist()

    raw = "|".join(values).encode("utf-8")

    return hashlib.sha256(raw).hexdigest()


# ============================================================
# Bedrock
# ============================================================

def create_bedrock_client():
    return boto3.client(
        "bedrock-runtime",
        region_name=AWS_REGION
    )


def get_embedding(client, text):
    response = client.invoke_model(
        modelId=EMBEDDING_MODEL,
        body=json.dumps({
            "inputText": str(text),
            "dimensions": 1024
        }),
        contentType="application/json",
        accept="application/json"
    )

    body = json.loads(
        response["body"].read()
    )

    return np.array(
        body["embedding"],
        dtype=np.float32
    )


import time

def build_embeddings(df):
    client = create_bedrock_client()

    embeddings = []

    total = len(df)

    for i, text in enumerate(
        df["customer_problem"]
    ):

        retries = 5
        success = False
        while not success and retries > 0:
            try:
                embedding = get_embedding(
                    client,
                    text
                )
                success = True
            except Exception as e:
                if "ThrottlingException" in str(e):
                    time.sleep(2)
                    retries -= 1
                    if retries == 0:
                        raise e
                else:
                    raise e

        embeddings.append(embedding)
        time.sleep(0.1)  # small buffer to prevent hitting rate limits

        if (i + 1) % 100 == 0 or i + 1 == total:
            print(
                f"Embedded {i + 1}/{total}"
            )

    return np.vstack(embeddings)


# ============================================================
# Main
# ============================================================

def main():

    print("Loading support cases...")

    df = pd.read_parquet(
        INPUT_PATH
    )

    required_columns = [
        "conversation_id",
        "tweet_ids",
        "customer_problem",
        "first_amazon_response",
        "full_transcript"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print(
        f"Total support cases: {len(df):,}"
    )

    # --------------------------------------------------------
    # Select historical retrieval corpus
    # --------------------------------------------------------

    sample_size = min(
        SAMPLE_SIZE,
        len(df)
    )

    memory_df = df.sample(
        n=sample_size,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    print(
        f"Cases selected for retrieval: "
        f"{len(memory_df):,}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Fingerprint the exact corpus
    # --------------------------------------------------------

    corpus_fingerprint = (
        create_corpus_fingerprint(
            memory_df
        )
    )

    print(
        f"Corpus fingerprint: "
        f"{corpus_fingerprint[:12]}..."
    )

    # --------------------------------------------------------
    # Save retrieval corpus
    # --------------------------------------------------------

    corpus_path = (
        OUTPUT_DIR /
        "retrieval_corpus.parquet"
    )

    memory_df.to_parquet(
        corpus_path,
        index=False
    )

    print(
        f"Saved corpus to {corpus_path}"
    )

    # --------------------------------------------------------
    # Build BM25
    # --------------------------------------------------------

    print("\nBuilding BM25 index...")

    tokenized_corpus = [
        tokenize(text)
        for text in memory_df[
            "customer_problem"
        ]
    ]

    bm25 = BM25Okapi(
        tokenized_corpus
    )

    bm25_path = (
        OUTPUT_DIR /
        "bm25.pkl"
    )

    with open(bm25_path, "wb") as f:
        pickle.dump(
            bm25,
            f
        )

    print(
        f"Saved BM25 index to {bm25_path}"
    )

    # --------------------------------------------------------
    # Embedding paths
    # --------------------------------------------------------

    embeddings_path = (
        OUTPUT_DIR /
        "embeddings_normalized.npy"
    )

    metadata_path = (
        OUTPUT_DIR /
        "metadata.json"
    )

    # --------------------------------------------------------
    # Check existing embedding cache
    # --------------------------------------------------------

    embeddings = None

    if (
        embeddings_path.exists()
        and metadata_path.exists()
    ):

        print(
            "\nFound existing embedding cache."
        )

        with open(
            metadata_path,
            "r"
        ) as f:
            old_metadata = json.load(f)

        same_corpus = (
            old_metadata.get(
                "corpus_fingerprint"
            )
            == corpus_fingerprint
        )

        same_size = (
            old_metadata.get(
                "sample_size"
            )
            == len(memory_df)
        )

        same_model = (
            old_metadata.get(
                "model"
            )
            == EMBEDDING_MODEL
        )

        if (
            same_corpus
            and same_size
            and same_model
        ):

            cached = np.load(
                embeddings_path
            )

            if len(cached) == len(memory_df):

                embeddings = cached

                print(
                    "Embedding cache matches "
                    "the current corpus. Reusing it."
                )

        if embeddings is None:

            print(
                "Embedding cache does not match "
                "the current corpus."
            )

            print(
                "Generating fresh embeddings..."
            )

    # --------------------------------------------------------
    # Generate embeddings if necessary
    # --------------------------------------------------------

    if embeddings is None:

        print(
            "\nGenerating Titan embeddings..."
        )

        embeddings = build_embeddings(
            memory_df
        )

        # Normalize once so cosine similarity
        # becomes a dot product during retrieval.
        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True
        )

        embeddings = (
            embeddings /
            np.maximum(norms, 1e-12)
        )

        np.save(
            embeddings_path,
            embeddings
        )

        print(
            f"Saved embeddings to "
            f"{embeddings_path}"
        )

    else:

        # Cached embeddings are already normalized.
        print(
            f"Embedding matrix shape: "
            f"{embeddings.shape}"
        )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = {
        "model": EMBEDDING_MODEL,
        "sample_size": len(memory_df),
        "random_state": RANDOM_STATE,
        "embedding_dimension": int(
            embeddings.shape[1]
        ),
        "corpus_fingerprint": corpus_fingerprint,
        "retrieval_text": "customer_problem",
        "stored_fields": [
            "conversation_id",
            "tweet_ids",
            "customer_problem",
            "first_amazon_response",
            "full_transcript"
        ]
    }

    with open(
        metadata_path,
        "w"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2
        )

    print(
        f"Saved metadata to {metadata_path}"
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 45)
    print("RETRIEVER BUILD COMPLETE")
    print("=" * 45)

    print(
        f"Cases indexed: {len(memory_df):,}"
    )

    print(
        f"BM25 index: {bm25_path}"
    )

    print(
        f"Embeddings: {embeddings_path}"
    )

    print(
        f"Corpus: {corpus_path}"
    )

    print(
        f"Embedding dimension: "
        f"{embeddings.shape[1]}"
    )


if __name__ == "__main__":
    main()
