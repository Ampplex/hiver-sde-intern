import hashlib
import json
import os
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

CASES = Path("data/processed/support_cases_train.parquet")
EMBEDDINGS = Path("data/processed/diagnostic_embeddings_train.npy")
IDS = Path("data/processed/diagnostic_embedding_ids.json")
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
MODEL = os.getenv("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")


def fingerprint(ids):
    return hashlib.sha256("|".join(ids).encode("utf-8")).hexdigest()


def get_embedding(client, text):
    response = client.invoke_model(
        modelId=MODEL,
        body=json.dumps({"inputText": str(text)[:8000], "dimensions": 1024}),
        accept="application/json",
        contentType="application/json",
    )
    return np.asarray(json.loads(response["body"].read())["embedding"], dtype=np.float32)


def main():
    df = pd.read_parquet(CASES)
    sample = df.sample(n=min(1000, len(df)), random_state=42).reset_index(drop=True)
    sample_ids = sample["conversation_id"].astype(str).tolist()
    fp = fingerprint(sample_ids)

    if len(sample) < 2:
        raise ValueError("Need at least two training cases for similarity diagnostics.")

    cache_ok = EMBEDDINGS.exists() and IDS.exists()
    if cache_ok:
        with open(IDS, "r", encoding="utf-8") as f:
            old = json.load(f)
        cache_ok = old.get("fingerprint") == fp and old.get("ids") == sample_ids

    if cache_ok:
        embeddings = np.load(EMBEDDINGS)
    else:
        client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
        embeddings = np.vstack([get_embedding(client, text) for text in sample["customer_problem"]])
        np.save(EMBEDDINGS, embeddings)
        with open(IDS, "w", encoding="utf-8") as f:
            json.dump({"fingerprint": fp, "ids": sample_ids}, f, indent=2)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-12)
    sims = normalized @ normalized.T
    np.fill_diagonal(sims, -np.inf)
    nn = np.max(sims, axis=1)
    second = np.partition(sims, -2, axis=1)[:, -2]

    print(f"Sample size: {len(sample):,}")

    for name, values in [("nearest", nn), ("second_nearest", second)]:
        print(f"\n{name}")
        for p in [50, 75, 90, 95, 99]:
            print(f"P{p}: {np.percentile(values, p):.4f}")

    print("\nThis script is diagnostic only. It does not tune thresholds on the golden set.")


if __name__ == "__main__":
    main()
