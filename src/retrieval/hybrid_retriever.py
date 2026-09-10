import json
import os
import pickle
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


class HybridRetriever:
    """Retrieve historical AmazonHelp cases using BM25 + dense RRF.

    When an intent is supplied, it is a HARD candidate filter. The intent is
    not treated as a third soft ranking signal.
    """

    def __init__(self, index_dir="data/processed/retriever"):
        index_dir = Path(index_dir)
        self.df = pd.read_parquet(index_dir / "retrieval_corpus.parquet")
        required = {
            "conversation_id", "intent_id", "customer_problem",
            "first_amazon_response", "amazonhelp_responses", "full_transcript",
        }
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Retrieval corpus missing columns: {sorted(missing)}")

        with open(index_dir / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        self.embeddings = np.asarray(np.load(index_dir / "embeddings_normalized.npy"), dtype=np.float32)
        if len(self.df) != len(self.embeddings):
            raise ValueError("Corpus and embedding counts do not match.")

        with open(index_dir / "metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        if self.metadata.get("intent_conditioning") != "hard_filter_before_BM25_dense_RRF":
            raise ValueError("Retriever metadata does not describe the expected intent-conditioned index.")

        region = os.getenv("AWS_REGION", "us-west-2")
        self.model_id = os.getenv("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
        self.client = boto3.client("bedrock-runtime", region_name=region)

    @staticmethod
    def tokenize(text):
        return str(text).lower().split()

    def embed(self, text):
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps({"inputText": str(text)[:8000], "dimensions": 1024}),
            contentType="application/json",
            accept="application/json",
        )
        body = json.loads(response["body"].read())
        embedding = np.asarray(body["embedding"], dtype=np.float32)
        return embedding / max(float(np.linalg.norm(embedding)), 1e-12)

    @staticmethod
    def _rrf(ranked_lists, rrf_k=60):
        scores = {}
        for ranked in ranked_lists:
            for rank, local_index in enumerate(ranked):
                scores[local_index] = scores.get(local_index, 0.0) + 1.0 / (rrf_k + rank + 1)
        return scores

    def _format_result(self, index, intent_id, rrf_score, dense_score, bm25_score):
        row = self.df.iloc[index]
        return {
            "intent_id": str(row["intent_id"]),
            "conversation_id": row["conversation_id"],
            "tweet_ids": row["tweet_ids"],
            "customer_problem": row["customer_problem"],
            "first_amazon_response": row["first_amazon_response"],
            "amazonhelp_responses": row["amazonhelp_responses"],
            "full_transcript": row["full_transcript"],
            "rrf_score": float(rrf_score),
            "dense_score": float(dense_score),
            "bm25_score": float(bm25_score),
            "intent_filter": intent_id,
        }

    def search(self, query, intent_id=None, top_k=5, candidate_k=50, rrf_k=60):
        if not str(query).strip():
            return []
        allowed = np.arange(len(self.df))
        if intent_id and intent_id != "uncertain":
            allowed = np.flatnonzero(self.df["intent_id"].astype(str).eq(str(intent_id)).to_numpy())
            if len(allowed) == 0:
                return []

        query_embedding = self.embed(query)
        bm25_all = np.asarray(self.bm25.get_scores(self.tokenize(query)), dtype=np.float32)
        dense_all = self.embeddings @ query_embedding
        bm25 = bm25_all[allowed]
        dense = dense_all[allowed]
        k = min(int(candidate_k), len(allowed))
        bm25_order = np.argsort(-bm25)[:k]
        dense_order = np.argsort(-dense)[:k]
        fused = self._rrf([bm25_order, dense_order], rrf_k=rrf_k)
        local = sorted(fused, key=fused.get, reverse=True)[:top_k]

        return [
            self._format_result(
                index=int(allowed[i]),
                intent_id=intent_id,
                rrf_score=fused[i],
                dense_score=dense_all[int(allowed[i])],
                bm25_score=bm25_all[int(allowed[i])],
            )
            for i in local
        ]

    def search_bm25_only(self, query, top_k=5):
        if not str(query).strip():
            return []
        scores = np.asarray(self.bm25.get_scores(self.tokenize(query)), dtype=np.float32)
        indices = np.argsort(-scores)[:top_k]
        return [
            self._format_result(
                index=int(i), intent_id=None, rrf_score=0.0,
                dense_score=0.0, bm25_score=scores[int(i)],
            )
            for i in indices
        ]
