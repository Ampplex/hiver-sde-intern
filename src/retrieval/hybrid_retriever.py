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
    """
    Hybrid historical-case retriever.

    Retrieval signals:
      1. BM25 lexical similarity
      2. Dense embedding similarity
      3. Intent-aware ranking

    These signals are combined using Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        index_dir="data/processed/retriever",
        classifier_dir="data/processed/classifier",
    ):

        index_dir = Path(index_dir)
        classifier_dir = Path(classifier_dir)

        self.df = pd.read_parquet(
            index_dir / "retrieval_corpus.parquet"
        )

        with open(
            index_dir / "bm25.pkl",
            "rb",
        ) as f:
            self.bm25 = pickle.load(f)

        self.embeddings = np.asarray(
            np.load(
                index_dir
                / "embeddings_normalized.npy"
            ),
            dtype=np.float32,
        )

        if len(self.df) != len(
            self.embeddings
        ):
            raise ValueError(
                "Corpus and embedding counts do not match."
            )

        self.prototypes = None
        self.intents = []

        prototype_path = (
            classifier_dir
            / "intent_prototypes.npy"
        )

        intent_list_path = (
            classifier_dir
            / "intent_list.json"
        )

        if (
            prototype_path.exists()
            and intent_list_path.exists()
        ):

            self.prototypes = np.load(
                prototype_path
            )

            with open(
                intent_list_path,
                "r",
                encoding="utf-8",
            ) as f:
                self.intents = json.load(f)

            if len(self.prototypes) != len(
                self.intents
            ):
                raise ValueError(
                    "Intent prototypes and intent list "
                    "do not match."
                )

        region = os.getenv(
            "AWS_REGION",
            "us-west-2",
        )

        self.model_id = os.getenv(
            "BEDROCK_EMBED_MODEL",
            "amazon.titan-embed-text-v2:0",
        )

        self.client = boto3.client(
            "bedrock-runtime",
            region_name=region,
        )

    @staticmethod
    def tokenize(text):
        return str(text).lower().split()

    def embed(self, text):

        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(
                {
                    "inputText": str(text),
                    "dimensions": 1024,
                }
            ),
            contentType="application/json",
            accept="application/json",
        )

        body = json.loads(
            response["body"].read()
        )

        embedding = np.asarray(
            body["embedding"],
            dtype=np.float32,
        )

        norm = np.linalg.norm(
            embedding
        )

        return embedding / max(
            float(norm),
            1e-12,
        )

    @staticmethod
    def reciprocal_rank_fusion(
        ranked_lists,
        rrf_k=60,
    ):

        scores = {}

        for ranked_list in ranked_lists:

            for rank, index in enumerate(
                ranked_list
            ):

                scores[index] = (
                    scores.get(index, 0.0)
                    + 1.0
                    / (
                        rrf_k
                        + rank
                        + 1
                    )
                )

        return scores

    def _format_result(
        self,
        index,
        intent_id,
        rrf_score,
        dense_score,
        bm25_score,
        intent_score,
    ):

        row = self.df.iloc[index]

        return {
            "intent_id": intent_id,
            "conversation_id": row[
                "conversation_id"
            ],
            "tweet_ids": row[
                "tweet_ids"
            ],
            "customer_problem": row[
                "customer_problem"
            ],
            "first_amazon_response": row[
                "first_amazon_response"
            ],
            "full_transcript": row[
                "full_transcript"
            ],
            "rrf_score": float(
                rrf_score
            ),
            "dense_score": float(
                dense_score
            ),
            "bm25_score": float(
                bm25_score
            ),
            "intent_score": float(
                intent_score
            ),
        }

    def search(
        self,
        query,
        intent_id=None,
        top_k=5,
        candidate_k=50,
        rrf_k=60,
    ):

        candidate_k = min(
            candidate_k,
            len(self.df),
        )

        # -----------------------------
        # BM25
        # -----------------------------

        bm25_scores = np.asarray(
            self.bm25.get_scores(
                self.tokenize(query)
            )
        )

        bm25_candidates = np.argsort(
            -bm25_scores
        )[:candidate_k]

        # -----------------------------
        # Dense retrieval
        # -----------------------------

        query_embedding = self.embed(
            query
        )

        dense_scores = (
            self.embeddings
            @ query_embedding
        )

        dense_candidates = np.argsort(
            -dense_scores
        )[:candidate_k]

        ranked_lists = [
            bm25_candidates,
            dense_candidates,
        ]

        # -----------------------------
        # Intent-aware retrieval
        # -----------------------------

        intent_scores = np.zeros(
            len(self.df),
            dtype=np.float32,
        )

        if (
            intent_id
            and intent_id != "uncertain"
            and self.prototypes is not None
            and intent_id in self.intents
        ):

            intent_index = (
                self.intents.index(
                    intent_id
                )
            )

            intent_prototype = (
                self.prototypes[
                    intent_index
                ]
            )

            intent_scores = (
                self.embeddings
                @ intent_prototype
            )

            intent_candidates = np.argsort(
                -intent_scores
            )[:candidate_k]

            ranked_lists.append(
                intent_candidates
            )

        # -----------------------------
        # RRF
        # -----------------------------

        rrf_scores = (
            self.reciprocal_rank_fusion(
                ranked_lists,
                rrf_k=rrf_k,
            )
        )

        final_indices = sorted(
            rrf_scores,
            key=rrf_scores.get,
            reverse=True,
        )[:top_k]

        return [
            self._format_result(
                index=int(index),
                intent_id=intent_id,
                rrf_score=rrf_scores[index],
                dense_score=dense_scores[index],
                bm25_score=bm25_scores[index],
                intent_score=intent_scores[index],
            )
            for index in final_indices
        ]

    def search_bm25_only(
        self,
        query,
        top_k=5,
    ):

        scores = np.asarray(
            self.bm25.get_scores(
                self.tokenize(query)
            )
        )

        indices = np.argsort(
            -scores
        )[:top_k]

        return [
            self._format_result(
                index=int(index),
                intent_id=None,
                rrf_score=0.0,
                dense_score=0.0,
                bm25_score=scores[index],
                intent_score=0.0,
            )
            for index in indices
        ]
