import json
import os
from pathlib import Path

import boto3
import numpy as np
from dotenv import load_dotenv

load_dotenv()


class IntentClassifier:

    def __init__(
        self,
        classifier_dir="data/processed/classifier",
    ):
        classifier_dir = Path(classifier_dir)

        self.prototypes = np.asarray(
            np.load(
                classifier_dir / "intent_prototypes.npy"
            ),
            dtype=np.float32,
        )

        with open(
            classifier_dir / "intent_list.json",
            "r",
            encoding="utf-8",
        ) as f:
            self.intents = json.load(f)

        if self.prototypes.ndim != 2:
            raise ValueError(
                "Intent prototypes must be a 2D matrix."
            )

        if len(self.prototypes) != len(self.intents):
            raise ValueError(
                "Number of prototypes does not match "
                "number of intents."
            )

        if len(self.intents) < 2:
            raise ValueError(
                "At least two intents are required."
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

        if embedding.shape[0] != self.prototypes.shape[1]:
            raise ValueError(
                f"Embedding dimension {embedding.shape[0]} "
                f"does not match classifier dimension "
                f"{self.prototypes.shape[1]}."
            )

        norm = np.linalg.norm(embedding)

        embedding = embedding / max(
            float(norm),
            1e-12,
        )

        return embedding

    def predict(
        self,
        text,
        threshold=None,
        margin=None,
    ):

        threshold = float(
            threshold
            if threshold is not None
            else os.getenv(
                "INTENT_SIMILARITY_THRESHOLD",
                "0.45",
            )
        )

        margin = float(
            margin
            if margin is not None
            else os.getenv(
                "INTENT_MARGIN_THRESHOLD",
                "0.05",
            )
        )

        embedding = self.embed(text)

        similarities = (
            self.prototypes @ embedding
        )

        order = np.argsort(
            -similarities
        )

        best_index = int(order[0])
        second_index = int(order[1])

        top1 = float(
            similarities[best_index]
        )

        top2 = float(
            similarities[second_index]
        )

        gap = top1 - top2

        uncertain = (
            top1 < threshold
            or gap < margin
        )

        return {
            "intent_id": (
                "uncertain"
                if uncertain
                else self.intents[best_index]
            ),
            "predicted_intent_id": self.intents[
                best_index
            ],
            "similarity_score": top1,
            "second_similarity_score": top2,
            "margin": gap,
            "uncertain": uncertain,
            "threshold": threshold,
            "margin_threshold": margin,
        }

    def predict_top_k(
        self,
        text,
        k=3,
    ):

        embedding = self.embed(text)

        similarities = (
            self.prototypes @ embedding
        )

        indices = np.argsort(
            -similarities
        )[:k]

        return [
            {
                "intent_id": self.intents[
                    int(index)
                ],
                "similarity_score": float(
                    similarities[index]
                ),
            }
            for index in indices
        ]
