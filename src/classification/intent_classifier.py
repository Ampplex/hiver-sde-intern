import json
import os
from pathlib import Path

import boto3
import numpy as np
from dotenv import load_dotenv

load_dotenv()


class IntentClassifier:
    def __init__(self, classifier_dir="data/processed/classifier"):
        d = Path(classifier_dir)
        self.prototypes = np.asarray(np.load(d / "intent_prototypes.npy"), dtype=np.float32)
        with open(d / "intent_list.json", "r", encoding="utf-8") as f:
            self.intents = json.load(f)
        if self.prototypes.ndim != 2 or len(self.prototypes) != len(self.intents):
            raise ValueError("Classifier prototype/list mismatch.")
        if len(self.intents) < 2:
            raise ValueError("At least two intents are required.")
        region = os.getenv("AWS_REGION", "us-west-2")
        self.model_id = os.getenv("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
        self.client = boto3.client("bedrock-runtime", region_name=region)

    def embed(self, text):
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps({"inputText": str(text)[:8000], "dimensions": self.prototypes.shape[1]}),
            contentType="application/json", accept="application/json",
        )
        body = json.loads(response["body"].read())
        embedding = np.asarray(body["embedding"], dtype=np.float32)
        if embedding.shape[0] != self.prototypes.shape[1]:
            raise ValueError("Embedding dimension does not match classifier.")
        return embedding / max(float(np.linalg.norm(embedding)), 1e-12)

    def predict(self, text, threshold=None, margin=None):
        threshold = float(threshold if threshold is not None else os.getenv("INTENT_SIMILARITY_THRESHOLD", "0.45"))
        margin_threshold = float(margin if margin is not None else os.getenv("INTENT_MARGIN_THRESHOLD", "0.05"))
        embedding = self.embed(text)
        sims = self.prototypes @ embedding
        order = np.argsort(-sims)
        top1, top2 = float(sims[order[0]]), float(sims[order[1]])
        gap = top1 - top2
        uncertain = top1 < threshold or gap < margin_threshold
        predicted = self.intents[int(order[0])]
        return {
            "intent_id": "uncertain" if uncertain else predicted,
            "predicted_intent_id": predicted,
            "similarity_score": top1,
            "second_similarity_score": top2,
            "margin": gap,
            "uncertain": uncertain,
            "threshold": threshold,
            "margin_threshold": margin_threshold,
        }
