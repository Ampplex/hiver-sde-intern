import json
import os
import random
import time

import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()


class CapabilityGuard:
    """Use an LLM to decide whether the agent can safely act autonomously."""

    def __init__(self, region=None, model_id=None):
        self.region = region or os.getenv("AWS_REGION", "us-west-2")
        self.model_id = model_id or os.getenv(
            "BEDROCK_MODEL_ID", "mistral.mistral-large-2407-v1:0"
        )
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=self.region,
            config=Config(retries={"max_attempts": 3, "mode": "standard"}),
        )

    @staticmethod
    def _clean_json(text):
        text = str(text).strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        return text.strip()

    def assess(self, customer_message, predicted_intent, historical_cases, draft_reply):
        evidence = []
        for case in historical_cases:
            evidence.append(
                {
                    "customer_problem": str(case.get("customer_problem", "")),
                    "amazonhelp_response": str(case.get("first_amazon_response", "")),
                    "dense_score": round(float(case.get("dense_score", 0.0)), 4),
                }
            )

        prompt = f"""
You are a safety/capability judge for an autonomous AmazonHelp customer-support agent.

Your job is NOT to judge whether the reply sounds good. Decide only whether this agent
can safely handle the customer's request autonomously with the capabilities it has.

AGENT CAPABILITIES:
- Can read the customer's message.
- Can retrieve historical AmazonHelp conversations.
- Can draft a response grounded in those historical conversations.
- Cannot access Amazon customer accounts, orders, billing records, delivery systems,
  refunds, private identity data, or other private backend state.
- Cannot perform account changes, issue refunds, change orders, or take other
  customer-specific operational actions.

CUSTOMER MESSAGE:
{customer_message}

PREDICTED INTENT:
{predicted_intent}

HISTORICAL EVIDENCE:
{json.dumps(evidence, ensure_ascii=False, indent=2)}

DRAFT RESPONSE:
{draft_reply if draft_reply else "<none>"}

Decision rule:
- AUTO_HANDLE only when the request can be resolved safely using the message,
  public/general knowledge demonstrated by the evidence, and the draft response,
  without private customer state or an unavailable operational action.
- ESCALATE when resolution requires looking up or changing customer-specific
  account/order/billing/delivery state, authentication, private information,
  a human-only operational action, or when the available evidence is insufficient
  or materially contradictory.
- A historical AmazonHelp message that asks the customer to contact support or
  provide details is evidence of how a human support workflow operated; it is NOT
  proof that this autonomous agent can perform that workflow.
- Do not escalate merely because the message is emotional, contains conversational
  words such as "also", or because the issue is unusual if it can still be safely
  answered from the evidence.

Return ONLY JSON:
{{"decision":"auto_handle or escalate","reason":"one concise reason"}}
"""

        last_error = None
        for attempt in range(6):
            try:
                response = self.client.converse(
                    modelId=self.model_id,
                    messages=[{"role": "user", "content": [{"text": prompt}]}],
                    inferenceConfig={"temperature": 0.0},
                )
                time.sleep(0.5 + random.uniform(0.1, 0.3))
                result = json.loads(
                    self._clean_json(response["output"]["message"]["content"][0]["text"])
                )
                decision = str(result.get("decision", "")).strip().lower()
                reason = str(result.get("reason", "")).strip()
                if decision not in {"auto_handle", "escalate"} or not reason:
                    raise ValueError("Capability guard returned invalid JSON.")
                return {"decision": decision, "reason": reason}
            except Exception as exc:
                last_error = exc
                delay = min(60.0, 2.0 * (2 ** attempt)) * random.uniform(0.8, 1.2)
                time.sleep(delay)

        return {
            "decision": "escalate",
            "reason": f"Capability guard failed: {type(last_error).__name__}",
        }
