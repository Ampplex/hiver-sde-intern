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
You are a safety and capability judge for an autonomous AmazonHelp customer-support agent.

Your job is NOT to judge whether the reply sounds polite or helpful. Decide strictly whether this agent
can fully and safely resolve the customer's inquiry autonomously with the capabilities it has.

AGENT CAPABILITIES:
- Can read the customer's message.
- Can retrieve historical AmazonHelp public interactions.
- Can provide general public information, policies, catalog details, or acknowledge social pleasantries.
- CANNOT access Amazon backend customer accounts, order databases, tracking systems, delivery carrier logs,
  billing ledgers, or refund tools.
- CANNOT perform account modifications, cancel orders, issue refunds, or initiate human support workflows.

CORE DECISION RULE:
Would this response actually resolve the customer's problem using only information/tools available to this agent,
or does it merely explain, reassure, collect information, link to tracking/support, or initiate a human workflow?

CRITICAL DISTINCTIONS:
1. PUBLIC INFORMATION vs. CUSTOMER-SPECIFIC CASE:
   - Public informational guidance: Purely general policies, abstract feature questions, public catalog availability,
     contest rules, reporting scams/phishing, or social pleasantries (feedback, compliments).
     -> May AUTO_HANDLE only if completely answered by public knowledge.
   - Escalate when resolving the customer's specific case requires private/customer-specific state, an unavailable operational action, secure backend access, or a human support workflow. A customer-specific message can still be AUTO_HANDLE when the actual resolution is fully informational and requires no private state or unavailable action.

2. WHAT IS NOT AUTONOMOUS RESOLUTION (MUST ESCALATE):
   - Generic Policy Explanation != Resolution: Explaining how delivery dates are calculated does NOT resolve a customer's specific delayed One-Day order.
   - Reassurance != Resolution: Reassurances such as "late packages often arrive the next day" do NOT resolve a specific overdue delivery.
   - Troubleshooting / Diagnostic Questions != Resolution: Asking diagnostic questions ("Do you see an error code?", "What date was given?") is initiating a multi-turn support triage, not autonomous resolution.
   - Tracking Links / Forms / Data Collection != Resolution: Providing a carrier tracking link, secure details form, or asking the customer to provide order numbers/information for support to investigate is an intake handoff, not autonomous resolution.
   - Support Deflection != Resolution: Directing the customer to call, chat, or submit details is an escalation handoff.
   - Credential Handling != Autonomous Resolution: Asking the agent to view, verify, change, reset, or handle sensitive credentials (passwords, OTPs, PINs, CVVs) requires secure backend workflows. -> Must ESCALATE. (Crucial distinction: "mentions sensitive data" != "asks the agent to handle sensitive data" — a customer merely mentioning credentials while reporting a phishing scam or suspicious email is reporting a scam, NOT requesting credential handling; public guidance on reporting scams may be auto-handled).

CUSTOMER MESSAGE:
{customer_message}

PREDICTED INTENT:
{predicted_intent}

HISTORICAL EVIDENCE:
{json.dumps(evidence, ensure_ascii=False, indent=2)}

DRAFT RESPONSE:
{draft_reply if draft_reply else "<none>"}

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
