import os
import re

ALLOWED_DECISIONS = {"auto_handle", "escalate"}

# Keep deterministic checks only for unambiguous sensitive-data handling.
# Semantic capability decisions are delegated to CapabilityGuard.
SENSITIVE_DATA_PATTERNS = [
    r"\b(?:otp|one[- ]time password|cvv|cvc|security code)\b",
    r"\b(?:password|passcode)\b.*\b(?:send|share|tell|provide)\b",
]


def looks_sensitive_data_request(text):
    text = str(text).lower()
    return any(re.search(pattern, text) for pattern in SENSITIVE_DATA_PATTERNS)


def looks_ambiguous(text):
    # Keep this deterministic check intentionally simple. Do not use generic
    # conversational words such as "and" or "also" as ambiguity signals.
    return len(str(text).split()) > 80


def validate_generation(output):
    if not isinstance(output, dict):
        return False, "Generator returned an invalid object."
    decision = str(output.get("decision", "")).strip().lower()
    reason = str(output.get("reason", "")).strip()
    reply = output.get("draft_reply")
    if decision not in ALLOWED_DECISIONS or not reason:
        return False, "Generator returned invalid decision/reason."
    if decision == "auto_handle":
        if not isinstance(reply, str) or not reply.strip():
            return False, "Auto-handle response is missing."
        if len(reply.strip()) > 1000:
            return False, "Auto-handle response is unexpectedly long."
    elif reply not in (None, ""):
        return False, "Escalation must not contain a sendable reply."
    return True, "valid"


def apply_safety_policy(
    customer_message,
    classification,
    historical_cases,
    llm_output,
    capability_result=None,
    retrieval_threshold=None,
    min_evidence_cases=None,
):
    retrieval_threshold = float(
        retrieval_threshold
        if retrieval_threshold is not None
        else os.getenv("RETRIEVAL_EVIDENCE_THRESHOLD", "0.55")
    )
    min_evidence_cases = int(
        min_evidence_cases
        if min_evidence_cases is not None
        else os.getenv("MIN_STRONG_EVIDENCE_CASES", "2")
    )
    intent = classification.get("intent_id")

    if intent in (None, "uncertain"):
        return {"decision": "escalate", "draft_reply": None, "reason": "Classifier uncertainty."}

    if looks_ambiguous(customer_message):
        return {
            "decision": "escalate",
            "draft_reply": None,
            "reason": "Message may contain multiple or ambiguous issues.",
        }

    # Deterministic security tripwire; do not delegate sensitive credential
    # handling to a probabilistic model.
    if looks_sensitive_data_request(customer_message):
        return {
            "decision": "escalate",
            "draft_reply": None,
            "reason": "Request involves sensitive authentication information.",
        }

    if not historical_cases:
        return {
            "decision": "escalate",
            "draft_reply": None,
            "reason": "No historical evidence was retrieved.",
        }

    strong = [
        c for c in historical_cases
        if float(c.get("dense_score", 0.0)) >= retrieval_threshold
    ]
    if len(strong) < min_evidence_cases:
        return {
            "decision": "escalate",
            "draft_reply": None,
            "reason": f"Insufficient historical evidence: {len(strong)} strong case(s), need {min_evidence_cases}.",
        }

    valid, reason = validate_generation(llm_output)
    if not valid:
        return {"decision": "escalate", "draft_reply": None, "reason": reason}

    if str(llm_output["decision"]).lower() != "auto_handle":
        return {
            "decision": "escalate",
            "draft_reply": None,
            "reason": llm_output["reason"],
        }

    if not isinstance(capability_result, dict):
        return {
            "decision": "escalate",
            "draft_reply": None,
            "reason": "Capability assessment was not available.",
        }

    capability_decision = str(capability_result.get("decision", "")).strip().lower()
    capability_reason = str(capability_result.get("reason", "")).strip()
    if capability_decision != "auto_handle":
        return {
            "decision": "escalate",
            "draft_reply": None,
            "reason": capability_reason or "Capability guard requires escalation.",
        }

    return {
        "decision": "auto_handle",
        "draft_reply": llm_output["draft_reply"].strip(),
        "reason": llm_output["reason"],
    }
