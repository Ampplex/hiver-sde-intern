import os
import re

ALLOWED_DECISIONS = {"auto_handle", "escalate"}

# Deterministic hard stops only. These patterns identify requests that
# explicitly require private/customer-specific state or secure handling.
ACCOUNT_SPECIFIC_PATTERNS = [
    r"\bcheck\s+(?:my\s+)?order\b",
    r"\blook\s+up\s+(?:my\s+)?order\b",
    r"\btrack\s+(?:my\s+)?order\b",
    r"\bcheck\s+(?:my\s+)?account\b",
    r"\baccess\s+(?:my\s+)?account\b",
    r"\border\s*(?:#|number|id)\s*[\w-]+\b",
    r"\bverify\s+(?:my\s+)?(?:identity|account)\b",
    r"\b(?:change|update)\s+(?:my\s+)?(?:payment|card)\b",
    r"\bpersonal\s+(?:information|details)\b",
    r"\b(?:check|where\s+is)\s+(?:my\s+)?refund\b",
    r"\b(?:cancel|refund|replace|exchange)\s+(?:my\s+)?(?:order|item)\b",
]

# A few high-signal customer-state cases that cannot be resolved without
# inspecting the private transaction/order record. Keep this deliberately
# small; this is a capability boundary, not a second intent classifier.
PRIVATE_STATE_PATTERNS = [
    r"\b(?:charged|billed)\b.*\b(?:after|despite)\b.*\b(?:cancel(?:ed|led)?|cancell?ation)\b",
    r"\b(?:refund|reimbursement)\b.*\b(?:not received|hasn't arrived|has not arrived|still waiting|missing)\b",
    r"\b(?:marked|says)\s+delivered\b.*\b(?:not received|didn't receive|did not receive|missing)\b",
    r"\b(?:package|order)\b.*\b(?:overdue|overdue for|late|hasn't arrived|has not arrived)\b",
]


def _matches_any(text, patterns):
    text = str(text).lower()
    return any(re.search(pattern, text) for pattern in patterns)


def looks_account_specific(text):
    return _matches_any(text, ACCOUNT_SPECIFIC_PATTERNS)


def looks_private_state(text):
    return _matches_any(text, PRIVATE_STATE_PATTERNS)


def looks_ambiguous(text):
    text = str(text).lower()
    if len(text.split()) > 80:
        return True
    patterns = [r"\banother issue\b", r"\bone more thing\b", r"\bseparate issue\b"]
    return any(re.search(p, text) for p in patterns)


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


def apply_safety_policy(customer_message, classification, historical_cases, llm_output, retrieval_threshold=None, min_evidence_cases=None):
    retrieval_threshold = float(retrieval_threshold if retrieval_threshold is not None else os.getenv("RETRIEVAL_EVIDENCE_THRESHOLD", "0.55"))
    min_evidence_cases = int(min_evidence_cases if min_evidence_cases is not None else os.getenv("MIN_STRONG_EVIDENCE_CASES", "2"))
    intent = classification.get("intent_id")

    if intent in (None, "uncertain"):
        return {"decision": "escalate", "draft_reply": None, "reason": "Classifier uncertainty."}
    if looks_ambiguous(customer_message):
        return {"decision": "escalate", "draft_reply": None, "reason": "Message may contain multiple or ambiguous issues."}
    if looks_account_specific(customer_message) or looks_private_state(customer_message):
        return {"decision": "escalate", "draft_reply": None, "reason": "Request requires customer-specific state or a secure lookup the agent cannot perform."}
    if not historical_cases:
        return {"decision": "escalate", "draft_reply": None, "reason": "No historical evidence was retrieved."}

    strong = [c for c in historical_cases if float(c.get("dense_score", 0.0)) >= retrieval_threshold]
    if len(strong) < min_evidence_cases:
        return {"decision": "escalate", "draft_reply": None, "reason": f"Insufficient historical evidence: {len(strong)} strong case(s), need {min_evidence_cases}."}

    valid, reason = validate_generation(llm_output)
    if not valid:
        return {"decision": "escalate", "draft_reply": None, "reason": reason}
    if str(llm_output["decision"]).lower() != "auto_handle":
        return {"decision": "escalate", "draft_reply": None, "reason": llm_output["reason"]}
    return {"decision": "auto_handle", "draft_reply": llm_output["draft_reply"].strip(), "reason": llm_output["reason"]}
