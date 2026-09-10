import os
import re

ALLOWED_DECISIONS = {"auto_handle", "escalate"}
ACCOUNT_SPECIFIC_PATTERNS = [
    r"\bcheck\s+(?:my\s+)?order\b", r"\blook\s+up\s+(?:my\s+)?order\b",
    r"\btrack\s+(?:my\s+)?order\b", r"\bcheck\s+(?:my\s+)?account\b",
    r"\baccess\s+(?:my\s+)?account\b", r"\border\s*(?:#|number|id)\s*[\w-]+\b",
    r"\bverify\s+(?:my\s+)?identity\b", r"\bverify\s+(?:my\s+)?account\b",
    r"\bconfirm\s+(?:my\s+)?identity\b", r"\bpersonal\s+(?:information|details)\b",
    r"\bchange\s+(?:my\s+)?payment\b", r"\bupdate\s+(?:my\s+)?payment\b",
    r"\bchange\s+(?:my\s+)?card\b", r"\bupdate\s+(?:my\s+)?card\b",
    r"\bcheck\s+(?:my\s+)?refund\b", r"\bwhere\s+is\s+(?:my\s+)?refund\b",
]


def looks_account_specific(text):
    return any(re.search(p, str(text).lower()) for p in ACCOUNT_SPECIFIC_PATTERNS)


def looks_ambiguous(text):
    text = str(text).lower()
    if len(text.split()) > 80:
        return True
    patterns = [r"\band\b.*\balso\b", r"\balso\b.*\band\b", r"\bplus\b", r"\banother issue\b", r"\bone more thing\b"]
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
    if looks_account_specific(customer_message):
        return {"decision": "escalate", "draft_reply": None, "reason": "Request may require account-specific or secure lookup."}
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
