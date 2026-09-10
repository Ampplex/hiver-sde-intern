import os
import re

ALLOWED_DECISIONS = {"auto_handle", "escalate"}

# Keep regex narrowly scoped to deterministic, high-signal requests. Do not use
# conversational words such as "also", "and", or "plus" as ambiguity signals.
EXPLICIT_ACCOUNT_PATTERNS = [
    r"\bcheck\s+(?:my\s+)?order\b",
    r"\blook\s+up\s+(?:my\s+)?order\b",
    r"\btrack\s+(?:my\s+)?order\b",
    r"\bcheck\s+(?:my\s+)?account\b",
    r"\baccess\s+(?:my\s+)?account\b",
    r"\border\s*(?:#|number|id)\s*[\w-]+\b",
    r"\bverify\s+(?:my\s+)?identity\b",
    r"\bverify\s+(?:my\s+)?account\b",
    r"\bconfirm\s+(?:my\s+)?identity\b",
    r"\bchange\s+(?:my\s+)?(?:payment|card)\b",
    r"\bupdate\s+(?:my\s+)?(?:payment|card)\b",
    r"\bpersonal\s+(?:information|details)\b",
]

# Explicit sensitive-data requests are hard stops regardless of intent.
SENSITIVE_DATA_PATTERNS = [
    r"\b(?:otp|one[- ]time password|cvv|cvc|security code)\b",
    r"\b(?:password|passcode)\b.*\b(?:send|share|tell|provide)\b",
]

# Case-specific state that this agent cannot inspect because it has no private
# order/account backend. These are semantic, intent-conditioned checks rather
# than a giant free-form regex list.
CASE_STATE_TERMS = {
    "billing_issue": (
        "charged", "charge", "billed", "billing", "double charged", "charged twice",
        "charged after", "charged despite", "canceled", "cancelled", "trial charge",
    ),
    "refund_issue": (
        "did not receive the refund", "haven't received the refund", "have not received the refund",
        "refund hasn't arrived", "refund has not arrived", "still waiting for my refund",
        "refund is missing", "refund not received", "where is my refund",
    ),
    "delivery_status": (
        "where is it", "where is my package", "where is my order", "hasn't arrived",
        "has not arrived", "not arrived", "was supposed to arrive", "out for delivery",
        "marked delivered", "says delivered", "not delivered", "not received",
        "hasn't been dispatched", "has not been dispatched", "not dispatched",
    ),
    "delivery_issue": (
        "marked delivered", "says delivered", "not delivered", "not received",
        "didn't receive", "did not receive", "missing package", "missing order",
    ),
    "delivery_speed_issue": (
        "my package", "my order", "was supposed to arrive", "didn't arrive", "did not arrive",
        "hasn't arrived", "has not arrived", "late package", "delivery is late",
    ),
    "preorder_issue": (
        "preordered", "pre-order", "preorder", "was supposed to be delivered",
        "delivery date changed", "now says delivery", "changed delivery date",
    ),
    "order_cancellation": (
        "my order", "order was cancelled", "order was canceled", "auto cancel", "automatically cancel",
    ),
    "return_refund_request": (
        "my order", "my item", "send it back", "return this", "replacement", "exchange this",
    ),
    "return_refund_issue": (
        "my return", "return status", "refund", "replacement hasn't", "replacement has not",
    ),
    "return_pickup_issue": (
        "my return", "pickup hasn't", "pickup has not", "pickup missed", "return pickup",
    ),
    "unauthorized_charge": (
        "charged", "charge", "don't recognize", "do not recognize", "not mine", "unauthorized",
    ),
    "multiple_charges": (
        "charged twice", "charged two times", "multiple charges", "duplicate charge", "double charged",
    ),
    "overcharged_payment": (
        "charged too much", "overcharged", "wrong amount", "incorrect amount", "charged more",
    ),
}

# Intents where the requested action is informational/social and does not
# inherently require private order/account state.
PUBLIC_INFORMATION_INTENTS = {
    "positive_feedback",
    "poor_service",
    "packaging_issue",
    "feature_request",
    "content_availability",
    "content_access_issue",
    "prime_membership_query",
    "prime_subscription_query",
    "policy_clarification",
    "delivery_info_inquiry",
    "return_policy_inquiry",
    "product_details_inquiry",
    "product_availability",
    "contest_status",
    "service_availability",
    "shipping_availability",
    "sale_notification_issue",
    "language_support_issue",
    "feature_availability",
    "compatibility_check",
}

# Public/social intents can be handled with one retrieved example. This is a
# deliberate exception to the global two-strong-case rule because many such
# interactions only need an acknowledgment or general guidance.
ONE_EVIDENCE_INTENTS = PUBLIC_INFORMATION_INTENTS


def _contains_any(text, terms):
    return any(term in text for term in terms)


def looks_account_specific(text):
    text = str(text).lower()
    return any(re.search(p, text) for p in EXPLICIT_ACCOUNT_PATTERNS)


def looks_sensitive_data_request(text):
    text = str(text).lower()
    return any(re.search(p, text) for p in SENSITIVE_DATA_PATTERNS)


def requires_private_state(text, intent):
    text = str(text).lower()

    # Explicit account/order/identity requests are deterministic hard stops.
    if looks_account_specific(text) or looks_sensitive_data_request(text):
        return True

    # For case-oriented intents, require escalation only when the message
    # actually describes a customer-specific state that the agent cannot inspect.
    terms = CASE_STATE_TERMS.get(str(intent), ())
    return bool(terms) and _contains_any(text, terms)


def looks_ambiguous(text):
    # Avoid brittle lexical triggers such as "also"/"and". Only treat clearly
    # multi-part messages as ambiguous using explicit issue markers.
    words = str(text).split()
    if len(words) > 80:
        return True
    text = str(text).lower()
    markers = ("another issue", "one more thing", "separate issue")
    return _contains_any(text, markers)


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


def looks_like_handoff(reply):
    """Reject auto-handle replies that merely initiate human support work."""
    if not reply:
        return False
    text = str(reply).lower()
    handoff_markers = (
        "provide your order",
        "provide order details",
        "send us your order",
        "share your order details",
        "share your details",
        "provide your information",
        "send your details",
        "contact us via phone or chat",
        "contact us here",
        "reach us via phone/chat",
        "we can look into this",
        "so we can look into",
        "our team can assist",
        "our team will look into",
    )
    return _contains_any(text, handoff_markers)


def apply_safety_policy(customer_message, classification, historical_cases, llm_output, retrieval_threshold=None, min_evidence_cases=None):
    retrieval_threshold = float(retrieval_threshold if retrieval_threshold is not None else os.getenv("RETRIEVAL_EVIDENCE_THRESHOLD", "0.55"))
    min_evidence_cases = int(min_evidence_cases if min_evidence_cases is not None else os.getenv("MIN_STRONG_EVIDENCE_CASES", "2"))
    intent = classification.get("intent_id")

    if intent in (None, "uncertain"):
        return {"decision": "escalate", "draft_reply": None, "reason": "Classifier uncertainty."}
    if looks_ambiguous(customer_message):
        return {"decision": "escalate", "draft_reply": None, "reason": "Message may contain multiple or ambiguous issues."}
    if requires_private_state(customer_message, intent):
        return {"decision": "escalate", "draft_reply": None, "reason": "Request requires customer-specific state or a secure lookup the agent cannot perform."}
    if not historical_cases and intent not in PUBLIC_INFORMATION_INTENTS:
        return {"decision": "escalate", "draft_reply": None, "reason": "No historical evidence was retrieved."}

    required_strong = 1 if intent in ONE_EVIDENCE_INTENTS else min_evidence_cases
    strong = [c for c in historical_cases if float(c.get("dense_score", 0.0)) >= retrieval_threshold]
    if len(strong) < required_strong and intent not in {"positive_feedback", "poor_service"}:
        return {"decision": "escalate", "draft_reply": None, "reason": f"Insufficient historical evidence: {len(strong)} strong case(s), need {required_strong}."}

    valid, reason = validate_generation(llm_output)
    if not valid:
        return {"decision": "escalate", "draft_reply": None, "reason": reason}
    if str(llm_output["decision"]).lower() != "auto_handle":
        return {"decision": "escalate", "draft_reply": None, "reason": llm_output["reason"]}
    if looks_like_handoff(llm_output.get("draft_reply")):
        return {"decision": "escalate", "draft_reply": None, "reason": "Draft is a support handoff/intake step rather than an autonomous resolution."}
    return {"decision": "auto_handle", "draft_reply": llm_output["draft_reply"].strip(), "reason": llm_output["reason"]}
