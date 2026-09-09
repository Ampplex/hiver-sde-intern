"""
Deterministic safety policy.

The LLM can draft a response, but it cannot override
these gates.

Important:
These regexes are safety gates only.
They are NOT used for intent classification.
"""

import os
import re


ALLOWED_DECISIONS = {
    "auto_handle",
    "escalate",
}


# These patterns indicate that the customer is asking the
# system to inspect, modify, verify, or act on private data.
#
# Merely mentioning "order", "refund", "payment", etc. is
# not sufficient to trigger escalation.
ACCOUNT_SPECIFIC_PATTERNS = [

    # Order/account lookup requests
    r"\bcheck\s+(?:my\s+)?order\b",
    r"\blook\s+up\s+(?:my\s+)?order\b",
    r"\btrack\s+(?:my\s+)?order\b",
    r"\bcheck\s+(?:my\s+)?account\b",
    r"\baccess\s+(?:my\s+)?account\b",

    # Explicit order identifiers
    r"\border\s*(?:#|number|id)\s*[\w-]+\b",

    # Personal/security information
    r"\bverify\s+(?:my\s+)?identity\b",
    r"\bverify\s+(?:my\s+)?account\b",
    r"\bconfirm\s+(?:my\s+)?identity\b",
    r"\bpersonal\s+(?:information|details)\b",

    # Payment/card actions
    r"\bchange\s+(?:my\s+)?payment\b",
    r"\bupdate\s+(?:my\s+)?payment\b",
    r"\bchange\s+(?:my\s+)?card\b",
    r"\bupdate\s+(?:my\s+)?card\b",

    # Refund actions requiring account lookup
    r"\bcheck\s+(?:my\s+)?refund\b",
    r"\bwhere\s+is\s+(?:my\s+)?refund\b",
]


def looks_account_specific(text):

    text = str(text).lower()

    return any(
        re.search(
            pattern,
            text,
        )
        for pattern
        in ACCOUNT_SPECIFIC_PATTERNS
    )


def looks_ambiguous(text):

    text = str(text).lower()

    words = text.split()

    if len(words) > 80:
        return True

    multi_issue_patterns = [
        r"\band\b.*\balso\b",
        r"\balso\b.*\band\b",
        r"\bplus\b",
        r"\banother issue\b",
        r"\bone more thing\b",
    ]

    return any(
        re.search(
            pattern,
            text,
        )
        for pattern
        in multi_issue_patterns
    )


def validate_generation(output):

    if not isinstance(output, dict):
        return (
            False,
            "Generator returned an invalid object.",
        )

    decision = str(
        output.get(
            "decision",
            "",
        )
    ).strip().lower()

    reply = output.get(
        "draft_reply"
    )

    reason = str(
        output.get(
            "reason",
            "",
        )
    ).strip()

    if decision not in ALLOWED_DECISIONS:
        return (
            False,
            "Generator returned an invalid decision.",
        )

    if not reason:
        return (
            False,
            "Generator did not provide a reason.",
        )

    if decision == "auto_handle":

        if not isinstance(
            reply,
            str,
        ):
            return (
                False,
                "Auto-handle response is not text.",
            )

        if not reply.strip():
            return (
                False,
                "Auto-handle response is empty.",
            )

        if len(reply.strip()) > 1000:
            return (
                False,
                "Auto-handle response is unexpectedly long.",
            )

    else:

        if reply not in (
            None,
            "",
        ):
            return (
                False,
                "Escalation must not contain a sendable reply.",
            )

    return True, "valid"


def apply_safety_policy(
    customer_message,
    classification,
    historical_cases,
    llm_output,
    retrieval_threshold=None,
    min_evidence_cases=None,
):

    if retrieval_threshold is None:
        retrieval_threshold = float(
            os.getenv(
                "RETRIEVAL_EVIDENCE_THRESHOLD",
                "0.55",
            )
        )

    if min_evidence_cases is None:
        min_evidence_cases = int(
            os.getenv(
                "MIN_STRONG_EVIDENCE_CASES",
                "2",
            )
        )

    intent = classification.get(
        "intent_id"
    )

    # --------------------------------------------------------
    # 1. Classification uncertainty
    # --------------------------------------------------------

    if intent in (
        None,
        "uncertain",
    ):

        return {
            "decision": "escalate",
            "draft_reply": None,
            "reason": "Classifier uncertainty.",
        }

    # --------------------------------------------------------
    # 2. Multiple/ambiguous issues
    # --------------------------------------------------------

    if looks_ambiguous(
        customer_message
    ):

        return {
            "decision": "escalate",
            "draft_reply": None,
            "reason": (
                "Message may contain multiple "
                "or ambiguous issues."
            ),
        }

    # --------------------------------------------------------
    # 3. Account-specific operations
    # --------------------------------------------------------

    if looks_account_specific(
        customer_message
    ):

        return {
            "decision": "escalate",
            "draft_reply": None,
            "reason": (
                "Request may require account-specific "
                "or secure lookup."
            ),
        }

    # --------------------------------------------------------
    # 4. No historical evidence
    # --------------------------------------------------------

    if not historical_cases:

        return {
            "decision": "escalate",
            "draft_reply": None,
            "reason": (
                "No historical evidence was retrieved."
            ),
        }

    # --------------------------------------------------------
    # 5. Require multiple strong historical cases
    # --------------------------------------------------------

    strong_cases = [
        case
        for case in historical_cases
        if float(
            case.get(
                "dense_score",
                0.0,
            )
        ) >= retrieval_threshold
    ]

    if len(strong_cases) < min_evidence_cases:

        return {
            "decision": "escalate",
            "draft_reply": None,
            "reason": (
                "Insufficient historical evidence: "
                f"{len(strong_cases)} strong case(s), "
                f"need {min_evidence_cases}."
            ),
        }

    # --------------------------------------------------------
    # 6. Validate LLM output
    # --------------------------------------------------------

    valid, reason = validate_generation(
        llm_output
    )

    if not valid:

        return {
            "decision": "escalate",
            "draft_reply": None,
            "reason": reason,
        }

    # --------------------------------------------------------
    # 7. LLM escalation recommendation
    # --------------------------------------------------------

    if (
        str(
            llm_output["decision"]
        ).lower()
        != "auto_handle"
    ):

        return {
            "decision": "escalate",
            "draft_reply": None,
            "reason": llm_output[
                "reason"
            ],
        }

    # --------------------------------------------------------
    # 8. Final auto-handle
    # --------------------------------------------------------

    return {
        "decision": "auto_handle",
        "draft_reply": (
            llm_output[
                "draft_reply"
            ].strip()
        ),
        "reason": llm_output[
            "reason"
        ],
    }
