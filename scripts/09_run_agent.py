import os

from src.agent.policy import apply_safety_policy
from src.agent.response_generator import ResponseGenerator
from src.classification.intent_classifier import IntentClassifier
from src.retrieval.hybrid_retriever import HybridRetriever


def main():

    print("Initializing support agent...")

    classifier = IntentClassifier()
    retriever = HybridRetriever()
    generator = ResponseGenerator()

    customer_message = input(
        "\nEnter customer message: "
    ).strip()

    if not customer_message:
        print("No customer message supplied.")
        return

    print("\n" + "=" * 60)
    print("NEW CUSTOMER MESSAGE")
    print("=" * 60)
    print(customer_message)

    # ============================================
    # 1. Classification
    # ============================================

    classification = classifier.predict(
        customer_message
    )

    print("\n[1] CLASSIFICATION")

    print(
        f"Intent: "
        f"{classification['intent_id']}"
    )

    print(
        f"Similarity: "
        f"{classification['similarity_score']:.4f}"
    )

    print(
        f"Second-best: "
        f"{classification['second_similarity_score']:.4f}"
    )

    print(
        f"Margin: "
        f"{classification['margin']:.4f}"
    )

    if classification["intent_id"] == "uncertain":

        print("\nDECISION: ESCALATE")
        print(
            "REASON: Classifier uncertainty."
        )

        return

    # ============================================
    # 2. Historical retrieval
    # ============================================

    historical_cases = retriever.search(
        query=customer_message,
        intent_id=classification["intent_id"],
        top_k=3,
    )

    print("\n[2] HISTORICAL EVIDENCE")

    for i, case in enumerate(
        historical_cases,
        start=1,
    ):

        print(
            f"\nCase {i}"
        )

        print(
            f"Dense score: "
            f"{case['dense_score']:.4f}"
        )

        print(
            f"RRF score: "
            f"{case['rrf_score']:.4f}"
        )

        print(
            f"AmazonHelp: "
            f"{case['first_amazon_response']}"
        )

    # ============================================
    # 3. Generate candidate response
    # ============================================

    llm_output = generator.generate_response(
        customer_message=customer_message,
        predicted_intent=classification[
            "intent_id"
        ],
        historical_cases=historical_cases,
    )

    # ============================================
    # 4. Deterministic safety policy
    # ============================================

    final = apply_safety_policy(
        customer_message=customer_message,
        classification=classification,
        historical_cases=historical_cases,
        llm_output=llm_output,
        retrieval_threshold=float(
            os.getenv(
                "RETRIEVAL_EVIDENCE_THRESHOLD",
                "0.55",
            )
        ),
        min_evidence_cases=int(
            os.getenv(
                "MIN_STRONG_EVIDENCE_CASES",
                "2",
            )
        ),
    )

    # ============================================
    # 5. Final result
    # ============================================

    print("\n" + "=" * 60)
    print("FINAL AGENT DECISION")
    print("=" * 60)

    print(
        f"Decision: "
        f"{final['decision'].upper()}"
    )

    print(
        f"Reason: "
        f"{final['reason']}"
    )

    if final["decision"] == "auto_handle":

        print(
            f"\nDraft reply:\n"
            f"{final['draft_reply']}"
        )


if __name__ == "__main__":
    main()
