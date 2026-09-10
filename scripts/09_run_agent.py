import os

from src.agent.policy import apply_safety_policy
from src.agent.response_generator import ResponseGenerator
from src.classification.intent_classifier import IntentClassifier
from src.retrieval.hybrid_retriever import HybridRetriever


def main():
    classifier = IntentClassifier()
    retriever = HybridRetriever()
    generator = ResponseGenerator()

    customer_message = input("\nEnter customer message: ").strip()
    if not customer_message:
        print("No customer message supplied.")
        return

    classification = classifier.predict(customer_message)
    print("\n[1] CLASSIFICATION")
    print(f"Intent: {classification['intent_id']}")
    print(f"Predicted intent: {classification['predicted_intent_id']}")
    print(f"Similarity: {classification['similarity_score']:.4f}")
    print(f"Second-best: {classification['second_similarity_score']:.4f}")
    print(f"Margin: {classification['margin']:.4f}")

    historical_cases = []
    if classification["intent_id"] != "uncertain":
        historical_cases = retriever.search(
            customer_message,
            intent_id=classification["intent_id"],
            top_k=3,
        )

    print("\n[2] HISTORICAL EVIDENCE")
    for i, case in enumerate(historical_cases, start=1):
        print(f"\nCase {i}: {case['conversation_id']}")
        print(f"Dense score: {case['dense_score']:.4f}")
        print(f"RRF score: {case['rrf_score']:.4f}")
        print(f"AmazonHelp: {case['first_amazon_response']}")

    llm_output = {
        "draft_reply": None,
        "decision": "escalate",
        "reason": "Classifier uncertainty." if classification["intent_id"] == "uncertain" else "Generation not attempted.",
    }
    if classification["intent_id"] != "uncertain" and historical_cases:
        llm_output = generator.generate_response(
            customer_message=customer_message,
            predicted_intent=classification["intent_id"],
            historical_cases=historical_cases,
        )

    final = apply_safety_policy(
        customer_message=customer_message,
        classification=classification,
        historical_cases=historical_cases,
        llm_output=llm_output,
        retrieval_threshold=float(os.getenv("RETRIEVAL_EVIDENCE_THRESHOLD", "0.55")),
        min_evidence_cases=int(os.getenv("MIN_STRONG_EVIDENCE_CASES", "2")),
    )

    print("\n" + "=" * 60)
    print("FINAL AGENT DECISION")
    print("=" * 60)
    print(f"Decision: {final['decision'].upper()}")
    print(f"Reason: {final['reason']}")
    if final["draft_reply"]:
        print(f"\nDraft reply:\n{final['draft_reply']}")


if __name__ == "__main__":
    main()
