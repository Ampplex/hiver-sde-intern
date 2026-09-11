import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.capability_guard import CapabilityGuard
from src.agent.policy import apply_safety_policy
from src.agent.response_generator import ResponseGenerator
from src.classification.intent_classifier import IntentClassifier
from src.retrieval.hybrid_retriever import HybridRetriever


def main():
    parser = argparse.ArgumentParser(description="Run the AmazonHelp Support Agent on an incoming query.")
    parser.add_argument("--query", "--message", "-m", dest="message", type=str, help="Customer message to evaluate.")
    args = parser.parse_args()

    customer_message = args.message
    if not customer_message:
        customer_message = input("\nEnter customer message: ").strip()
    if not customer_message:
        print("No customer message supplied.")
        return

    classifier = IntentClassifier()
    retriever = HybridRetriever()
    generator = ResponseGenerator()
    capability_guard = CapabilityGuard()

    print("\n" + "=" * 60)
    print("INCOMING CUSTOMER MESSAGE")
    print("=" * 60)
    print(customer_message)

    classification = classifier.predict(customer_message)
    print("\n[1] CLASSIFICATION")
    print(f"Status:           {classification['intent_id']}")
    print(f"Predicted intent: {classification['predicted_intent_id']}")
    print(f"Top-1 similarity: {classification['similarity_score']:.4f}")
    print(f"Top-2 similarity: {classification['second_similarity_score']:.4f}")
    print(f"Margin:           {classification['margin']:.4f}")

    historical_cases = []
    if classification["intent_id"] != "uncertain":
        historical_cases = retriever.search(
            customer_message,
            intent_id=classification["intent_id"],
            top_k=3,
        )

    print("\n[2] HISTORICAL RETRIEVAL")
    print(f"Retrieved cases:  {len(historical_cases)}")
    for i, case in enumerate(historical_cases, start=1):
        print(f"\nCase {i} (Conv ID: {case['conversation_id']})")
        print(f"  Dense score:    {case['dense_score']:.4f}")
        print(f"  BM25 score:     {case['bm25_score']:.4f}")
        print(f"  RRF score:      {case['rrf_score']:.4f}")
        print(f"  Customer:       {case['customer_problem'][:100]}...")
        print(f"  AmazonHelp:     {case['first_amazon_response'][:100]}...")

    llm_output = {
        "draft_reply": None,
        "decision": "escalate",
        "reason": "Classifier uncertainty." if classification["intent_id"] == "uncertain" else "No historical evidence retrieved.",
    }
    capability_result = None

    if classification["intent_id"] != "uncertain" and historical_cases:
        print("\n[3] RESPONSE GENERATION (Mistral Large)")
        llm_output = generator.generate_response(
            customer_message=customer_message,
            predicted_intent=classification["intent_id"],
            historical_cases=historical_cases,
        )
        print(f"Generator recommended: {llm_output.get('decision')}")
        print(f"Generator reason:      {llm_output.get('reason')}")

        if str(llm_output.get("decision", "")).lower() == "auto_handle" and llm_output.get("draft_reply"):
            print("\n[4] CAPABILITY GUARD SAFETY CHECK (Mistral Large)")
            capability_result = capability_guard.assess(
                customer_message=customer_message,
                predicted_intent=classification["intent_id"],
                historical_cases=historical_cases,
                draft_reply=llm_output.get("draft_reply"),
            )
            print(f"CapabilityGuard decision: {capability_result.get('decision')}")
            print(f"CapabilityGuard reason:   {capability_result.get('reason')}")

    final = apply_safety_policy(
        customer_message=customer_message,
        classification=classification,
        historical_cases=historical_cases,
        llm_output=llm_output,
        capability_result=capability_result,
        retrieval_threshold=float(os.getenv("RETRIEVAL_EVIDENCE_THRESHOLD", "0.55")),
        min_evidence_cases=int(os.getenv("MIN_STRONG_EVIDENCE_CASES", "1")),
    )

    print("\n" + "=" * 60)
    print("FINAL AGENT DECISION")
    print("=" * 60)
    print(f"Decision: {final['decision'].upper()}")
    print(f"Reason:   {final['reason']}")
    if final["draft_reply"]:
        print(f"\nDraft reply:\n{final['draft_reply']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
