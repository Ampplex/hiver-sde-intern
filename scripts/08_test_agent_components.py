from src.classification.intent_classifier import IntentClassifier
from src.retrieval.hybrid_retriever import HybridRetriever


QUERY = (
    "My Prime package was supposed to arrive "
    "yesterday but it is still delayed"
)


def main():

    print("Loading components...")

    classifier = IntentClassifier()
    retriever = HybridRetriever()

    print("\nQUERY")
    print(QUERY)

    classification = classifier.predict(
        QUERY
    )

    print("\nCLASSIFICATION")

    print(
        f"Intent: "
        f"{classification['intent_id']}"
    )

    print(
        f"Similarity: "
        f"{classification['similarity_score']:.4f}"
    )

    print(
        f"Margin: "
        f"{classification['margin']:.4f}"
    )

    if classification["intent_id"] == "uncertain":

        print(
            "\nClassifier is uncertain."
        )

        return

    results = retriever.search(
        query=QUERY,
        intent_id=classification[
            "intent_id"
        ],
        top_k=5,
    )

    print(
        f"\nRetrieved {len(results)} cases."
    )

    for i, result in enumerate(
        results,
        start=1,
    ):

        print(
            "\n" + "-" * 60
        )

        print(
            f"CASE {i}"
        )

        print(
            f"Conversation: "
            f"{result['conversation_id']}"
        )

        print(
            f"Dense: "
            f"{result['dense_score']:.4f}"
        )

        print(
            f"BM25: "
            f"{result['bm25_score']:.4f}"
        )

        print(
            f"RRF: "
            f"{result['rrf_score']:.4f}"
        )

        print(
            f"\nCustomer:\n"
            f"{result['customer_problem']}"
        )

        print(
            f"\nAmazonHelp:\n"
            f"{result['first_amazon_response']}"
        )


if __name__ == "__main__":
    main()
