import json
import os
import re
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

CASES_PATH = Path(
    "data/processed/support_cases_train.parquet"
)

TAXONOMY_PATH = Path(
    "data/processed/intent_taxonomy.json"
)

ASSIGNMENTS_PATH = Path(
    "data/processed/intent_assignments.parquet"
)

EMBEDDINGS_PATH = Path(
    "data/processed/intent_assignment_embeddings.npy"
)

EMBEDDING_IDS_PATH = Path(
    "data/processed/intent_assignment_ids.json"
)

AWS_REGION = os.getenv(
    "AWS_REGION",
    "us-west-2",
)

BEDROCK_MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID",
    "mistral.mistral-large-2407-v1:0",
)

BEDROCK_EMBED_MODEL = os.getenv(
    "BEDROCK_EMBED_MODEL",
    "amazon.titan-embed-text-v2:0",
)

# Discovery is performed only on training data.
DISCOVERY_SAMPLE_SIZE = int(
    os.getenv(
        "INTENT_DISCOVERY_SAMPLE_SIZE",
        "1500",
    )
)

# Cosine similarity threshold used to form semantic clusters.
DISCOVERY_SIMILARITY_THRESHOLD = float(
    os.getenv(
        "INTENT_DISCOVERY_MATCH_THRESHOLD",
        "0.62",
    )
)

MIN_CLUSTER_SIZE = int(
    os.getenv(
        "MIN_CANDIDATE_EXAMPLES",
        "3",
    )
)

# Number of training cases used to build the final prototypes.
# This avoids requiring another embedding call for every training case.
ASSIGNMENT_SAMPLE_SIZE = int(
    os.getenv(
        "INTENT_ASSIGNMENT_SAMPLE_SIZE",
        "3000",
    )
)

ASSIGNMENT_THRESHOLD = float(
    os.getenv(
        "INTENT_ASSIGNMENT_THRESHOLD",
        "0.45",
    )
)


# ============================================================
# BEDROCK
# ============================================================

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
)


# ============================================================
# EMBEDDINGS
# ============================================================

def get_embedding(text):
    text = str(text)[:8000]

    body = json.dumps(
        {
            "inputText": text,
            "dimensions": 1024,
        }
    )

    response = bedrock.invoke_model(
        body=body,
        modelId=BEDROCK_EMBED_MODEL,
        accept="application/json",
        contentType="application/json",
    )

    result = json.loads(
        response["body"].read()
    )

    embedding = np.asarray(
        result["embedding"],
        dtype=np.float32,
    )

    norm = np.linalg.norm(embedding)

    if norm > 0:
        embedding = embedding / norm

    return embedding


def normalize_matrix(matrix):
    matrix = np.asarray(
        matrix,
        dtype=np.float32,
    )

    norms = np.linalg.norm(
        matrix,
        axis=1,
        keepdims=True,
    )

    return matrix / np.maximum(
        norms,
        1e-12,
    )


# ============================================================
# TEXT / INTENT HELPERS
# ============================================================

def clean_intent_name(name):
    name = str(name).strip().lower()

    name = re.sub(
        r"[^a-z0-9]+",
        "_",
        name,
    )

    name = re.sub(
        r"_+",
        "_",
        name,
    ).strip("_")

    return name


def unique_intent_name(
    requested_name,
    existing_names,
):
    base = clean_intent_name(
        requested_name
    )

    if not base:
        base = "unknown_support_issue"

    name = base
    counter = 2

    while name in existing_names:
        name = f"{base}_{counter}"
        counter += 1

    return name


# ============================================================
# LLM INTENT NAMING
# ============================================================

def name_intent(examples):
    problems = [
        str(example["customer_problem"])
        for example in examples
    ]

    prompt = f"""
You are defining a customer-support intent taxonomy
from real AmazonHelp support conversations.

The examples below belong to the same semantic cluster.

Examples:

{json.dumps(
    problems,
    indent=2,
    ensure_ascii=False,
)}

Identify the underlying customer-support problem.

Rules:

1. Use only evidence present in the examples.
2. Do not invent policies, products, or issues.
3. Ignore emotional wording unless it represents the actual problem.
4. Create a short snake_case intent name.
5. The intent should describe the customer problem, not the response.
6. Keep the description concise.
7. Prefer a general intent when examples clearly represent
   the same underlying support problem.
8. Do not mention AmazonHelp in the intent name.

Return ONLY valid JSON:

{{
    "intent_name": "delivery_status",
    "description": "Customers asking about the status or expected delivery of an order."
}}
"""

    try:
        response = bedrock.converse(
            modelId=BEDROCK_MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "text": prompt,
                        }
                    ],
                }
            ],
            inferenceConfig={
                "temperature": 0.0,
            },
        )

        text = (
            response["output"]
            ["message"]
            ["content"][0]
            ["text"]
            .strip()
        )

        if text.startswith("```json"):
            text = text[7:-3].strip()
        elif text.startswith("```"):
            text = text[3:-3].strip()

        result = json.loads(text)

        if not result.get("intent_name"):
            return None

        if not result.get("description"):
            return None

        return result

    except Exception as exc:
        print(
            f"Intent naming failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return None


# ============================================================
# DISCOVERY
# ============================================================

def discover_taxonomy(df):
    """
    Discover semantic clusters from a deterministic,
    training-only sample.

    Unlike greedy discovery, clustering is independent
    of the order in which examples are processed.
    """

    sample_size = min(
        DISCOVERY_SAMPLE_SIZE,
        len(df),
    )

    sample = (
        df.sample(
            n=sample_size,
            random_state=42,
        )
        .reset_index(drop=True)
    )

    print(
        f"\nGenerating embeddings for "
        f"{len(sample):,} discovery examples..."
    )

    embeddings = []

    for _, row in tqdm(
        sample.iterrows(),
        total=len(sample),
        desc="Discovery embeddings",
    ):
        try:
            embeddings.append(
                get_embedding(
                    row["customer_problem"]
                )
            )
        except Exception as exc:
            print(
                f"Embedding failed: "
                f"{type(exc).__name__}: {exc}"
            )
            embeddings.append(
                np.zeros(
                    1024,
                    dtype=np.float32,
                )
            )

    embeddings = normalize_matrix(
        np.vstack(embeddings)
    )

    # Cosine distance = 1 - cosine similarity.
    distance_threshold = (
        1.0 - DISCOVERY_SIMILARITY_THRESHOLD
    )

    print(
        "\nClustering discovery examples..."
    )

    try:
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="average",
        )
    except TypeError:
        # Compatibility with older sklearn versions.
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            affinity="cosine",
            linkage="average",
        )

    labels = clusterer.fit_predict(
        embeddings
    )

    sample["cluster_id"] = labels

    clusters = []

    for cluster_id in sorted(
        sample["cluster_id"].unique()
    ):
        indices = np.where(
            labels == cluster_id
        )[0]

        if len(indices) < MIN_CLUSTER_SIZE:
            continue

        examples = []

        for index in indices:
            row = sample.iloc[index]

            examples.append(
                {
                    "conversation_id": row[
                        "conversation_id"
                    ],
                    "tweet_ids": row[
                        "tweet_ids"
                    ],
                    "customer_problem": row[
                        "customer_problem"
                    ],
                    "first_amazon_response": row[
                        "first_amazon_response"
                    ],
                    "full_transcript": row[
                        "full_transcript"
                    ],
                }
            )

        cluster_embedding = embeddings[
            indices
        ].mean(axis=0)

        norm = np.linalg.norm(
            cluster_embedding
        )

        if norm > 0:
            cluster_embedding /= norm

        clusters.append(
            {
                "cluster_id": int(cluster_id),
                "indices": indices.tolist(),
                "examples": examples,
                "prototype": cluster_embedding,
            }
        )

    print(
        f"Semantic clusters discovered: "
        f"{len(clusters)}"
    )

    intents = {}

    for cluster in tqdm(
        clusters,
        desc="Naming intents",
    ):
        result = name_intent(
            cluster["examples"]
        )

        if result is None:
            continue

        intent_id = unique_intent_name(
            result["intent_name"],
            intents,
        )

        intents[intent_id] = {
            "description": result[
                "description"
            ],
            "prototype": cluster[
                "prototype"
            ],
            "discovery_examples": cluster[
                "examples"
            ],
        }

        print(
            f"\nDiscovered intent: "
            f"{intent_id} "
            f"({len(cluster['examples'])} discovery examples)"
        )

    return intents, sample, embeddings


# ============================================================
# ASSIGN TRAINING CASES
# ============================================================

def assign_training_cases(
    df,
    intents,
    discovery_sample,
    discovery_embeddings,
):
    """
    Assign a deterministic sample of training cases
    to the frozen discovered taxonomy.

    Cases below the threshold remain unassigned.
    """

    if len(intents) < 2:
        raise RuntimeError(
            "Need at least two discovered intents."
        )

    # Start with a deterministic training sample.
    sample_size = min(
        ASSIGNMENT_SAMPLE_SIZE,
        len(df),
    )

    assignment_df = df.sample(
        n=sample_size,
        random_state=43,
    )

    # Always include discovery examples so every
    # discovered intent has representation.
    assignment_df = pd.concat(
        [
            assignment_df,
            discovery_sample,
        ],
        ignore_index=True,
    ).drop_duplicates(
        subset=["conversation_id"]
    ).reset_index(drop=True)

    print(
        f"\nAssigning "
        f"{len(assignment_df):,} training cases "
        f"to the frozen taxonomy..."
    )

    intent_ids = list(
        intents.keys()
    )

    prototypes = normalize_matrix(
        np.vstack(
            [
                intents[intent_id]["prototype"]
                for intent_id in intent_ids
            ]
        )
    )

    discovery_embeddings_by_id = {
        str(row["conversation_id"]): embedding
        for row, embedding
        in zip(
            discovery_sample.to_dict(
                orient="records"
            ),
            discovery_embeddings,
        )
    }

    rows = []
    assigned_embeddings = []
    assigned_ids = []

    for _, row in tqdm(
        assignment_df.iterrows(),
        total=len(assignment_df),
        desc="Assigning intents",
    ):
        cid = str(
            row["conversation_id"]
        )

        embedding = (
            discovery_embeddings_by_id.get(
                cid
            )
        )

        if embedding is None:
            try:
                embedding = get_embedding(
                    row["customer_problem"]
                )
            except Exception as exc:
                print(
                    f"\nEmbedding failed for "
                    f"{cid}: "
                    f"{type(exc).__name__}"
                )
                continue

        similarities = (
            prototypes @ embedding
        )

        best_index = int(
            np.argmax(similarities)
        )

        best_score = float(
            similarities[best_index]
        )

        if best_score < ASSIGNMENT_THRESHOLD:
            continue

        intent_id = intent_ids[
            best_index
        ]

        rows.append(
            {
                "conversation_id": row[
                    "conversation_id"
                ],
                "tweet_ids": row[
                    "tweet_ids"
                ],
                "intent_id": intent_id,
                "similarity_score": best_score,
                "customer_problem": row[
                    "customer_problem"
                ],
                "first_amazon_response": row[
                    "first_amazon_response"
                ],
                "full_transcript": row[
                    "full_transcript"
                ],
            }
        )

        assigned_embeddings.append(
            embedding
        )

        assigned_ids.append(cid)

    assignments = pd.DataFrame(
        rows
    )

    if assignments.empty:
        raise RuntimeError(
            "No training cases were assigned."
        )

    return (
        assignments,
        np.vstack(
            assigned_embeddings
        ).astype(np.float32),
        assigned_ids,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Loading training-only support cases..."
    )

    df = pd.read_parquet(
        CASES_PATH
    )

    required = {
        "conversation_id",
        "tweet_ids",
        "customer_problem",
        "first_amazon_response",
        "full_transcript",
    }

    missing = (
        required - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing)}"
        )

    df = df.dropna(
        subset=["customer_problem"]
    ).copy()

    df["customer_problem"] = (
        df["customer_problem"]
        .astype(str)
        .str.strip()
    )

    df = df[
        df["customer_problem"].str.len() > 0
    ]

    if df["conversation_id"].duplicated().any():
        raise ValueError(
            "Duplicate conversation_id detected."
        )

    print(
        f"Training cases available: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Phase 1: semantic discovery
    # --------------------------------------------------------

    (
        intents,
        discovery_sample,
        discovery_embeddings,
    ) = discover_taxonomy(df)

    if len(intents) < 2:
        raise RuntimeError(
            "Discovery produced fewer than two intents."
        )

    # --------------------------------------------------------
    # Phase 2: assign training sample
    # --------------------------------------------------------

    (
        assignments_df,
        assignment_embeddings,
        assignment_ids,
    ) = assign_training_cases(
        df,
        intents,
        discovery_sample,
        discovery_embeddings,
    )

    # --------------------------------------------------------
    # Rebuild prototypes using assigned cases
    # --------------------------------------------------------

    final_prototypes = []

    for intent_id in intents:

        mask = (
            assignments_df["intent_id"]
            == intent_id
        ).to_numpy()

        if not mask.any():
            # Fall back to discovery prototype.
            prototype = intents[
                intent_id
            ]["prototype"]
        else:
            prototype = (
                assignment_embeddings[
                    mask
                ].mean(axis=0)
            )

        norm = np.linalg.norm(
            prototype
        )

        if norm > 0:
            prototype = prototype / norm

        final_prototypes.append(
            prototype
        )

    final_prototypes = np.vstack(
        final_prototypes
    ).astype(np.float32)

    # --------------------------------------------------------
    # Taxonomy metadata
    # --------------------------------------------------------

    taxonomy = {}

    for index, intent_id in enumerate(
        intents
    ):

        examples = assignments_df[
            assignments_df["intent_id"]
            == intent_id
        ]

        taxonomy[intent_id] = {
            "description": intents[
                intent_id
            ]["description"],
            "example_count": int(
                len(examples)
            ),
            "discovery_example_count": int(
                len(
                    intents[
                        intent_id
                    ][
                        "discovery_examples"
                    ]
                )
            ),
            "conversations": (
                examples[
                    "conversation_id"
                ].tolist()
            ),
        }

    # --------------------------------------------------------
    # Save taxonomy
    # --------------------------------------------------------

    TAXONOMY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        TAXONOMY_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            taxonomy,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Save assignments
    # --------------------------------------------------------

    ASSIGNMENTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    assignments_df.to_parquet(
        ASSIGNMENTS_PATH,
        index=False,
    )

    # Save embeddings so script 07 does not
    # need to re-call Bedrock unnecessarily.
    np.save(
        EMBEDDINGS_PATH,
        assignment_embeddings,
    )

    with open(
        EMBEDDING_IDS_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            assignment_ids,
            f,
            indent=2,
        )

    coverage = (
        len(assignments_df)
        / len(df)
    )

    print("\n" + "=" * 60)
    print("INTENT DISCOVERY COMPLETE")
    print("=" * 60)

    print(
        f"Discovered intents: "
        f"{len(intents)}"
    )

    print(
        f"Training sample assigned: "
        f"{len(assignments_df):,}"
    )

    print(
        f"Assignment coverage: "
        f"{coverage:.1%}"
    )

    print(
        f"\nTaxonomy saved to:\n"
        f"{TAXONOMY_PATH}"
    )

    print(
        f"\nAssignments saved to:\n"
        f"{ASSIGNMENTS_PATH}"
    )

    print(
        f"\nAssignment embeddings saved to:\n"
        f"{EMBEDDINGS_PATH}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Review and freeze intent_taxonomy.json "
        "before manually labeling the golden set."
    )


if __name__ == "__main__":
    main()
