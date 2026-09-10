import hashlib
import json
import os
import random
import time
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from botocore.config import Config
from dotenv import load_dotenv
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

CASES_PATH = Path("data/processed/support_cases_train.parquet")

TAXONOMY_PATH = Path("data/processed/intent_taxonomy.json")
TAXONOMY_META_PATH = Path("data/processed/intent_taxonomy_meta.json")
ASSIGNMENTS_PATH = Path("data/processed/intent_assignments.parquet")
UNCOVERED_PATH = Path("data/processed/uncovered_training_cases.parquet")

EMBEDDINGS_PATH = Path(
    "data/processed/intent_assignment_embeddings.npy"
)
EMBEDDING_IDS_PATH = Path(
    "data/processed/intent_assignment_ids.json"
)

AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
BEDROCK_MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID",
    "mistral.mistral-large-2407-v1:0",
)
BEDROCK_EMBED_MODEL = os.getenv(
    "BEDROCK_EMBED_MODEL",
    "amazon.titan-embed-text-v2:0",
)

DISCOVERY_SAMPLE_SIZE = int(
    os.getenv("INTENT_DISCOVERY_SAMPLE_SIZE", "1500")
)

DISCOVERY_SIMILARITY_THRESHOLD = float(
    os.getenv("INTENT_DISCOVERY_MATCH_THRESHOLD", "0.50")
)

MIN_CLUSTER_SIZE = int(
    os.getenv("INTENT_MIN_CLUSTER_SIZE", "2")
)

ASSIGNMENT_THRESHOLD = float(
    os.getenv("INTENT_ASSIGNMENT_THRESHOLD", "0.45")
)

EXISTING_INTENT_CANDIDATE_SIMILARITY = float(
    os.getenv("EXISTING_INTENT_CANDIDATE_SIMILARITY", "0.50")
)

EXISTING_INTENT_TOP_K = int(
    os.getenv("EXISTING_INTENT_TOP_K", "5")
)

CORE_PROBLEM_BATCH_SIZE = int(
    os.getenv("CORE_PROBLEM_BATCH_SIZE", "20")
)

MAX_LLM_RETRIES = int(
    os.getenv("BEDROCK_MAX_RETRIES", "6")
)

MAX_EMBED_RETRIES = int(
    os.getenv("BEDROCK_EMBED_MAX_RETRIES", "10")
)

FAIL_ON_UNCOVERED = os.getenv(
    "FAIL_ON_UNCOVERED", "1"
) == "1"


# ============================================================
# BEDROCK CLIENT (SYNCHRONOUS & CONTROLLED)
# ============================================================

_bedrock_client = None


def get_bedrock_client():
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=AWS_REGION,
            config=Config(
                connect_timeout=30,
                read_timeout=120,
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        )
    return _bedrock_client


def clean_json_text(text):
    text = str(text).strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def call_llm(prompt):
    client = get_bedrock_client()
    last_error = None
    base_delay = 2.0
    max_delay = 60.0

    for attempt in range(MAX_LLM_RETRIES):
        try:
            response = client.converse(
                modelId=BEDROCK_MODEL_ID,
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": prompt}],
                    }
                ],
                inferenceConfig={
                    "temperature": 0.0,
                },
            )
            text = response["output"]["message"]["content"][0]["text"]
            # Polite pacing between LLM requests to avoid throttling bursts
            time.sleep(0.3 + random.uniform(0.1, 0.2))
            return json.loads(clean_json_text(text))

        except Exception as exc:
            last_error = exc
            delay = min(max_delay, base_delay * (2 ** attempt)) * random.uniform(0.8, 1.2)
            print(f"\n[LLM retry {attempt + 1}/{MAX_LLM_RETRIES}] {type(exc).__name__}: {exc}. Backing off {delay:.1f}s...")
            time.sleep(delay)

    raise last_error


def normalize_vector(vector):
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("Zero embedding encountered.")
    return vector / norm


def normalize_matrix(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("Zero embedding encountered.")
    return matrix / norms


def get_embedding(text):
    client = get_bedrock_client()
    last_error = None
    base_delay = 1.0
    max_delay = 30.0

    for attempt in range(MAX_EMBED_RETRIES):
        try:
            response = client.invoke_model(
                modelId=BEDROCK_EMBED_MODEL,
                body=json.dumps(
                    {
                        "inputText": str(text)[:8000],
                        "dimensions": 1024,
                    }
                ),
                accept="application/json",
                contentType="application/json",
            )
            result = json.loads(response["body"].read())
            # Polite pacing between embeddings (~50ms)
            time.sleep(0.04 + random.uniform(0.01, 0.03))
            return normalize_vector(result["embedding"])

        except Exception as exc:
            last_error = exc
            delay = min(max_delay, base_delay * (2 ** attempt)) * random.uniform(0.8, 1.2)
            print(f"\n[Embed retry {attempt + 1}/{MAX_EMBED_RETRIES}] {type(exc).__name__}: {exc}. Backing off {delay:.1f}s...")
            time.sleep(delay)

    raise last_error


# ============================================================
# CORE-PROBLEM EXTRACTION
# ============================================================

CORE_PROBLEMS_CACHE_PATH = Path("data/processed/core_problems_cache.json")


def extract_core_problems(batch):
    examples = []
    for idx, text in batch:
        examples.append(
            {
                "idx": idx,
                "customer_problem": text,
            }
        )

    prompt = f"""
You are extracting the underlying operational customer-support
problem from AmazonHelp customer messages.

For every input message, identify what the customer actually needs
help with.

Ignore:
- emotional tone
- politeness
- insults
- unnecessary product names
- Prime/non-Prime wording unless operationally important
- wording differences such as late/delayed/not arrived

Focus on the underlying support operation.

Examples:

"Where is my package?"
→ delivery status / order tracking

"Cancel this order before it ships"
→ order cancellation

"Can I change where my package is being delivered?"
→ delivery address change

Return ONLY JSON in this format:

{{
  "items": [
    {{
      "idx": 0,
      "object": "order",
      "operation": "delivery_status",
      "state": "not_received",
      "requested_action": "track_order",
      "core_problem": "Customer wants the status of an order that has not arrived."
    }}
  ]
}}

Rules:
- core_problem must describe the underlying customer problem.
- Keep materially different operations separate.
- Do not mention how AmazonHelp should respond.
- Do not invent policies or resolutions.
- Use short, stable descriptions.
- Return exactly one item for every input idx: {[item[0] for item in batch]}.

INPUTS:

{json.dumps(examples, indent=2, ensure_ascii=False)}
"""

    by_idx = {}
    try:
        result = call_llm(prompt)
        items = result.get("items", [])
        for item in items:
            try:
                idx = int(item["idx"])
                by_idx[idx] = {
                    "object": str(item.get("object", "")).strip(),
                    "operation": str(item.get("operation", "")).strip(),
                    "state": str(item.get("state", "")).strip(),
                    "requested_action": str(item.get("requested_action", "")).strip(),
                    "core_problem": str(item.get("core_problem", "")).strip(),
                }
            except Exception:
                continue
    except Exception as exc:
        print(f"\n[Warning] Batch LLM call error: {exc}")

    missing = [
        idx for idx, _ in batch
        if idx not in by_idx or not by_idx[idx]["core_problem"]
    ]

    if missing:
        print(f"\n[Info] LLM omitted {len(missing)} items. Retrying missing items...")
        missing_batch = [item for item in batch if item[0] in missing]
        for m_idx, m_text in missing_batch:
            single_prompt = f"""
Identify the underlying customer support operation for this message:
"{m_text}"

Return ONLY JSON:
{{
  "object": "order",
  "operation": "delivery_status",
  "state": "not_received",
  "requested_action": "track_order",
  "core_problem": "Customer wants the status of an order that has not arrived."
}}
"""
            try:
                single_res = call_llm(single_prompt)
                by_idx[m_idx] = {
                    "object": str(single_res.get("object", "order")).strip(),
                    "operation": str(single_res.get("operation", "support_request")).strip(),
                    "state": str(single_res.get("state", "open")).strip(),
                    "requested_action": str(single_res.get("requested_action", "help")).strip(),
                    "core_problem": str(single_res.get("core_problem", m_text[:200])).strip(),
                }
            except Exception:
                by_idx[m_idx] = {
                    "object": "order",
                    "operation": "support_inquiry",
                    "state": "open",
                    "requested_action": "assist",
                    "core_problem": m_text[:200].strip(),
                }

    return by_idx


def extract_all_core_problems(sample):
    """
    Synchronous batch LLM extraction with disk caching and graceful fallback.
    """
    results = {}
    if CORE_PROBLEMS_CACHE_PATH.exists():
        try:
            with open(CORE_PROBLEMS_CACHE_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
                results = {int(k): v for k, v in saved.items()}
            print(f"Loaded {len(results)} cached core problems from {CORE_PROBLEMS_CACHE_PATH}")
        except Exception as exc:
            print(f"Could not load cache: {exc}")

    rows = list(
        enumerate(
            sample["customer_problem"].tolist()
        )
    )

    needed_rows = [r for r in rows if r[0] not in results]

    if needed_rows:
        batches = [
            needed_rows[i:i + CORE_PROBLEM_BATCH_SIZE]
            for i in range(
                0,
                len(needed_rows),
                CORE_PROBLEM_BATCH_SIZE,
            )
        ]

        for batch in tqdm(batches, desc="Extracting core problems (sync)"):
            batch_results = extract_core_problems(batch)
            results.update(batch_results)
            try:
                with open(CORE_PROBLEMS_CACHE_PATH, "w", encoding="utf-8") as f:
                    json.dump({str(k): v for k, v in results.items()}, f, indent=2, ensure_ascii=False)
            except Exception:
                pass

    core_text = []
    for idx in range(len(sample)):
        item = results.get(idx) or {
            "object": "order",
            "operation": "customer_support",
            "state": "open",
            "requested_action": "assist",
            "core_problem": str(sample.iloc[idx]["customer_problem"])[:200],
        }
        results[idx] = item
        canonical = (
            f"object: {item['object']}\n"
            f"operation: {item['operation']}\n"
            f"state: {item['state']}\n"
            f"requested_action: {item['requested_action']}\n"
            f"core_problem: {item['core_problem']}"
        )
        core_text.append(canonical)

    sample = sample.copy()
    sample["core_object"] = [results[i]["object"] for i in range(len(sample))]
    sample["core_operation"] = [results[i]["operation"] for i in range(len(sample))]
    sample["core_state"] = [results[i]["state"] for i in range(len(sample))]
    sample["core_requested_action"] = [results[i]["requested_action"] for i in range(len(sample))]
    sample["core_problem"] = [results[i]["core_problem"] for i in range(len(sample))]
    sample["core_problem_text"] = core_text

    return sample


# ============================================================
# INTENT NAMING
# ============================================================

def clean_intent_name(name):
    import re
    name = re.sub(
        r"[^a-z0-9]+",
        "_",
        str(name).strip().lower(),
    )
    return re.sub(
        r"_+",
        "_",
        name,
    ).strip("_")


def name_intent(examples, existing_names):
    prompt_examples = []
    for example in examples[:8]:
        prompt_examples.append(
            {
                "customer_problem": example["customer_problem"],
                "core_problem": example["core_problem"],
                "operation": example["core_operation"],
                "state": example["core_state"],
                "requested_action": example["core_requested_action"],
            }
        )

    prompt = f"""
Define ONE customer-support intent for these examples.

The intent must represent the underlying operational problem,
not wording, tone, product names, or response style.

Examples:

{json.dumps(prompt_examples, indent=2, ensure_ascii=False)}

Existing intents:

{json.dumps(sorted(existing_names), indent=2)}

Rules:
- Generalize different wording when the operational problem is the same.
- Keep tracking/status, cancellation, address changes,
  payment issues, etc. separate when the operation differs.
- Do not split because of Prime/non-Prime.
- Do not split late/delayed/not-arrived wording when the
  operational problem is the same.
- Do not invent unsupported problems.
- Use a short snake_case name.
- Never create _2, _3, etc.

Return ONLY:

{{
  "intent_name": "delivery_status",
  "description": "Customers asking about the status or expected delivery of an order."
}}
"""

    result = call_llm(prompt)
    intent_name = clean_intent_name(result.get("intent_name", ""))
    description = str(result.get("description", "")).strip()

    if not intent_name or not description:
        raise ValueError("Invalid intent naming response.")

    return intent_name, description


# ============================================================
# DEDUPLICATION
# ============================================================

def compare_with_existing(
    candidate_name,
    candidate_description,
    candidate_examples,
    candidate_prototype,
    intents,
):
    if not intents:
        return {
            "decision": "new",
            "matched_intent": None,
            "similarity": None,
            "reason": "First intent.",
        }

    ids = list(intents)
    prototypes = normalize_matrix(
        np.vstack([intents[i]["prototype"] for i in ids])
    )
    candidate_prototype = normalize_vector(candidate_prototype)
    similarities = prototypes @ candidate_prototype
    order = np.argsort(-similarities)
    best_idx = int(order[0])
    best_score = float(similarities[best_idx])
    best_id = ids[best_idx]

    if best_score < EXISTING_INTENT_CANDIDATE_SIMILARITY:
        return {
            "decision": "new",
            "matched_intent": None,
            "similarity": best_score,
            "reason": "No plausible semantic match.",
        }

    candidates = []
    for idx in order[:EXISTING_INTENT_TOP_K]:
        idx = int(idx)
        candidates.append(
            {
                "intent_id": ids[idx],
                "description": intents[ids[idx]]["description"],
                "prototype_similarity": round(float(similarities[idx]), 4),
            }
        )

    prompt = f"""
You are consolidating a customer-support taxonomy.

NEW CANDIDATE:
Name: {candidate_name}
Description: {candidate_description}

Examples:
{json.dumps(
    [
        {
            "customer_problem": e["customer_problem"],
            "core_problem": e["core_problem"],
            "operation": e["core_operation"],
            "state": e["core_state"],
        }
        for e in candidate_examples[:8]
    ],
    indent=2,
    ensure_ascii=False,
)}

CLOSEST EXISTING INTENTS:
{json.dumps(candidates, indent=2, ensure_ascii=False)}

Decide whether the candidate describes the SAME underlying
operational customer problem as one existing intent.

Merge when:
- only wording differs
- the emotional tone differs
- Prime/non-Prime differs
- product names differ
- late/delayed/not-arrived wording differs

Keep separate when the requested operation materially differs,
such as:
- tracking vs cancellation
- address change vs delivery delay
- payment issue vs delivery issue
- membership charge vs order status

Return ONLY:

{{
  "decision": "existing",
  "intent_id": "delivery_status",
  "reason": "Same underlying delivery-status problem."
}}

or:

{{
  "decision": "new",
  "intent_id": null,
  "reason": "Different operational support problem."
}}
"""

    try:
        result = call_llm(prompt)
        decision = str(result.get("decision", "")).lower()
        matched = result.get("intent_id")

        if decision == "existing" and matched in intents:
            return {
                "decision": "existing",
                "matched_intent": matched,
                "similarity": best_score,
                "reason": str(result.get("reason", "")),
            }

        if decision == "new":
            return {
                "decision": "new",
                "matched_intent": None,
                "similarity": best_score,
                "reason": str(result.get("reason", "")),
            }

    except Exception as exc:
        print(f"Dedup arbitration failed: {exc}")

    if best_score >= 0.70:
        return {
            "decision": "existing",
            "matched_intent": best_id,
            "similarity": best_score,
            "reason": "LLM arbitration failed; strong semantic match.",
        }

    return {
        "decision": "new",
        "matched_intent": None,
        "similarity": best_score,
        "reason": "LLM arbitration failed and similarity was not decisive.",
    }


# ============================================================
# DISCOVERY
# ============================================================

def discover_taxonomy(df):
    sample_n = min(
        DISCOVERY_SAMPLE_SIZE,
        len(df),
    )

    sample = (
        df.sample(
            n=sample_n,
            random_state=42,
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # STEP 1: CORE PROBLEM EXTRACTION (SYNC)
    # --------------------------------------------------------
    sample = extract_all_core_problems(sample)
    print(f"Core problems extracted: {len(sample):,}")

    DISCOVERY_CORE_EMB_PATH = Path("data/processed/discovery_core_embeddings.npy")
    DISCOVERY_RAW_EMB_PATH = Path("data/processed/discovery_raw_embeddings.npy")

    rows = sample.to_dict("records")

    if DISCOVERY_CORE_EMB_PATH.exists():
        print(f"Loading cached core embeddings from {DISCOVERY_CORE_EMB_PATH}...")
        core_embeddings = np.load(DISCOVERY_CORE_EMB_PATH)
    else:
        print(f"Embedding {len(sample):,} normalized core problems synchronously...")
        core_embeddings = []
        for row in tqdm(rows, desc="Embedding core problems (sync)"):
            core_embeddings.append(get_embedding(row["core_problem_text"]))
        core_embeddings = normalize_matrix(np.vstack(core_embeddings))
        np.save(DISCOVERY_CORE_EMB_PATH, core_embeddings)
        print(f"Saved {len(core_embeddings)} core embeddings to disk.")

    if DISCOVERY_RAW_EMB_PATH.exists():
        print(f"Loading cached raw embeddings from {DISCOVERY_RAW_EMB_PATH}...")
        raw_embeddings = np.load(DISCOVERY_RAW_EMB_PATH)
    else:
        print(f"Embedding {len(sample):,} raw customer problems synchronously...")
        raw_embeddings = []
        for row in tqdm(rows, desc="Embedding raw customer problems (sync)"):
            raw_embeddings.append(get_embedding(row["customer_problem"]))
        raw_embeddings = normalize_matrix(np.vstack(raw_embeddings))
        np.save(DISCOVERY_RAW_EMB_PATH, raw_embeddings)
        print(f"Saved {len(raw_embeddings)} raw embeddings to disk.")

    # --------------------------------------------------------
    # STEP 3: CLUSTER NORMALIZED CORE PROBLEMS
    # --------------------------------------------------------
    distance_threshold = 1.0 - DISCOVERY_SIMILARITY_THRESHOLD

    try:
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="average",
        )
    except TypeError:
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            affinity="cosine",
            linkage="average",
        )

    labels = clusterer.fit_predict(core_embeddings)
    sample["cluster_id"] = labels
    cluster_ids = sorted(sample["cluster_id"].unique())

    print()
    print(f"Discovery samples: {len(sample):,}")
    print(f"Raw clusters: {len(cluster_ids):,}")

    # --------------------------------------------------------
    # STEP 4: REMOVE TINY CLUSTERS
    # --------------------------------------------------------
    clusters = []
    for cluster_id in cluster_ids:
        idxs = np.where(labels == cluster_id)[0]
        if len(idxs) < MIN_CLUSTER_SIZE:
            continue

        examples = sample.iloc[idxs].to_dict("records")
        prototype = normalize_vector(raw_embeddings[idxs].mean(axis=0))
        clusters.append(
            {
                "cluster_id": int(cluster_id),
                "size": len(idxs),
                "examples": examples,
                "prototype": prototype,
            }
        )

    clusters.sort(
        key=lambda x: (
            -x["size"],
            x["cluster_id"],
        )
    )

    print(f"Clusters after min-size filter: {len(clusters):,}")

    # --------------------------------------------------------
    # STEP 5: NAME + DEDUP (SYNC)
    # --------------------------------------------------------
    intents = {}
    audit = []

    for cluster in tqdm(clusters, desc="Naming + semantic deduplication"):
        name, description = name_intent(
            cluster["examples"],
            intents.keys(),
        )

        match = compare_with_existing(
            candidate_name=name,
            candidate_description=description,
            candidate_examples=cluster["examples"],
            candidate_prototype=cluster["prototype"],
            intents=intents,
        )

        if match["decision"] == "existing" or name in intents:
            target_intent = match["matched_intent"] if (match["decision"] == "existing" and match["matched_intent"] in intents) else name
            existing = intents[target_intent]
            old_count = existing["discovery_count"]
            new_count = cluster["size"]

            existing["prototype"] = normalize_vector(
                (existing["prototype"] * old_count + cluster["prototype"] * new_count)
                / (old_count + new_count)
            )
            existing["discovery_count"] += new_count
            existing["merged_cluster_count"] += 1
            existing["discovery_examples"].extend(cluster["examples"][:8])

            action = "merge"

        else:
            intent_id = name
            intents[intent_id] = {
                "description": description,
                "prototype": cluster["prototype"],
                "discovery_examples": cluster["examples"][:8],
                "discovery_count": cluster["size"],
                "merged_cluster_count": 1,
            }
            target_intent = intent_id
            action = "create"

        audit.append(
            {
                "cluster_id": cluster["cluster_id"],
                "cluster_size": cluster["size"],
                "candidate_name": name,
                "action": action,
                "target_intent": target_intent,
                "prototype_similarity": match["similarity"],
                "reason": match["reason"],
            }
        )

    print(f"Final intents after dedup: {len(intents):,}")

    return (
        intents,
        sample,
        raw_embeddings,
        audit,
    )


# ============================================================
# ASSIGN FULL TRAINING SET (SYNC)
# ============================================================

def assign_training_cases(
    df,
    intents,
    discovery_sample,
    discovery_raw_embeddings,
):
    intent_ids = list(intents)
    prototypes = normalize_matrix(
        np.vstack([intents[i]["prototype"] for i in intent_ids])
    )

    discovery_cache = {
        str(row["conversation_id"]): emb
        for row, emb in zip(
            discovery_sample.to_dict("records"),
            discovery_raw_embeddings,
        )
    }

    rows = df.reset_index(drop=True).to_dict("records")

    assigned_rows = []
    uncovered_rows = []
    assigned_embeddings = []
    assigned_ids = []

    for row in tqdm(rows, desc="Assigning training cases (sync)"):
        cid = str(row["conversation_id"])

        try:
            if cid in discovery_cache:
                embedding = discovery_cache[cid]
            else:
                embedding = get_embedding(row["customer_problem"])
        except Exception as exc:
            uncovered_rows.append(
                {
                    **row,
                    "best_intent_id": None,
                    "similarity_score": None,
                    "uncovered_reason": f"embedding_failed:{type(exc).__name__}",
                }
            )
            continue

        similarities = prototypes @ embedding
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        intent_id = intent_ids[best_idx]

        row_out = {
            **row,
            "intent_id": intent_id,
            "similarity_score": best_score,
        }

        if best_score >= ASSIGNMENT_THRESHOLD or FAIL_ON_UNCOVERED:
            assigned_rows.append(row_out)
            assigned_embeddings.append(embedding)
            assigned_ids.append(cid)
        else:
            uncovered_rows.append(
                {
                    **row_out,
                    "best_intent_id": intent_id,
                    "uncovered_reason": "below_assignment_threshold",
                }
            )

    assignments = pd.DataFrame(assigned_rows)
    uncovered = pd.DataFrame(uncovered_rows)

    if assignments.empty:
        raise RuntimeError("No training cases were assigned.")

    return (
        assignments,
        uncovered,
        np.vstack(assigned_embeddings).astype(np.float32),
        assigned_ids,
    )


# ============================================================
# MAIN
# ============================================================

def taxonomy_hash(taxonomy):
    payload = json.dumps(
        taxonomy,
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main():
    print("Loading training-only support cases...")
    df = pd.read_parquet(CASES_PATH)

    required = {
        "conversation_id",
        "tweet_ids",
        "customer_problem",
        "first_amazon_response",
        "amazonhelp_responses",
        "full_transcript",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.dropna(subset=["customer_problem"]).copy()
    df["customer_problem"] = (
        df["customer_problem"].astype(str).str.strip()
    )
    df = df[df["customer_problem"].str.len() > 0].reset_index(drop=True)

    if df["conversation_id"].duplicated().any():
        raise ValueError("Duplicate conversation_id detected.")

    # --------------------------------------------------------
    # DISCOVERY
    # --------------------------------------------------------
    (
        intents,
        discovery_sample,
        discovery_raw_embeddings,
        audit,
    ) = discover_taxonomy(df)

    if len(intents) < 2:
        raise RuntimeError("Discovery produced fewer than two intents.")

    # --------------------------------------------------------
    # ASSIGNMENT
    # --------------------------------------------------------
    (
        assignments,
        uncovered,
        assignment_embeddings,
        assignment_ids,
    ) = assign_training_cases(
        df,
        intents,
        discovery_sample,
        discovery_raw_embeddings,
    )

    # --------------------------------------------------------
    # FINAL TAXONOMY
    # --------------------------------------------------------
    final_taxonomy = {}
    for intent_id in intents:
        examples = assignments[assignments["intent_id"] == intent_id]
        if examples.empty:
            raise RuntimeError(f"Intent has zero assigned cases: {intent_id}")

        final_taxonomy[intent_id] = {
            "description": intents[intent_id]["description"],
            "example_count": int(len(examples)),
            "discovery_example_count": int(intents[intent_id]["discovery_count"]),
            "merged_cluster_count": int(intents[intent_id]["merged_cluster_count"]),
            "conversations": [str(x) for x in examples["conversation_id"].tolist()],
        }

    coverage = len(assignments) / len(df)
    meta = {
        "taxonomy_hash": taxonomy_hash(final_taxonomy),
        "training_cases": int(len(df)),
        "assigned_cases": int(len(assignments)),
        "uncovered_cases": int(len(uncovered)),
        "coverage": float(coverage),
        "discovery_sample_size": int(len(discovery_sample)),
        "discovery_similarity_threshold": DISCOVERY_SIMILARITY_THRESHOLD,
        "assignment_threshold": ASSIGNMENT_THRESHOLD,
        "candidate_min_cluster_size": MIN_CLUSTER_SIZE,
        "semantic_dedup_audit": audit,
    }

    TAXONOMY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TAXONOMY_PATH, "w", encoding="utf-8") as f:
        json.dump(final_taxonomy, f, indent=2, ensure_ascii=False)

    with open(TAXONOMY_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    assignments.to_parquet(ASSIGNMENTS_PATH, index=False)
    uncovered.to_parquet(UNCOVERED_PATH, index=False)

    np.save(EMBEDDINGS_PATH, assignment_embeddings)
    with open(EMBEDDING_IDS_PATH, "w", encoding="utf-8") as f:
        json.dump(assignment_ids, f, indent=2)

    print("\n" + "=" * 60)
    print("INTENT DISCOVERY COMPLETE")
    print("=" * 60)
    print(f"Final intents: {len(final_taxonomy):,}")
    print(f"Training cases: {len(df):,}")
    print(f"Assigned cases: {len(assignments):,}")
    print(f"Uncovered cases: {len(uncovered):,}")
    print(f"Coverage: {coverage:.1%}")
    print(f"Taxonomy hash: {meta['taxonomy_hash']}")

    if FAIL_ON_UNCOVERED and len(uncovered) > 0:
        raise RuntimeError(f"{len(uncovered)} training cases remain uncovered.")


if __name__ == "__main__":
    main()
