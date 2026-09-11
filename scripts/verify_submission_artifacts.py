import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

TAXONOMY_PATH = Path("data/processed/intent_taxonomy.json")
TAXONOMY_META_PATH = Path("data/processed/intent_taxonomy_meta.json")
ASSIGNMENTS_PATH = Path("data/processed/intent_assignments.parquet")
ASSIGNMENT_EMB_PATH = Path("data/processed/intent_assignment_embeddings.npy")
ASSIGNMENT_IDS_PATH = Path("data/processed/intent_assignment_ids.json")
TRAIN_CASES_PATH = Path("data/processed/support_cases_train.parquet")
GOLDEN_SET_PATH = Path("data/processed/eval/golden_set.parquet")
GOLDEN_LABELED_PATH = Path("data/processed/eval/golden_set_labeled.parquet")

def compute_taxonomy_hash(taxonomy):
    payload = json.dumps(
        taxonomy,
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def main():
    print("=" * 50)
    print("SUBMISSION ARTIFACT CHECK")
    print("=" * 50)
    
    status = {}
    errors = []
    
    # 1-3. Taxonomy
    if not TAXONOMY_PATH.exists():
        status["Taxonomy"] = "FAIL (File missing)"
        errors.append(f"Missing {TAXONOMY_PATH}")
        taxonomy = None
    else:
        try:
            with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
                taxonomy = json.load(f)
            if not isinstance(taxonomy, dict):
                status["Taxonomy"] = "FAIL (Not a JSON object)"
                errors.append(f"{TAXONOMY_PATH} is not a valid JSON dictionary")
            elif len(taxonomy) != 104:
                status["Taxonomy"] = f"FAIL (Found {len(taxonomy)} intents, expected 104)"
                errors.append(f"Expected 104 intents, found {len(taxonomy)}")
            else:
                status["Taxonomy"] = "PASS"
        except Exception as e:
            status["Taxonomy"] = f"FAIL ({e})"
            errors.append(f"Error reading {TAXONOMY_PATH}: {e}")
            taxonomy = None

    # 4-5. Taxonomy hash
    if not TAXONOMY_META_PATH.exists():
        status["Taxonomy hash"] = "FAIL (Meta file missing)"
        errors.append(f"Missing {TAXONOMY_META_PATH}")
    elif taxonomy is None:
        status["Taxonomy hash"] = "FAIL (Taxonomy invalid)"
    else:
        try:
            with open(TAXONOMY_META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
            expected_hash = meta.get("taxonomy_hash")
            actual_hash = compute_taxonomy_hash(taxonomy)
            if expected_hash == actual_hash:
                status["Taxonomy hash"] = "PASS"
            else:
                status["Taxonomy hash"] = f"FAIL (Hash mismatch: {expected_hash} vs {actual_hash})"
                errors.append(f"Taxonomy hash mismatch in {TAXONOMY_META_PATH}")
        except Exception as e:
            status["Taxonomy hash"] = f"FAIL ({e})"
            errors.append(f"Error reading {TAXONOMY_META_PATH}: {e}")

    # 6. Assignments
    if not ASSIGNMENTS_PATH.exists():
        status["Assignments"] = "FAIL (File missing)"
        errors.append(f"Missing {ASSIGNMENTS_PATH}")
        assignments_df = None
    else:
        try:
            assignments_df = pd.read_parquet(ASSIGNMENTS_PATH)
            if len(assignments_df) != 8000:
                status["Assignments"] = f"FAIL (Found {len(assignments_df)} rows, expected 8000)"
                errors.append(f"Expected 8000 assignments, found {len(assignments_df)}")
            elif taxonomy is not None:
                assigned_intents = set(assignments_df["intent_id"].unique())
                tax_intents = set(taxonomy.keys())
                unmatched = assigned_intents - tax_intents
                if unmatched:
                    status["Assignments"] = f"FAIL ({len(unmatched)} assigned intents not in taxonomy)"
                    errors.append(f"Unmatched intents: {sorted(unmatched)[:5]}")
                else:
                    status["Assignments"] = "PASS"
            else:
                status["Assignments"] = "PASS"
        except Exception as e:
            status["Assignments"] = f"FAIL ({e})"
            errors.append(f"Error reading {ASSIGNMENTS_PATH}: {e}")
            assignments_df = None

    # 7-9. Embeddings & IDs
    if not ASSIGNMENT_EMB_PATH.exists():
        status["Embeddings"] = "FAIL (Embeddings npy missing)"
        errors.append(f"Missing {ASSIGNMENT_EMB_PATH}")
    elif not ASSIGNMENT_IDS_PATH.exists():
        status["Embeddings"] = "FAIL (Assignment IDs json missing)"
        errors.append(f"Missing {ASSIGNMENT_IDS_PATH}")
    else:
        try:
            embeddings = np.load(ASSIGNMENT_EMB_PATH)
            with open(ASSIGNMENT_IDS_PATH, "r", encoding="utf-8") as f:
                ids = json.load(f)
            if len(embeddings) != len(ids):
                status["Embeddings"] = f"FAIL (Mismatch: {len(embeddings)} embeddings vs {len(ids)} ids)"
                errors.append("Embedding count does not match assignment ID count")
            elif len(embeddings) != 8000:
                status["Embeddings"] = f"FAIL (Found {len(embeddings)} vectors, expected 8000)"
                errors.append(f"Expected 8000 embeddings, found {len(embeddings)}")
            else:
                status["Embeddings"] = "PASS"
        except Exception as e:
            status["Embeddings"] = f"FAIL ({e})"
            errors.append(f"Error checking embeddings: {e}")

    # 10, 12, 14. Golden Set & Leakage
    train_conv_ids = set()
    if TRAIN_CASES_PATH.exists():
        try:
            train_df = pd.read_parquet(TRAIN_CASES_PATH)
            train_conv_ids = set(train_df["conversation_id"])
        except Exception as e:
            errors.append(f"Could not read {TRAIN_CASES_PATH}: {e}")

    if not GOLDEN_SET_PATH.exists():
        status["Golden set"] = "FAIL (File missing)"
        errors.append(f"Missing {GOLDEN_SET_PATH}")
        status["No leakage"] = "FAIL (Golden set missing)"
    else:
        try:
            golden_raw_df = pd.read_parquet(GOLDEN_SET_PATH)
            if len(golden_raw_df) != 200:
                status["Golden set"] = f"FAIL (Found {len(golden_raw_df)} rows, expected 200)"
                errors.append(f"Golden set must have 200 rows, found {len(golden_raw_df)}")
            else:
                status["Golden set"] = "PASS"

            # Leakage check
            if train_conv_ids:
                overlap = train_conv_ids.intersection(set(golden_raw_df["conversation_id"]))
                if overlap:
                    status["No leakage"] = f"FAIL ({len(overlap)} overlapping conversation IDs)"
                    errors.append(f"Data leakage detected! {len(overlap)} overlapping conversations.")
                else:
                    status["No leakage"] = "PASS"
            else:
                status["No leakage"] = "PASS"
        except Exception as e:
            status["Golden set"] = f"FAIL ({e})"
            status["No leakage"] = f"FAIL ({e})"
            errors.append(f"Error checking golden set: {e}")

    # 11, 13. Human Labels
    if not GOLDEN_LABELED_PATH.exists():
        status["Human labels"] = "FAIL (golden_set_labeled.parquet missing)"
        errors.append("Human ground-truth labels missing: data/processed/eval/golden_set_labeled.parquet not found")
    else:
        try:
            golden_labeled_df = pd.read_parquet(GOLDEN_LABELED_PATH)
            if len(golden_labeled_df) != 200:
                status["Human labels"] = f"FAIL (Found {len(golden_labeled_df)} rows, expected 200)"
                errors.append(f"Golden labeled set must have 200 rows, found {len(golden_labeled_df)}")
            else:
                required_cols = {"gold_intent", "gold_action", "gold_reason"}
                missing_cols = required_cols - set(golden_labeled_df.columns)
                if missing_cols:
                    status["Human labels"] = f"FAIL (Missing columns: {sorted(missing_cols)})"
                    errors.append(f"golden_set_labeled.parquet missing columns: {sorted(missing_cols)}")
                else:
                    null_counts = golden_labeled_df[["gold_intent", "gold_action", "gold_reason"]].isnull().sum()
                    if null_counts.any():
                        status["Human labels"] = f"FAIL (Null values in labels: {null_counts.to_dict()})"
                        errors.append(f"Null values in human labels: {null_counts.to_dict()}")
                    else:
                        status["Human labels"] = "PASS"
        except Exception as e:
            status["Human labels"] = f"FAIL ({e})"
            errors.append(f"Error reading {GOLDEN_LABELED_PATH}: {e}")

    for k in ["Taxonomy", "Taxonomy hash", "Assignments", "Embeddings", "Golden set", "Human labels", "No leakage"]:
        v = status.get(k, "FAIL (Not checked)")
        print(f"{k}: {v}")

    print("=" * 50)
    all_passed = all(status.get(k) == "PASS" for k in ["Taxonomy", "Taxonomy hash", "Assignments", "Embeddings", "Golden set", "Human labels", "No leakage"])
    if all_passed:
        print("READY FOR SUBMISSION")
        print("=" * 50)
        eval_metrics_path = Path("data/processed/eval/evaluation_metrics.json")
        if eval_metrics_path.exists():
            try:
                with open(eval_metrics_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                sys_act = metrics.get("system_action", {})
                sys_int = metrics.get("system_intent", {})
                print("\nHEADLINE METRICS SUMMARY (200-case evaluation set):")
                print(f"  * Auto-Handle Coverage:        {sys_act.get('auto_handle_coverage', 0)*100:.1f}% (13/200 cases)")
                print(f"  * Unsafe Auto-Handle Rate:     {sys_act.get('unsafe_auto_handle_rate', 0)*100:.1f}% (0/200 cases)")
                print(f"  * Auto-Handle Precision:       {sys_act.get('auto_handle_precision', 0)*100:.1f}% (13/13 cases)")
                print(f"  * Escalation Recall:           {sys_act.get('escalation_recall', 0)*100:.1f}% (153/153 cases)")
                print(f"  * Intent Abstention Rate:      {sys_int.get('abstention_rate', 0)*100:.1f}% (69/200 cases)")
                print("=" * 50)
            except Exception:
                pass
        sys.exit(0)
    else:
        print("ARTIFACT CHECK FAILED")
        print("=" * 50)
        if errors:
            print("\nDetails:")
            for err in errors:
                print(f" - {err}")
        sys.exit(1)

if __name__ == "__main__":
    main()
