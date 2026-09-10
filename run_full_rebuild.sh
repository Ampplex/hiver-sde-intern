#!/usr/bin/env bash
set -e

echo "============================================================"
echo "HIVER TAKE-HOME: FULL REBUILD PIPELINE (TRANSPARENCY / AUDIT)"
echo "============================================================"
echo "NOTE: This path rebuilds the expensive LLM taxonomy discovery"
echo "from scratch via AWS Bedrock APIs (~60-90 minutes)."
echo "This is NOT the quick reproduction path."
echo "============================================================"

# Step 1: Create golden eval split (200 cases) and training set (8,000 cases)
echo ""
echo "--> [1/6] Creating 200 golden / 8,000 training case split..."
python scripts/10_create_eval_set.py

# Step 2: Discover intent taxonomy from scratch via LLM + clustering + assignment
echo ""
echo "--> [2/6] Running LLM-assisted intent discovery and assignment (04_discover_intents.py)..."
python scripts/04_discover_intents.py

# Step 3: Run cosine similarity distribution diagnostics
echo ""
echo "--> [3/6] Running embedding similarity diagnostics..."
python scripts/05_diagnose_similarity.py

# Step 4: Build hybrid retriever
echo ""
echo "--> [4/6] Building hybrid retriever..."
python scripts/06_build_retriever.py

# Step 5: Build intent classifier
echo ""
echo "--> [5/6] Building intent classifier centroids..."
python scripts/07_build_classifier.py

# Step 6: Smoke test agent components
echo ""
echo "--> [6/6] Testing agent components..."
python scripts/08_test_agent_components.py

echo ""
echo "============================================================"
echo "FULL REBUILD STAGE 1-6 COMPLETE"
echo "============================================================"
echo "STOPPING HERE: Human labeling is required before running evaluation."
echo "To evaluate:"
echo " 1. Prepare labeling template: python scripts/10b_prepare_human_labels.py prepare"
echo " 2. Hand-label the 200 golden cases in data/processed/eval/golden_labels_manual.csv"
echo " 3. Finalize golden set:       python scripts/10b_prepare_human_labels.py finalize"
echo " 4. Run evaluation suite:      python scripts/11_run_evaluation.py"
echo "                               python scripts/12_llm_judge.py"
echo "                               python scripts/13_judge_human_agreement.py"
echo "============================================================"
