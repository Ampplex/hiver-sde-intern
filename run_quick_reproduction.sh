#!/usr/bin/env bash
set -e

echo "============================================================"
echo "HIVER TAKE-HOME: QUICK REPRODUCTION PATH"
echo "============================================================"
echo "Using frozen submission artifacts to reproduce headline results in <15 min."
echo "No expensive LLM taxonomy discovery will be rerun."
echo "============================================================"

# Step 1: Validate required frozen artifacts exist and are consistent
python scripts/verify_submission_artifacts.py

# Step 2: Build retriever from frozen assignments and embeddings (<5 sec)
echo ""
echo "--> [1/6] Building Hybrid Retriever from frozen embeddings..."
python scripts/06_build_retriever.py

# Step 3: Build intent classifier from frozen assignments and embeddings (<5 sec)
echo ""
echo "--> [2/6] Building Intent Classifier centroids..."
python scripts/07_build_classifier.py

# Check if AWS credentials are configured for live LLM inference
HAS_AWS=0
if python -c '
import os, boto3
from dotenv import load_dotenv
load_dotenv()
try:
    session = boto3.Session()
    creds = session.get_credentials()
    if creds and creds.access_key:
        exit(0)
    exit(1)
except Exception:
    exit(1)
' 2>/dev/null; then
    HAS_AWS=1
fi

if [ "$HAS_AWS" -eq 1 ]; then
    # Step 4: Smoke test agent components (<5 sec)
    echo ""
    echo "--> [3/6] Testing agent components via Bedrock..."
    python scripts/08_test_agent_components.py

    # Step 5: Run end-to-end evaluation on the 200-case audited golden set (~2-3 min)
    echo ""
    echo "--> [4/6] Running system evaluation on 200 golden cases..."
    python scripts/11_run_evaluation.py

    # Step 6: Run LLM judge on auto-handled responses (~2-3 min)
    echo ""
    echo "--> [5/6] Running LLM judge evaluation..."
    python scripts/12_llm_judge.py
else
    echo ""
    echo "============================================================"
    echo "NOTE: AWS credentials not detected in .env or environment."
    echo "Skipping live LLM inference (steps 3-5)."
    echo "To run live LLM evaluation, copy .env.example to .env and configure AWS credentials."
    echo "Using pre-computed submission evaluation artifacts."
    echo "============================================================"
fi

# Step 7: Evaluate human-judge agreement (<5 sec)
if [ -f "data/processed/eval/human_judge_scores.csv" ]; then
    echo ""
    echo "--> [6/6] Computing LLM-vs-Human judge agreement..."
    python scripts/13_judge_human_agreement.py
else
    echo ""
    echo "--> [6/6] Skipping 13_judge_human_agreement.py (data/processed/eval/human_judge_scores.csv not present)."
fi

echo ""
echo "============================================================"
echo "QUICK REPRODUCTION COMPLETE"
echo "============================================================"
echo "Headline metrics saved to:"
echo " - data/processed/eval/evaluation_metrics.json"
echo " - data/processed/eval/system_predictions.parquet"
if [ -f "data/processed/eval/llm_judge_scores.parquet" ]; then
    echo " - data/processed/eval/llm_judge_scores.parquet"
fi
if [ -f "data/processed/eval/judge_human_agreement.json" ]; then
    echo " - data/processed/eval/judge_human_agreement.json"
fi
echo "============================================================"
