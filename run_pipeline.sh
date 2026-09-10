#!/bin/bash
set -e

echo "Running 04_discover_intents.py..."
python scripts/04_discover_intents.py

echo "Running 05_diagnose_similarity.py..."
python scripts/05_diagnose_similarity.py

echo "Running 06_build_retriever.py..."
python scripts/06_build_retriever.py

echo "Running 07_build_classifier.py..."
python scripts/07_build_classifier.py

echo "Running 08_test_agent_components.py..."
python scripts/08_test_agent_components.py

echo "Running 10b_prepare_human_labels.py prepare..."
python scripts/10b_prepare_human_labels.py prepare

echo "=================================================="
echo "NOW LABEL THE 200 GOLDEN CASES in data/processed/eval/golden_labels_manual.csv"
echo "=================================================="
read -p "Press Enter after manual labels are complete..."

echo "Running 10b_prepare_human_labels.py finalize..."
python scripts/10b_prepare_human_labels.py finalize

echo "Running 11_run_evaluation.py..."
python scripts/11_run_evaluation.py

echo "Running 12_llm_judge.py..."
python scripts/12_llm_judge.py

echo "Running 13_judge_human_agreement.py..."
python scripts/13_judge_human_agreement.py

echo "Pipeline execution finished successfully!"
