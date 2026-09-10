import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

JUDGE_PATH = Path("data/processed/eval/llm_judge_scores.parquet")
HUMAN_PATH = Path("data/processed/eval/human_judge_scores.csv")
OUTPUT_PATH = Path("data/processed/eval/judge_human_agreement.json")
METRICS = ["correctness", "groundedness", "resolution_appropriateness", "completeness", "communication_quality", "overall"]
MIN_HUMAN_OVERLAP = 40


def validate_scores(df, columns, source):
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"{source} missing columns: {sorted(missing)}")
    for col in columns:
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isna().any() or (values < 1).any() or (values > 5).any():
            raise ValueError(f"{source}.{col} contains invalid scores.")


def main():
    llm = pd.read_parquet(JUDGE_PATH)
    human = pd.read_csv(HUMAN_PATH)
    validate_scores(llm, [f"system_{m}" for m in METRICS], "LLM judge")
    validate_scores(human, METRICS, "Human judge")
    if human["conversation_id"].duplicated().any():
        raise ValueError("Duplicate human conversation IDs.")

    merged = llm.merge(human[["conversation_id", *METRICS]], on="conversation_id", how="inner", validate="one_to_one")
    if len(merged) < MIN_HUMAN_OVERLAP:
        raise ValueError(f"Need at least {MIN_HUMAN_OVERLAP} overlapping examples; found {len(merged)}.")

    results = {"n_overlap": int(len(merged)), "metrics": {}}
    for metric in METRICS:
        a = merged[f"system_{metric}"].astype(int)
        b = merged[metric].astype(int)
        rho = spearmanr(a, b).statistic
        kappa = cohen_kappa_score(a, b, weights="quadratic")
        mae = (a - b).abs().mean()
        results["metrics"][metric] = {"spearman": float(rho), "quadratic_kappa": float(kappa), "mae": float(mae)}
        print(f"{metric:<30} Spearman={rho:.3f} QuadraticKappa={kappa:.3f} MAE={mae:.3f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Agreement results: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
