import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score


JUDGE_PATH = (
    "data/processed/eval/llm_judge_scores.parquet"
)

HUMAN_PATH = (
    "data/processed/eval/human_judge_scores.csv"
)

METRICS = [
    "correctness",
    "groundedness",
    "resolution_appropriateness",
    "completeness",
    "communication_quality",
    "overall",
]


def validate_scores(
    df,
    columns,
    source_name,
):
    for column in columns:

        if column not in df.columns:
            raise ValueError(
                f"{source_name} is missing column: "
                f"{column}"
            )

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if values.isna().any():
            raise ValueError(
                f"{source_name}.{column} contains "
                "missing or non-numeric values."
            )

        if (
            (values < 1)
            | (values > 5)
        ).any():

            raise ValueError(
                f"{source_name}.{column} contains "
                "scores outside 1-5."
            )


def main():

    llm = pd.read_parquet(
        JUDGE_PATH
    )

    human = pd.read_csv(
        HUMAN_PATH
    )

    if "conversation_id" not in human.columns:
        raise ValueError(
            "human_judge_scores.csv must contain "
            "conversation_id."
        )

    if human["conversation_id"].duplicated().any():
        raise ValueError(
            "human_judge_scores.csv contains duplicate "
            "conversation_id values."
        )

    validate_scores(
        llm,
        [
            f"system_{metric}"
            for metric in METRICS
        ],
        "LLM judge",
    )

    validate_scores(
        human,
        METRICS,
        "Human judge",
    )

    merged = llm.merge(
        human,
        on="conversation_id",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) < 20:
        raise ValueError(
            "Need at least 20 overlapping "
            "human-judged examples."
        )

    print(
        f"Agreement subset: n={len(merged)}"
    )

    print("\n=== SYSTEM JUDGE / HUMAN AGREEMENT ===")

    for metric in METRICS:

        llm_scores = pd.to_numeric(
            merged[
                f"system_{metric}"
            ],
            errors="coerce",
        ).astype(int)

        human_scores = pd.to_numeric(
            merged[metric],
            errors="coerce",
        ).astype(int)

        rho_result = spearmanr(
            llm_scores,
            human_scores,
        )

        kappa = cohen_kappa_score(
            llm_scores,
            human_scores,
            weights="quadratic",
        )

        mae = (
            (llm_scores - human_scores)
            .abs()
            .mean()
        )

        print(
            f"{metric:<30}"
            f"Spearman={rho_result.statistic:.3f} "
            f"QuadraticKappa={kappa:.3f} "
            f"MAE={mae:.3f}"
        )


if __name__ == "__main__":
    main()
