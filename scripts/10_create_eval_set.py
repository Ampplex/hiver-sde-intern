from pathlib import Path

import pandas as pd


SOURCE = Path(
    "data/processed/support_cases.parquet"
)

EVAL_DIR = Path(
    "data/processed/eval"
)

GOLDEN_PATH = (
    EVAL_DIR / "golden_set.parquet"
)

TRAIN_PATH = Path(
    "data/processed/support_cases_train.parquet"
)


def main():

    print("Loading support cases...")

    df = pd.read_parquet(
        SOURCE
    )

    required = {
        "conversation_id",
        "customer_problem",
        "first_amazon_response",
        "full_transcript",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if df["conversation_id"].duplicated().any():
        raise ValueError(
            "conversation_id is duplicated. "
            "Expected one support case per conversation."
        )

    if len(df) < 200:
        raise ValueError(
            f"Need at least 200 cases, "
            f"found {len(df)}."
        )

    EVAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------
    # Golden set
    # ------------------------------------------

    golden_df = df.sample(
        n=200,
        random_state=42,
    ).copy()

    golden_ids = set(
        golden_df["conversation_id"]
    )

    # ------------------------------------------
    # Training set
    # ------------------------------------------

    train_df = df[
        ~df["conversation_id"].isin(
            golden_ids
        )
    ].copy()

    golden_df = golden_df.reset_index(
        drop=True
    )

    train_df = train_df.reset_index(
        drop=True
    )

    # ------------------------------------------
    # Safety assertions
    # ------------------------------------------

    overlap = (
        set(golden_df["conversation_id"])
        &
        set(train_df["conversation_id"])
    )

    if overlap:
        raise RuntimeError(
            f"Data leakage detected: "
            f"{len(overlap)} conversations overlap."
        )

    # ------------------------------------------
    # Save
    # ------------------------------------------

    golden_df.to_parquet(
        GOLDEN_PATH,
        index=False,
    )

    train_df.to_parquet(
        TRAIN_PATH,
        index=False,
    )

    print(
        f"Golden evaluation set: "
        f"{len(golden_df)}"
    )

    print(
        f"Training/retrieval set: "
        f"{len(train_df):,}"
    )

    print(
        "Verified: zero conversation overlap."
    )


if __name__ == "__main__":
    main()
