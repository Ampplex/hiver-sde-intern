import pandas as pd
from pathlib import Path
from langdetect import detect
from tqdm import tqdm


INPUT_PATH = Path(
    "data/processed/amazon_conversations.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/support_cases.parquet"
)


def check_en(text):

    try:
        return detect(
            str(text)
        ) == "en"
    except Exception:
        return False


def main():

    print(
        "Loading conversations..."
    )

    df = pd.read_parquet(
        INPUT_PATH
    )

    required = {
        "conversation_id",
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
    }

    missing = (
        required - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    # Explicit timestamp parsing.
    df["created_at"] = pd.to_datetime(
        df["created_at"],
        errors="coerce",
        utc=True,
    )

    invalid_dates = df[
        "created_at"
    ].isna().sum()

    if invalid_dates:
        print(
            f"Warning: {invalid_dates:,} rows "
            "have invalid timestamps and will be removed."
        )

        df = df[
            df["created_at"].notna()
        ].copy()

    # Deterministic ordering:
    # timestamp first, tweet_id as tie-breaker.
    df = df.sort_values(
        by=[
            "conversation_id",
            "created_at",
            "tweet_id",
        ]
    )

    cases = []

    grouped = df.groupby(
        "conversation_id",
        sort=False,
    )

    print(
        f"Processing {len(grouped):,} conversations..."
    )

    for cid, group in tqdm(
        grouped,
        total=len(grouped),
        desc="Building cases",
    ):

        has_inbound = bool(
            group["inbound"].any()
        )

        has_outbound = bool(
            (~group["inbound"]).any()
        )

        if not (
            has_inbound
            and has_outbound
        ):
            continue

        outbound = group[
            ~group["inbound"]
        ]

        first_outbound = outbound.iloc[0]

        first_outbound_time = (
            first_outbound[
                "created_at"
            ]
        )

        # Customer messages received before the first
        # brand response form the problem statement.
        problem_msgs = group[
            group["inbound"]
            &
            (
                group["created_at"]
                <= first_outbound_time
            )
        ]

        if problem_msgs.empty:
            continue

        customer_problem = " ".join(
            problem_msgs[
                "text"
            ]
            .astype(str)
            .str.strip()
        )

        if not customer_problem.strip():
            continue

        transcript_lines = []

        for _, row in group.iterrows():

            speaker = (
                "Customer"
                if row["inbound"]
                else "AmazonHelp"
            )

            transcript_lines.append(
                f"{speaker}: {row['text']}"
            )

        full_transcript = (
            "\n\n".join(
                transcript_lines
            )
        )

        cases.append(
            {
                "conversation_id": cid,
                "tweet_ids": group[
                    "tweet_id"
                ].tolist(),
                "customer_problem": (
                    customer_problem
                ),
                "first_amazon_response": (
                    first_outbound["text"]
                ),
                "full_transcript": (
                    full_transcript
                ),
            }
        )

    cases_df = pd.DataFrame(
        cases
    )

    if cases_df.empty:
        raise RuntimeError(
            "No support cases were extracted."
        )

    print(
        f"Extracted {len(cases_df):,} "
        "support cases."
    )

    # Language filtering.
    tqdm.pandas()

    cases_df["is_english"] = (
        cases_df[
            "customer_problem"
        ]
        .progress_apply(
            check_en
        )
    )

    english_cases = cases_df[
        cases_df["is_english"]
    ].drop(
        columns=["is_english"]
    ).copy()

    print(
        f"English support cases: "
        f"{len(english_cases):,}"
    )

    # One row per conversation.
    if (
        english_cases[
            "conversation_id"
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Duplicate conversation_id detected."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    english_cases.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"Saved to:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
