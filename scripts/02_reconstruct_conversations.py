import pandas as pd
from pathlib import Path


RAW_PATH = Path(
    "data/raw/twcs.csv"
)

OUTPUT_PATH = Path(
    "data/processed/amazon_conversations.parquet"
)


def main():

    print(
        "Loading raw Twitter support dataset..."
    )

    df = pd.read_csv(
        RAW_PATH
    )

    required = {
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "in_response_to_tweet_id",
    }

    missing = (
        required - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    print(
        f"Loaded {len(df):,} tweets."
    )

    # --------------------------------------------------------
    # Normalize identifiers
    # --------------------------------------------------------

    df["tweet_id"] = (
        pd.to_numeric(
            df["tweet_id"],
            errors="coerce",
        )
        .astype("Int64")
    )

    df["in_response_to_tweet_id"] = (
        pd.to_numeric(
            df["in_response_to_tweet_id"],
            errors="coerce",
        )
        .astype("Int64")
    )

    df = df[
        df["tweet_id"].notna()
    ].copy()

    # --------------------------------------------------------
    # Parse timestamps before any chronological operation.
    # --------------------------------------------------------

    df["created_at"] = pd.to_datetime(
        df["created_at"],
        errors="coerce",
        utc=True,
    )

    # --------------------------------------------------------
    # Union-Find
    # --------------------------------------------------------

    print(
        "Building conversation graph..."
    )

    parent = {}

    def find(x):

        root = x

        while parent[root] != root:
            root = parent[root]

        while parent[x] != x:
            next_node = parent[x]
            parent[x] = root
            x = next_node

        return root

    def union(a, b):

        root_a = find(a)
        root_b = find(b)

        if root_a != root_b:
            parent[root_b] = root_a

    for tweet_id in df[
        "tweet_id"
    ]:

        tweet_id = int(
            tweet_id
        )

        parent[tweet_id] = tweet_id

    for row in df[
        [
            "tweet_id",
            "in_response_to_tweet_id",
        ]
    ].itertuples(
        index=False
    ):

        tweet_id = int(
            row.tweet_id
        )

        parent_id = row.in_response_to_tweet_id

        if pd.notna(parent_id):

            parent_id = int(
                parent_id
            )

            if parent_id in parent:
                union(
                    tweet_id,
                    parent_id,
                )

    print(
        "Assigning conversation IDs..."
    )

    df["conversation_id"] = [
        find(int(tweet_id))
        for tweet_id
        in df["tweet_id"]
    ]

    # --------------------------------------------------------
    # Keep conversations involving AmazonHelp.
    # --------------------------------------------------------

    amazon_ids = set(
        df.loc[
            df["author_id"].astype(str)
            == "AmazonHelp",
            "conversation_id",
        ]
    )

    print(
        f"AmazonHelp conversations: "
        f"{len(amazon_ids):,}"
    )

    conv_df = df[
        df["conversation_id"].isin(
            amazon_ids
        )
    ].copy()

    # Deterministic chronological ordering.
    conv_df = conv_df.sort_values(
        by=[
            "conversation_id",
            "created_at",
            "tweet_id",
        ]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    conv_df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"Tweets retained: "
        f"{len(conv_df):,}"
    )

    print(
        f"Saved to:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
