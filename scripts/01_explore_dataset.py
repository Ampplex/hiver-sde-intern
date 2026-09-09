import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw/twcs.csv")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 100_000


def inspect_dataset():
    df = pd.read_csv(DATA_PATH, nrows=5)

    print("Columns:")
    print(df.columns.tolist())

    print("\nSample:")
    print(df.head().to_string(index=False))


def find_support_accounts():
    stats = {}

    for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE):

        outbound = chunk[chunk["inbound"] == False]

        counts = outbound.groupby("author_id").size()

        for author_id, count in counts.items():
            stats[author_id] = stats.get(author_id, 0) + count

    result = (
        pd.Series(stats, name="outbound_tweets")
        .sort_values(ascending=False)
        .to_frame()
    )

    result.index.name = "author_id"

    return result


def extract_brand(brand_author_id):

    chunks = []

    for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE):

        selected = chunk[
            chunk["author_id"].astype(str) == str(brand_author_id)
        ]

        if not selected.empty:
            chunks.append(selected)

    brand_df = pd.concat(chunks, ignore_index=True)

    brand_df["created_at"] = pd.to_datetime(
        brand_df["created_at"],
        errors="coerce"
    )

    brand_df = brand_df.sort_values("created_at")

    return brand_df


def main():

    # 1. Understand dataset
    inspect_dataset()

    # 2. Find candidate support accounts
    support_accounts = find_support_accounts()

    print("\nTop support accounts:")
    print(support_accounts.head(30))

    support_accounts.to_csv(
        OUTPUT_DIR / "support_accounts.csv"
    )

    # 3. After inspection, select ONE brand
    BRAND_AUTHOR_ID = "AmazonHelp"

    if BRAND_AUTHOR_ID is None:
        print("\nSelect a brand from support_accounts.csv")
        return

    # 4. Extract selected brand
    brand_df = extract_brand(BRAND_AUTHOR_ID)

    # 5. Save
    output = OUTPUT_DIR / f"brand_{BRAND_AUTHOR_ID}.parquet"

    brand_df.to_parquet(output, index=False)

    print(f"\nSaved: {output}")
    print(f"Rows: {len(brand_df):,}")


if __name__ == "__main__":
    main()