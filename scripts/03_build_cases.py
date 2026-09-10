import pandas as pd
from pathlib import Path
from langdetect import DetectorFactory, detect
from tqdm import tqdm

DetectorFactory.seed = 42

INPUT_PATH = Path("data/processed/amazon_conversations.parquet")
OUTPUT_PATH = Path("data/processed/support_cases.parquet")
BRAND_ID = "AmazonHelp"


def check_en(text):
    try:
        return detect(str(text)) == "en"
    except Exception:
        return False


def main():
    print("Loading conversations...")
    df = pd.read_parquet(INPUT_PATH)

    required = {
        "conversation_id", "tweet_id", "author_id", "inbound",
        "created_at", "text",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df = df[df["created_at"].notna()].copy()
    df = df.sort_values(["conversation_id", "created_at", "tweet_id"])

    cases = []
    grouped = df.groupby("conversation_id", sort=False)

    for cid, group in tqdm(grouped, total=len(grouped), desc="Building cases"):
        brand_mask = group["author_id"].astype(str).eq(BRAND_ID)
        amazon_outbound = group[(~group["inbound"]) & brand_mask]

        if amazon_outbound.empty or not group["inbound"].any():
            continue

        first_brand_response = amazon_outbound.iloc[0]
        first_response_time = first_brand_response["created_at"]

        problem_msgs = group[
            group["inbound"] & (group["created_at"] <= first_response_time)
        ]
        if problem_msgs.empty:
            continue

        customer_problem = " ".join(
            problem_msgs["text"].astype(str).str.strip()
        ).strip()
        if not customer_problem:
            continue

        transcript_lines = []
        amazonhelp_responses = []
        for _, row in group.iterrows():
            if bool(row["inbound"]):
                speaker = "Customer"
            elif str(row["author_id"]) == BRAND_ID:
                speaker = BRAND_ID
                amazonhelp_responses.append(str(row["text"]).strip())
            else:
                speaker = f"Other:{row['author_id']}"
            transcript_lines.append(f"{speaker}: {row['text']}")

        cases.append({
            "conversation_id": cid,
            "tweet_ids": group["tweet_id"].tolist(),
            "customer_problem": customer_problem,
            "first_amazon_response": str(first_brand_response["text"]),
            "amazonhelp_responses": amazonhelp_responses,
            "full_transcript": "\n\n".join(transcript_lines),
        })

    cases_df = pd.DataFrame(cases)
    if cases_df.empty:
        raise RuntimeError("No support cases were extracted.")

    tqdm.pandas()
    cases_df["is_english"] = cases_df["customer_problem"].progress_apply(check_en)
    cases_df = cases_df[cases_df["is_english"]].drop(columns=["is_english"]).copy()

    if cases_df["conversation_id"].duplicated().any():
        raise RuntimeError("Duplicate conversation_id detected.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cases_df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Extracted English AmazonHelp support cases: {len(cases_df):,}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
