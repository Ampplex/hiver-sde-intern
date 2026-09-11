import sys
from pathlib import Path
import pandas as pd

def inspect_cases():
    target_path = Path("data/processed/support_cases_train.parquet")
    if not target_path.exists():
        target_path = Path("data/processed/support_cases.parquet")
        
    if not target_path.exists():
        print("Error: Neither support_cases_train.parquet nor support_cases.parquet found in data/processed/")
        sys.exit(1)
        
    print(f"Loading {target_path}...")
    df = pd.read_parquet(target_path)
    
    print("\n--- Basic Stats ---")
    print(f"Total Support Cases: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    
    print("\n--- Customer Problem Examples ---")
    for i, problem in enumerate(df['customer_problem'].head(3)):
        print(f"Example {i+1}: {problem}")
        
    if 'full_transcript' in df.columns:
        print("\n--- Conversation Lengths ---")
        turns = df['full_transcript'].str.count('\n\n') + 1
        print(f"Average turns per conversation: {turns.mean():.2f}")
        print(f"Max turns in a conversation: {turns.max()}")
        print(f"Min turns in a conversation: {turns.min()}")
    
    print("\n--- Duplicates Analysis ---")
    exact_duplicates = df['customer_problem'].duplicated().sum()
    print(f"Exact duplicate problems: {exact_duplicates} out of {len(df):,}")
    
    short_problems = df[df['customer_problem'].str.len() < 10]
    print(f"Very short problems (<10 chars): {len(short_problems)}")
    if len(short_problems) > 0:
        print("Examples of short problems:")
        print(short_problems['customer_problem'].head(3).tolist())

if __name__ == "__main__":
    inspect_cases()
