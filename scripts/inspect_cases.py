import pandas as pd

def inspect_cases():
    print("Loading support_cases.parquet...")
    df = pd.read_parquet("data/processed/support_cases.parquet")
    
    print("\n--- Basic Stats ---")
    print(f"Total Support Cases: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    print("\n--- Customer Problem Examples ---")
    for i, problem in enumerate(df['customer_problem'].head(3)):
        print(f"Example {i+1}: {problem}")
        
    print("\n--- Conversation Lengths ---")
    # Number of newlines in full transcript generally correlates with turns
    turns = df['full_transcript'].str.count('\n\n') + 1
    print(f"Average turns per conversation: {turns.mean():.2f}")
    print(f"Max turns in a conversation: {turns.max()}")
    print(f"Min turns in a conversation: {turns.min()}")
    
    print("\n--- Duplicates Analysis ---")
    exact_duplicates = df['customer_problem'].duplicated().sum()
    print(f"Exact duplicate problems: {exact_duplicates} out of {len(df)}")
    
    # Check for short/empty problems
    short_problems = df[df['customer_problem'].str.len() < 10]
    print(f"Very short problems (<10 chars): {len(short_problems)}")
    if len(short_problems) > 0:
        print("Examples of short problems:")
        print(short_problems['customer_problem'].head(3).tolist())

if __name__ == "__main__":
    inspect_cases()
