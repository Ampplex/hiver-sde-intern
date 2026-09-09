import pandas as pd
import numpy as np
import json
import os
import boto3
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
BEDROCK_EMBED_MODEL = os.getenv("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")

# Initialize Bedrock client
bedrock = boto3.client('bedrock-runtime', region_name=AWS_REGION)

def get_embedding(text):
    body = json.dumps({"inputText": text, "dimensions": 1024})
    response = bedrock.invoke_model(
        body=body,
        modelId=BEDROCK_EMBED_MODEL,
        accept='application/json',
        contentType='application/json'
    )
    result = json.loads(response['body'].read())
    return result['embedding']

def main():
    print("Loading support cases...")
    df = pd.read_parquet("data/processed/support_cases_train.parquet")
    
    sample_size = min(1000, len(df))
    df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    embeddings_path = Path("data/processed/diagnostic_embeddings_train.npy")
    
    if embeddings_path.exists():
        print("Loading cached embeddings...")
        embeddings = np.load(embeddings_path)
    else:
        print(f"Generating embeddings for {sample_size} cases (this takes a few minutes)...")
        embeddings_list = []
        for i, text in enumerate(df['customer_problem']):
            if (i + 1) % 100 == 0:
                print(f"  Embedded {i + 1}/{sample_size}...")
            
            try:
                emb = get_embedding(str(text)[:8000])
                embeddings_list.append(emb)
            except Exception as e:
                print(f"  Error on case {i}: {e}")
                embeddings_list.append(np.zeros(1024).tolist())
                
        embeddings = np.array(embeddings_list)
        np.save(embeddings_path, embeddings)
        print(f"Saved embeddings to {embeddings_path}")
        
    print("\nCalculating pairwise cosine similarities...")
    # Normalize vectors
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1e-10
    normalized_embeddings = embeddings / norms
    
    # Calculate similarity matrix (dot product of normalized vectors)
    sim_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
    
    # Ignore self-similarity by setting diagonal to 0
    np.fill_diagonal(sim_matrix, 0.0)
    
    # For each case, find the max similarity to any other case (nearest neighbor)
    nearest_neighbor_sims = np.max(sim_matrix, axis=1)
    
    print("\n--- Nearest Neighbor Similarity Distribution ---")
    print(f"Min:  {np.min(nearest_neighbor_sims):.4f}")
    print(f"Mean: {np.mean(nearest_neighbor_sims):.4f}")
    print(f"P50:  {np.percentile(nearest_neighbor_sims, 50):.4f}")
    print(f"P75:  {np.percentile(nearest_neighbor_sims, 75):.4f}")
    print(f"P90:  {np.percentile(nearest_neighbor_sims, 90):.4f}")
    print(f"P95:  {np.percentile(nearest_neighbor_sims, 95):.4f}")
    print(f"P99:  {np.percentile(nearest_neighbor_sims, 99):.4f}")
    print(f"Max:  {np.max(nearest_neighbor_sims):.4f}")

    # We want to form clusters. If we want at least a few clusters to reach 3 items,
    # what threshold would group at least 3 items together?
    sims_sorted = np.sort(sim_matrix, axis=1)[:, ::-1]
    second_nearest_sims = sims_sorted[:, 1]
    
    print("\n--- Similarity to 2nd Nearest Neighbor (required for cluster of 3) ---")
    print(f"Mean: {np.mean(second_nearest_sims):.4f}")
    print(f"P50:  {np.percentile(second_nearest_sims, 50):.4f}")
    print(f"P75:  {np.percentile(second_nearest_sims, 75):.4f}")
    print(f"P90:  {np.percentile(second_nearest_sims, 90):.4f}")
    
    print("\n--- Step 3: Manual Pair Check Samples ---")
    # For different buckets, print a couple of pairs to visually inspect
    buckets = [(0.80, 0.90), (0.70, 0.80), (0.60, 0.70), (0.50, 0.60), (0.40, 0.50)]
    
    # Nearest neighbor indices
    nn_indices = np.argmax(sim_matrix, axis=1)
    
    np.random.seed(42)
    
    for lower, upper in buckets:
        print(f"\nBucket {lower:.2f} - {upper:.2f}:")
        # Find pairs in this bucket
        valid_pairs = np.where((nearest_neighbor_sims >= lower) & (nearest_neighbor_sims < upper))[0]
        
        if len(valid_pairs) == 0:
            print("  No pairs found.")
            continue
            
        # Sample up to 3 pairs
        sample_indices = np.random.choice(valid_pairs, min(3, len(valid_pairs)), replace=False)
        
        for idx in sample_indices:
            neighbor_idx = nn_indices[idx]
            sim = nearest_neighbor_sims[idx]
            text1 = df.iloc[idx]['customer_problem'].replace('\n', ' ')
            text2 = df.iloc[neighbor_idx]['customer_problem'].replace('\n', ' ')
            print(f"  Score: {sim:.3f}")
            print(f"  A: {text1}")
            print(f"  B: {text2}")
            print(f"  ---")

if __name__ == "__main__":
    main()
