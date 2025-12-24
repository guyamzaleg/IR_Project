import pickle
import pandas as pd
import os
from inverted_index_gcp import InvertedIndex

def load_index():
    """Load the inverted index from local disk."""
    print("Loading inverted index...")
    index = InvertedIndex.read_index("data/postings_gcp", "index")
    print(f"✓ Index loaded: {len(index.df)} terms")
    return index

def load_pagerank():
    """Load PageRank scores from CSV files."""
    print("Loading PageRank...")
    pr_files = [f"data/{f}" for f in os.listdir("data") if f.endswith('.csv.gz')]
    
    if not pr_files:
        print("⚠ No PageRank files found")
        return {}
    
    dfs = []
    for file in pr_files:
        df = pd.read_csv(file, header=None, names=['doc_id', 'pagerank'])
        dfs.append(df)
    
    pr_df = pd.concat(dfs)
    pr_dict = dict(zip(pr_df['doc_id'].astype(int), pr_df['pagerank']))
    print(f"✓ PageRank loaded: {len(pr_dict)} documents")
    return pr_dict
    
    # index = InvertedIndex.read_index("postings_gcp", "index", bucket_name=BUCKET_NAME)
    # print(f"✓ Index loaded: {len(index.df)} terms")
    # print(f"✓ Mode: {'LOCAL' if WORK_LOCALLY else 'REMOTE (GCS)'}")
    # return index

# def load_pagerank():
#     """Load PageRank scores from CSV."""
#     print("Loading PageRank...")
    