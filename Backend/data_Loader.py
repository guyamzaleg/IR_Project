import pickle
import pandas as pd
import os
from inverted_index_gcp import InvertedIndex

def load_index(type):
    if type == "text":
        index = InvertedIndex.read_index("data/postings_gcp", "index")
    if type == "title":
        index = InvertedIndex.read_index("data/postings_gcp", "title_index")
    if type == "anchor":
        index = InvertedIndex.read_index("data/postings_gcp", "anchor_index")
    print(f"✓ {type.capitalize()} Index loaded: {len(index.df)} terms")
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
    
def load_pageviews() -> dict:
    """Load PageViews from pickle file."""
    print("Loading PageViews...")
    path = 'pv/pageview.pkl'
    
    try:
        with open(path, 'rb') as f:
            pageviews = pickle.load(f)
        print(f"✓ PageViews: {len(pageviews)} documents")
        return pageviews
    except FileNotFoundError:
        print(f"⚠️ PageViews not found: {path}")
        return {}


def load_doc_titles() -> dict:
    """Load document titles from even/odd pickle files."""
    print("Loading document titles...")
    
    even_path = 'id_title/even_id_title_dict.pkl'
    odd_path = 'id_title/uneven_id_title_dict.pkl'
    
    titles = {}
    
    for path in [even_path, odd_path]:
        try:
            with open(path, 'rb') as f:
                titles.update(pickle.load(f))
        except FileNotFoundError:
            print(f"⚠️ Not found: {path}")
    
    print(f"✓ Titles: {len(titles)} documents")
    return titles

def load_embeddings(field='title') -> tuple:
    """Load embeddings and doc_ids."""
    import numpy as np
    doc_ids = np.load(f'embeddings/{field}/{field}_doc_ids.npy')
    embeddings = np.load(f'embeddings/{field}/{field}_embeddings.npy')
    return doc_ids, embeddings