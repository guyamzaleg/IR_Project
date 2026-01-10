import pickle
import pandas as pd
import os
import numpy as np
from inverted_index_gcp import InvertedIndex

def load_index(index_type):
    """Load inverted index: 'text', 'title', or 'anchor'"""
    index_map = {
        "text": ("data/postings_gcp/text", "text_index"),
        "title": ("data/postings_gcp/title", "title_index"),
        "anchor": ("data/postings_gcp/anchor", "anchor_index")
    }
    base_dir, index_name = index_map[index_type]
    index = InvertedIndex.read_index(base_dir, index_name)
    print(f"✓ {index_type.capitalize()} index: {len(index.df)} terms")
    return index

def load_all_indexes():
    """Load all three indexes"""
    return {
        'text': load_index('text'),
        'title': load_index('title'),
        'anchor': load_index('anchor')
    }

def load_pagerank():
    """Load PageRank scores"""
    print("Loading PageRank...")
    pr_files = [os.path.join("data/pr", f) for f in os.listdir("data/pr") if f.endswith('.csv.gz')]
    
    if not pr_files:
        return {}
    
    dfs = [pd.read_csv(f, header=None, names=['doc_id', 'pagerank']) for f in pr_files]
    pr_df = pd.concat(dfs, ignore_index=True)
    pr_dict = dict(zip(pr_df['doc_id'].astype(int), pr_df['pagerank']))
    
    print(f"✓ PageRank: {len(pr_dict)} documents")
    return pr_dict

def load_pageviews():
    """Load PageViews"""
    print("Loading PageViews...")
    path = 'pv/pageview.pkl'
    
    try:
        with open('data/mappings/doc_id_to_title.pkl', 'rb') as f:
            titles = pickle.load(f)
        print(f"✓ Titles: {len(titles)} documents")
        return titles
    except FileNotFoundError:
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