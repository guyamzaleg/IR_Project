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
    try:
        with open('data/pv/pageview.pkl', 'rb') as f:
            pv = pickle.load(f)
        print(f"✓ PageViews: {len(pv)} documents")
        return pv
    except FileNotFoundError:
        return {}

def load_doc_titles():
    """Load document titles"""
    print("Loading titles...")
    try:
        with open('data/mappings/doc_id_to_title.pkl', 'rb') as f:
            titles = pickle.load(f)
        print(f"✓ Titles: {len(titles)} documents")
        return titles
    except FileNotFoundError:
        return {}

def load_embeddings(field='title'):
    """Load embeddings"""
    print(f"Loading {field} embeddings...")
    try:
        doc_ids = np.load(f'data/embeddings/{field}/{field}_doc_ids.npy')
        embeddings = np.load(f'data/embeddings/{field}/{field}_embeddings.npy')
        print(f"✓ Embeddings: {len(doc_ids)} documents")
        return doc_ids, embeddings
    except FileNotFoundError:
        return None, None

def load_all_data():
    """Load everything"""
    print("=" * 60)
    data = {
        'indexes': load_all_indexes(),
        'pagerank': load_pagerank(),
        'pageviews': load_pageviews(),
        'titles': load_doc_titles()
    }
    
    doc_ids, embeddings = load_embeddings('title')
    if doc_ids is not None:
        data['embeddings'] = {'doc_ids': doc_ids, 'vectors': embeddings}
    
    print("=" * 60)
    return data