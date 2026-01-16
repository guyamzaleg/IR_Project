import pickle
import pandas as pd
import os
import numpy as np
from inverted_index_gcp import InvertedIndex

def load_index(type):
    if type == "text":
        index = InvertedIndex.read_index("data/postings_gcp/text", "text_index")
    if type == "title":
        index = InvertedIndex.read_index("data/postings_gcp/title", "title_index")
    if type == "anchor":
        index = InvertedIndex.read_index("data/postings_gcp/anchor", "anchor_index")
    print(f"✓ {type.capitalize()} Index loaded: {len(index.df)} terms")
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
            pageviews = pickle.load(f)
        print(f"✓ PageViews: {len(pageviews)} documents")
        return pageviews
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

def load_faiss_index(index_name='text'):
    """
    Load a FAISS index and its corresponding doc_ids mapping.

    Args:
        index_name: 'text' or 'title'

    Returns:
        tuple: (faiss_index, doc_ids_array) or (None, None) if not found
    """
    import faiss
    from pathlib import Path

    index_path = Path(f'data/embeddings/{index_name}_vector/{index_name}_index.faiss')
    docids_path = Path(f'data/embeddings/{index_name}_vector/doc_ids.npy')

    if not index_path.exists():
        print(f"⚠ FAISS index not found: {index_path}")
        return None, None

    if not docids_path.exists():
        print(f"⚠ Doc IDs mapping not found: {docids_path}")
        return None, None

    print(f"Loading FAISS index: {index_name}...")
    faiss_index = faiss.read_index(str(index_path))

    # doc_ids.npy was saved as memmap (see vector_indexer.py), load accordingly
    doc_ids = np.memmap(str(docids_path), dtype=np.int32, mode='r')

    print(f"✓ FAISS {index_name}: {faiss_index.ntotal} vectors, {len(doc_ids)} doc_ids")
    return faiss_index, doc_ids


def load_all_data():
    """Load everything"""
    print("=" * 60)

    # Load FAISS indexes
    text_faiss, text_faiss_docids = load_faiss_index('text')
    title_faiss, title_faiss_docids = load_faiss_index('title')

    data = {
        'indexes': load_all_indexes(),
        'pagerank': load_pagerank(),
        'pageviews': load_pageviews(),
        'titles': load_doc_titles(),
        'text_faiss': text_faiss,
        'text_faiss_docids': text_faiss_docids,
        'title_faiss': title_faiss,
        'title_faiss_docids': title_faiss_docids,
    }

    print("=" * 60)
    return data