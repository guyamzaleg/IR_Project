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


def load_embeddings(field='title'):
    """
    Load embeddings from parquet files into a dict for fast lookup.

    Returns:
    --------
        dict: {doc_id: np.array} mapping document IDs to embedding vectors
    """
    import pyarrow.parquet as pq
    from pathlib import Path

    print(f"Loading {field} embeddings...")
    embeddings_dir = Path(f'data/embeddings/{field}')

    if not embeddings_dir.exists():
        print(f"⚠ Embeddings directory not found: {embeddings_dir}")
        return {}

    parquet_files = list(embeddings_dir.glob('*.parquet'))
    if not parquet_files:
        print(f"⚠ No parquet files found in {embeddings_dir}")
        return {}

    embeddings_dict = {}
    for pf in parquet_files:
        table = pq.read_table(pf)
        doc_ids = table['doc_id'].to_pylist()
        vectors = table['embedding'].to_pylist()
        for doc_id, vec in zip(doc_ids, vectors):
            embeddings_dict[doc_id] = np.array(vec, dtype=np.float32)

    print(f"✓ {field.capitalize()} Embeddings: {len(embeddings_dict)} documents")
    return embeddings_dict


# def load_embeddings(field='title') -> tuple:
#     """Load embeddings and doc_ids."""
#     import numpy as np
#     doc_ids = np.load(f'embeddings/{field}/{field}_doc_ids.npy')
#     embeddings = np.load(f'embeddings/{field}/{field}_embeddings.npy')
#     return doc_ids, embeddings

def load_all_data(load_emb=True):
    """Load everything"""
    print("=" * 60)
    data = {
        'indexes': load_all_indexes(),
        'pagerank': load_pagerank(),
        'pageviews': load_pageviews(),
        'titles': load_doc_titles()
    }

    if load_emb:
        data['embeddings'] = load_embeddings('title')

    print("=" * 60)
    return data