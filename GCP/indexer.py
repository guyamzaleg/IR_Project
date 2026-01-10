import pyspark
import sys
from collections import Counter, OrderedDict, defaultdict
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
import re
from operator import itemgetter
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from pathlib import Path
import pickle
from google.cloud import storage
import hashlib

# For embeddings (lazy loaded in workers)
import gensim.downloader as api
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, ArrayType, FloatType

def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()

from inverted_index_gcp import InvertedIndex

english_stopwords = frozenset([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 
    'there', 'when', 'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most', 
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 
    'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 
    'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', 
    "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 
    'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 
    'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 
    'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
])

corpus_stopwords = ["category", "references", "also", "external", "links", 
                    "may", "first", "see", "history", "people", "one", "two", 
                    "part", "thumb", "including", "second", "following", 
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

NUM_BUCKETS = 124
def token2bucket_id(token):
  return int(_hash(token),16) % NUM_BUCKETS

def word_count(text, id):
  ''' Count the frequency of each word in `text` (tf) that is not included in
  `all_stopwords` and return entries that will go into our posting lists.
  Parameters:
  -----------
    text: str
      Text of one document
    id: int
      Document id
  Returns:
  --------
    List of tuples
      A list of (token, (doc_id, tf)) pairs
      for example: [("Anarchism", (12, 5)), ...]
  '''
  tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
  tf_counter = Counter(token for token in tokens if token not in all_stopwords)
  return [(token, (id, tf)) for token, tf in tf_counter.items()]

def reduce_word_counts(unsorted_pl):
  ''' Returns a sorted posting list by wiki_id.
  Parameters:
  -----------
    unsorted_pl: list of tuples
      A list of (wiki_id, tf) tuples
  Returns:
  --------
    list of tuples
      A sorted posting list.
  '''
  return sorted(unsorted_pl, key=lambda x: x[0])

def calculate_df(postings):
  ''' Takes a posting list RDD and calculate the df for each token.
  Parameters:
  -----------
    postings: RDD
      An RDD where each element is a (token, posting_list) pair.
  Returns:
  --------
    RDD
      An RDD where each element is a (token, df) pair.
  '''
  return postings.map(lambda x: (x[0], len(x[1])))

def partition_postings_and_write(postings, bucket_name, base_dir='postings_gcp'):
  ''' A function that partitions the posting lists into buckets, writes out
  all posting lists in a bucket to disk, and returns the posting locations for
  each bucket. Partitioning should be done through the use of `token2bucket`
  above. Writing to disk should use the function  `write_a_posting_list`, a
  static method implemented in inverted_index_colab.py under the InvertedIndex
  class.
  Parameters:
  -----------
    postings: RDD
      An RDD where each item is a (w, posting_list) pair.
  Returns:
  --------
    RDD
      An RDD where each item is a posting locations dictionary for a bucket. The
      posting locations maintain a list for each word of file locations and
      offsets its posting list was written to. See `write_a_posting_list` for
      more details.
  '''
  return (postings
          .map(lambda x: (token2bucket_id(x[0]), x))
          .groupByKey()
          .map(lambda x: InvertedIndex.write_a_posting_list(
              (x[0], list(x[1])), 
              base_dir,        # This is the directory prefix
              bucket_name        
          )))

def construct_inverted_index(pairs, client, bucket_name, index_prefix='postings_gcp', apply_filter=True):
    word_counts = pairs.flatMap(lambda x: word_count(x[0], x[1]))
    postings = word_counts.groupByKey().mapValues(reduce_word_counts)
    # filtering postings and calculate df - relevant for text only
    if (apply_filter):
        postings = postings.filter(lambda x: len(x[1])>50)
    w2df = calculate_df(postings)
    w2df_dict = w2df.collectAsMap()
    # partition posting lists and write out
    _ = partition_postings_and_write(postings, bucket_name, index_prefix).collect()

    super_posting_locs = defaultdict(list)
    for blob in client.list_blobs(bucket_name, prefix=index_prefix):
        if not blob.name.endswith("pickle"):
            continue
        with blob.open("rb") as f:
            posting_locs = pickle.load(f)
            for k, v in posting_locs.items():
                super_posting_locs[k].extend(v)
    
    # Create inverted index instance
    inverted = InvertedIndex()
    # Adding the posting locations dictionary to the inverted index
    inverted.posting_locs = super_posting_locs
    # Add the token - df dictionary to the inverted index
    inverted.df = w2df_dict
    return inverted

def write_inverted_index(inverted_index, client, bucket_name, index_name, index_prefix='postings_gcp'):
    inverted_index.write_index('.', index_name)
    # upload to gs
    index_src = f"{index_name}.pkl"
    index_dst = f'{index_prefix}/{index_src}'
    
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(index_dst)
    blob.upload_from_filename(index_src)
    
    print(f"✅ Uploaded {index_src} to gs://{bucket_name}/{index_dst}")
    
    return index_src, f'gs://{bucket_name}/{index_dst}'

def generate_graph(pages):
  ''' Compute the directed graph generated by wiki links.
  Parameters:
  -----------
    pages: RDD
      An RDD where each row consists of one wikipedia articles with 'id' and
      'anchor_text'.
  Returns:
  --------
    edges: RDD
      An RDD where each row represents an edge in the directed graph created by
      the wikipedia links. The first entry should the source page id and the
      second entry is the destination page id. No duplicates should be present.
    vertices: RDD
      An RDD where each row represents a vetrix (node) in the directed graph
      created by the wikipedia links. No duplicates should be present.
  '''
# Extract edges from (source_id, dest_id) for valid links
  edges = pages.flatMap(
        lambda x: [(x['id'], dest['id']) for dest in x['anchor_text'] if 'id' in dest and dest['id'] is not None]
    ).distinct()

# Extract vertices directly from edges
  vertices = edges.flatMap(lambda x: x).distinct().map(lambda v: (v,))

  return edges, vertices

def process_page_views(pv_path, bucket_name=None):
    """
    Download, process, and save page view statistics.
    
    Parameters:
    -----------
        pv_path: str - URL to page views data file
        bucket_name: str - Optional GCS bucket name to upload to
    
    Returns:
    --------
        dict - Dictionary mapping doc_id to page view count
    """
    from pathlib import Path
    from collections import Counter, defaultdict
    import pickle
    
    p = Path(pv_path)
    pv_name = p.name
    pv_temp = f'{p.stem}-4dedup.txt'
    
    # Download the file (2.3GB)
    import subprocess
    subprocess.run(['wget', '-N', pv_path], check=True)
    
    # Filter for English pages and extract article ID and page views
    # Keep just two fields: article ID (3) and monthly total page views (5)
    # Remove lines with non-digit values
    filter_cmd = f"bzcat {pv_name} | grep '^en\\.wikipedia' | cut -d' ' -f3,5 | grep -P '^\\d+\\s\\d+$' > {pv_temp}"
    subprocess.run(filter_cmd, shell=True, check=True)
    
    # Create a Counter that sums up page views for the same article
    wid2pv = Counter()
    with open(pv_temp, 'rt') as f:
        for line in f:
            parts = line.split(' ')
            wid2pv.update({int(parts[0]): int(parts[1])})
    
    # Convert to defaultdict
    page_view_dict = defaultdict(int)
    for doc_id, view in wid2pv.items():
        page_view_dict[doc_id] = view
    
    # Save locally
    with open("pageview.pkl", 'wb') as f:
        pickle.dump(page_view_dict, f)
    
    # Upload to GCS if bucket specified
    if bucket_name:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob('pv/pageview.pkl')
        blob.upload_from_filename('pageview.pkl')
        print(f"✅ Page views uploaded to gs://{bucket_name}/pv/pageview.pkl")
    
    return page_view_dict

def create_title_mappings(doc_title_pairs, bucket_name=None):
    """
    Create a mapping from document ID to title for UI display.
    
    Parameters:
    -----------
        doc_title_pairs: RDD - RDD of (title, doc_id) pairs
        bucket_name: str - Optional GCS bucket name to upload to
    
    Returns:
    --------
        dict - Dictionary mapping doc_id to title string
    """
    import pickle
    
    print("Creating document ID to title mappings...")
    
    # Create mapping: doc_id -> title
    title_dict = {}
    for title, doc_id in doc_title_pairs.collect():
        title_dict[int(doc_id)] = title
    
    print(f"Created mappings for {len(title_dict)} documents")
    
    # Save locally
    with open('doc_id_to_title.pkl', 'wb') as f:
        pickle.dump(title_dict, f)
    
    print("✅ Saved title mappings locally as doc_id_to_title.pkl")
    
    # Upload to GCS if bucket specified
    if bucket_name:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob('mappings/doc_id_to_title.pkl')
        blob.upload_from_filename('doc_id_to_title.pkl')
        print(f"✅ Title mappings uploaded to gs://{bucket_name}/mappings/doc_id_to_title.pkl")
    
    return title_dict

# Global helper for embedding computation (used by mapPartitions)
# This loads model once per worker process, not per task
_embedding_model_cache = {}

def _compute_embeddings_for_partition(rows_iter, model_name, re_pattern, re_flags, stopwords_set):
    """
    Helper function to compute embeddings for one partition.
    Loads model once per Python worker process for efficiency.
    Auto-installs gensim on worker if not present.
    """
    # Use process-level cache to load model only once per worker
    cache_key = f"model_{model_name}"
    
    if cache_key not in _embedding_model_cache:
        # Auto-install gensim on worker if needed
        try:
            import gensim.downloader as api_local
        except ImportError:
            print(f"[Worker] Installing gensim...")
            import subprocess
            import sys
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-q", "gensim", 
                "--root-user-action=ignore"
            ])
            import gensim.downloader as api_local
            print(f"[Worker] ✅ gensim installed")
        
        print(f"[Worker] Loading {model_name}...")
        _embedding_model_cache[cache_key] = api.load(model_name)
        _embedding_model_cache['re_word'] = re.compile(re_pattern, re_flags)
        _embedding_model_cache['stopwords'] = stopwords_set
        print(f"[Worker] ✅ Model loaded")
    
    kv_model = _embedding_model_cache[cache_key]
    re_word = _embedding_model_cache['re_word']
    stop = _embedding_model_cache['stopwords']
    
    # Process each document in this partition
    for title, doc_id in rows_iter:
        # Tokenize and filter
        tokens = [m.group() for m in re_word.finditer(title.lower())]
        tokens = [t for t in tokens if t not in stop and t in kv_model]
        
        if not tokens:
            continue
        
        # Use get_mean_vector: faster than building list + np.mean
        vec = kv_model.get_mean_vector(tokens, pre_normalize=False).astype('float32')
        
        # Yield (doc_id, embedding_as_list)
        yield (int(doc_id), vec.tolist())
    
def create_embeddings(doc_title_pairs, bucket_name, RE_WORD, all_stopwords, 
                      model_name='glove-wiki-gigaword-100', output_prefix='title'):
    """
    Create document embeddings using word2vec in a distributed, memory-efficient way.
    
    This function:
    - Loads the model once per partition (NOT broadcast) to avoid OOM
    - Writes distributed Parquet output to GCS (NO collect to driver)
    - Uses get_mean_vector for efficiency
    
    Args:
        doc_title_pairs: RDD of (title, doc_id) pairs
        bucket_name: GCS bucket name
        RE_WORD: regex pattern for tokenization
        all_stopwords: set of stopwords to filter
        model_name: gensim model name (default: glove-wiki-gigaword-100)
        output_prefix: output folder prefix (default: 'title')
        
    Returns:
        str: GCS path where embeddings were saved
        
    Prerequisites:
        - gensim must be installed on ALL cluster workers
        - Run: !pip install gensim OR add to cluster initialization
        
    Recommended Spark settings for large models:
        spark.executor.cores=1
        spark.task.cpus=1
        spark.python.worker.reuse=true
    """
    print(f"Creating embeddings using model: {model_name}")
    print(f"Processing {doc_title_pairs.count()} documents...")
    
    # Serialize RE_WORD pattern and stopwords for use in workers
    re_pattern = RE_WORD.pattern
    re_flags = RE_WORD.flags
    stopwords_set = set(all_stopwords)
    
    print("Computing embeddings in parallel...")
    
    # Use lambda to pass arguments to the helper function
    embeddings_rdd = doc_title_pairs.mapPartitions(
        lambda rows: _compute_embeddings_for_partition(
            rows, model_name, re_pattern, re_flags, stopwords_set
        )
    )
    
    # Create DataFrame with proper schema
    schema = StructType([
        StructField("doc_id", IntegerType(), False),
        StructField("embedding", ArrayType(FloatType()), False),
    ])
    
    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(embeddings_rdd, schema=schema)
    
    # Write distributed to GCS as Parquet (NO collect - avoids driver OOM)
    output_path = f"gs://{bucket_name}/embeddings/{output_prefix}_parquet/"
    print(f"Writing embeddings to {output_path}...")
    
    df.write.mode("overwrite").parquet(output_path)
    
    # Get count without collecting all data
    num_embeddings = df.count()
    
    print(f"✅ Successfully created embeddings for {num_embeddings:,} documents")
    print(f"✅ Saved to: {output_path}")
    print(f"\nTo read later:")
    print(f"  df = spark.read.parquet('{output_path}')")
    print(f"  # or in Python: pd.read_parquet('{output_path}')")
    
    return output_path