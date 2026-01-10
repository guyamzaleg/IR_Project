from collections import Counter
import math
import numpy as np
import pandas as pd
import nltk
from nltk import PorterStemmer
from nltk.corpus import stopwords
import re
import string
from typing import List, Dict
from inverted_index_gcp import *

def query_tfidf(query, index):
    """
    Calculates the TF-IDF for a given query.

    Args:
        query: A list of tokens representing the query.
        index: An inverted index object.

    Returns:
        A dictionary where keys are query tokens and values are their TF-IDF scores.
    """
    query_tfidf = {}
    N = len(index.df)  # Total number of documents
    for token in set(query):  # Iterate through unique query tokens
        tf = query.count(token)  # Term frequency in the query
        df = index.df.get(token, 0)  # Document frequency of the term
        if df == 0:
          idf = 0
        else:
          idf = math.log(N / df, 10)  # Inverse document frequency
        query_tfidf[token] = tf * idf

    return query_tfidf

def calc_cosine(query_tfidf, doc_tfidf):
    """
    Calculates the cosine similarity between a query and a document.
    """
    dot_product = 0
    query_magnitude = 0
    doc_magnitude = 0

    for term, tfidf in query_tfidf.items():
        dot_product += tfidf * doc_tfidf.get(term, 0)  # Handle terms not in the document
        query_magnitude += tfidf ** 2

    for tfidf in doc_tfidf.values():
        doc_magnitude += tfidf ** 2

    if query_magnitude == 0 or doc_magnitude == 0:
        return 0

    return dot_product / (math.sqrt(query_magnitude) * math.sqrt(doc_magnitude))


def cosine_similarity(tokenized_query, index):
    """
    Returns a dictionary of candidates with cosine similarity scores.

    Args:
        tokenized_query (list): A list of tokens representing the query.
        index: The inverted index object.

    Output:
    - A dictionary where:
        - key: doc_id
        - value: cosine similarity score for the document
    """
    query_tfidf_scores = query_tfidf(tokenized_query, index)
    N = len(index.df)
    candidates_dict = {}
    results = Counter()

    for token in tokenized_query:
        pl = index.read_a_posting_list("data", token)
        if pl == []:
            continue
        else:
            candidates_dict[token] = pl
            for doc_id, tf in candidates_dict[token]:
                doc_tfidf = {}
                df = index.df.get(token, 0)
                if df == 0:
                    idf = 0
                else:
                    idf = math.log(N / df, 10)
                tfidf_value = tf * idf
                doc_tfidf[token] = tfidf_value
                cosine = calc_cosine(query_tfidf_scores, doc_tfidf)
                results[doc_id] += cosine

    return results

def BM25_score(tokenized_query, index, doc_num, doc_lengths, avg_doc_length, k1=1.2, k3=1.5, b=0.75):
    """
    Calculates BM25 scores for documents based on a given query and an inverted index.
    """
    bm25_scores = Counter()
    candidates_dict = {}
    
    # Count query term frequencies
    query_term_freq = Counter(tokenized_query)

    for token in tokenized_query:
        pl = index.read_a_posting_list("data", token)
        if pl == []:
            continue
        
        candidates_dict[token] = pl
        df = index.df[token]
        
        idf = np.log((doc_num - df + 0.5) / (df + 0.5) + 1.0)  # Modern IDF formula
        # idf = np.log((doc_num + 1) / df)  # Original IDF formula
        
        # Query term weight
        tf_iq = query_term_freq[token]
        query_weight = ((k3 + 1) * tf_iq) / (k3 + tf_iq)
        
        for doc_id, tf in candidates_dict[token]:
            try:
                # Document length normalization
                if doc_lengths:
                    doc_len = doc_lengths.get(doc_id, avg_doc_length)
                else:
                    doc_len = avg_doc_length
                
                B = 1 - b + b * (doc_len / avg_doc_length)
                
                # Document term weight (p1)
                doc_weight = ((k1 + 1) * tf) / (B * k1 + tf)
                
                # Combine all components
                bm25_scores[doc_id] += idf * doc_weight * query_weight
            except:
                pass

    return bm25_scores


def word_count_score(tokenized_query, index):
    """
    Calculates the number of query terms present in each candidate document.

    Args:
        tokenized_query (list): A list of tokens representing the query.
        index: The inverted index object.

    Returns:
        dict: A dictionary where keys are document IDs and values are the number of
              query terms found in that document.
    """

    candidates_dict = {}  # Initialize a regular dictionary
    doc_term_counts = Counter()
    for token in tokenized_query:
        pl = index.read_a_posting_list("data", token)
        if pl == []:
            continue
        else:
            candidates_dict[token] = pl
            for doc_id, tf in candidates_dict[token]:
                doc_term_counts[doc_id] += 1
    return doc_term_counts


def tf_count_score(tokenized_query, index):
    """
    Calculates the total term frequency (tf) for each document in the candidates posting list.

    Args:
        tokenized_query (list): A list of tokens representing the query.
        index: The inverted index object.

    Returns:
        dict: A dictionary where keys are document IDs and values are their total tf score.
    """
    candidates_dict = {}  # Initialize a regular dictionary
    doc_tf_scores = Counter()

    for token in tokenized_query:
        pl = index.read_a_posting_list("data", token)
        if pl == []:
            continue
        else:
            candidates_dict[token] = pl
            for doc_id, tf in candidates_dict[token]:
                doc_tf_scores[doc_id] += tf

    return doc_tf_scores


def pagerank_score(doc_ids: List[int], pagerank_dict: Dict[int, float]) -> Counter:
    """
    Rank documents purely by PageRank score (query-independent).
    
    Args:
        doc_ids: List of document IDs to rank (or empty list for all docs)
        pagerank_dict: Dictionary mapping doc_id to PageRank score
    
    Returns:
        Counter with doc_id as key and PageRank score as value
    """
    scores = Counter()
    
    # If no specific docs provided, use all available
    if not doc_ids:
        doc_ids = list(pagerank_dict.keys())
    
    for doc_id in doc_ids:
        scores[doc_id] = pagerank_dict.get(doc_id, 0.0)
    
    return scores


def pageviews_score(doc_ids: List[int], pageviews_dict: Dict[int, int]) -> Counter:
    """
    Rank documents purely by PageViews count (query-independent).
    
    Args:
        doc_ids: List of document IDs to rank (or empty list for all docs)
        pageviews_dict: Dictionary mapping doc_id to pageview count
    
    Returns:
        Counter with doc_id as key and pageview count as value
    """
    scores = Counter()
    
    # If no specific docs provided, use all available
    if not doc_ids:
        doc_ids = list(pageviews_dict.keys())
    
    for doc_id in doc_ids:
        scores[doc_id] = pageviews_dict.get(doc_id, 0)
    
    return scores

