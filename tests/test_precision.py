#!/usr/bin/env python3
"""
Test script for evaluating search engine performance using Precision@10
Queries the search engine with queries from queries_train.json and compares
results against ground truth to calculate precision metrics.
"""

import json
import sys
import os
from pathlib import Path
from collections import defaultdict
import statistics
from datetime import datetime

# Add parent directory to path to import search_frontend modules
sys.path.append(str(Path(__file__).parent.parent))

# Import the real search engine
from search import SearchEngine

# Old imports (commented out - no longer using manual BM25 implementation)
# from Backend.tokenizer import tokenize
# from Backend.data_Loader import load_index, load_pagerank
# import math

# ============== CONFIGURATION ==============
QUERIES_FILE = "queries_train.json"
OUTPUT_DIR = "tests/results"
K = 10  # For Precision@10

# Global search engine instance (NEW: using real SearchEngine class)
search_engine = None

# Old global variables (commented out - no longer needed)
# inverted_index = None
# pagerank_dict = None
# N_DOCS = 6000000  # Approximate Wikipedia size


# ============== IMPLEMENTATION ==============

def initialize_search_engine():
    """Initialize the search engine by loading index and pagerank data."""
    global search_engine
    
    # NEW: Initialize real SearchEngine instance
    if search_engine is None:
        print("Initializing search engine...")
        search_engine = SearchEngine()
        print("✓ Search engine initialized successfully!")
    
    # OLD implementation (commented out):
    # global inverted_index, pagerank_dict
    # if inverted_index is None or pagerank_dict is None:
    #     print("Initializing search engine...")
    #     inverted_index = load_index()
    #     pagerank_dict = load_pagerank()
    #     print("✓ Search engine initialized successfully!")


def load_queries(filepath):
    """
    Load queries and ground truth from JSON file
    
    Args:
        filepath: Path to queries_train.json
    
    Returns:
        dict: {query_text: [relevant_doc_ids]}
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            queries_dict = json.load(f)
        print(f"✓ Loaded {len(queries_dict)} queries from {filepath}")
        return queries_dict
    except FileNotFoundError:
        print(f"Error: File {filepath} not found!")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {filepath}: {e}")
        sys.exit(1)


def query_search_engine(query_text, top_k=10):
    """
    Query the real search engine and return top K document IDs
    
    NEW: Uses the actual search_basic method from SearchEngine class
    
    Args:
        query_text: String query
        top_k: Number of results to retrieve (default: 10)
    
    Returns:
        list: Document IDs (as strings) of top K results
    """
    global search_engine
    
    # NEW: Call the real search_basic method from SearchEngine
    # search_basic returns a list of [doc_id, title] pairs
    results = search_engine.search_basic(query_text)
    
    # Extract just the doc_ids and convert to strings, limit to top_k
    doc_ids = [str(doc_id) for doc_id, _ in results[:top_k]]
    
    return doc_ids

    # ========== OLD IMPLEMENTATION (COMMENTED OUT) ==========
    # """
    # Advanced hybrid search using BM25 + PageRank + query features
    # 
    # This implementation combines multiple IR best practices:
    # - BM25 scoring (state-of-the-art probabilistic retrieval)
    # - PageRank integration for authority boost
    # - Query term weighting (IDF-based importance)
    # - Document length normalization
    # - Multi-signal fusion
    # """
    # global inverted_index, pagerank_dict, N_DOCS
    # 
    # # Tokenize the query
    # query_tokens = tokenize(query_text)
    # if not query_tokens:
    #     return []
    # 
    # # BM25 Parameters (tuned for Wikipedia-scale corpus)
    # k1 = 1.5  # Term frequency saturation parameter
    # b = 0.75  # Document length normalization
    # 
    # # Average document length (estimated for Wikipedia)
    # avgdl = 500  # Average tokens per Wikipedia article
    # 
    # # Calculate query term weights using IDF
    # query_term_weights = {}
    # for term in set(query_tokens):
    #     if term in inverted_index.df:
    #         df = inverted_index.df[term]
    #         # IDF with smoothing
    #         idf = math.log((N_DOCS - df + 0.5) / (df + 0.5) + 1.0)
    #         query_term_weights[term] = idf
    #     else:
    #         query_term_weights[term] = 0.0
    # 
    # # Count query term frequencies for query boosting
    # query_term_freq = defaultdict(int)
    # for term in query_tokens:
    #     query_term_freq[term] += 1
    # 
    # # Collect candidate documents with BM25 scores
    # doc_scores = defaultdict(float)
    # doc_lengths = {}  # Store document lengths for normalization
    # 
    # for term in query_tokens:
    #     if term not in inverted_index.posting_locs:
    #         continue
    #     
    #     # Get IDF weight for this term
    #     idf = query_term_weights.get(term, 0.0)
    #     if idf == 0:
    #         continue
    #     
    #     # Read posting list
    #     posting_list = inverted_index.read_a_posting_list("data/postings_gcp", term)
    #     
    #     for doc_id, tf in posting_list:
    #         # Estimate document length (TF can be proxy)
    #         if doc_id not in doc_lengths:
    #             doc_lengths[doc_id] = tf * 10  # Rough estimate
    #         
    #         # BM25 scoring formula
    #         # BM25 = IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / avgdl)))
    #         dl = doc_lengths.get(doc_id, avgdl)
    #         numerator = tf * (k1 + 1)
    #         denominator = tf + k1 * (1 - b + b * (dl / avgdl))
    #         bm25_component = idf * (numerator / denominator)
    #         
    #         # Query term frequency boost (repeated terms matter more)
    #         query_boost = 1.0 + 0.5 * math.log(1 + query_term_freq[term])
    #         
    #         doc_scores[doc_id] += bm25_component * query_boost
    # 
    # if not doc_scores:
    #     return []
    # 
    # # Normalize BM25 scores to [0, 1] range
    # max_bm25 = max(doc_scores.values())
    # min_bm25 = min(doc_scores.values())
    # bm25_range = max_bm25 - min_bm25
    # 
    # if bm25_range > 0:
    #     for doc_id in doc_scores:
    #         doc_scores[doc_id] = (doc_scores[doc_id] - min_bm25) / bm25_range
    # 
    # # Integrate PageRank scores for authority boost
    # final_scores = {}
    # 
    # # Calculate PageRank boost parameters
    # if pagerank_dict:
    #     pagerank_values = [pr for pr in pagerank_dict.values() if pr > 0]
    #     if pagerank_values:
    #         max_pr = max(pagerank_values)
    #         min_pr = min(pagerank_values)
    #         pr_range = max_pr - min_pr if max_pr > min_pr else 1.0
    #     else:
    #         max_pr, min_pr, pr_range = 1.0, 0.0, 1.0
    # else:
    #     max_pr, min_pr, pr_range = 1.0, 0.0, 1.0
    # 
    # # Hybrid scoring: weighted combination of BM25 and PageRank
    # # Weights: 80% relevance (BM25), 20% authority (PageRank)
    # alpha = 0.80  # BM25 weight
    # beta = 0.20   # PageRank weight
    # 
    # for doc_id, bm25_score in doc_scores.items():
    #     # Get normalized PageRank
    #     pr_score = 0.5  # Default for documents without PageRank
    #     if pagerank_dict and doc_id in pagerank_dict:
    #         pr_raw = pagerank_dict[doc_id]
    #         if pr_range > 0:
    #             pr_score = (pr_raw - min_pr) / pr_range
    #     
    #     # Combined score with tuned weights
    #     final_scores[doc_id] = alpha * bm25_score + beta * pr_score
    # 
    # # Sort by final score (highest first) and take top K
    # sorted_docs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    # 
    # # Return list of document IDs as strings (to match ground truth format)
    # return [str(doc_id) for doc_id, _ in sorted_docs]


def calculate_precision_at_k(retrieved_docs, relevant_docs, k=10):
    """
    Calculate Precision@K metric
    
    Args:
        retrieved_docs: List of retrieved document IDs (strings)
        relevant_docs: List of ground truth relevant document IDs (strings)
        k: Cut-off rank (default: 10)
    
    Returns:
        tuple: (precision_score, list of hit doc IDs)
    """
    # Take first K documents from retrieved results
    top_k_docs = retrieved_docs[:k]
    
    # Convert to sets for efficient intersection
    top_k_set = set(top_k_docs)
    relevant_set = set(relevant_docs)
    
    # Find hits (documents that are both retrieved and relevant)
    hits = top_k_set.intersection(relevant_set)
    
    # Calculate precision: relevant_in_topK / K
    precision = len(hits) / k if k > 0 else 0.0
    
    return precision, list(hits)


def run_precision_test(queries_file):
    """
    Main function to run precision tests on all queries
    
    Args:
        queries_file: Path to queries_train.json
    
    Returns:
        dict: Results for each query with precision scores
    """
    # Load queries
    queries_dict = load_queries(queries_file)
    
    # Initialize search engine
    initialize_search_engine()
    
    # Store results
    results = {
        'per_query': {},
        'summary': {}
    }
    
    precision_scores = []
    
    print(f"\nTesting queries:")
    print("=" * 80)
    
    # Test each query
    for idx, (query_text, relevant_docs) in enumerate(queries_dict.items(), 1):
        # Query the search engine
        retrieved_docs = query_search_engine(query_text, top_k=K)
        
        # Calculate Precision@K
        precision, hits = calculate_precision_at_k(retrieved_docs, relevant_docs, k=K)
        
        # Store results
        results['per_query'][query_text] = {
            'precision_at_10': precision,
            'relevant_found': len(hits),
            'total_relevant': len(relevant_docs),
            'total_retrieved': len(retrieved_docs),
            'hit_article_ids': sorted(hits),  # Article IDs that matched
            'retrieved_ids': retrieved_docs
        }
        
        precision_scores.append(precision)
        
        # Print progress
        print(f"[{idx:2d}/{len(queries_dict)}] {query_text[:50]:50s} | P@10 = {precision:.3f} | Hits: {len(hits)}/10")
        if hits:
            print(f"         Hit Article IDs: {', '.join(sorted(hits)[:5])}{'...' if len(hits) > 5 else ''}")
    
    print("=" * 80)
    
    # Calculate summary statistics
    results['summary'] = {
        'total_queries': len(queries_dict),
        'mean_precision': statistics.mean(precision_scores),
        'median_precision': statistics.median(precision_scores),
        'stdev_precision': statistics.stdev(precision_scores) if len(precision_scores) > 1 else 0.0,
        'min_precision': min(precision_scores),
        'max_precision': max(precision_scores),
        'perfect_scores': sum(1 for p in precision_scores if p == 1.0),
        'zero_scores': sum(1 for p in precision_scores if p == 0.0)
    }
    
    return results


def generate_report(results):
    """
    Generate comprehensive test report in Markdown format
    
    Args:
        results: Dictionary of test results
    
    Outputs:
        - Console summary
        - Markdown report with timestamp in a timestamped directory
    """
    summary = results['summary']
    per_query = results['per_query']
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamp_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Print summary to console
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total Queries Tested: {summary['total_queries']}")
    print(f"Mean Precision@10:    {summary['mean_precision']:.4f}")
    print(f"Median Precision@10:  {summary['median_precision']:.4f}")
    print(f"Std Deviation:        {summary['stdev_precision']:.4f}")
    print(f"Min Precision@10:     {summary['min_precision']:.4f}")
    print(f"Max Precision@10:     {summary['max_precision']:.4f}")
    print(f"\nPerfect Scores (P@10 = 1.0): {summary['perfect_scores']} queries")
    print(f"Zero Scores (P@10 = 0.0):     {summary['zero_scores']} queries")
    print("=" * 80)
    
    # Create timestamped output directory
    output_dir = os.path.join(OUTPUT_DIR, timestamp_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save Markdown report
    md_path = os.path.join(output_dir, "precision_results.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        # Header with timestamp
        f.write("# Search Engine Precision@10 Evaluation Report\n\n")
        f.write(f"**Generated:** {timestamp}\n\n")
        f.write("---\n\n")
        
        # Summary Statistics
        f.write("## Summary Statistics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Total Queries Tested | {summary['total_queries']} |\n")
        f.write(f"| Mean Precision@10 | {summary['mean_precision']:.4f} |\n")
        f.write(f"| Median Precision@10 | {summary['median_precision']:.4f} |\n")
        f.write(f"| Standard Deviation | {summary['stdev_precision']:.4f} |\n")
        f.write(f"| Min Precision@10 | {summary['min_precision']:.4f} |\n")
        f.write(f"| Max Precision@10 | {summary['max_precision']:.4f} |\n")
        f.write(f"| Perfect Scores (P@10 = 1.0) | {summary['perfect_scores']} queries |\n")
        f.write(f"| Zero Scores (P@10 = 0.0) | {summary['zero_scores']} queries |\n")
        f.write("\n---\n\n")
        
        # Per-Query Results
        f.write("## Detailed Results by Query\n\n")
        
        for idx, (query, data) in enumerate(per_query.items(), 1):
            f.write(f"### {idx}. {query}\n\n")
            f.write(f"- **Precision@10:** {data['precision_at_10']:.4f}\n")
            f.write(f"- **Relevant Found:** {data['relevant_found']}/10 retrieved\n")
            f.write(f"- **Total Relevant in Ground Truth:** {data['total_relevant']}\n")
            f.write(f"- **Hit Article IDs:** {', '.join(data['hit_article_ids']) if data['hit_article_ids'] else 'None'}\n")
            f.write(f"- **Retrieved IDs (Top 10):** {', '.join(data['retrieved_ids'][:10]) if data['retrieved_ids'] else 'None'}\n")
            f.write("\n")
        
        f.write("---\n\n")
        f.write(f"*Report generated on {timestamp}*\n")
    
    print(f"\n✓ Markdown report saved to: {md_path}")


def main():
    """Main entry point for the test script."""
    print("=" * 80)
    print("SEARCH ENGINE PRECISION@10 EVALUATION")
    print("=" * 80)
    print()
    
    # Run tests
    results = run_precision_test(QUERIES_FILE)
    
    # Generate report
    generate_report(results)
    
    print("\n✓ Testing complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
