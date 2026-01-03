"""
Search Engine Query Processing Module

This module provides the main SearchEngine class that orchestrates
search operations by combining ranking algorithms and data sources.

Single Responsibility: Coordinate search operations and combine signals
"""

import math
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

from Backend.ranking import BM25_score, word_count_score, cosine_similarity
from Backend.tokenizer import tokenize, og_tokenize
from Backend.data_Loader import load_index, load_pagerank, load_pageviews, load_doc_titles
from inverted_index_gcp import InvertedIndex

# Constants
N_DOCS = 6348910  # Wikipedia corpus size
DEFAULT_AVGDL = 500  # Average document length estimate

# BM25 tuning parameters
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75

# Hybrid scoring weights
DEFAULT_BM25_WEIGHT = 0.80
DEFAULT_PAGERANK_WEIGHT = 0.20


class SearchEngine:
    """
    Main search engine orchestrator.
    
    Responsibilities:
    - Load and cache data sources (indices, pagerank, etc.)
    - Coordinate different search strategies
    - Combine ranking signals (BM25, PageRank, PageViews)
    - Return formatted results
    """
    
    def __init__(self):
        """Initialize search engine and load all necessary data."""
        print("🔧 Initializing Search Engine...")
        
        # Load indices
        self.text_index = load_index("text")
        self.title_index = load_index("title")
        self.anchor_index = load_index("anchor")
        
        # Load auxiliary data
        self.pagerank_dict = load_pagerank()
        self.pageviews_dict = load_pageviews()
        self.doc_titles_dict = load_doc_titles()
        
        # Precompute PageRank normalization parameters
        self._precompute_pagerank_params()
        
        # Precompute PageViews normalization
        self._precompute_pageviews_params()
        
        print("✅ Search Engine Initialized!")
    
    def _precompute_pagerank_params(self):
        """Precompute PageRank normalization parameters for efficiency."""
        if self.pagerank_dict:
            pagerank_values = [pr for pr in self.pagerank_dict.values() if pr > 0]
            if pagerank_values:
                self.pr_max = max(pagerank_values)
                self.pr_min = min(pagerank_values)
                self.pr_range = self.pr_max - self.pr_min if self.pr_max > self.pr_min else 1.0
            else:
                self.pr_max, self.pr_min, self.pr_range = 1.0, 0.0, 1.0
        else:
            self.pr_max, self.pr_min, self.pr_range = 1.0, 0.0, 1.0
    
    def _precompute_pageviews_params(self):
        """Precompute PageViews normalization parameters."""
        if self.pageviews_dict:
            self.pv_max = max(self.pageviews_dict.values())
        else:
            self.pv_max = 1.0
    
    def _get_normalized_pagerank(self, doc_id: int) -> float:
        """
        Get normalized PageRank score for a document.
        
        Args:
            doc_id: Document ID
        
        Returns:
            Normalized PageRank score [0, 1]
        """
        if not self.pagerank_dict or doc_id not in self.pagerank_dict:
            return 0.5  # Neutral default
        
        pr_raw = self.pagerank_dict[doc_id]
        if self.pr_range > 0:
            return (pr_raw - self.pr_min) / self.pr_range
        return 0.5
    
    def _get_normalized_pageviews(self, doc_id: int) -> float:
        """
        Get normalized PageViews score for a document.
        
        Args:
            doc_id: Document ID
        
        Returns:
            Normalized PageViews score [0, 1]
        """
        if not self.pageviews_dict or doc_id not in self.pageviews_dict:
            return 0.0
        
        return self.pageviews_dict[doc_id] / self.pv_max if self.pv_max > 0 else 0.0
    
    def _format_results(self, doc_ids: List[int]) -> List[Tuple[str, str]]:
        """
        Format document IDs with titles.
        
        Args:
            doc_ids: List of document IDs
        
        Returns:
            List of (doc_id_str, title) tuples
        """
        results = []
        for doc_id in doc_ids:
            title = self.doc_titles_dict.get(doc_id, f"Document {doc_id}")
            results.append((str(doc_id), title))
        return results
    
    # ========================================================================
    # PRIMARY SEARCH METHOD (Main endpoint)
    # ========================================================================
    
    def search(self, query: str, top_k: int = 100) -> List[Tuple[str, str]]:
        # Tokenize query
        tokens = tokenize(query)
        if not tokens:
            return []
        
        # Get BM25 scores from text and title indices
        text_scores = self._get_text_bm25_scores(tokens, top_n=500)
        title_scores = self._get_title_bm25_scores(tokens, top_n=500)
        anchor_scores = self._get_anchor_scores(tokens, top_n=500)
        
        # Combine signals with tuned weights
        combined_scores = self._combine_all_signals(
            text_scores, 
            title_scores,
            anchor_scores,
            text_weight=1.5,
            title_weight=1.2,
            anchor_weight=0.8,
            pr_weight=0.4,
            pv_weight=0.6
        )
        
        # Sort and take top K
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        doc_ids = [doc_id for doc_id, _ in sorted_docs]
        
        return self._format_results(doc_ids)
    
    # ========================================================================
    # BASIC SEARCH (Simplified hybrid approach)
    # ========================================================================
    
    def search_basic(self, query: str, top_k: int = 10) -> List[List]:
        """
        Basic hybrid search for testing - BM25 + PageRank.
        
        Simplified version optimized for quick queries.
        Returns format: [[doc_id, title], ...]
        
        Args:
            query: Search query string
            top_k: Number of results (default: 10)
        
        Returns:
            List of [doc_id, title] pairs
        """
        tokens = tokenize(query)
        if not tokens:
            return []
        
        # Calculate BM25 scores
        doc_scores = self._calculate_bm25_scores(tokens)
        
        if not doc_scores:
            return []
        
        # Normalize BM25 scores
        doc_scores = self._normalize_scores(doc_scores)
        
        # Combine with PageRank
        final_scores = {}
        for doc_id, bm25_score in doc_scores.items():
            pr_score = self._get_normalized_pagerank(doc_id)
            final_scores[doc_id] = (DEFAULT_BM25_WEIGHT * bm25_score + 
                                   DEFAULT_PAGERANK_WEIGHT * pr_score)
        
        # Sort and format
        sorted_docs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for doc_id, _ in sorted_docs:
            title = self.doc_titles_dict.get(doc_id, f"Article {doc_id}")
            results.append([int(doc_id), title])
        
        return results
    
    # ========================================================================
    # PARTIAL SEARCH METHODS (Body, Title, Anchor)
    # ========================================================================
    
    def search_body(self, query: str, top_k: int = 100) -> List[Tuple[str, str]]:
        return self._search_partial_index(query, self.text_index, top_k)
    
    def search_title(self, query: str, top_k: int = 100) -> List[Tuple[str, str]]:
        return self._search_partial_index(query, self.title_index, top_k)
    
    def search_anchor(self, query: str, top_k: int = 100) -> List[Tuple[str, str]]:
        return self._search_partial_index(query, self.anchor_index, top_k)
    
    # ========================================================================
    # PARAMETERIZED SEARCH (For experimentation/tuning)
    # ========================================================================
    
    def search_custom(
        self, 
        query: str,
        text_weight: float = 0.65,
        title_weight: float = 0.25,
        anchor_weight: float = 0.1,
        pr_weight: float = 1.0,
        pv_weight: float = 1.0,
        k1: float = 1.2,
        b: float = 0.5,
        top_k: int = 100
    ) -> List[Tuple[str, str]]:
        """
        Customizable search with adjustable weights and parameters.
        
        Useful for experimentation and parameter tuning.
        
        Args:
            query: Search query
            text_weight: Weight for text index BM25 scores
            title_weight: Weight for title index scores
            anchor_weight: Weight for anchor index scores
            pr_weight: Weight for PageRank
            pv_weight: Weight for PageViews
            k1: BM25 k1 parameter
            b: BM25 b parameter
            top_k: Number of results
        
        Returns:
            List of (doc_id, title) tuples
        """
        tokens = tokenize(query)
        if not tokens:
            return []
        
        # Get scores from each index
        text_scores = BM25_score(
            tokens, self.text_index, N_DOCS, 
            {}, DEFAULT_AVGDL, k1=k1, b=b
        ).most_common(500)
        
        title_scores = word_count_score(tokens, self.title_index).most_common(500)
        anchor_scores = word_count_score(tokens, self.anchor_index).most_common(500)
        
        # Normalize scores
        text_scores = self._normalize_score_list(text_scores)
        title_scores = self._normalize_score_list(title_scores)
        anchor_scores = self._normalize_score_list(anchor_scores)
        
        # Convert to dicts
        text_dict = dict(text_scores)
        title_dict = dict(title_scores)
        anchor_dict = dict(anchor_scores)
        
        # Combine all signals
        all_doc_ids = set(text_dict) | set(title_dict) | set(anchor_dict)
        
        final_scores = {}
        for doc_id in all_doc_ids:
            score = (
                text_dict.get(doc_id, 0.0) * text_weight +
                title_dict.get(doc_id, 0.0) * title_weight +
                anchor_dict.get(doc_id, 0.0) * anchor_weight +
                self._get_normalized_pagerank(doc_id) * pr_weight +
                self._get_normalized_pageviews(doc_id) * pv_weight
            )
            final_scores[doc_id] = score
        
        # Sort and return top K
        sorted_docs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        doc_ids = [doc_id for doc_id, _ in sorted_docs]
        
        return self._format_results(doc_ids)
    
    # ========================================================================
    # UTILITY METHODS (For individual endpoints)
    # ========================================================================
    
    def get_pagerank(self, doc_ids: List[int]) -> List[float]:
        """
        Get PageRank scores for a list of document IDs.
        
        Args:
            doc_ids: List of document IDs
        
        Returns:
            List of PageRank scores
        """
        return [self.pagerank_dict.get(doc_id, 0.0) for doc_id in doc_ids]
    
    def get_pageviews(self, doc_ids: List[int]) -> List[int]:
        """
        Get PageView counts for a list of document IDs.
        
        Args:
            doc_ids: List of document IDs
        
        Returns:
            List of PageView counts
        """
        return [self.pageviews_dict.get(doc_id, 0) for doc_id in doc_ids]
    
    def get_doc_titles(self, doc_ids: List[int]) -> List[str]:
        """
        Get document titles for a list of document IDs.
        
        Args:
            doc_ids: List of document IDs
        
        Returns:
            List of document titles
        """
        return [self.doc_titles_dict.get(doc_id, f"Document {doc_id}") for doc_id in doc_ids]
    
    # ========================================================================
    # PRIVATE HELPER METHODS
    # ========================================================================
    
    def _calculate_bm25_scores(self, tokens: List[str]) -> Dict[int, float]:
        """Calculate BM25 scores for query tokens."""
        doc_scores = defaultdict(float)
        query_term_freq = defaultdict(int)
        
        # Count query term frequencies
        for term in tokens:
            query_term_freq[term] += 1
        
        # Calculate IDF weights
        query_term_weights = {}
        for term in set(tokens):
            if term in self.text_index.df:
                df = self.text_index.df[term]
                idf = math.log((N_DOCS - df + 0.5) / (df + 0.5) + 1.0)
                query_term_weights[term] = idf
            else:
                query_term_weights[term] = 0.0
        
        # Score documents
        for term in tokens:
            if term not in self.text_index.posting_locs:
                continue
            
            idf = query_term_weights.get(term, 0.0)
            if idf == 0:
                continue
            
            posting_list = self.text_index.read_a_posting_list("data/postings_gcp", term)
            
            for doc_id, tf in posting_list:
                # BM25 formula
                numerator = tf * (DEFAULT_K1 + 1)
                denominator = tf + DEFAULT_K1 * (1 - DEFAULT_B + DEFAULT_B * (tf * 10 / DEFAULT_AVGDL))
                bm25_component = idf * (numerator / denominator)
                
                # Query boost
                query_boost = 1.0 + 0.5 * math.log(1 + query_term_freq[term])
                
                doc_scores[doc_id] += bm25_component * query_boost
        
        return doc_scores
    
    def _normalize_scores(self, scores: Dict[int, float]) -> Dict[int, float]:
        """Normalize scores to [0, 1] range."""
        if not scores:
            return {}
        
        max_score = max(scores.values())
        min_score = min(scores.values())
        score_range = max_score - min_score
        
        if score_range > 0:
            return {doc_id: (score - min_score) / score_range for doc_id, score in scores.items()}
        return scores
    
    def _normalize_score_list(self, score_list: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """Normalize a list of (doc_id, score) tuples."""
        if not score_list:
            return []
        
        max_score = score_list[0][1]  # Assuming sorted
        if max_score > 0:
            return [(doc_id, score / max_score) for doc_id, score in score_list]
        return score_list
    
    def _get_text_bm25_scores(self, tokens: List[str], top_n: int = 500) -> Dict[int, float]:
        """Get BM25 scores from text index."""
        scores = BM25_score(tokens, self.text_index, N_DOCS, {}, DEFAULT_AVGDL, k1=1.2, b=0.6)
        top_scores = scores.most_common(top_n)
        return dict(self._normalize_score_list(top_scores))
    
    def _get_title_bm25_scores(self, tokens: List[str], top_n: int = 500) -> Dict[int, float]:
        """Get BM25 scores from title index."""
        scores = BM25_score(tokens, self.title_index, N_DOCS, {}, DEFAULT_AVGDL, k1=1.5, b=0.4)
        top_scores = scores.most_common(top_n)
        return dict(self._normalize_score_list(top_scores))
    
    def _get_anchor_scores(self, tokens: List[str], top_n: int = 500) -> Dict[int, float]:
        """Get word count scores from anchor index."""
        scores = word_count_score(tokens, self.anchor_index)
        top_scores = scores.most_common(top_n)
        return dict(self._normalize_score_list(top_scores))
    
    def _combine_all_signals(
        self, 
        text_scores: Dict[int, float],
        title_scores: Dict[int, float],
        anchor_scores: Dict[int, float],
        text_weight: float = 1.5,
        title_weight: float = 1.2,
        anchor_weight: float = 0.8,
        pr_weight: float = 0.4,
        pv_weight: float = 0.6
    ) -> Dict[int, float]:
        """Combine text, title, anchor, PageRank, and PageViews signals."""
        all_docs = set(text_scores) | set(title_scores) | set(anchor_scores)
        
        combined = {}
        for doc_id in all_docs:
            score = (
                text_scores.get(doc_id, 0.0) * text_weight +
                title_scores.get(doc_id, 0.0) * title_weight +
                anchor_scores.get(doc_id, 0.0) * anchor_weight +
                self._get_normalized_pagerank(doc_id) * pr_weight +
                self._get_normalized_pageviews(doc_id) * pv_weight
            )
            combined[doc_id] = score
        
        return combined
    
    def _search_partial_index(
        self, 
        query: str, 
        index: InvertedIndex, 
        top_k: int = 100
    ) -> List[Tuple[str, str]]:
        """Search a single index using cosine similarity."""
        tokens = og_tokenize(query)
        if not tokens:
            return []
        
        scores = cosine_similarity(tokens, index)
        top_docs = scores.most_common(top_k)
        doc_ids = [doc_id for doc_id, _ in top_docs]
        
        return self._format_results(doc_ids)