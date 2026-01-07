import math
from collections import defaultdict
from typing import List, Tuple, Dict

from Backend.ranking import BM25_score, word_count_score, cosine_similarity
from Backend.tokenizer import tokenize
from Backend.data_Loader import load_all_data
from inverted_index_gcp import InvertedIndex

N_DOCS = 6348910
DEFAULT_AVGDL = 500
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75

class SearchEngine:
    """Main search engine with hybrid ranking."""
    
    def __init__(self):
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
        
        self._precompute_normalization()
        
        print("✅ Search Engine Ready!")
    
    def _precompute_normalization(self):
        """Precompute normalization parameters."""
        # PageRank normalization
        if self.pagerank_dict:
            pr_values = list(self.pagerank_dict.values())
            self.pr_max = max(pr_values)
            self.pr_min = min(pr_values)
            self.pr_range = self.pr_max - self.pr_min if self.pr_max > self.pr_min else 1.0
        else:
            self.pr_max, self.pr_min, self.pr_range = 1.0, 0.0, 1.0
        
        # PageViews normalization
        self.pv_max = max(self.pageviews_dict.values()) if self.pageviews_dict else 1.0
    
    # ========================================================================
    # MAIN SEARCH METHODS
    # ========================================================================
    
    def search(self, query: str, top_k: int = 100) -> List[Tuple[str, str]]:
        """Main hybrid search combining all signals."""
        tokens = tokenize(query)
        if not tokens:
            return []
        
        # Get scores from all indices
        text_scores = self._get_text_scores(tokens, top_n=500)
        title_scores = self._get_title_scores(tokens, top_n=500)
        anchor_scores = self._get_anchor_scores(tokens, top_n=500)
        
        # Combine with weights
        combined = self._combine_signals(
            text_scores, title_scores, anchor_scores,
            text_w=1.5, title_w=1.2, anchor_w=0.8, pr_w=0.4, pv_w=0.6
        )
        
        # Sort and return
        sorted_docs = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return self._format_results([doc_id for doc_id, _ in sorted_docs])
    
    def search_body(self, query: str, top_k: int = 100) -> List[Tuple[str, str]]:
        """Search text index only with cosine similarity."""
        return self._search_single_index(query, self.text_index, top_k)
    
    def search_title(self, query: str, top_k: int = 100) -> List[Tuple[str, str]]:
        return self._search_partial_index(query, self.title_index, top_k)
    
    def search_anchor(self, query: str, top_k: int = 100) -> List[Tuple[str, str]]:
        return self._search_partial_index(query, self.anchor_index, top_k)
    
    # ========================================================================
    # HELPER METHODS - SCORING
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
        return dict(self._normalize_list(scores.most_common(top_n)))
    
    def _get_title_scores(self, tokens: List[str], top_n: int) -> Dict[int, float]:
        """Get BM25 scores from title index."""
        scores = BM25_score(tokens, self.title_index, N_DOCS, {}, DEFAULT_AVGDL, k1=1.5, b=0.4)
        return dict(self._normalize_list(scores.most_common(top_n)))
    
    def _get_anchor_scores(self, tokens: List[str], top_n: int) -> Dict[int, float]:
        """Get word count scores from anchor index."""
        scores = word_count_score(tokens, self.anchor_index)
        return dict(self._normalize_list(scores.most_common(top_n)))
    
    def _combine_signals(
        self, 
        text_scores: Dict[int, float],
        title_scores: Dict[int, float],
        anchor_scores: Dict[int, float],
        text_w: float, title_w: float, anchor_w: float,
        pr_w: float, pv_w: float
    ) -> Dict[int, float]:
        """Combine all ranking signals with weights."""
        all_docs = set(text_scores) | set(title_scores) | set(anchor_scores)
        
        combined = {}
        for doc_id in all_docs:
            combined[doc_id] = (
                text_scores.get(doc_id, 0.0) * text_w +
                title_scores.get(doc_id, 0.0) * title_w +
                anchor_scores.get(doc_id, 0.0) * anchor_w +
                self._norm_pr(doc_id) * pr_w +
                self._norm_pv(doc_id) * pv_w
            )
        
        return combined
    
    def _search_single_index(self, query: str, index: InvertedIndex, top_k: int) -> List[Tuple[str, str]]:
        """Search single index using cosine similarity."""
        tokens = tokenize(query)
        if not tokens:
            return []
        
        scores = cosine_similarity(tokens, index)
        doc_ids = [doc_id for doc_id, _ in scores.most_common(top_k)]
        
        return self._format_results(doc_ids)
    
    # ========================================================================
    # HELPER METHODS - NORMALIZATION
    # ========================================================================
    
    def _norm_pr(self, doc_id: int) -> float:
        """Get normalized PageRank [0, 1]."""
        if not self.pagerank_dict or doc_id not in self.pagerank_dict:
            return 0.5
        pr_raw = self.pagerank_dict[doc_id]
        return (pr_raw - self.pr_min) / self.pr_range if self.pr_range > 0 else 0.5
    
    def _norm_pv(self, doc_id: int) -> float:
        """Get normalized PageViews [0, 1]."""
        if not self.pageviews_dict or doc_id not in self.pageviews_dict:
            return 0.0
        return self.pageviews_dict[doc_id] / self.pv_max if self.pv_max > 0 else 0.0
    
    def _normalize_list(self, score_list: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """Normalize list of (doc_id, score) tuples."""
        if not score_list:
            return []
        max_score = score_list[0][1]
        if max_score > 0:
            return [(doc_id, score / max_score) for doc_id, score in score_list]
        return score_list
    
    def _format_results(self, doc_ids: List[int]) -> List[Tuple[str, str]]:
        """Format results as [(doc_id, title), ...]."""
        results = []
        for doc_id in doc_ids:
            title = self.doc_titles_dict.get(doc_id, f"Document {doc_id}")
            results.append((str(doc_id), title))
        return results
    
    # ========================================================================
    # PUBLIC UTILITY METHODS
    # ========================================================================
    
    def get_pagerank(self, doc_ids: List[int]) -> List[float]:
        """Get PageRank scores for doc IDs."""
        return [self.pagerank_dict.get(doc_id, 0.0) for doc_id in doc_ids]
    
    def get_pageviews(self, doc_ids: List[int]) -> List[int]:
        """Get PageView counts for doc IDs."""
        return [self.pageviews_dict.get(doc_id, 0) for doc_id in doc_ids]
    
    def get_doc_titles(self, doc_ids: List[int]) -> List[str]:
        """Get titles for doc IDs."""
        return [self.doc_titles_dict.get(doc_id, f"Document {doc_id}") for doc_id in doc_ids]