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
        
        data = load_all_data()
        
        self.text_index = data['indexes']['text']
        self.title_index = data['indexes']['title']
        self.anchor_index = data['indexes']['anchor']
        self.pagerank_dict = data['pagerank']
        self.pageviews_dict = data['pageviews']
        self.doc_titles_dict = data['titles']
        self.embeddings = data.get('embeddings')
        
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
        """Search title index only with cosine similarity."""
        return self._search_single_index(query, self.title_index, top_k)
    
    def search_anchor(self, query: str, top_k: int = 100) -> List[Tuple[str, str]]:
        """Search anchor index only with cosine similarity."""
        return self._search_single_index(query, self.anchor_index, top_k)
    
    # ========================================================================
    # HELPER METHODS - SCORING
    # ========================================================================
    
    def _get_text_scores(self, tokens: List[str], top_n: int) -> Dict[int, float]:
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