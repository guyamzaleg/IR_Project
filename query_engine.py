import math
from collections import defaultdict, Counter
from typing import List, Tuple, Dict

from Backend.ranking import BM25_score, word_count_score, cosine_similarity, tf_count_score
from Backend.tokenizer import tokenize
from Backend.data_Loader import load_all_data#load_index, load_pagerank, load_pageviews, load_doc_titles
from inverted_index_gcp import InvertedIndex

N_DOCS = 6348910
DEFAULT_AVGDL = 500

CONFIG = {
    'n_docs': N_DOCS,
    'default_avgdl': DEFAULT_AVGDL,
    'bm25_text': {'k1': 1.2, 'b': 0.9, 'k3': 2.0},
    'bm25_title': {'k1': 1.2, 'b': 0.9, 'k3': 2.0},
    'weights': {
        'text': 0.6,
        'title': 0.2,
        'anchor': 0.05,
        'pagerank': 0.05,
        'pageviews': 0.1
    },
    'ranking_methods': {
        'text': 'BM25',        # Options: 'BM25', 'cosine', 'word_count', 'tf_count'
        'title': 'BM25',       # Options: 'BM25', 'cosine', 'word_count', 'tf_count'
        'anchor': 'word_count' # Options: 'BM25', 'cosine', 'word_count', 'tf_count'
    },
    'top_n_candidates': 500,
    'use_stemming': False
}

class SearchEngine:
    """Main search engine with hybrid ranking."""
    
    def __init__(self, config: Dict = CONFIG):
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

        # Precompute document lengths and averages
        self.text_doc_lengths = self.text_index.DL if hasattr(self.text_index, 'DL') else {}
        self.text_avg_dl = sum(self.text_doc_lengths.values()) / len(self.text_doc_lengths) if self.text_doc_lengths else DEFAULT_AVGDL
    
        self.title_doc_lengths = self.title_index.DL if hasattr(self.title_index, 'DL') else {}
        self.title_avg_dl = sum(self.title_doc_lengths.values()) / len(self.title_doc_lengths) if self.title_doc_lengths else DEFAULT_AVGDL
    
    # ========================================================================
    # MAIN SEARCH METHODS
    # ========================================================================
    # def search_basic(self, query: str, top_k: int = 10) -> List[List]:
    #     """
    #     Basic hybrid search for testing - BM25 + PageRank.
        
    #     Simplified version optimized for quick queries.
    #     Returns format: [[doc_id, title], ...]
        
    #     Args:
    #         query: Search query string
    #         top_k: Number of results (default: 10)
        
    #     Returns:
    #         List of [doc_id, title] pairs
    #     """
    #     tokens = tokenize(query,  self.config['use_stemming'])
    #     if not tokens:
    #         return []
        
    #     # Calculate BM25 scores
    #     doc_scores = self._calculate_bm25_scores(tokens)
        
    #     if not doc_scores:
    #         return []
        
    #     # Normalize BM25 scores
    #     doc_scores = self._normalize_scores(doc_scores)

    #     # Combine with PageRank
    #     final_scores = {}
    #     for doc_id, bm25_score in doc_scores.items():
    #         pr_score = self._get_normalized_pagerank(doc_id)
    #         final_scores[doc_id] = (0.8 * bm25_score + 
    #                                0.2 * pr_score)
        
    #     # Sort and format
    #     sorted_docs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
    #     results = []
    #     for doc_id, _ in sorted_docs:
    #         title = self.doc_titles_dict.get(doc_id, f"Article {doc_id}")
    #         results.append([int(doc_id), title])
        
    #     return results
    
    def search(self, query: str, top_k: int = 100) -> List[Tuple[str, str]]:
        """Main hybrid search combining all signals."""
        tokens = tokenize(query,  self.config['use_stemming'])
        if not tokens:
            return []
        
        # Get scores from all indices
        text_scores = self._get_text_scores(tokens, self.config['top_n_candidates'])
        title_scores = self._get_title_scores(tokens, top_n=self.config['top_n_candidates'])
        anchor_scores = self._get_anchor_scores(tokens, top_n=self.config['top_n_candidates'])
        
        # Combine with weights
        combined = self._combine_signals(
            text_scores, title_scores, anchor_scores,
            text_w=self.config['weights']['text'],
            title_w=self.config['weights']['title'],
            anchor_w=self.config['weights']['anchor'],
            pr_w=self.config['weights']['pagerank'],
            pv_w=self.config['weights']['pageviews']
        )
        
        # Sort and return
        sorted_docs = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return self._format_results([doc_id for doc_id, _ in sorted_docs])
    
    def search_body(self, query: str, top_k: int = 100) -> List[Tuple[str, str]]:
        """Search text index only with cosine similarity."""
        return self._search_single_index(query, self.text_index, top_k)
    
    def search_title(self, query: str, top_k: int = 100) -> List[Tuple[str, str]]:
        return self._search_single_index(query, self.title_index, top_k)
        # return self._search_partial_index(query, self.title_index, top_k)
    
    def search_anchor(self, query: str, top_k: int = 100) -> List[Tuple[str, str]]:
        return self._search_single_index(query, self.anchor_index, top_k)
        # return self._search_partial_index(query, self.anchor_index, top_k)
    
    # ========================================================================
    # HELPER METHODS - SCORING
    # ========================================================================
    
    def _get_text_scores(self, tokens: List[str], top_n: int) -> Dict[int, float]:
        """Get scores from text index using configured ranking method."""
        method = self.config['ranking_methods']['text']
        
        if method == 'BM25':
            scores = BM25_score(
                tokens, self.text_index, N_DOCS, 
                self.text_doc_lengths, self.text_avg_dl, 
                k1=self.config['bm25_text']['k1'], 
                k3=self.config['bm25_text']['k3'], 
                b=self.config['bm25_text']['b']
            )
        elif method == 'cosine':
            scores = cosine_similarity(tokens, self.text_index)
        elif method == 'word_count':
            scores = word_count_score(tokens, self.text_index)
        elif method == 'tf_count':
            scores = tf_count_score(tokens, self.text_index)
        else:
            raise ValueError(f"Unknown ranking method: {method}")
        
        return dict(self._normalize_list(scores.most_common(top_n)))
    
    def _get_title_scores(self, tokens: List[str], top_n: int) -> Dict[int, float]:
        """Get scores from title index using configured ranking method."""
        method = self.config['ranking_methods']['title']
        
        if method == 'BM25':
            scores = BM25_score(
                tokens, self.title_index, N_DOCS, 
                self.title_doc_lengths, self.title_avg_dl, 
                k1=self.config['bm25_title']['k1'], 
                k3=self.config['bm25_title']['k3'], 
                b=self.config['bm25_title']['b']
            )
        elif method == 'cosine':
            scores = cosine_similarity(tokens, self.title_index)
        elif method == 'word_count':
            scores = word_count_score(tokens, self.title_index)
        elif method == 'tf_count':
            scores = tf_count_score(tokens, self.title_index)
        else:
            raise ValueError(f"Unknown ranking method: {method}")

        return dict(self._normalize_list(scores.most_common(top_n)))
    
    def _get_anchor_scores(self, tokens: List[str], top_n: int) -> Dict[int, float]: 
        """Get scores from anchor index using configured ranking method."""
        method = self.config['ranking_methods']['anchor']
        
        if method == 'BM25':
            # Anchor index doesn't have DL, use default
            scores = BM25_score(
                tokens, self.anchor_index, N_DOCS, 
                {}, DEFAULT_AVGDL, 
                k1=self.config['bm25_text']['k1'], 
                k3=self.config['bm25_text']['k3'], 
                b=self.config['bm25_text']['b']
            )
        elif method == 'cosine':
            scores = cosine_similarity(tokens, self.anchor_index)
        elif method == 'word_count':
            scores = word_count_score(tokens, self.anchor_index)
        elif method == 'tf_count':
            scores = tf_count_score(tokens, self.anchor_index)
        else:
            raise ValueError(f"Unknown ranking method: {method}")
            
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
        tokens = tokenize(query, False)
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