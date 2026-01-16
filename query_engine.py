import math
from collections import defaultdict, Counter
from typing import List, Tuple, Dict
import numpy as np

from Backend.ranking import BM25_score, word_count_score, cosine_similarity, tf_count_score, ann_search
from Backend.tokenizer import tokenize
from Backend.data_Loader import load_all_data
from inverted_index_gcp import InvertedIndex

N_DOCS = 6348910
DEFAULT_AVGDL = 500
EMBEDDING_MODEL = 'glove-wiki-gigaword-100'

CONFIG = {
    'n_docs': N_DOCS,
    'default_avgdl': DEFAULT_AVGDL,
    'bm25_text': {'k1': 1.2, 'b': 0.9, 'k3': 2.0},
    'bm25_title': {'k1': 1.2, 'b': 0.9, 'k3': 2.0},
    'weights': {
        'text_bm25': 0.7,
        'title_bm25': 0.2,
        'text_ann': 0.12,
        'title_ann': 0.0,
        # 'anchor': 0.05,
        # 'pagerank': 0.05,
        'pageviews': 0.1,
    },
    'ranking_methods': {
        'text': 'BM25',        # Best performance (grid search validated)
        'title': 'BM25',       # Best performance (grid search validated)
        # 'anchor': 'word_count' # Fastest with same accuracy (grid search validated)
    },
    'retrieval': {
        'top_k': 750,          # Number of ANN candidates to retrieve
        'nprobe': 128,          # IVF clusters to probe (higher = more accurate)
        'top_n_candidates': 750,
    },
    
    'use_stemming': False
}

class SearchEngine:
    """Main search engine with hybrid ranking."""

    def __init__(self, config: Dict = CONFIG):
        print("🔧 Initializing Search Engine...")

        self.config = config

        data = load_all_data()

        self.text_index = data['indexes']['text']
        self.title_index = data['indexes']['title']
        self.anchor_index = data['indexes']['anchor']
        self.pagerank_dict = data['pagerank']
        self.pageviews_dict = data['pageviews']
        self.doc_titles_dict = data['titles']

        # FAISS indexes for ANN search
        self.text_faiss = data.get('text_faiss')
        self.text_faiss_docids = data.get('text_faiss_docids')
        self.title_faiss = data.get('title_faiss')
        self.title_faiss_docids = data.get('title_faiss_docids')

        # Load GloVe model for query embeddings (needed for both dict-based and FAISS-based)
        needs_embeddings = (
            self.text_faiss is not None or self.title_faiss is not None
        )
        if needs_embeddings:
            print(f"Loading embedding model: {EMBEDDING_MODEL}...")
            import gensim.downloader as api
            self.embedding_model = api.load(EMBEDDING_MODEL)
            print(f"✓ Embedding model loaded ({len(self.embedding_model)} words)")
        else:
            self.embedding_model = None

        self._precompute_normalization()

        print("✅ Search Engine Ready!")
    # ========================================================================
    # MAIN SEARCH METHODS
    # ========================================================================
    def search(self, query: str, top_k: int = 100) -> List[Tuple[str, str]]:
        """
        Main hybrid search.

        Pipeline:
        1. Tokenize query
        2. BM25 retrieval from text and title indexes
        3. ANN retrieval from FAISS indexes
        4. Union candidates from BM25 and ANN
        5. Normalize and blend all signals with weights
        6. Return top-k results
        """
        tokens = tokenize(query, self.config['use_stemming'])
        if not tokens:
            return []

        # ---- BM25 retrieval ----
        text_bm25_scores = self._get_text_scores(tokens, self.config['retrieval']['top_n_candidates'])
        title_bm25_scores = self._get_title_scores(tokens, top_n=self.config['retrieval']['top_n_candidates'])

        # ---- ANN retrieval (FAISS) ----
        query_emb = self._compute_query_embedding(tokens)
        text_ann_scores = self._get_ann_scores(query_emb, self.text_faiss, self.text_faiss_docids)

        # ---- Combine scores ----
        combined = self._combine_scores(
            text_bm25_scores,
            title_bm25_scores,
            text_ann_scores,
        )

        # Sort and return
        sorted_docs = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return self._format_results([doc_id for doc_id, _ in sorted_docs])
    
    def search_body(self, query: str, top_k: int = 100) -> List[Tuple[str, str]]:
        """Search text index only with cosine similarity."""
        return self._search_single_index(query, self.text_index, top_k)
    
    def search_title(self, query: str, top_k: int = 100) -> List[Tuple[str, str]]:
        return self._search_single_index(query, self.title_index, top_k)
    
    def search_anchor(self, query: str, top_k: int = 100) -> List[Tuple[str, str]]:
        return self._search_single_index(query, self.anchor_index, top_k)
    
    # ========================================================================
    # HELPER METHODS - SCORING
    # ========================================================================
    def _compute_query_embedding(self, tokens: List[str]) -> np.ndarray:
        """Compute query embedding as mean of token vectors (same as indexing)."""
        if not self.embedding_model:
            return None
        valid_tokens = [t for t in tokens if t in self.embedding_model]
        if not valid_tokens:
            return None
        return self.embedding_model.get_mean_vector(valid_tokens, pre_normalize=False).astype(np.float32)

    def _combine_scores(self, 
                        text_bm25_scores,
                        title_bm25_scores,
                        text_ann_scores
                        ) -> Dict[int, float]:
        weights = self.config['weights']
        all_candidates = (
            set(text_bm25_scores.keys()) |
            set(title_bm25_scores.keys()) |
            set(text_ann_scores.keys())
        )

        # ---- Blend signals ----
        combined = {}
        for doc_id in all_candidates:
            combined[doc_id] = (
                text_bm25_scores.get(doc_id, 0.0) * weights['text_bm25'] +
                title_bm25_scores.get(doc_id, 0.0) * weights['title_bm25'] +
                text_ann_scores.get(doc_id, 0.0) * weights['text_ann'] +
                self._norm_pv(doc_id) * weights['pageviews']
            )
        return combined
    
    def _get_text_scores(self, tokens: List[str], top_n: int) -> Dict[int, float]:
        """Get scores from text index using configured ranking method."""
        scores = BM25_score(
                tokens, self.text_index, N_DOCS, 
                self.text_doc_lengths, self.text_avg_dl, 
                k1=self.config['bm25_text']['k1'], 
                k3=self.config['bm25_text']['k3'], 
                b=self.config['bm25_text']['b']
                )
        
        return dict(self._normalize_list(scores.most_common(top_n)))
    
    def _get_title_scores(self, tokens: List[str], top_n: int) -> Dict[int, float]:
        """Get scores from title index using configured ranking method."""
        scores = BM25_score(
            tokens, self.title_index, N_DOCS, 
            self.title_doc_lengths, self.title_avg_dl, 
            k1=self.config['bm25_title']['k1'], 
            k3=self.config['bm25_title']['k3'], 
            b=self.config['bm25_title']['b']
            )

        return dict(self._normalize_list(scores.most_common(top_n)))
    
    def _get_anchor_scores(self, tokens: List[str], top_n: int) -> Dict[int, float]: 
        """Get scores from anchor index using word_count (fastest, same accuracy as BM25)."""
        # Grid search showed word_count is 25% faster than BM25 with identical P@10
        # since anchor weight is only 0.05, the ranking method has minimal impact
        scores = word_count_score(tokens, self.anchor_index)
            
        return dict(self._normalize_list(scores.most_common(top_n)))
    
    def _get_ann_scores(self, query_emb: np.ndarray, faiss_index, faiss_docids) -> Dict[int, float]:
        ann_raw = ann_search(
            query_emb, faiss_index, faiss_docids,
            top_k=self.config['retrieval']['top_k'], nprobe=self.config['retrieval']['nprobe']
            )
        return self._normalize_ann_scores(ann_raw)
    
    def _search_single_index(self, query: str, index: InvertedIndex, top_k: int) -> List[Tuple[str, str]]:
        """Search single index using BM25 (best performing method)."""
        tokens = tokenize(query, False)
        if not tokens:
            return []
        
        # Determine which index and use appropriate doc lengths
        if index is self.text_index:
            doc_lengths, avg_dl = self.text_doc_lengths, self.text_avg_dl
            bm25_params = self.config['bm25_text']
        elif index is self.title_index:
            doc_lengths, avg_dl = self.title_doc_lengths, self.title_avg_dl
            bm25_params = self.config['bm25_title']
        else:  # anchor index
            doc_lengths, avg_dl = {}, DEFAULT_AVGDL
            bm25_params = self.config['bm25_text']
        
        scores = BM25_score(
            tokens, index, N_DOCS,
            doc_lengths, avg_dl,
            k1=bm25_params['k1'],
            k3=bm25_params['k3'],
            b=bm25_params['b']
        )
        doc_ids = [doc_id for doc_id, _ in scores.most_common(top_k)]
        
        return self._format_results(doc_ids)
    
    # ========================================================================
    # HELPER METHODS - NORMALIZATION
    # ========================================================================   
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

    def _normalize_ann_scores(self, scores: Counter) -> Dict[int, float]:
        """Normalize ANN scores (cosine similarities) to [0, 1] range."""
        if not scores:
            return {}
        # Cosine similarities are in [-1, 1], but typically positive for similar docs
        max_score = max(scores.values())
        min_score = min(scores.values())
        score_range = max_score - min_score
        if score_range > 0:
            return {doc_id: (s - min_score) / score_range for doc_id, s in scores.items()}
        return {doc_id: 1.0 for doc_id in scores}  # All same score
    
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
    
    def get_doc_titles(self, doc_ids: List[int]) -> List[str]:
        """Get titles for doc IDs."""
        return [self.doc_titles_dict.get(doc_id, f"Document {doc_id}") for doc_id in doc_ids]