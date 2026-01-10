"""
Pure evaluation metrics for search engine.
Calculates P@5, P@10, F1@30, HM(P@5,F1@30), AP, and retrieval time.
"""
import time
import json
from typing import List, Dict, Tuple, Callable
import numpy as np


class SearchEvaluator:
    """Calculate search quality metrics."""
    
    def __init__(self, ground_truth_file: str = "queries_train.json"):
        with open(ground_truth_file, 'r') as f:
            self.ground_truth = json.load(f)
        
        # Convert doc_ids to strings
        self.ground_truth = {
            query: [str(doc_id) for doc_id in doc_ids]
            for query, doc_ids in self.ground_truth.items()
        }
    
    def precision_at_k(self, results: List[str], relevant: List[str], k: int) -> float:
        """Precision@K = (relevant in top-k) / k"""
        if not results or k == 0:
            return 0.0
        top_k = results[:k]
        relevant_set = set(relevant)
        num_relevant = sum(1 for doc_id in top_k if doc_id in relevant_set)
        return num_relevant / k
    
    def recall_at_k(self, results: List[str], relevant: List[str], k: int) -> float:
        """Recall@K = (relevant in top-k) / total_relevant"""
        if not relevant:
            return 0.0
        top_k = results[:k]
        relevant_set = set(relevant)
        num_relevant = sum(1 for doc_id in top_k if doc_id in relevant_set)
        return num_relevant / len(relevant)
    
    def f1_at_k(self, results: List[str], relevant: List[str], k: int) -> float:
        """F1@K = harmonic mean of P@K and R@K"""
        p = self.precision_at_k(results, relevant, k)
        r = self.recall_at_k(results, relevant, k)
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)
    
    def average_precision(self, results: List[str], relevant: List[str]) -> float:
        """Average Precision"""
        if not relevant:
            return 0.0
        
        relevant_set = set(relevant)
        num_relevant = 0
        sum_precisions = 0.0
        
        for i, doc_id in enumerate(results, 1):
            if doc_id in relevant_set:
                num_relevant += 1
                sum_precisions += num_relevant / i
        
        return sum_precisions / len(relevant) if num_relevant > 0 else 0.0
    
    def evaluate_single_query(self, query: str, results: List[Tuple[str, str]]) -> Dict[str, float]:
        """
        Evaluate one query. Returns all required metrics.
        
        Returns:
            {
                'P@5': float,
                'P@10': float, 
                'F1@30': float,
                'HM_P5_F30': float,  # Harmonic mean of P@5 and F1@30
                'AP': float
            }
        """
        result_ids = [doc_id for doc_id, _ in results]
        
        if query not in self.ground_truth:
            return {'P@5': 0.0, 'P@10': 0.0, 'F1@30': 0.0, 'HM_P5_F30': 0.0, 'AP': 0.0}
        
        relevant = self.ground_truth[query]
        
        p5 = self.precision_at_k(result_ids, relevant, 5)
        p10 = self.precision_at_k(result_ids, relevant, 10)
        f30 = self.f1_at_k(result_ids, relevant, 30)
        ap = self.average_precision(result_ids, relevant)
        
        # Harmonic mean of P@5 and F1@30 (main quality metric)
        hm = 2 * (p5 * f30) / (p5 + f30) if (p5 + f30) > 0 else 0.0
        
        return {
            'P@5': p5,
            'P@10': p10,
            'F1@30': f30,
            'HM_P5_F30': hm,
            'AP': ap
        }
    
    def evaluate_search_method(
        self, 
        search_func: Callable[[str], List[Tuple[str, str]]], 
        queries: List[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate search method on queries. Measures quality AND time.
        
        Returns:
            {
                'avg_P@5': float,
                'avg_P@10': float,
                'avg_F1@30': float,
                'avg_HM_P5_F30': float,  # MAIN QUALITY METRIC
                'avg_AP': float,
                'avg_time': float,       # MAIN EFFICIENCY METRIC
                'max_time': float,
                'num_queries': int
            }
        """
        if queries is None:
            queries = list(self.ground_truth.keys())
        
        all_p5, all_p10, all_f30, all_hm, all_ap = [], [], [], [], []
        times = []
        
        for query in queries:
            # Measure time
            start = time.time()
            results = search_func(query)
            elapsed = time.time() - start
            times.append(elapsed)
            
            # Measure quality
            metrics = self.evaluate_single_query(query, results)
            all_p5.append(metrics['P@5'])
            all_p10.append(metrics['P@10'])
            all_f30.append(metrics['F1@30'])
            all_hm.append(metrics['HM_P5_F30'])
            all_ap.append(metrics['AP'])
        
        return {
            'avg_P@5': np.mean(all_p5),
            'avg_P@10': np.mean(all_p10),
            'avg_F1@30': np.mean(all_f30),
            'avg_HM_P5_F30': np.mean(all_hm),  # Main quality metric
            'avg_AP': np.mean(all_ap),
            'avg_time': np.mean(times),         # Main efficiency metric
            'max_time': np.max(times),
            'num_queries': len(queries)
        }
