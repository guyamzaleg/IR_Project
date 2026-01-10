#!/usr/bin/env python3
"""
Comprehensive Search Engine Evaluation Test Suite

Evaluates search engine performance using multiple metrics:
- Average Precision@10 (AP@10)
- Precision@5 (P@5)
- F1@30
- Harmonic Mean of P@5 and F1@30

Tests against queries_train.json ground truth data.
"""

import json
import sys
import os
from pathlib import Path
from collections import defaultdict
import statistics
from datetime import datetime
from typing import Dict, List, Tuple, Set

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import search engine
from query_engine import SearchEngine, CONFIG


class SearchEngineEvaluator:
    
    def __init__(self, queries_file: str = "queries_train.json"):

        self.queries_file = queries_file
        self.queries_dict = {}
        self.search_engine = None
        self.results = {
            'per_query': {},
            'summary': {}
        }
        
    def load_queries(self) -> Dict[str, List[str]]:
        queries_path = Path(self.queries_file)
        
        if not queries_path.exists():
            raise FileNotFoundError(f"Queries file not found: {queries_path}")
        
        print(f"Loading queries from {queries_path}...")
        
        with open(queries_path, 'r', encoding='utf-8') as f:
            self.queries_dict = json.load(f)
        
        print(f"✓ Loaded {len(self.queries_dict)} queries")
        return self.queries_dict
    
    def initialize_search_engine(self):
        """Initialize the search engine instance."""
        if self.search_engine is None:
            print("Initializing search engine...")
            self.search_engine = SearchEngine(CONFIG)
            print("✓ Search engine initialized successfully!")
    
    def query_search(self, query_text: str, top_k: int = 100) -> List[str]:
        # Use search_basic method which returns [(doc_id, title), ...]
        results = self.search_engine.search(query_text)
        
        # Extract doc_ids and convert to strings
        doc_ids = [str(doc_id) for doc_id, _ in results[:top_k]]
        
        return doc_ids
    
    @staticmethod
    def calculate_precision_at_k(retrieved_docs: List[str], 
                                  relevant_docs: List[str], 
                                  k: int = 10) -> Tuple[float, Set[str]]:
        retrieved_set = set(retrieved_docs[:k])
        relevant_set = set(relevant_docs)
        hits = retrieved_set & relevant_set
        
        precision = len(hits) / k if k > 0 else 0.0
        return precision, hits
    
    @staticmethod
    def calculate_average_precision_at_k(retrieved_docs: List[str],
                                          relevant_docs: List[str],
                                          k: int = 10) -> float:
        relevant_set = set(relevant_docs)
        retrieved_k = retrieved_docs[:k]
        
        if not relevant_set:
            return 0.0
        
        precision_sum = 0.0
        num_relevant_found = 0
        
        for i, doc_id in enumerate(retrieved_k, 1):
            if doc_id in relevant_set:
                num_relevant_found += 1
                precision_at_i = num_relevant_found / i
                precision_sum += precision_at_i
        
        if num_relevant_found == 0:
            return 0.0
        
        # Average over relevant documents found (up to k)
        ap = precision_sum / min(len(relevant_set), k)
        return ap
    
    @staticmethod
    def calculate_recall_at_k(retrieved_docs: List[str],
                               relevant_docs: List[str],
                               k: int = 30) -> float:
        retrieved_set = set(retrieved_docs[:k])
        relevant_set = set(relevant_docs)
        
        if not relevant_set:
            return 0.0
        
        true_positives = len(retrieved_set & relevant_set)
        recall = true_positives / len(relevant_set)
        
        return recall
    
    @staticmethod
    def calculate_f1_at_k(retrieved_docs: List[str],
                          relevant_docs: List[str],
                          k: int = 30) -> float:
        retrieved_set = set(retrieved_docs[:k])
        relevant_set = set(relevant_docs)
        
        if not relevant_set:
            return 0.0
        
        true_positives = len(retrieved_set & relevant_set)
        
        # Precision@K
        precision = true_positives / k if k > 0 else 0.0
        
        # Recall@K
        recall = true_positives / len(relevant_set)
        
        # F1 Score
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        return f1
    
    @staticmethod
    def calculate_harmonic_mean(p5: float, f1_30: float) -> float:
        if p5 + f1_30 > 0:
            return 2 * (p5 * f1_30) / (p5 + f1_30)
        return 0.0
    
    def evaluate_all_queries(self) -> Dict:
        # Ensure queries are loaded
        if not self.queries_dict:
            self.load_queries()
        
        # Initialize search engine
        self.initialize_search_engine()
        
        # Storage for all metrics
        ap10_scores = []
        p5_scores = []
        p10_scores = []
        f1_30_scores = []
        harmonic_mean_scores = []
        
        print(f"\n{'='*80}")
        print("SEARCH ENGINE COMPREHENSIVE EVALUATION")
        print(f"{'='*80}")
        print(f"\nEvaluating {len(self.queries_dict)} queries...\n")
        
        # Evaluate each query
        for idx, (query_text, relevant_docs) in enumerate(self.queries_dict.items(), 1):
            # Query search engine (get top 100 for full evaluation)
            retrieved_docs = self.query_search(query_text, top_k=100)
            
            # Calculate all metrics
            ap10 = self.calculate_average_precision_at_k(retrieved_docs, relevant_docs, k=10)
            p5, hits_5 = self.calculate_precision_at_k(retrieved_docs, relevant_docs, k=5)
            p10, hits_10 = self.calculate_precision_at_k(retrieved_docs, relevant_docs, k=10)
            f1_30 = self.calculate_f1_at_k(retrieved_docs, relevant_docs, k=30)
            harmonic_mean = self.calculate_harmonic_mean(p5, f1_30)
            
            # Store per-query results
            self.results['per_query'][query_text] = {
                'average_precision_at_10': ap10,
                'precision_at_5': p5,
                'precision_at_10': p10,
                'f1_at_30': f1_30,
                'harmonic_mean_p5_f1_30': harmonic_mean,
                'relevant_found_at_5': len(hits_5),
                'relevant_found_at_10': len(hits_10),
                'total_relevant': len(relevant_docs),
                'total_retrieved': len(retrieved_docs)
            }
            
            # Accumulate scores
            ap10_scores.append(ap10)
            p5_scores.append(p5)
            p10_scores.append(p10)
            f1_30_scores.append(f1_30)
            harmonic_mean_scores.append(harmonic_mean)
            
            # Progress output
            status = "✓" if harmonic_mean > 0.1 else "✗"
            print(f"[{idx:2d}/{len(self.queries_dict)}] {status} {query_text[:45]:45s} | "
                  f"AP@10={ap10:.3f} P@5={p5:.3f} F1@30={f1_30:.3f} HM={harmonic_mean:.3f}")
        
        print(f"\n{'='*80}")
        
        # Calculate summary statistics
        self.results['summary'] = {
            'total_queries': len(self.queries_dict),
            'mean_average_precision_at_10': statistics.mean(ap10_scores),
            'mean_precision_at_5': statistics.mean(p5_scores),
            'mean_precision_at_10': statistics.mean(p10_scores),
            'mean_f1_at_30': statistics.mean(f1_30_scores),
            'mean_harmonic_mean': statistics.mean(harmonic_mean_scores),
            'median_average_precision_at_10': statistics.median(ap10_scores),
            'median_harmonic_mean': statistics.median(harmonic_mean_scores),
            'std_dev_average_precision_at_10': statistics.stdev(ap10_scores) if len(ap10_scores) > 1 else 0,
            'std_dev_harmonic_mean': statistics.stdev(harmonic_mean_scores) if len(harmonic_mean_scores) > 1 else 0,
            'min_average_precision_at_10': min(ap10_scores),
            'max_average_precision_at_10': max(ap10_scores),
            'min_harmonic_mean': min(harmonic_mean_scores),
            'max_harmonic_mean': max(harmonic_mean_scores),
            'queries_passing_ap10_threshold': sum(1 for x in ap10_scores if x > 0.1),
            'queries_passing_hm_threshold': sum(1 for x in harmonic_mean_scores if x > 0.1)
        }
        
        return self.results
    
    def print_summary(self):
        """Print summary statistics to console."""
        summary = self.results['summary']
        
        print(f"\n{'='*80}")
        print("EVALUATION SUMMARY")
        print(f"{'='*80}\n")
        
        print("📊 PRIMARY METRICS:")
        print(f"   Mean Average Precision@10:  {summary['mean_average_precision_at_10']:.4f} "
              f"{'✓ PASS' if summary['mean_average_precision_at_10'] > 0.1 else '✗ FAIL'} (threshold: > 0.1)")
        print(f"   Mean Harmonic Mean (P@5 & F1@30): {summary['mean_harmonic_mean']:.4f}\n")
        
        print("📈 DETAILED METRICS:")
        print(f"   Mean Precision@5:           {summary['mean_precision_at_5']:.4f}")
        print(f"   Mean Precision@10:          {summary['mean_precision_at_10']:.4f}")
        print(f"   Mean F1@30:                 {summary['mean_f1_at_30']:.4f}\n")
        
        print("📉 STATISTICS:")
        print(f"   Median AP@10:               {summary['median_average_precision_at_10']:.4f}")
        print(f"   Std Dev AP@10:              {summary['std_dev_average_precision_at_10']:.4f}")
        print(f"   Min/Max AP@10:              {summary['min_average_precision_at_10']:.4f} / "
              f"{summary['max_average_precision_at_10']:.4f}\n")
        
        print("🎯 THRESHOLD ANALYSIS:")
        print(f"   Queries passing AP@10 > 0.1: {summary['queries_passing_ap10_threshold']}/{summary['total_queries']}")
        print(f"   Queries passing HM > 0.1:    {summary['queries_passing_hm_threshold']}/{summary['total_queries']}\n")
        
        print(f"{'='*80}\n")
    
    def save_results(self, output_dir: str = "tests/results"):
        """
        Save detailed results to files.
        
        Args:
            output_dir: Directory to save results
        """
        # Create timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir) / timestamp
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        json_path = output_path / "evaluation_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # Save Markdown report
        md_path = output_path / "evaluation_report.md"
        self._generate_markdown_report(md_path)
        
        print(f"✓ Results saved to: {output_path}")
        print(f"  - JSON: {json_path}")
        print(f"  - Report: {md_path}")
        
        return output_path
    
    def _generate_markdown_report(self, filepath: Path):
        """Generate comprehensive Markdown report."""
        summary = self.results['summary']
        per_query = self.results['per_query']
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# Search Engine Comprehensive Evaluation Report\n\n")
            f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            
            f.write("## Executive Summary\n\n")
            
            # Pass/Fail status
            ap10_pass = summary['mean_average_precision_at_10'] > 0.1
            f.write(f"**Overall Status: {'✅ PASS' if ap10_pass else '❌ FAIL'}**\n\n")
            
            f.write("### Primary Metrics\n\n")
            f.write("| Metric | Value | Status |\n")
            f.write("|--------|-------|--------|\n")
            f.write(f"| Mean Average Precision@10 | {summary['mean_average_precision_at_10']:.4f} | "
                   f"{'✅ PASS' if ap10_pass else '❌ FAIL'} (> 0.1) |\n")
            f.write(f"| Mean Harmonic Mean (P@5 & F1@30) | {summary['mean_harmonic_mean']:.4f} | - |\n\n")
            
            f.write("## Detailed Statistics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Total Queries Tested | {summary['total_queries']} |\n")
            f.write(f"| Mean Precision@5 | {summary['mean_precision_at_5']:.4f} |\n")
            f.write(f"| Mean Precision@10 | {summary['mean_precision_at_10']:.4f} |\n")
            f.write(f"| Mean F1@30 | {summary['mean_f1_at_30']:.4f} |\n")
            f.write(f"| Median AP@10 | {summary['median_average_precision_at_10']:.4f} |\n")
            f.write(f"| Std Dev AP@10 | {summary['std_dev_average_precision_at_10']:.4f} |\n")
            f.write(f"| Min AP@10 | {summary['min_average_precision_at_10']:.4f} |\n")
            f.write(f"| Max AP@10 | {summary['max_average_precision_at_10']:.4f} |\n")
            f.write(f"| Queries Passing AP@10 > 0.1 | {summary['queries_passing_ap10_threshold']}/{summary['total_queries']} |\n")
            f.write(f"| Queries Passing HM > 0.1 | {summary['queries_passing_hm_threshold']}/{summary['total_queries']} |\n\n")
            
            f.write("## Per-Query Results\n\n")
            f.write("| # | Query | AP@10 | P@5 | P@10 | F1@30 | HM |\n")
            f.write("|---|-------|-------|-----|------|-------|----|\n")
            
            for idx, (query, data) in enumerate(per_query.items(), 1):
                f.write(f"| {idx} | {query[:40]}... | "
                       f"{data['average_precision_at_10']:.3f} | "
                       f"{data['precision_at_5']:.3f} | "
                       f"{data['precision_at_10']:.3f} | "
                       f"{data['f1_at_30']:.3f} | "
                       f"{data['harmonic_mean_p5_f1_30']:.3f} |\n")
            
            f.write("\n## Metric Definitions\n\n")
            f.write("- **AP@10**: Average Precision at rank 10\n")
            f.write("- **P@5**: Precision at rank 5\n")
            f.write("- **P@10**: Precision at rank 10\n")
            f.write("- **F1@30**: F1 score at rank 30\n")
            f.write("- **HM**: Harmonic Mean of P@5 and F1@30\n")
    
    def run_full_evaluation(self, save_results: bool = True) -> Dict:
        # Load queries
        self.load_queries()
        
        # Run evaluation
        results = self.evaluate_all_queries()
        
        # Print summary
        self.print_summary()
        
        # Save results
        if save_results:
            self.save_results()
        
        return results


def main():
    """Main entry point for running evaluations."""
    print("\n" + "="*80)
    print("SEARCH ENGINE COMPREHENSIVE EVALUATION SUITE")
    print("="*80 + "\n")
    
    # Create evaluator
    evaluator = SearchEngineEvaluator(queries_file="queries_train.json")
    
    # Run full evaluation
    results = evaluator.run_full_evaluation(save_results=True)
    
    print("\n✅ Evaluation complete!\n")
    
    return results


if __name__ == "__main__":
    main()