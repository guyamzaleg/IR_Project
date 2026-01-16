"""
Comprehensive grid search optimization for search engine.
Clean implementation with single responsibility - updates query engine config.
"""
import itertools
import pandas as pd
from typing import Dict, List
from datetime import datetime
import os

import query_engine
from query_engine import SearchEngine
from Backend.analysis import SearchEvaluator

# ============================================================================
# GRID SEARCH CONFIGURATION
# ============================================================================

GRID_CONFIG = {
    # Ranking methods to test
    'ranking_methods': [
        'BM25',           # BM25 ranking
        'cosine',         # TF-IDF cosine
        'word_count',     # binary ranking
        'tf_count',       # term frequency ranking
    ],
    
    # BM25 hyperparameters
    'bm25': {
        'k1': [0.8, 1.2, 1.5, 1.8, 2.0],
        'b': [0.3, 0.5, 0.75, 0.9],
        'k3': [0.5, 1.0, 1.5, 2.0],
    },
    
    # Tokenization
    'use_stemming': [True, False],

    # weights (normalized to sum to 1.0)
    'weights': {
        'text_bm25':  [0.60, 0.70, 0.80],
        'title_bm25': [0.10, 0.15, 0.20],
        'pv':       [0.00, 0.05, 0.10],
        'text_ann':   [0.125, 0.15, 0.20],
        'title_ann':  [0.00, 0.05, 0.10],
    },

    'retrieval': {
        'top_k': [250, 500, 750],          # Number of ANN candidates to retrieve
        'nprobe': [32, 64,128],          # IVF clusters to probe (higher = more accurate)
        'top_n_candidates': [100, 250, 500, 750, 1000] # Number of candidates to re-rank BM25
    },
}

# Evaluation settings
TIME_LIMIT = 35.0  # Max time per query in seconds
SAMPLE_SIZE = None  # Use all queries (or set to int for subset)

# ============================================================================
# GRID SEARCH OPTIMIZER
# ============================================================================

class GridSearch:
    """Clean grid search - updates query_engine.CONFIG for each test."""
    
    def __init__(self, engine: SearchEngine, evaluator: SearchEvaluator):
        self.engine = engine
        self.evaluator = evaluator
        self.results = []
    
    # ========================================================================
    # Ranking Methods
    # ========================================================================
    
    def _ranking_methods(self, queries: List[str]):
        """Test different ranking method combinations."""
        print("\n🎯 Testing Ranking Method Combinations\n")
        
        methods = GRID_CONFIG['ranking_methods']
        
        # Test all combinations of ranking methods for text, title, and anchor
        all_combos = list(itertools.product(methods, methods, methods))
        
        print(f"  Testing {len(all_combos)} method combinations...\n")
        
        tested = 0
        for text_method, title_method, anchor_method in all_combos:
            self.engine.config['ranking_methods']['text'] = text_method
            self.engine.config['ranking_methods']['title'] = title_method
            self.engine.config['ranking_methods']['anchor'] = anchor_method
            
            config_name = f"rank_t-{text_method}_ti-{title_method}_a-{anchor_method}"
            self._test_config(config_name, queries)
            tested += 1
            
            if tested % 20 == 0:
                print(f"  Tested {tested}/{len(all_combos)} configs...")
        
        print(f"\n✓ Ranking method testing complete: {tested} configurations tested\n")
    # ========================================================================
    # BM25 HYPERPARAMETER TUNING
    # ========================================================================
    
    def _BM25(self, queries: List[str]):
        """Test BM25 with different hyperparameters"""
        print("\n🔧 BM25 Hyperparameter Tuning\n")
         
        tested = 0
        total = (len(GRID_CONFIG['bm25']['k1']) * 
                len(GRID_CONFIG['bm25']['b']) * 
                len(GRID_CONFIG['bm25']['k3']) 
                )
        
        print(f"  Testing {total} configurations...\n")
        
        for k1 in GRID_CONFIG['bm25']['k1']:
            for b in GRID_CONFIG['bm25']['b']:
                for k3 in GRID_CONFIG['bm25']['k3']:
                    self.engine.config['bm25_text']['k1'] = k1
                    self.engine.config['bm25_text']['b'] = b
                    self.engine.config['bm25_text']['k3'] = k3
                    self.engine.config['bm25_title']['k1'] = k1
                    self.engine.config['bm25_title']['b'] = b
                    self.engine.config['bm25_title']['k3'] = k3
                    config_name = f"bm25_k1{k1}_b{b}_k3{k3}"
                        
                    self._test_config(config_name, queries)
                    tested += 1

                    if tested % 20 == 0:
                        print(f"  Progress: {tested}/{total} configs tested...")    
                        
        print(f"✓ BM25 tuning complete: {tested} configurations tested\n")
    
    # ========================================================================
    # WEIGHT OPTIMIZATION
    # ========================================================================
    
    def _weights(self, queries: List[str]):
        """Test ranking with different weight combinations that sum to 1.0."""
        print("\n⚖️  Weight Optimization\n")
        
        # Generate all weight combinations
        all_combos = list(itertools.product(
            GRID_CONFIG['weights']['text_bm25'],
            GRID_CONFIG['weights']['title_bm25'],
            GRID_CONFIG['weights']['pv'],
            GRID_CONFIG['weights']['text_ann'],
            GRID_CONFIG['weights']['title_ann'],
        ))
        
        # Filter to only keep combinations that sum to ~1.0 (within tolerance)
        weight_configs = [
            combo for combo in all_combos 
            if abs(sum(combo) - 1.0) < 0.5  # Allow small rounding errors
        ]
        
        print(f"  Testing {len(weight_configs)} weight combinations (sum=1.0)...\n")
        print(f"  (Filtered from {len(all_combos)} total combinations)\n")
        
        tested = 0
        # for text_w, title_w, anchor_w, pr_w, pv in weight_configs:
        for text_bm25_w, title_bm25_w, pv, text_ann_w, title_ann_w in weight_configs:
            self.engine.config['weights']['text_bm25'] = text_bm25_w
            self.engine.config['weights']['title_bm25'] = title_bm25_w
            self.engine.config['weights']['pv'] = pv
            self.engine.config['weights']['text_ann'] = text_ann_w
            self.engine.config['weights']['title_ann'] = title_ann_w
            config_name = "weights_textbm25{:.2f}_titlebm25{:.2f}_pv{:.2f}_textann{:.2f}_titleann{:.2f}".format(
                text_bm25_w, title_bm25_w, pv, text_ann_w, title_ann_w
            )
            
            self._test_config(config_name, queries)
            tested += 1
            
            if tested % 50 == 0:
                print(f"  Tested {tested}/{len(weight_configs)} configs...")
        
        print(f"\n✓ Weight optimization complete: {tested} configurations tested\n")

    # ========================================================================
    # Stemming
    # ========================================================================
    
    def _stemming(self, queries: List[str]):
        """Test stemming."""
        print("\n🔧 Test stemming\n")
        
        self.engine.config['use_stemming'] = True                
        config_name = f"stem{True}"
        self._test_config(config_name, queries)

        self.engine.config['use_stemming'] = False                
        config_name = f"stem{False}"
        self._test_config(config_name, queries)
                       
        
        print(f"✓ Stemming configurations tested\n")

    # ========================================================================
    # retrieval
    # ========================================================================
    
    def _retrieval(self, queries: List[str]):
        """Test Embedding with different hyperparameters"""
        print("\n🔧 Embeddings Hyperparameter Tuning\n")
         
        tested = 0
        total = (len(GRID_CONFIG['retrieval']['top_k']) * 
                len(GRID_CONFIG['retrieval']['nprobe']) *
                len(GRID_CONFIG['retrieval']['top_n_candidates'])
                )
        
        print(f"  Testing {total} configurations...\n")
        
        for top_k in GRID_CONFIG['retrieval']['top_k']:
            for nprobe in GRID_CONFIG['retrieval']['nprobe']:
                for top_n_candidates in GRID_CONFIG['retrieval']['top_n_candidates']:
                    self.engine.config['retrieval']['top_k'] = top_k
                    self.engine.config['retrieval']['nprobe'] = nprobe
                    self.engine.config['retrieval']['top_n_candidates'] = top_n_candidates
                    config_name = f"embeddings_top_k{top_k}_nprobe{nprobe}_topn{top_n_candidates}"     
                    self._test_config(config_name, queries)
                    tested += 1   
                        
        print(f"✓ Embeddings tuning complete: {tested} configurations tested\n")

    # ========================================================================
    # CORE TESTING LOGIC
    # ========================================================================
    
    def _test_config(self, config_name: str, queries: List[str]):
        """Test a single configuration by updating query engine CONFIG."""

        search_func = self.engine.search
        metrics = self.evaluator.evaluate_search_method(search_func, queries)
        
        # Only keep configs within time limit
        if metrics['max_time'] <= TIME_LIMIT:
            result = {
                'config_name': config_name,
                **metrics
            }
            self.results.append(result)
    
    def save_results(self, df: pd.DataFrame, save_dir: str = "tests/results"):
        """Save results to CSV with summary."""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save full results
        filepath = f"{save_dir}/grid_search_{timestamp}.csv"
        df.to_csv(filepath, index=False)
        print(f"\n✅ Full results saved to {filepath}")
        
        # Save top 20 configs
        top_path = f"{save_dir}/top20_configs_{timestamp}.txt"
        with open(top_path, 'w') as f:
            f.write("TOP 20 CONFIGURATIONS\n")
            f.write("=" * 100 + "\n\n")
            for idx, (i, row) in enumerate(df.head(20).iterrows(), 1):
                f.write(f"#{idx}: {row['config_name']}\n")
                f.write(f"     HM(P@5,F1@30)={row['avg_HM_P5_F30']:.4f} | "
                       f"P@10={row['avg_P@10']:.4f} | "
                       f"Time={row['avg_time']:.3f}s\n\n")
        
        print(f"✅ Top 20 configs saved to {top_path}")
        
        return filepath

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_test(test):
    engine = SearchEngine()
    evaluator = SearchEvaluator("queries_train.json")
    
    optimizer = GridSearch(engine, evaluator)
    
    # Get all queries
    queries = list(evaluator.ground_truth.keys())
    
    # Run Test
    # if test == 'phase1':
    #     optimizer._phase1_single_index_ranking(queries)
    if test == 'BM25':
        optimizer._BM25(queries)
    
    if test == 'weights':
        optimizer._weights(queries)

    if test == 'stemming':
        optimizer._stemming(queries)
    
    if test == 'ranking':
        optimizer._ranking_methods(queries)
    
    if test == 'retrieval':
        optimizer._retrieval(queries)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(optimizer.results)
    df = df.sort_values('avg_P@10', ascending=False)
    
    # Save results
    filepath = optimizer.save_results(df, save_dir=f"tests/results/{test}")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"{test.upper()} COMPLETE")
    print(f"{'='*80}")
    print(f"\nTop 5 ranking:")
    print(df.head(5)[['config_name', 'avg_HM_P5_F30', 'avg_P@10', 'avg_time']].to_string(index=False))
    
    # Identify best method
    best = df.iloc[0]
    print(f"\n🏆 BEST: {best['config_name']}")
    print(f"   HM(P@5,F1@30): {best['avg_HM_P5_F30']:.4f}")
    print(f"{'='*80}\n")
    
    return df


if __name__ == "__main__":
    results_df = run_test("retrieval")
