"""
Live Search Engine Evaluation
Tests the deployed search engine at http://35.223.209.4:8080/
Evaluates using P@10, P@5, F1@30, and Harmonic Mean metrics
"""

import json
import requests
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import pandas as pd

# Configuration
ENGINE_URL = "http://35.223.209.4:8080/search"
QUERIES_FILE = Path(__file__).parent.parent / "queries_train.json"
RESULTS_DIR = Path(__file__).parent / "results" / "live_engine"
TIMEOUT = 30  # seconds

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def load_ground_truth():
    """Load ground truth queries and relevant documents"""
    with open(QUERIES_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def query_search_engine(query, top_n=100):
    """
    Query the live search engine and return results
    Returns: (doc_ids, response_time)
    """
    try:
        start_time = time.time()
        response = requests.get(
            ENGINE_URL,
            params={'query': query, 'n': top_n},
            timeout=TIMEOUT
        )
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            results = response.json()
            
            # Extract document IDs from results - handle different formats
            doc_ids = []
            if isinstance(results, list):
                for item in results:
                    if isinstance(item, dict):
                        # Format: [{"doc_id": "123", ...}, ...] or [{"id": "123", ...}, ...]
                        doc_id = item.get('doc_id', item.get('id', item.get('wiki_id', '')))
                        doc_ids.append(str(doc_id))
                    elif isinstance(item, (str, int)):
                        # Format: ["123", "456", ...] or [123, 456, ...]
                        doc_ids.append(str(item))
                    elif isinstance(item, (list, tuple)) and len(item) > 0:
                        # Format: [[doc_id, score], ...] or [(doc_id, score), ...]
                        doc_ids.append(str(item[0]))
            elif isinstance(results, dict):
                # Format: {"results": [...]} or similar
                if 'results' in results:
                    doc_ids = [str(item) for item in results['results']]
                elif 'documents' in results:
                    doc_ids = [str(item) for item in results['documents']]
                    
            return doc_ids, response_time
        else:
            print(f"Error: Status {response.status_code} for query: {query}")
            return [], response_time
            
    except requests.exceptions.Timeout:
        print(f"Timeout for query: {query}")
        return [], TIMEOUT
    except Exception as e:
        print(f"Error querying '{query}': {e}")
        print(f"Response type: {type(response.json()) if 'response' in locals() else 'N/A'}")
        if 'response' in locals():
            try:
                sample = response.json()
                if isinstance(sample, list) and len(sample) > 0:
                    print(f"First item sample: {sample[0]}")
                    print(f"First item type: {type(sample[0])}")
            except:
                pass
        return [], 0.0


def calculate_precision_at_k(retrieved, relevant, k):
    """Calculate Precision@k"""
    if k == 0:
        return 0.0
    retrieved_at_k = retrieved[:k]
    relevant_at_k = [doc for doc in retrieved_at_k if doc in relevant]
    return len(relevant_at_k) / k


def calculate_f1_at_k(retrieved, relevant, k):
    """Calculate F1@k"""
    retrieved_at_k = set(retrieved[:k])
    relevant_set = set(relevant)
    
    tp = len(retrieved_at_k & relevant_set)
    precision = tp / k if k > 0 else 0
    recall = tp / len(relevant_set) if len(relevant_set) > 0 else 0
    
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def calculate_average_precision_at_k(retrieved, relevant, k):
    """Calculate Average Precision@k"""
    if not relevant:
        return 0.0
    
    retrieved_at_k = retrieved[:k]
    score = 0.0
    num_hits = 0.0
    
    for i, doc in enumerate(retrieved_at_k):
        if doc in relevant:
            num_hits += 1.0
            precision_at_i = num_hits / (i + 1.0)
            score += precision_at_i
    
    return score / min(len(relevant), k) if len(relevant) > 0 else 0.0


def harmonic_mean(a, b):
    """Calculate harmonic mean of two values"""
    if a + b == 0:
        return 0.0
    return 2 * (a * b) / (a + b)


def evaluate_engine(ground_truth):
    """
    Evaluate the live search engine on all queries
    Returns: dict with per-query and summary results
    """
    results = {
        'per_query': {},
        'summary': {},
        'timestamp': datetime.now().isoformat()
    }
    
    all_p5 = []
    all_p10 = []
    all_f1_30 = []
    all_ap10 = []
    all_hm = []
    all_times = []
    
    print("=" * 80)
    print(f"Evaluating Live Search Engine: {ENGINE_URL}")
    print("=" * 80)
    
    for i, (query, relevant_docs) in enumerate(ground_truth.items(), 1):
        print(f"\n[{i}/{len(ground_truth)}] Querying: {query[:60]}...")
        
        # Convert relevant docs to strings
        relevant_docs = [str(doc) for doc in relevant_docs]
        
        # Query the engine
        retrieved_docs, response_time = query_search_engine(query, top_n=100)
        
        # Calculate metrics
        p5 = calculate_precision_at_k(retrieved_docs, relevant_docs, 5)
        p10 = calculate_precision_at_k(retrieved_docs, relevant_docs, 10)
        f1_30 = calculate_f1_at_k(retrieved_docs, relevant_docs, 30)
        ap10 = calculate_average_precision_at_k(retrieved_docs, relevant_docs, 10)
        hm = harmonic_mean(p5, f1_30)
        
        # Store results
        results['per_query'][query] = {
            'precision_at_5': p5,
            'precision_at_10': p10,
            'f1_at_30': f1_30,
            'average_precision_at_10': ap10,
            'harmonic_mean_p5_f1_30': hm,
            'response_time': response_time,
            'relevant_found_at_10': len([d for d in retrieved_docs[:10] if d in relevant_docs]),
            'total_relevant': len(relevant_docs),
            'total_retrieved': len(retrieved_docs)
        }
        
        all_p5.append(p5)
        all_p10.append(p10)
        all_f1_30.append(f1_30)
        all_ap10.append(ap10)
        all_hm.append(hm)
        all_times.append(response_time)
        
        print(f"  P@5: {p5:.3f} | P@10: {p10:.3f} | F1@30: {f1_30:.3f} | HM: {hm:.3f} | Time: {response_time:.3f}s")
        
        # Small delay to avoid overwhelming the server
        time.sleep(0.5)
    
    # Calculate summary statistics
    results['summary'] = {
        'total_queries': len(ground_truth),
        'mean_precision_at_5': sum(all_p5) / len(all_p5),
        'mean_precision_at_10': sum(all_p10) / len(all_p10),
        'mean_f1_at_30': sum(all_f1_30) / len(all_f1_30),
        'mean_average_precision_at_10': sum(all_ap10) / len(all_ap10),
        'mean_harmonic_mean': sum(all_hm) / len(all_hm),
        'mean_response_time': sum(all_times) / len(all_times),
        'median_response_time': sorted(all_times)[len(all_times)//2],
        'max_response_time': max(all_times),
        'min_response_time': min(all_times)
    }
    
    return results


def plot_results(results, output_dir):
    """Generate visualization plots for the evaluation results"""
    
    # Extract data
    queries = list(results['per_query'].keys())
    p5_values = [results['per_query'][q]['precision_at_5'] for q in queries]
    p10_values = [results['per_query'][q]['precision_at_10'] for q in queries]
    f1_values = [results['per_query'][q]['f1_at_30'] for q in queries]
    hm_values = [results['per_query'][q]['harmonic_mean_p5_f1_30'] for q in queries]
    ap10_values = [results['per_query'][q]['average_precision_at_10'] for q in queries]
    times = [results['per_query'][q]['response_time'] for q in queries]
    
    # Create short query names for plotting
    query_names = [q[:40] + "..." if len(q) > 40 else q for q in queries]
    
    # Plot 1: Performance Metrics by Query
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # P@10 by query
    ax = axes[0, 0]
    bars = ax.barh(range(len(queries)), p10_values, color='#2E86AB')
    ax.set_yticks(range(len(queries)))
    ax.set_yticklabels(query_names, fontsize=8)
    ax.set_xlabel('Precision@10', fontweight='bold')
    ax.set_title('Precision@10 by Query', fontweight='bold', fontsize=14)
    ax.axvline(results['summary']['mean_precision_at_10'], color='red', 
              linestyle='--', linewidth=2, label=f'Mean: {results["summary"]["mean_precision_at_10"]:.3f}')
    ax.legend()
    ax.set_xlim(0, 1.0)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    # Harmonic Mean by query
    ax = axes[0, 1]
    bars = ax.barh(range(len(queries)), hm_values, color='#A23B72')
    ax.set_yticks(range(len(queries)))
    ax.set_yticklabels(query_names, fontsize=8)
    ax.set_xlabel('Harmonic Mean (P@5, F1@30)', fontweight='bold')
    ax.set_title('Harmonic Mean by Query', fontweight='bold', fontsize=14)
    ax.axvline(results['summary']['mean_harmonic_mean'], color='red', 
              linestyle='--', linewidth=2, label=f'Mean: {results["summary"]["mean_harmonic_mean"]:.3f}')
    ax.legend()
    ax.set_xlim(0, 1.0)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    # Response time by query
    ax = axes[1, 0]
    bars = ax.barh(range(len(queries)), times, color='#F18F01')
    ax.set_yticks(range(len(queries)))
    ax.set_yticklabels(query_names, fontsize=8)
    ax.set_xlabel('Response Time (seconds)', fontweight='bold')
    ax.set_title('Response Time by Query', fontweight='bold', fontsize=14)
    ax.axvline(results['summary']['mean_response_time'], color='red', 
              linestyle='--', linewidth=2, label=f'Mean: {results["summary"]["mean_response_time"]:.3f}s')
    ax.legend()
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    # Metrics comparison
    ax = axes[1, 1]
    metric_names = ['P@5', 'P@10', 'F1@30', 'AP@10', 'HM']
    metric_values = [
        results['summary']['mean_precision_at_5'],
        results['summary']['mean_precision_at_10'],
        results['summary']['mean_f1_at_30'],
        results['summary']['mean_average_precision_at_10'],
        results['summary']['mean_harmonic_mean']
    ]
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    bars = ax.bar(metric_names, metric_values, color=colors)
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Average Metric Scores', fontweight='bold', fontsize=14)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    output_file = output_dir / "live_engine_performance.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved performance plot to {output_file}")
    plt.close()
    
    # Plot 2: Performance vs Time Trade-off
    fig, ax = plt.subplots(figsize=(12, 8))
    
    scatter = ax.scatter(times, hm_values, c=p10_values, cmap='viridis', 
                        s=200, alpha=0.7, edgecolors='black', linewidth=1.5)
    
    # Add query labels for interesting points
    for i, (t, hm, p10, q) in enumerate(zip(times, hm_values, p10_values, queries)):
        if hm > 0.6 or hm < 0.3 or t > results['summary']['mean_response_time'] * 1.5:
            ax.annotate(q[:25] + "...", xy=(t, hm), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=7, alpha=0.7,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    ax.set_xlabel('Response Time (seconds)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Harmonic Mean (P@5, F1@30)', fontweight='bold', fontsize=12)
    ax.set_title('Performance vs Speed Trade-off (Live Engine)\nColor = P@10', 
                fontweight='bold', fontsize=14)
    
    # Add reference lines
    ax.axhline(results['summary']['mean_harmonic_mean'], color='red', 
              linestyle='--', alpha=0.5, label=f'Mean HM: {results["summary"]["mean_harmonic_mean"]:.3f}')
    ax.axvline(results['summary']['mean_response_time'], color='blue', 
              linestyle='--', alpha=0.5, label=f'Mean Time: {results["summary"]["mean_response_time"]:.3f}s')
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('P@10', fontweight='bold')
    
    plt.tight_layout()
    output_file = output_dir / "live_engine_tradeoff.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved trade-off plot to {output_file}")
    plt.close()
    
    # Plot 3: Metric Distributions
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    metrics_data = [
        (p5_values, 'P@5', '#2E86AB'),
        (p10_values, 'P@10', '#A23B72'),
        (f1_values, 'F1@30', '#F18F01'),
        (ap10_values, 'AP@10', '#C73E1D'),
        (hm_values, 'HM', '#6A994E'),
        (times, 'Response Time (s)', '#BC4B51')
    ]
    
    for idx, (data, label, color) in enumerate(metrics_data):
        ax = axes[idx // 3, idx % 3]
        ax.hist(data, bins=15, color=color, alpha=0.7, edgecolor='black')
        ax.axvline(sum(data)/len(data), color='red', linestyle='--', 
                  linewidth=2, label=f'Mean: {sum(data)/len(data):.3f}')
        ax.set_xlabel(label, fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title(f'{label} Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = output_dir / "live_engine_distributions.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved distributions plot to {output_file}")
    plt.close()


def save_results(results, output_dir):
    """Save evaluation results to JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"live_evaluation_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved detailed results to {output_file}")
    return output_file


def print_summary(results):
    """Print evaluation summary"""
    print("\n" + "=" * 80)
    print("LIVE ENGINE EVALUATION SUMMARY")
    print("=" * 80)
    
    summary = results['summary']
    print(f"\nTotal Queries Tested: {summary['total_queries']}")
    print(f"\n{'Metric':<30} {'Value':>10}")
    print("-" * 42)
    print(f"{'Mean Precision@5':<30} {summary['mean_precision_at_5']:>10.4f}")
    print(f"{'Mean Precision@10':<30} {summary['mean_precision_at_10']:>10.4f}")
    print(f"{'Mean F1@30':<30} {summary['mean_f1_at_30']:>10.4f}")
    print(f"{'Mean Average Precision@10':<30} {summary['mean_average_precision_at_10']:>10.4f}")
    print(f"{'Mean Harmonic Mean':<30} {summary['mean_harmonic_mean']:>10.4f}")
    print(f"\n{'Mean Response Time':<30} {summary['mean_response_time']:>9.3f}s")
    print(f"{'Median Response Time':<30} {summary['median_response_time']:>9.3f}s")
    print(f"{'Min Response Time':<30} {summary['min_response_time']:>9.3f}s")
    print(f"{'Max Response Time':<30} {summary['max_response_time']:>9.3f}s")
    
    # Find best and worst queries
    queries = results['per_query']
    best_p10 = max(queries.items(), key=lambda x: x[1]['precision_at_10'])
    worst_p10 = min(queries.items(), key=lambda x: x[1]['precision_at_10'])
    best_hm = max(queries.items(), key=lambda x: x[1]['harmonic_mean_p5_f1_30'])
    worst_hm = min(queries.items(), key=lambda x: x[1]['harmonic_mean_p5_f1_30'])
    
    print(f"\n{'Best P@10:':<30} {best_p10[1]['precision_at_10']:.3f} - {best_p10[0][:40]}")
    print(f"{'Worst P@10:':<30} {worst_p10[1]['precision_at_10']:.3f} - {worst_p10[0][:40]}")
    print(f"{'Best HM:':<30} {best_hm[1]['harmonic_mean_p5_f1_30']:.3f} - {best_hm[0][:40]}")
    print(f"{'Worst HM:':<30} {worst_hm[1]['harmonic_mean_p5_f1_30']:.3f} - {worst_hm[0][:40]}")
    
    print("=" * 80)


def main():
    """Main execution function"""
    print("Live Search Engine Evaluation Tool")
    print(f"Target Engine: {ENGINE_URL}\n")
    
    # Create output directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load ground truth
    print("Loading ground truth queries...")
    ground_truth = load_ground_truth()
    print(f"Loaded {len(ground_truth)} queries\n")
    
    # Evaluate engine
    results = evaluate_engine(ground_truth)
    
    # Print summary
    print_summary(results)
    
    # Save results
    save_results(results, RESULTS_DIR)
    
    # Generate plots
    print("\nGenerating visualization plots...")
    plot_results(results, RESULTS_DIR)
    
    print("\n✓ Evaluation complete!")
    print(f"All results saved to: {RESULTS_DIR.absolute()}")


if __name__ == "__main__":
    main()
