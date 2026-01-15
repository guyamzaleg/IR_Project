"""
Search Engine Performance Visualization
Generates plots for:
1. P@10 performance across major implementation versions
2. Average retrieval time across major implementation versions
3. Harmonic Mean vs Time trade-offs from grid search results
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Define base directory
BASE_DIR = Path(__file__).parent / "results"


def extract_precision_from_md(md_file):
    """Extract mean P@10 from precision_results.md file"""
    try:
        with open(md_file, 'r', encoding='utf-8') as f:
            for line in f:
                if "| Mean Precision@10 |" in line:
                    # Format: | Mean Precision@10 | 0.7033 |
                    parts = line.split('|')
                    if len(parts) >= 3:
                        return float(parts[2].strip())
    except Exception as e:
        print(f"Error reading {md_file}: {e}")
    return None


def extract_from_evaluation_json(json_file):
    """Extract metrics from evaluation_results.json file"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Calculate mean P@10 from per_query data
            p10_values = []
            for query_data in data.get('per_query', {}).values():
                if 'precision_at_10' in query_data:
                    p10_values.append(query_data['precision_at_10'])
            
            mean_p10 = sum(p10_values) / len(p10_values) if p10_values else None
            return mean_p10
    except Exception as e:
        print(f"Error reading {json_file}: {e}")
    return None


def get_version_data():
    """
    Scan results directory and extract P@10 for each timestamped version.
    Returns a list of (timestamp, mean_p10, version_name) tuples.
    """
    versions = []
    
    # Scan all timestamped directories
    for item in sorted(BASE_DIR.iterdir()):
        if item.is_dir() and item.name[0].isdigit():  # Timestamped directories
            timestamp_str = item.name
            
            # Try to parse timestamp
            try:
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            except:
                continue
            
            # Check for evaluation_results.json first, then precision_results.md
            json_file = item / "evaluation_results.json"
            md_file = item / "precision_results.md"
            
            mean_p10 = None
            if json_file.exists():
                mean_p10 = extract_from_evaluation_json(json_file)
            elif md_file.exists():
                mean_p10 = extract_precision_from_md(md_file)
            
            if mean_p10 is not None:
                # Create a readable version name
                version_name = timestamp.strftime("%m/%d %H:%M")
                versions.append((timestamp, mean_p10, version_name))
    
    return sorted(versions, key=lambda x: x[0])


def plot_p10_evolution(versions, output_dir):
    """
    Plot 1: P@10 performance across major implementation versions
    """
    if not versions:
        print("No version data found")
        return
    
    timestamps, p10_values, version_names = zip(*versions)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot line
    ax.plot(range(len(versions)), p10_values, marker='o', linewidth=2, 
            markersize=8, color='#2E86AB', label='Mean P@10')
    
    # Highlight major milestones
    if len(versions) > 0:
        # First version
        ax.scatter(0, p10_values[0], color='red', s=200, zorder=5, 
                  label=f'Initial: {p10_values[0]:.3f}', marker='*')
        # Final version
        ax.scatter(len(versions)-1, p10_values[-1], color='green', s=200, 
                  zorder=5, label=f'Final: {p10_values[-1]:.3f}', marker='*')
    
    # Add value labels
    for i, (ts, p10, name) in enumerate(versions):
        ax.annotate(f'{p10:.3f}', 
                   xy=(i, p10), 
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center',
                   fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    # Styling
    ax.set_xlabel('Version (chronological)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Precision@10', fontsize=12, fontweight='bold')
    ax.set_title('Search Engine Performance Evolution - P@10 Across Versions', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(range(len(versions)))
    ax.set_xticklabels(version_names, rotation=45, ha='right')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    # Add improvement percentage
    if len(versions) > 1:
        improvement = ((p10_values[-1] - p10_values[0]) / p10_values[0]) * 100
        ax.text(0.02, 0.98, f'Total Improvement: {improvement:+.1f}%', 
               transform=ax.transAxes,
               fontsize=11,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_file = output_dir / "p10_evolution.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved P@10 evolution plot to {output_file}")
    plt.close()


def plot_time_evolution(output_dir):
    """
    Plot 2: Average retrieval time across major implementation versions
    Note: Time data is extracted from grid search results
    """
    # Collect time data from grid search CSVs
    time_data = []
    
    search_dirs = {
        'BM25': BASE_DIR / 'BM25',
        'Query Boost': BASE_DIR / 'query_boost',
        'Ranking': BASE_DIR / 'ranking',
        'Stemming': BASE_DIR / 'stemming',
        'Weights': BASE_DIR / 'weights'
    }
    
    for version_name, search_dir in search_dirs.items():
        if search_dir.exists():
            for csv_file in search_dir.glob('grid_search_*.csv'):
                try:
                    df = pd.read_csv(csv_file)
                    if 'avg_time' in df.columns:
                        # Get timestamp from filename
                        timestamp_str = csv_file.stem.split('_')[-2] + '_' + csv_file.stem.split('_')[-1]
                        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                        
                        # Get average of all avg_time values in this grid search
                        mean_time = df['avg_time'].mean()
                        min_time = df['avg_time'].min()
                        max_time = df['avg_time'].max()
                        
                        time_data.append({
                            'timestamp': timestamp,
                            'version': version_name,
                            'mean_time': mean_time,
                            'min_time': min_time,
                            'max_time': max_time
                        })
                except Exception as e:
                    print(f"Error reading {csv_file}: {e}")
    
    if not time_data:
        print("No time data found in grid search results")
        return
    
    # Sort by timestamp
    time_data = sorted(time_data, key=lambda x: x['timestamp'])
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    timestamps = [d['timestamp'] for d in time_data]
    mean_times = [d['mean_time'] for d in time_data]
    min_times = [d['min_time'] for d in time_data]
    max_times = [d['max_time'] for d in time_data]
    versions = [d['version'] for d in time_data]
    
    x = range(len(time_data))
    
    # Plot mean time with error bars
    ax.plot(x, mean_times, marker='o', linewidth=2, markersize=8, 
           color='#A23B72', label='Mean Retrieval Time')
    
    # Add shaded region for min-max range
    ax.fill_between(x, min_times, max_times, alpha=0.2, color='#A23B72',
                    label='Min-Max Range')
    
    # Add value labels
    for i, mean_time in enumerate(mean_times):
        ax.annotate(f'{mean_time:.3f}s', 
                   xy=(i, mean_time), 
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center',
                   fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.5))
    
    # Styling
    ax.set_xlabel('Optimization Phase', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Retrieval Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Search Engine Retrieval Time Evolution Across Optimization Phases', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(versions, rotation=45, ha='right')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / "time_evolution.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved time evolution plot to {output_file}")
    plt.close()


def plot_hm_vs_time_grid_search(output_dir):
    """
    Plot 3: Harmonic Mean vs Time trade-off from grid search results
    Shows the relationship between performance (HM) and efficiency (time)
    """
    # Collect all grid search data
    all_data = []
    
    search_dirs = {
        'BM25': BASE_DIR / 'BM25',
        'Query Boost': BASE_DIR / 'query_boost',
        'Ranking': BASE_DIR / 'ranking',
        'Stemming': BASE_DIR / 'stemming',
        'Weights': BASE_DIR / 'weights'
    }
    
    for category, search_dir in search_dirs.items():
        if search_dir.exists():
            for csv_file in search_dir.glob('grid_search_*.csv'):
                try:
                    df = pd.read_csv(csv_file)
                    if 'avg_HM_P5_F30' in df.columns and 'avg_time' in df.columns:
                        df['category'] = category
                        df['avg_P@10'] = df.get('avg_P@10', 0)  # Add P@10 if available
                        all_data.append(df)
                except Exception as e:
                    print(f"Error reading {csv_file}: {e}")
    
    if not all_data:
        print("No grid search data found")
        return
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Create scatter plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: HM vs Time
    for category in combined_df['category'].unique():
        cat_data = combined_df[combined_df['category'] == category]
        ax1.scatter(cat_data['avg_time'], cat_data['avg_HM_P5_F30'], 
                   alpha=0.6, s=50, label=category)
    
    ax1.set_xlabel('Average Retrieval Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Harmonic Mean (P@5, F1@30)', fontsize=12, fontweight='bold')
    ax1.set_title('Performance vs Efficiency Trade-off\n(Harmonic Mean vs Time)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add Pareto frontier
    pareto_data = combined_df.sort_values('avg_time')
    pareto_points = []
    max_hm = -1
    for _, row in pareto_data.iterrows():
        if row['avg_HM_P5_F30'] > max_hm:
            max_hm = row['avg_HM_P5_F30']
            pareto_points.append((row['avg_time'], row['avg_HM_P5_F30']))
    
    if pareto_points:
        pareto_x, pareto_y = zip(*pareto_points)
        ax1.plot(pareto_x, pareto_y, 'r--', linewidth=2, alpha=0.5, 
                label='Pareto Frontier')
    
    # Plot 2: P@10 vs Time
    if 'avg_P@10' in combined_df.columns and combined_df['avg_P@10'].sum() > 0:
        for category in combined_df['category'].unique():
            cat_data = combined_df[combined_df['category'] == category]
            ax2.scatter(cat_data['avg_time'], cat_data['avg_P@10'], 
                       alpha=0.6, s=50, label=category)
        
        ax2.set_xlabel('Average Retrieval Time (seconds)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Mean Precision@10', fontsize=12, fontweight='bold')
        ax2.set_title('Performance vs Efficiency Trade-off\n(P@10 vs Time)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Add Pareto frontier for P@10
        pareto_data_p10 = combined_df.sort_values('avg_time')
        pareto_points_p10 = []
        max_p10 = -1
        for _, row in pareto_data_p10.iterrows():
            if row['avg_P@10'] > max_p10:
                max_p10 = row['avg_P@10']
                pareto_points_p10.append((row['avg_time'], row['avg_P@10']))
        
        if pareto_points_p10:
            pareto_x, pareto_y = zip(*pareto_points_p10)
            ax2.plot(pareto_x, pareto_y, 'r--', linewidth=2, alpha=0.5, 
                    label='Pareto Frontier')
    else:
        ax2.text(0.5, 0.5, 'P@10 data not available', 
                transform=ax2.transAxes, ha='center', va='center',
                fontsize=14)
    
    plt.tight_layout()
    output_file = output_dir / "hm_time_tradeoff.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved HM vs Time trade-off plot to {output_file}")
    plt.close()


def plot_grid_search_heatmaps(output_dir):
    """
    Bonus Plot 4: Heatmaps of grid search results for each optimization phase
    """
    search_dirs = {
        'BM25': BASE_DIR / 'BM25',
        'Ranking': BASE_DIR / 'ranking',
    }
    
    for category, search_dir in search_dirs.items():
        if search_dir.exists():
            for csv_file in search_dir.glob('grid_search_*.csv'):
                try:
                    df = pd.read_csv(csv_file)
                    
                    if len(df) < 5:  # Skip if too few configurations
                        continue
                    
                    # Create figure with subplots for different metrics
                    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                    fig.suptitle(f'{category} Grid Search Results\n{csv_file.stem}', 
                               fontsize=16, fontweight='bold')
                    
                    # Sort by HM
                    df_sorted = df.sort_values('avg_HM_P5_F30', ascending=False)
                    
                    # Plot 1: Top configs by HM
                    top_n = min(20, len(df_sorted))
                    top_configs = df_sorted.head(top_n)
                    
                    ax = axes[0, 0]
                    y_pos = range(top_n)
                    ax.barh(y_pos, top_configs['avg_HM_P5_F30'], color='#2E86AB')
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(top_configs['config_name'], fontsize=8)
                    ax.set_xlabel('Harmonic Mean (P@5, F1@30)', fontweight='bold')
                    ax.set_title(f'Top {top_n} Configurations by HM', fontweight='bold')
                    ax.invert_yaxis()
                    ax.grid(True, alpha=0.3, axis='x')
                    
                    # Plot 2: Time vs HM scatter
                    ax = axes[0, 1]
                    scatter = ax.scatter(df['avg_time'], df['avg_HM_P5_F30'], 
                                       c=df.get('avg_P@10', df['avg_HM_P5_F30']), 
                                       cmap='viridis', s=100, alpha=0.6)
                    ax.set_xlabel('Average Time (seconds)', fontweight='bold')
                    ax.set_ylabel('Harmonic Mean', fontweight='bold')
                    ax.set_title('Time vs Performance Trade-off', fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    plt.colorbar(scatter, ax=ax, label='P@10' if 'avg_P@10' in df.columns else 'HM')
                    
                    # Plot 3: Metric comparison
                    ax = axes[1, 0]
                    metrics = ['avg_P@5', 'avg_P@10', 'avg_F1@30', 'avg_HM_P5_F30']
                    available_metrics = [m for m in metrics if m in df.columns]
                    
                    if available_metrics:
                        metric_means = [df[m].mean() for m in available_metrics]
                        bars = ax.bar(range(len(available_metrics)), metric_means, 
                                     color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
                        ax.set_xticks(range(len(available_metrics)))
                        ax.set_xticklabels([m.replace('avg_', '') for m in available_metrics], 
                                          rotation=45, ha='right')
                        ax.set_ylabel('Average Value', fontweight='bold')
                        ax.set_title('Average Metric Values Across All Configs', fontweight='bold')
                        ax.grid(True, alpha=0.3, axis='y')
                        
                        # Add value labels
                        for i, (bar, val) in enumerate(zip(bars, metric_means)):
                            ax.text(i, val, f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
                    
                    # Plot 4: Time distribution
                    ax = axes[1, 1]
                    ax.hist(df['avg_time'], bins=20, color='#F18F01', alpha=0.7, edgecolor='black')
                    ax.axvline(df['avg_time'].mean(), color='red', linestyle='--', 
                              linewidth=2, label=f'Mean: {df["avg_time"].mean():.3f}s')
                    ax.axvline(df['avg_time'].median(), color='green', linestyle='--', 
                              linewidth=2, label=f'Median: {df["avg_time"].median():.3f}s')
                    ax.set_xlabel('Average Time (seconds)', fontweight='bold')
                    ax.set_ylabel('Frequency', fontweight='bold')
                    ax.set_title('Retrieval Time Distribution', fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    plt.tight_layout()
                    output_file = output_dir / f"{category}_{csv_file.stem}_analysis.png"
                    plt.savefig(output_file, dpi=300, bbox_inches='tight')
                    print(f"Saved grid search analysis to {output_file}")
                    plt.close()
                    
                except Exception as e:
                    print(f"Error creating heatmap for {csv_file}: {e}")


def main():
    """Main function to generate all plots"""
    print("=" * 80)
    print("Search Engine Performance Visualization")
    print("=" * 80)
    
    # Create output directory
    output_dir = BASE_DIR / "plots"
    output_dir.mkdir(exist_ok=True)
    
    # Get version data
    print("\n1. Collecting version data...")
    versions = get_version_data()
    print(f"   Found {len(versions)} versions with P@10 data")
    
    # Generate plots
    print("\n2. Generating P@10 evolution plot...")
    plot_p10_evolution(versions, output_dir)
    
    print("\n3. Generating time evolution plot...")
    plot_time_evolution(output_dir)
    
    print("\n4. Generating HM vs Time trade-off plot...")
    plot_hm_vs_time_grid_search(output_dir)
    
    print("\n5. Generating detailed grid search analysis plots...")
    plot_grid_search_heatmaps(output_dir)
    
    print("\n" + "=" * 80)
    print(f"All plots saved to: {output_dir.absolute()}")
    print("=" * 80)
    
    # Print summary statistics
    if versions:
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        timestamps, p10_values, version_names = zip(*versions)
        print(f"\nInitial P@10: {p10_values[0]:.4f} ({version_names[0]})")
        print(f"Final P@10:   {p10_values[-1]:.4f} ({version_names[-1]})")
        improvement = ((p10_values[-1] - p10_values[0]) / p10_values[0]) * 100
        print(f"Improvement:  {improvement:+.2f}%")
        print(f"\nBest P@10:    {max(p10_values):.4f}")
        print(f"Worst P@10:   {min(p10_values):.4f}")
        print(f"Mean P@10:    {sum(p10_values)/len(p10_values):.4f}")
        print("=" * 80)


if __name__ == "__main__":
    main()
