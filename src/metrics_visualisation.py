import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

base_dir_path=os.getcwd()
df = pd.read_excel(os.path.join(base_dir_path,"metrics/metrics.xlsx"))

# Create a configuration label for better readability
df['config'] = df['index_type'] + ' + ' + df['embeddings_type'].str.replace('all_product_fields_768dim', '768dim').str.replace('product_fields_exclude_desc_384dim', '384dim')

# Set up the plotting style
plt.style.use('default')
sns.set_palette("husl")

# Function to create comprehensive charts for each dataset
def create_dataset_charts(dataset_name):
    data_subset = df[df['dataset'] == dataset_name]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{dataset_name.title()} Dataset Performance Metrics', fontsize=16, fontweight='bold')
    
    # HITS@k metrics
    hits_metrics = ['HITS@k_1', 'HITS@k_5', 'HITS@k_10']
    for i, metric in enumerate(hits_metrics):
        ax = axes[0, i]
        bars = ax.bar(range(len(data_subset)), data_subset[metric], 
                     color=sns.color_palette("husl", len(data_subset)))
        ax.set_title(f'{metric}', fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_xticks(range(len(data_subset)))
        ax.set_xticklabels(data_subset['config'], rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # MRR metrics
    mrr_metrics = ['MRR_1', 'MRR_5', 'MRR_10']
    for i, metric in enumerate(mrr_metrics):
        ax = axes[1, i]
        bars = ax.bar(range(len(data_subset)), data_subset[metric], 
                     color=sns.color_palette("husl", len(data_subset)))
        ax.set_title(f'{metric}', fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_xticks(range(len(data_subset)))
        ax.set_xticklabels(data_subset['config'], rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig

# Function to create comparison line charts
def create_comparison_charts():
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Balanced vs Imbalanced Dataset Comparison', fontsize=16, fontweight='bold')
    
    metrics = ['HITS@k_1', 'HITS@k_5', 'HITS@k_10', 'MRR_1', 'MRR_5', 'MRR_10']
    
    for i, metric in enumerate(metrics):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        balanced_data = df[df['dataset'] == 'balanced']
        imbalanced_data = df[df['dataset'] == 'imbalanced']
        
        x_pos = range(len(balanced_data))
        
        ax.plot(x_pos, balanced_data[metric], 'o-', linewidth=2, markersize=8, 
               label='Balanced', color='#2E8B57')
        ax.plot(x_pos, imbalanced_data[metric], 's-', linewidth=2, markersize=8, 
               label='Imbalanced', color='#DC143C')
        
        ax.set_title(f'{metric}', fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(balanced_data['config'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# Function to create heatmap
def create_heatmap():
    # Pivot the data for heatmap
    pivot_data = df.pivot_table(index='config', columns='dataset', values=['HITS@k_1', 'HITS@k_5', 'HITS@k_10'])
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle('Performance Heatmap: Balanced vs Imbalanced', fontsize=16, fontweight='bold')
    
    metrics = ['HITS@k_1', 'HITS@k_5', 'HITS@k_10']
    
    for i, metric in enumerate(metrics):
        sns.heatmap(pivot_data[metric], annot=True, cmap='RdYlGn', 
                   cbar_kws={'label': 'Score'}, ax=axes[i])
        axes[i].set_title(f'{metric}', fontweight='bold')
        axes[i].set_xlabel('Dataset Type')
        axes[i].set_ylabel('Configuration')
    
    plt.tight_layout()
    return fig

# Generate all charts
if __name__ == "__main__":
    # Create individual dataset charts
    balanced_fig = create_dataset_charts('balanced')
    balanced_fig.savefig(os.path.join(base_dir_path,'metrics/balanced_dataset_performance.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    imbalanced_fig = create_dataset_charts('imbalanced')
    imbalanced_fig.savefig(os.path.join(base_dir_path,'metrics/imbalanced_dataset_performance.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create comparison charts
    comparison_fig = create_comparison_charts()
    comparison_fig.savefig(os.path.join(base_dir_path,'metrics/dataset_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create heatmap
    heatmap_fig = create_heatmap()
    heatmap_fig.savefig(os.path.join(base_dir_path,'metrics/performance_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n=== PERFORMANCE SUMMARY ===")
    print("\nBalanced Dataset - Best Performers:")
    balanced_subset = df[df['dataset'] == 'balanced']
    for metric in ['HITS@k_1', 'HITS@k_5', 'HITS@k_10']:
        best_idx = balanced_subset[metric].idxmax()
        best_config = balanced_subset.loc[best_idx, 'config']
        best_score = balanced_subset.loc[best_idx, metric]
        print(f"{metric}: {best_config} ({best_score:.3f})")
    
    print("\nImbalanced Dataset - Best Performers:")
    imbalanced_subset = df[df['dataset'] == 'imbalanced']
    for metric in ['HITS@k_1', 'HITS@k_5', 'HITS@k_10']:
        best_idx = imbalanced_subset[metric].idxmax()
        best_config = imbalanced_subset.loc[best_idx, 'config']
        best_score = imbalanced_subset.loc[best_idx, metric]
        print(f"{metric}: {best_config} ({best_score:.3f})")