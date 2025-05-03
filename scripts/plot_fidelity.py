#!/usr/bin/env python
"""
Generate plots from fidelity evaluation results.

This script reads previously generated fidelity evaluation results from CSV files
and creates visualizations for each category of metrics:
1. Statistical tests (KS test, Chi-squared)
2. Correlation metrics (Pearson, Spearman, correlation matrix distance)
3. Distribution metrics (KL divergence)
4. Likelihood-based metrics (BN Likelihood, GM Log Likelihood)
5. Detection-based metrics (Logistic Detection, SVC Detection)
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def setup_argparse():
    """Set up command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Generate plots from fidelity evaluation results"
    )
    
    # Required arguments
    parser.add_argument(
        "--results_path",
        type=str,
        required=True,
        help="Path to the fidelity_results.csv file"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save generated plots (defaults to same directory as results file)"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["statistical", "correlation", "distribution", "likelihood", "detection"],
        choices=["statistical", "correlation", "distribution", "likelihood", "detection", "all"],
        help="Categories of fidelity metrics to plot"
    )
    parser.add_argument(
        "--plot_formats",
        type=str,
        nargs="+",
        default=["png"],
        choices=["png", "pdf", "svg"],
        help="Output formats for the plots"
    )
    parser.add_argument(
        "--style",
        type=str,
        default="whitegrid",
        choices=["darkgrid", "whitegrid", "dark", "white", "ticks"],
        help="Seaborn plot style"
    )
    
    return parser


def get_custom_color_palette(num_colors):
    """
    Get a custom color palette with visually appealing colors
    
    Args:
        num_colors (int): Number of colors needed
        
    Returns:
        list: List of colors in RGB tuples
    """
    # Define a visually appealing color palette
    custom_colors = [
        # Teal
        (0.00, 0.63, 0.64),
        # Crimson
        (0.75, 0.12, 0.24),
        # Dark orange
        (0.85, 0.37, 0.01),
        # Forest green
        (0.13, 0.55, 0.13),
        # Ochre
        (0.80, 0.52, 0.25),
        # Burgundy
        (0.55, 0.00, 0.26),
        # Navy
        (0.00, 0.21, 0.41),
        # Dark red
        (0.64, 0.08, 0.18),
        # Dark green
        (0.00, 0.39, 0.25),
        # Brown
        (0.55, 0.27, 0.07),
        # Dark slate
        (0.28, 0.24, 0.55)
    ]
    
    # If we need more colors than in our custom palette, extend it with a seaborn color palette
    if num_colors > len(custom_colors):
        additional_colors = sns.color_palette("tab20", num_colors)
        result_colors = custom_colors + additional_colors
        return result_colors[:num_colors]
    else:
        return custom_colors[:num_colors]


def get_metric_display_info():
    """
    Get display information for metrics
    
    Returns:
        dict: Dictionary with metric display names, categories, and goal information
    """
    return {
        # Statistical metrics
        "KSComplement": {
            "display_name": "Kolmogorov-Smirnov Complement",
            "category": "statistical",
            "goal": "maximize",
            "description": "1 - KS statistic; measures similarity of distributions (higher is better)",
            "min_val": 0.0,
            "max_val": 1.0,
            "title": "Distribution Similarity via KS Test (higher is better)"
        },
        "CSTest": {
            "display_name": "Chi-Squared Test",
            "category": "statistical",
            "goal": "maximize",
            "description": "Chi-squared test p-value; higher values indicate more similar distributions",
            "min_val": 0.0,
            "max_val": 1.0,
            "title": "Distribution Similarity via Chi-Squared Test (higher is better)"
        },
        
        # Correlation metrics
        "PearsonCorrelation": {
            "display_name": "Pearson Correlation Similarity",
            "category": "correlation",
            "goal": "maximize",
            "description": "How well linear correlations are preserved (higher is better)",
            "min_val": 0.0,
            "max_val": 1.0,
            "title": "Linear Correlation Preservation (higher is better)"
        },
        "SpearmanCorrelation": {
            "display_name": "Spearman Rank Correlation",
            "category": "correlation",
            "goal": "maximize",
            "description": "How well monotonic relationships are preserved (higher is better)",
            "min_val": 0.0,
            "max_val": 1.0,
            "title": "Monotonic Relationship Preservation (higher is better)"
        },
        "KendallCorrelation": {
            "display_name": "Kendall Rank Correlation",
            "category": "correlation",
            "goal": "maximize",
            "description": "How well concordance between pairs is preserved (higher is better)",
            "min_val": 0.0,
            "max_val": 1.0,
            "title": "Pair Concordance Preservation (higher is better)"
        },
        "CorrelationMatrixDistance": {
            "display_name": "Correlation Matrix Similarity",
            "category": "correlation",
            "goal": "maximize",
            "description": "Overall similarity between correlation matrices (higher is better)",
            "min_val": 0.0,
            "max_val": 1.0,
            "title": "Correlation Structure Similarity (higher is better)"
        },
        "MixedTypeCorrelation": {
            "display_name": "Mixed-Type Correlation",
            "category": "correlation",
            "goal": "maximize",
            "description": "Preservation of correlations between mixed data types (higher is better)",
            "min_val": 0.0,
            "max_val": 1.0,
            "title": "Mixed-Type Correlation Preservation (higher is better)"
        },
        
        # Distribution metrics
        "ContinuousKLDivergence": {
            "display_name": "Continuous KL Divergence",
            "category": "distribution",
            "goal": "maximize",
            "description": "Kullback-Leibler divergence for continuous data (higher is better)",
            "min_val": 0.0,
            "max_val": 1.0,
            "title": "Continuous Distribution Similarity via KL Divergence (higher is better)"
        },
        "DiscreteKLDivergence": {
            "display_name": "Discrete KL Divergence",
            "category": "distribution",
            "goal": "maximize",
            "description": "Kullback-Leibler divergence for discrete data (higher is better)",
            "min_val": 0.0,
            "max_val": 1.0,
            "title": "Discrete Distribution Similarity via KL Divergence (higher is better)"
        },
        
        # Likelihood metrics
        "BNLikelihood": {
            "display_name": "Bayesian Network Likelihood",
            "category": "likelihood",
            "goal": "maximize",
            "description": "Likelihood of synthetic data given a BN fit on real data (higher is better)",
            "min_val": 0.0,
            "max_val": 1.0,
            "title": "Bayesian Network Data Likelihood Score (higher is better)"
        },
        "BNLogLikelihood": {
            "display_name": "Bayesian Network Log Likelihood",
            "category": "likelihood",
            "goal": "maximize",
            "description": "Log-likelihood of synthetic data given a BN fit on real data (higher is better)",
            "min_val": -np.inf,
            "max_val": 0.0,
            "normalize": True,
            "title": "Bayesian Network Log Likelihood Score (higher is better)"
        },
        "GMLogLikelihood": {
            "display_name": "Gaussian Mixture Log Likelihood",
            "category": "likelihood",
            "goal": "maximize",
            "description": "Log-likelihood of synthetic data given a GM fit on real data (higher is better)",
            "min_val": -np.inf,
            "max_val": np.inf,
            "normalize": True,
            "title": "Gaussian Mixture Model Log Likelihood Score (higher is better)"
        },
        
        # Detection metrics
        "LogisticDetection": {
            "display_name": "Logistic Regression Detection",
            "category": "detection",
            "goal": "maximize",
            "description": "1 - AUC for logistic classifier; higher means less distinguishable from real data",
            "min_val": 0.0,
            "max_val": 1.0,
            "title": "Synthetic Data Concealment from Logistic Regression (higher is better)"
        },
        "SVCDetection": {
            "display_name": "SVC Detection",
            "category": "detection",
            "goal": "maximize",
            "description": "1 - AUC for SVM classifier; higher means less distinguishable from real data",
            "min_val": 0.0,
            "max_val": 1.0,
            "title": "Synthetic Data Concealment from SVM (higher is better)"
        }
    }


def normalize_metric_value(value, metric_info):
    """
    Normalize metric value to [0, 1] range
    
    Args:
        value (float): Metric value
        metric_info (dict): Metric information
        
    Returns:
        float: Normalized value
    """
    if np.isnan(value):
        return np.nan
    
    if not metric_info.get("normalize", False):
        return value
    
    min_val = metric_info.get("min_val", 0.0)
    max_val = metric_info.get("max_val", 1.0)
    
    # Handle infinite bounds
    if min_val == -np.inf and max_val == np.inf:
        # Normalize using sigmoid
        return 1 / (1 + np.exp(-value))
    elif min_val == -np.inf:
        # Normalize using exponential
        return 1.0 - np.exp(value - max_val)
    elif max_val == np.inf:
        # Normalize using exponential
        return np.exp(value - min_val)
    else:
        # Linear normalization
        return (value - min_val) / (max_val - min_val)


def plot_metric_bars(results_df, metrics, output_dir, plot_formats, metric_info):
    """
    Create bar charts for each metric
    
    Args:
        results_df (pd.DataFrame): Results dataframe
        metrics (list): List of metrics to plot
        output_dir (str): Output directory
        plot_formats (list): List of output formats
        metric_info (dict): Metric information dictionary
    """
    datasets = results_df['Dataset'].values
    colors = get_custom_color_palette(len(datasets))
    
    # Creating a mapping of dataset to color
    color_map = dict(zip(datasets, colors))
    
    for metric in metrics:
        if metric not in results_df.columns:
            continue
            
        # Get metric display information
        display_info = metric_info.get(metric, {
            "display_name": metric,
            "category": "unknown",
            "goal": "unknown",
            "description": "",
            "min_val": 0.0,
            "max_val": 1.0,
            "title": f"{metric} (unknown)"
        })
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create bar chart
        bars = plt.bar(
            results_df['Dataset'], 
            results_df[metric], 
            color=[color_map[dataset] for dataset in datasets]
        )
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                # Format the value based on its magnitude
                if abs(height) < 0.01 or abs(height) >= 1000:
                    value_text = f'{height:.2e}'
                elif abs(height) < 1:
                    value_text = f'{height:.3f}'
                else:
                    value_text = f'{height:.2f}'

                # NEW: if this is a negative‐value bar for log‐likelihood metrics,
                # place the label just below the bar; otherwise above
                if metric in ["BNLogLikelihood", "GMLogLikelihood"] or height < 0:
                    y = height - 0.01
                    va = 'top'
                else:
                    y = max(height + 0.01, 0.01)
                    va = 'bottom'

                plt.text(
                    bar.get_x() + bar.get_width() / 2.,
                    y,
                    value_text,
                    ha='center',
                    va=va,
                    fontsize=10
                )
        
        # Set y-axis limits based on metric type
        min_val = display_info.get("min_val", 0.0)
        max_val = display_info.get("max_val", 1.0)
        
        if min_val != -np.inf and max_val != np.inf:
            # Add some padding to the limits
            y_min = min(min_val, results_df[metric].min() * 0.9)
            y_max = max(max_val, results_df[metric].max() * 1.1)
            plt.ylim(y_min, y_max)
        
        # Set labels and title
        plt.xlabel('Synthetic Data Model', fontsize=12)
        plt.ylabel(f'{display_info["display_name"]}', fontsize=12)
        
        # Use the new title field instead of constructing from display_name and goal
        plt.title(f'{display_info["title"]}', fontsize=14)
        
        # Add grid for readability
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot in each format
        for fmt in plot_formats:
            filename = os.path.join(output_dir, f"{metric}.{fmt}")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            
        plt.close()


def plot_correlation_matrix_comparison(results_df, output_dir, plot_formats):
    """
    Create a visualization comparing correlation matrices if CorrelationMatrixDistance is present
    
    Args:
        results_df (pd.DataFrame): Results dataframe
        output_dir (str): Output directory
        plot_formats (list): List of output formats
    """
    # Check if CorrelationMatrixDistance is present
    if 'CorrelationMatrixDistance' not in results_df.columns:
        return
    
    # Get best model based on CorrelationMatrixDistance
    best_model = results_df.loc[results_df['CorrelationMatrixDistance'].idxmax(), 'Dataset']
    
    # Create a bar chart highlighting the best model
    plt.figure(figsize=(10, 6))
    
    # Get datasets and scores
    datasets = results_df['Dataset'].values
    scores = results_df['CorrelationMatrixDistance'].values
    
    # Create custom color list where best model is highlighted
    colors = ['lightgray'] * len(datasets)
    best_idx = np.where(datasets == best_model)[0][0]
    colors[best_idx] = '#1f77b4'  # Highlight color
    
    # Create bar chart
    bars = plt.bar(datasets, scores, color=colors)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.01,
            f'{height:.3f}',
            ha='center', 
            va='bottom',
            fontsize=10
        )
    
    # Set labels and title
    plt.xlabel('Synthetic Data Model', fontsize=12)
    plt.ylabel('Correlation Matrix Similarity Score', fontsize=12)
    plt.title(f'Correlation Matrix Similarity (Best: {best_model})', fontsize=14)
    
    # Add grid for readability
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    
    # Save the plot in each format
    for fmt in plot_formats:
        filename = os.path.join(output_dir, f"correlation_matrix_comparison.{fmt}")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
    plt.close()


def plot_normalized_radar(results_df, metrics_by_category, output_dir, plot_formats, metric_info):
    """
    Create radar charts for each category of metrics
    
    Args:
        results_df (pd.DataFrame): Results dataframe
        metrics_by_category (dict): Dictionary of metrics grouped by category
        output_dir (str): Output directory
        plot_formats (list): List of output formats
        metric_info (dict): Metric information dictionary
    """
    datasets = results_df['Dataset'].values
    colors = get_custom_color_palette(len(datasets))
    
    for category, metrics in metrics_by_category.items():
        # Filter metrics that exist in the results
        available_metrics = [m for m in metrics if m in results_df.columns]
        
        if not available_metrics:
            continue
            
        # Normalize metrics to [0, 1] range
        normalized_df = results_df.copy()
        for metric in available_metrics:
            normalized_df[metric] = normalized_df[metric].apply(
                lambda x: normalize_metric_value(x, metric_info.get(metric, {}))
            )
        
        # Count number of metrics
        n_metrics = len(available_metrics)
        
        # If not enough metrics, skip radar chart
        if n_metrics < 3:
            continue
        
        # Create radar chart
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Set angle for each metric
        angles = np.linspace(0, 2*np.pi, n_metrics, endpoint=False).tolist()
        # Close the polygon
        angles += angles[:1]
        
        # Add display names on the radar chart
        display_names = [metric_info.get(m, {}).get("display_name", m) for m in available_metrics]
        display_names += display_names[:1]  # Close the polygon
        
        # Plot each dataset
        for i, dataset in enumerate(datasets):
            # Get values for this dataset
            values = normalized_df.loc[normalized_df['Dataset'] == dataset, available_metrics].values.flatten().tolist()
            # Close the polygon
            values += values[:1]
            
            # Plot the dataset line
            ax.plot(angles, values, color=colors[i], linewidth=2, label=dataset)
            # Fill area
            ax.fill(angles, values, color=colors[i], alpha=0.1)
        
        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Set labels
        plt.xticks(angles[:-1], display_names[:-1], fontsize=10)
        
        # Draw y-axis labels (0.2, 0.4, 0.6, 0.8, 1.0)
        ax.set_rlabel_position(0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
        
        # Set y-axis limits
        plt.ylim(0, 1)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Add title
        plt.title(f"{category.title()} Metrics Comparison", fontsize=14)
        
        # Save the plot in each format
        for fmt in plot_formats:
            filename = os.path.join(output_dir, f"radar_{category}.{fmt}")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            
        plt.close()


def plot_heatmap(results_df, categories, output_dir, plot_formats, metric_info):
    """
    Create a heatmap with all metrics, with datasets on the x-axis
    
    Args:
        results_df (pd.DataFrame): Results dataframe
        categories (list): List of metric categories
        output_dir (str): Output directory
        plot_formats (list): List of output formats
        metric_info (dict): Metric information dictionary
    """
    # Get all metrics
    all_metrics = []
    for category in categories:
        all_metrics.extend([m for m in results_df.columns if m != 'Dataset' and 
                            metric_info.get(m, {}).get('category') == category])
    
    if not all_metrics:
        return
    
    # Normalize all metrics to [0, 1] scale
    heatmap_df = results_df.copy()
    for metric in all_metrics:
        heatmap_df[metric] = heatmap_df[metric].apply(
            lambda x: normalize_metric_value(x, metric_info.get(metric, {}))
        )
    
    # Sort metrics by category
    sorted_metrics = []
    for category in categories:
        category_metrics = [m for m in all_metrics if metric_info.get(m, {}).get('category') == category]
        sorted_metrics.extend(category_metrics)
    
    # Replace metric names with display names
    display_names = {m: metric_info.get(m, {}).get('display_name', m) for m in sorted_metrics}
    
    # Create a transposed pivot table with metrics as rows and datasets as columns
    # First, create a new dataframe with the metrics we want
    pivot_data = []
    
    for metric in sorted_metrics:
        row_data = {'Metric': display_names[metric], 
                   'Category': metric_info.get(metric, {}).get('category', 'unknown')}
        
        for _, dataset_row in heatmap_df.iterrows():
            dataset = dataset_row['Dataset']
            value = dataset_row[metric]
            row_data[dataset] = value
        
        pivot_data.append(row_data)
    
    # Create the pivot dataframe
    pivot_df = pd.DataFrame(pivot_data)
    
    # Store category information for each metric for drawing lines later
    category_to_metrics = {}
    for category in categories:
        category_metrics = [display_names[m] for m in sorted_metrics if metric_info.get(m, {}).get('category') == category]
        category_to_metrics[category] = category_metrics
    
    # Set only the metric name as index (not using hierarchical index)
    pivot_df = pivot_df.set_index('Metric')
    # Remove the Category column before creating the heatmap
    if 'Category' in pivot_df.columns:
        pivot_df = pivot_df.drop(columns=['Category'])
    
    # Create heatmap with appropriate dimensions
    plt.figure(figsize=(max(12, len(results_df) + 2), max(8, len(sorted_metrics))))
    
    # Create the heatmap with transposed axes
    ax = sns.heatmap(
        pivot_df, 
        annot=True, 
        cmap='YlGnBu', 
        linewidths=0.5, 
        vmin=0, 
        vmax=1,
        fmt='.2f',
        cbar_kws={'label': 'Normalized Score (higher is better)'}
    )
    
    # Add horizontal lines between metric categories
    category_boundaries = []
    current_pos = 0
    
    # Get unique categories in the order they appear
    unique_categories = []
    for category in categories:
        category_metrics = [display_names[m] for m in sorted_metrics if metric_info.get(m, {}).get('category') == category]
        if category_metrics and category not in unique_categories:
            unique_categories.append(category)
    
    # Calculate boundaries
    for category in unique_categories:
        category_metrics = [display_names[m] for m in sorted_metrics if metric_info.get(m, {}).get('category') == category]
        current_pos += len(category_metrics)
        category_boundaries.append(current_pos)
    
    # Add horizontal lines (excluding the last boundary)
    for boundary in category_boundaries[:-1]:
        plt.axhline(y=boundary, color='black', linestyle='-', linewidth=2)
    
    # Add category labels on the y-axis
    current_pos = 0
    for category in unique_categories:
        category_metrics = [display_names[m] for m in sorted_metrics if metric_info.get(m, {}).get('category') == category]
        if category_metrics:
            mid_point = current_pos + len(category_metrics) / 2
            plt.text(
                -1.5, 
                mid_point,
                category.upper(), 
                verticalalignment='center',
                horizontalalignment='right',
                fontsize=12,
                fontweight='bold',
                rotation=90
            )
            current_pos += len(category_metrics)
    
    # Set title
    plt.title('Synthetic Data Fidelity Metrics Comparison', fontsize=16, pad=20)
    
    # Improve tick label sizes and orientation
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=10)
    # Remove the y-axis label (which would show "Metric")
    plt.ylabel('')
    plt.tight_layout()
    
    # Save the plot in each format
    for fmt in plot_formats:
        filename = os.path.join(output_dir, f"fidelity_heatmap.{fmt}")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
    plt.close()


def plot_aggregated_scores(results_df, categories, output_dir, plot_formats, metric_info):
    """
    Create a bar chart with aggregated scores by category
    
    Args:
        results_df (pd.DataFrame): Results dataframe
        categories (list): List of metric categories
        output_dir (str): Output directory
        plot_formats (list): List of output formats
        metric_info (dict): Metric information dictionary
    """
    datasets = results_df['Dataset'].values
    colors = get_custom_color_palette(len(datasets))
    
    # Create a new dataframe for aggregated scores
    agg_scores = []
    
    for _, row in results_df.iterrows():
        dataset = row['Dataset']
        category_scores = {}
        
        for category in categories:
            # Get metrics for this category
            category_metrics = [m for m in results_df.columns if m != 'Dataset' and 
                               metric_info.get(m, {}).get('category') == category]
            
            if category_metrics:
                # Normalize and average scores
                scores = []
                for metric in category_metrics:
                    value = row[metric]
                    if not np.isnan(value):
                        norm_value = normalize_metric_value(value, metric_info.get(metric, {}))
                        scores.append(norm_value)
                
                if scores:
                    category_scores[category] = np.mean(scores)
                else:
                    category_scores[category] = np.nan
            else:
                category_scores[category] = np.nan
        
        # Calculate overall score
        valid_scores = [score for score in category_scores.values() if not np.isnan(score)]
        overall_score = np.mean(valid_scores) if valid_scores else np.nan
        
        # Add overall score
        category_scores['overall'] = overall_score
        category_scores['Dataset'] = dataset
        
        agg_scores.append(category_scores)
    
    agg_df = pd.DataFrame(agg_scores)
    
    # Melt the dataframe for easier plotting
    melted_df = pd.melt(
        agg_df, 
        id_vars=['Dataset'], 
        value_vars=categories + ['overall'],
        var_name='Category', 
        value_name='Score'
    )
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Create grouped bar chart
    ax = sns.barplot(
        x='Dataset', 
        y='Score', 
        hue='Category', 
        data=melted_df,
        palette='tab10'
    )
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=9)
    
    # Set labels and title
    plt.xlabel('Synthetic Data Model', fontsize=12)
    plt.ylabel('Aggregated Score (0-1)', fontsize=12)
    plt.title('Aggregated Fidelity Scores by Category', fontsize=14)
    
    # Improve legend
    plt.legend(title='Category', fontsize=10, title_fontsize=12)
    
    # Add grid for readability
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot in each format
    for fmt in plot_formats:
        filename = os.path.join(output_dir, f"aggregated_scores.{fmt}")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
    plt.close()


def main():
    """Main function to generate plots from CSV results"""
    # Parse command line arguments
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Set output directory if not specified
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.results_path)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set plot style
    sns.set_style(args.style)
    
    # Read results from CSV file
    print(f"Reading evaluation results from {args.results_path}")
    results_df = pd.read_csv(args.results_path)
    
    # Get metric information
    metric_info = get_metric_display_info()
    
    # Set up categories to plot
    categories = args.metrics
    if "all" in categories:
        categories = ["statistical", "correlation", "distribution", "likelihood", "detection"]
    
    # Group metrics by category
    metrics_by_category = {
        category: [
            metric for metric in results_df.columns
            if metric != 'Dataset' and metric_info.get(metric, {}).get('category') == category
        ]
        for category in categories
    }
    
    # Flatten to get all metrics
    all_metrics = [m for cat in metrics_by_category.values() for m in cat]
    
    # Plot individual metric bar charts
    print("Generating individual metric bar charts...")
    plot_metric_bars(results_df, all_metrics, args.output_dir, args.plot_formats, metric_info)
    
    # Plot special correlation matrix comparison if it exists
    if 'correlation' in categories and 'CorrelationMatrixDistance' in results_df.columns:
        print("Generating correlation matrix comparison...")
        plot_correlation_matrix_comparison(results_df, args.output_dir, args.plot_formats)
    
    # Plot radar charts by category
    print("Generating radar charts...")
    plot_normalized_radar(results_df, metrics_by_category, args.output_dir, args.plot_formats, metric_info)
    
    # Plot overall heatmap
    print("Generating heatmap...")
    plot_heatmap(results_df, categories, args.output_dir, args.plot_formats, metric_info)
    
    # Plot aggregated scores
    print("Generating aggregated scores chart...")
    plot_aggregated_scores(results_df, categories, args.output_dir, args.plot_formats, metric_info)
    
    print(f"All plots generated and saved to {args.output_dir}")


if __name__ == "__main__":
    main()

# Example usage:
# python scripts/generate_fidelity_plots.py --results_path results/fidelity_results/fidelity_results.csv --output_dir fidelity_plots --metrics statistical correlation distribution detection likelihood
# Example usage:
# python scripts/generate_fidelity_plots.py --results_path results/fidelity/fidelity_results.csv --output_dir results/fidelity/plots --metrics statistical distribution likelihood