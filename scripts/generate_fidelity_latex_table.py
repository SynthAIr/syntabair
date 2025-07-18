#!/usr/bin/env python
"""
Generate LaTeX table from fidelity evaluation results.

This script reads previously generated fidelity evaluation results from CSV files
and creates a LaTeX table with proper formatting, highlighting best performers,
and indicating optimization direction for each metric.
"""

import os
import argparse
import pandas as pd
import numpy as np


def setup_argparse():
    """Set up command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Generate LaTeX table from fidelity evaluation results"
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
        "--output_path",
        type=str,
        default=None,
        help="Path to save the LaTeX table (defaults to same directory as results file)"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["statistical", "correlation", "distribution", "likelihood", "detection"],
        choices=["statistical", "correlation", "distribution", "likelihood", "detection", "all"],
        help="Categories of fidelity metrics to include in table"
    )
    parser.add_argument(
        "--table_title",
        type=str,
        default="Synthetic Data Fidelity Metrics",
        help="Title for the LaTeX table"
    )
    parser.add_argument(
        "--table_label",
        type=str,
        default="tab:fidelity_metrics",
        help="Label for the LaTeX table"
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Number of decimal places for metric values"
    )
    parser.add_argument(
        "--highlight_best",
        action="store_true",
        default=True,
        help="Bold the best value for each metric"
    )
    parser.add_argument(
        "--show_arrows",
        action="store_true",
        default=False,
        help="Show arrows indicating optimization direction"
    )
    parser.add_argument(
        "--alternating_colors",
        action="store_true",
        default=True,
        help="Use alternating row colors"
    )
    
    return parser


def get_metric_display_info():
    """
    Get display information for metrics including optimization direction
    
    Returns:
        dict: Dictionary with metric display names, categories, and optimization info
    """
    return {
        # Statistical metrics
        "KSComplement": {
            "display_name": "KS Complement",
            "category": "statistical",
            "goal": "maximize",
            "description": "Kolmogorov-Smirnov complement (higher is better)"
        },
        "CSTest": {
            "display_name": "Chi-Squared Test",
            "category": "statistical", 
            "goal": "maximize",
            "description": "Chi-squared test p-value (higher is better)"
        },
        
        # Correlation metrics
        "PearsonCorrelation": {
            "display_name": "Pearson Correlation",
            "category": "correlation",
            "goal": "maximize",
            "description": "Pearson correlation preservation (higher is better)"
        },
        "SpearmanCorrelation": {
            "display_name": "Spearman Correlation", 
            "category": "correlation",
            "goal": "maximize",
            "description": "Spearman correlation preservation (higher is better)"
        },
        "KendallCorrelation": {
            "display_name": "Kendall Correlation",
            "category": "correlation",
            "goal": "maximize", 
            "description": "Kendall correlation preservation (higher is better)"
        },
        "CorrelationMatrixDistance": {
            "display_name": "Correlation Matrix Similarity",
            "category": "correlation",
            "goal": "maximize",
            "description": "Correlation matrix similarity (higher is better)"
        },
        "MixedTypeCorrelation": {
            "display_name": "Mixed-Type Correlation",
            "category": "correlation",
            "goal": "maximize",
            "description": "Mixed-type correlation preservation (higher is better)"
        },
        
        # Distribution metrics
        "ContinuousKLDivergence": {
            "display_name": "Continuous KL Divergence",
            "category": "distribution",
            "goal": "maximize",
            "description": "Continuous KL divergence score (higher is better)"
        },
        "DiscreteKLDivergence": {
            "display_name": "Discrete KL Divergence", 
            "category": "distribution",
            "goal": "maximize",
            "description": "Discrete KL divergence score (higher is better)"
        },
        
        # Likelihood metrics
        "BNLogLikelihood": {
            "display_name": "BN Log Likelihood",
            "category": "likelihood",
            "goal": "maximize",
            "description": "Bayesian Network log likelihood (higher is better)"
        },
        "GMLogLikelihood": {
            "display_name": "GM Log Likelihood",
            "category": "likelihood", 
            "goal": "maximize",
            "description": "Gaussian Mixture log likelihood (higher is better)"
        },
        
        # Detection metrics
        "LogisticDetection": {
            "display_name": "Logistic Detection",
            "category": "detection",
            "goal": "maximize",
            "description": "Logistic regression detection score (higher is better)"
        },
        "SVCDetection": {
            "display_name": "SVC Detection",
            "category": "detection",
            "goal": "maximize", 
            "description": "SVM detection score (higher is better)"
        }
    }


def format_metric_value(value, precision, is_best=False):
    """
    Format a metric value for LaTeX table
    
    Args:
        value (float): Metric value
        precision (int): Number of decimal places
        is_best (bool): Whether this is the best value for the metric
        
    Returns:
        str: Formatted value string
    """
    if pd.isna(value):
        formatted = "N/A"
    elif abs(value) < 10**(-precision):
        # Use scientific notation for very small numbers
        formatted = f"{value:.{precision-1}e}"
    else:
        formatted = f"{value:.{precision}f}"
    
    # Bold the best value
    if is_best and not pd.isna(value):
        formatted = f"\\textbf{{{formatted}}}"
    
    return formatted


def determine_best_values(results_df, metric_info):
    """
    Determine the best value for each metric based on optimization direction
    
    Args:
        results_df (pd.DataFrame): Results dataframe
        metric_info (dict): Metric information dictionary
        
    Returns:
        dict: Dictionary mapping metric names to best dataset names
    """
    best_values = {}
    
    for column in results_df.columns:
        if column == 'Dataset':
            continue
            
        # Get metric info
        info = metric_info.get(column, {})
        goal = info.get('goal', 'maximize')
        
        # Skip if all values are NaN
        if results_df[column].isna().all():
            continue
        
        # Find best value
        if goal == 'maximize':
            best_idx = results_df[column].idxmax()
        else:  # minimize
            best_idx = results_df[column].idxmin()
        
        if not pd.isna(best_idx):
            best_values[column] = results_df.loc[best_idx, 'Dataset']
    
    return best_values


def get_optimization_arrow(goal):
    """
    Get the LaTeX arrow symbol for optimization direction
    
    Args:
        goal (str): 'maximize' or 'minimize'
        
    Returns:
        str: LaTeX arrow symbol
    """
    if goal == 'maximize':
        return r'$\uparrow$'
    else:
        return r'$\downarrow$'


def escape_latex_special_chars(text):
    """
    Escape special LaTeX characters in text
    
    Args:
        text (str): Input text
        
    Returns:
        str: Escaped text
    """
    # Define characters that need escaping in LaTeX
    special_chars = {
        '&': r'\&',
        '%': r'\%', 
        '$': r'\$',
        '#': r'\#',
        '^': r'\textasciicircum{}',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '\\': r'\textbackslash{}'
    }
    
    for char, replacement in special_chars.items():
        text = text.replace(char, replacement)
    
    return text


def generate_latex_table(results_df, metric_info, categories, args, best_values):
    """
    Generate the complete LaTeX table
    
    Args:
        results_df (pd.DataFrame): Results dataframe
        metric_info (dict): Metric information dictionary  
        categories (list): List of metric categories to include
        args: Command line arguments
        best_values (dict): Dictionary of best values for each metric
        
    Returns:
        str: Complete LaTeX table code
    """
    # Get datasets (columns)
    datasets = results_df['Dataset'].tolist()
    
    # Get metrics to include
    metrics_to_include = []
    for category in categories:
        category_metrics = [
            col for col in results_df.columns 
            if col != 'Dataset' and metric_info.get(col, {}).get('category') == category
        ]
        metrics_to_include.extend(category_metrics)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_metrics = []
    for metric in metrics_to_include:
        if metric not in seen:
            unique_metrics.append(metric)
            seen.add(metric)
    
    metrics_to_include = unique_metrics
    
    if not metrics_to_include:
        return "% No metrics found for the specified categories\n"
    
    # Start building the LaTeX table
    latex_lines = []
    
    # Table environment and setup
    latex_lines.append(r'\begin{table}[!htbp]')
    latex_lines.append(r'\centering')
    latex_lines.append(r'\scriptsize')
    latex_lines.append(r'\setlength{\tabcolsep}{6pt}')
    latex_lines.append(r'\renewcommand{\arraystretch}{1.2}')
    
    # Caption and label
    escaped_title = escape_latex_special_chars(args.table_title)
    latex_lines.append(f'\\caption{{{escaped_title}}}')
    latex_lines.append(f'\\label{{{args.table_label}}}')
    
    # Column specification
    num_cols = len(datasets) + 1  # +1 for metric names
    col_spec = 'l' + 'c' * len(datasets)
    latex_lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
    
    # Top rule
    latex_lines.append(r'\toprule')
    
    # Header row
    if args.alternating_colors:
        latex_lines.append(r'\rowcolor{blue!15}')
    
    header_row = [r'\textbf{Metric}']
    for dataset in datasets:
        escaped_dataset = escape_latex_special_chars(dataset)
        header_row.append(f'\\textbf{{{escaped_dataset}}}')
    
    latex_lines.append(' & '.join(header_row) + r' \\')
    latex_lines.append(r'\midrule')
    
    # Data rows
    for i, metric in enumerate(metrics_to_include):
        # Alternating row colors
        if args.alternating_colors:
            if i % 2 == 0:
                latex_lines.append(r'\rowcolor{gray!5}')
            else:
                latex_lines.append(r'\rowcolor{gray!20}')
        
        # Get metric display info
        info = metric_info.get(metric, {})
        display_name = info.get('display_name', metric)
        goal = info.get('goal', 'maximize')
        
        # Escape special characters in display name
        escaped_name = escape_latex_special_chars(display_name)
        
        # Start row with metric name and arrow
        row_cells = []
        if args.show_arrows:
            arrow = get_optimization_arrow(goal)
            row_cells.append(f'\\textbf{{{escaped_name}}} ({arrow})')
        else:
            row_cells.append(f'\\textbf{{{escaped_name}}}')
        
        # Add values for each dataset
        best_dataset = best_values.get(metric)
        
        for dataset in datasets:
            # Get value for this dataset and metric
            dataset_row = results_df[results_df['Dataset'] == dataset]
            if not dataset_row.empty:
                value = dataset_row[metric].iloc[0]
                is_best = (args.highlight_best and dataset == best_dataset)
                formatted_value = format_metric_value(value, args.precision, is_best)
                row_cells.append(formatted_value)
            else:
                row_cells.append('N/A')
        
        latex_lines.append(' & '.join(row_cells) + r' \\')
    
    # Bottom rule and close table
    latex_lines.append(r'\bottomrule')
    latex_lines.append(r'\end{tabular}')
    latex_lines.append(r'\end{table}')
    
    return '\n'.join(latex_lines)


def main():
    """Main function to generate LaTeX table from CSV results"""
    # Parse command line arguments
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Set output path if not specified
    if args.output_path is None:
        base_dir = os.path.dirname(args.results_path)
        args.output_path = os.path.join(base_dir, "fidelity_results_table.tex")
    
    # Read results from CSV file
    print(f"Reading evaluation results from {args.results_path}")
    results_df = pd.read_csv(args.results_path)
    
    # Get metric information
    metric_info = get_metric_display_info()
    
    # Set up categories to include
    categories = args.metrics
    if "all" in categories:
        categories = ["statistical", "correlation", "distribution", "likelihood", "detection"]
    
    # Determine best values for each metric
    print("Determining best values for each metric...")
    best_values = determine_best_values(results_df, metric_info)
    
    # Generate LaTeX table
    print("Generating LaTeX table...")
    latex_table = generate_latex_table(results_df, metric_info, categories, args, best_values)
    
    # Save to file
    with open(args.output_path, 'w') as f:
        f.write(latex_table)
    
    print(f"LaTeX table saved to {args.output_path}")
    
    # Also print to console for immediate viewing
    print("\n" + "="*50)
    print("GENERATED LATEX TABLE:")
    print("="*50)
    print(latex_table)
    print("="*50)


if __name__ == "__main__":
    main()

# Example usage:
# python scripts/generate_fidelity_latex_table.py --results_path results/fidelity/fidelity_results.csv --output_path results/fidelity/fidelity_table.tex --metrics statistical correlation distribution --table_title "Synthetic Data Fidelity Metrics" --precision 4 --highlight_best --show_arrows --alternating_colors