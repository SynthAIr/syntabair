#!/usr/bin/env python
"""
Generate plots from privacy evaluation results.

This script reads previously generated privacy evaluation results from CSV files
and creates visualizations for two key privacy metrics:
1. DCRBaselineProtection - measures how synthetic data compares to random data in terms of 
   distance to the original data
2. DCROverfittingProtection - measures whether synthetic data is overfit to training data
   by comparing distances to training vs. validation data
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
        description="Generate plots from privacy evaluation results"
    )
    
    # Required arguments (at least one)
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing evaluation results CSV files"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save generated plots (defaults to results_dir/plots)"
    )
    parser.add_argument(
        "--plot_types",
        type=str,
        nargs="+",
        default=["scores", "details", "comparison"],
        choices=["scores", "details", "comparison", "all"],
        help="Types of plots to generate"
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


def plot_privacy_scores(results_df, output_dir, plot_formats):
    """
    Create bar charts for privacy scores
    
    Args:
        results_df (pd.DataFrame): Results dataframe with combined metrics
        output_dir (str): Output directory
        plot_formats (list): List of output formats
    """
    # Check for baseline protection metric
    if 'BaselineProtectionScore' in results_df.columns:
        datasets = results_df['Dataset']
        baseline_scores = results_df['BaselineProtectionScore']
        
        # Set up the figure for baseline protection
        plt.figure(figsize=(10, 6))
        
        # Get custom colors
        colors = get_custom_color_palette(len(datasets))
        
        # Plot baseline protection bars
        bars = plt.bar(datasets, baseline_scores, color=colors)
        
        # Add labels, title
        plt.xlabel('Synthetic Data Model', fontsize=12)
        plt.ylabel('Privacy Score (higher is better)', fontsize=12)
        plt.title('Baseline Protection Score (higher is better)', fontsize=14)
        # plt.xticks(rotation=45, ha='right')
        
        # Add value annotations on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.01,
                f'{height:.2f}',
                ha='center', 
                va='bottom',
                fontsize=10
            )
        
        plt.ylim(0, 1.1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save the plot in each format
        for fmt in plot_formats:
            filename = os.path.join(output_dir, f"baseline_protection_score.{fmt}")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            
        plt.close()
    
    # Check for overfitting protection metric
    if 'OverfittingProtectionScore' in results_df.columns:
        datasets = results_df['Dataset']
        overfitting_scores = results_df['OverfittingProtectionScore']
        
        # Set up the figure for overfitting protection
        plt.figure(figsize=(10, 6))
        
        # Get custom colors - use a different color scheme for this metric
        colors = get_custom_color_palette(len(datasets))
        
        # Plot overfitting protection bars
        bars = plt.bar(datasets, overfitting_scores, color=colors)
        
        # Add labels, title
        plt.xlabel('Synthetic Data Model', fontsize=12)
        plt.ylabel('Privacy Score (higher is better)', fontsize=12)
        plt.title('Overfitting Protection Score (higher is better)', fontsize=14)
        # plt.xticks(rotation=45, ha='right')
        
        # Add value annotations on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.01,
                f'{height:.2f}',
                ha='center', 
                va='bottom',
                fontsize=10
            )
        
        plt.ylim(0, 1.1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save the plot in each format
        for fmt in plot_formats:
            filename = os.path.join(output_dir, f"overfitting_protection_score.{fmt}")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            
        plt.close()
    
    # If both metrics are present, also plot the average
    if 'BaselineProtectionScore' in results_df.columns and 'OverfittingProtectionScore' in results_df.columns:
        datasets = results_df['Dataset']
        baseline_scores = results_df['BaselineProtectionScore']
        overfitting_scores = results_df['OverfittingProtectionScore']
        
        # Calculate average score
        avg_scores = (baseline_scores + overfitting_scores) / 2
        
        # Set up the figure for average score
        plt.figure(figsize=(10, 6))
        
        # Get custom colors
        colors = get_custom_color_palette(len(datasets))
        
        # Plot bars
        bars = plt.bar(datasets, avg_scores, color=colors)
        
        # Add labels and title
        plt.xlabel('Synthetic Data Model', fontsize=12)
        plt.ylabel('Average Privacy Score', fontsize=12)
        plt.title('Average Privacy Protection Score (higher is better)', fontsize=14)
        # plt.xticks(rotation=45, ha='right')
        
        # Add value annotations on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.01,
                f'{height:.2f}',
                ha='center', 
                va='bottom',
                fontsize=10
            )
        
        plt.ylim(0, 1.1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save the plot in each format
        for fmt in plot_formats:
            filename = os.path.join(output_dir, f"average_privacy_score.{fmt}")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            
        plt.close()


def plot_baseline_details(results_df, output_dir, plot_formats):
    """
    Create detailed plots for baseline protection
    
    Args:
        results_df (pd.DataFrame): Results dataframe
        output_dir (str): Output directory
        plot_formats (list): List of output formats
    """
    if 'SyntheticMedianDCR' in results_df.columns and 'RandomMedianDCR' in results_df.columns:
        datasets = results_df['Dataset']
        synthetic_dcr = results_df['SyntheticMedianDCR']
        random_dcr = results_df['RandomMedianDCR']
        
        # Set up the figure
        plt.figure(figsize=(12, 8))
        
        # Set positions and width for the bars
        x = np.arange(len(datasets))
        width = 0.35
        
        # Get custom colors - one per dataset
        colors = get_custom_color_palette(len(datasets))
        
        # Plot bars - use the same color for each dataset but different patterns
        bars1 = plt.bar(x - width/2, synthetic_dcr, width, 
                      color=colors, label='Synthetic Data', hatch='')
        bars2 = plt.bar(x + width/2, random_dcr, width, 
                      color=colors, label='Random Data', hatch='////')
        
        # Add labels, title, and legend
        plt.xlabel('Synthetic Data Model', fontsize=12)
        plt.ylabel('Median DCR to Real Data', fontsize=12)
        plt.title('Distance to Closest Record (DCR) Comparison', fontsize=14)
        plt.xticks(x, datasets)
        plt.legend()
        
        # Add value annotations on bars
        for bar in bars1:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.01,
                f'{height:.3f}',
                ha='center', 
                va='bottom',
                fontsize=10
            )
        
        for bar in bars2:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.01,
                f'{height:.3f}',
                ha='center', 
                va='bottom',
                fontsize=10
            )
        
        # Set y-axis limits with some padding
        max_val = max(synthetic_dcr.max(), random_dcr.max())
        plt.ylim(0, max_val * 1.2)
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save the plot in each format
        for fmt in plot_formats:
            filename = os.path.join(output_dir, f"baseline_protection_dcr_comparison.{fmt}")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            
        plt.close()
        
        # Plot the ratio of synthetic to random DCR
        plt.figure(figsize=(10, 6))
        
        # Calculate ratio
        ratio = synthetic_dcr / random_dcr
        
        # Get custom colors - same as before for consistency
        # Use the same colors as above
        
        # Plot bars
        bars = plt.bar(datasets, ratio, color=colors)
        
        # Add a horizontal line at y=1.0
        # plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Equal DCR')
        
        # Add labels and title
        plt.xlabel('Synthetic Data Model', fontsize=12)
        plt.ylabel('Synthetic DCR / Random DCR Ratio', fontsize=12)
        plt.title('Ratio of Synthetic to Random DCR (lower is better)', fontsize=14)
        # plt.xticks(rotation=45, ha='right')
        # plt.legend()
        
        # Add value annotations on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.01,
                f'{height:.2f}',
                ha='center', 
                va='bottom',
                fontsize=10
            )
        
        # Set y-axis limits with some padding
        plt.ylim(0, ratio.max() * 1.2)
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save the plot in each format
        for fmt in plot_formats:
            filename = os.path.join(output_dir, f"baseline_protection_dcr_ratio.{fmt}")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            
        plt.close()


def plot_overfitting_details(results_df, output_dir, plot_formats):
    """
    Create detailed plots for overfitting protection
    
    Args:
        results_df (pd.DataFrame): Results dataframe
        output_dir (str): Output directory
        plot_formats (list): List of output formats
    """
    if 'CloserToTraining' in results_df.columns and 'CloserToHoldout' in results_df.columns:
        datasets = results_df['Dataset']
        closer_to_training = results_df['CloserToTraining']
        closer_to_holdout = results_df['CloserToHoldout']
        
        # Get custom colors - same color per dataset for consistency
        colors = get_custom_color_palette(len(datasets))
        
        # For the stacked bar, we need a different approach to maintain dataset coloring
        # First create side-by-side bars with consistent dataset colors
        plt.figure(figsize=(14, 8))
        
        # Set positions and width for the bars
        x = np.arange(len(datasets))
        width = 0.35
        
        # Create the side-by-side bars
        bars1 = plt.bar(x - width/2, closer_to_training, width, color=colors, hatch='', 
                        label='Closer to Training Data', edgecolor='black', linewidth=0.5)
        bars2 = plt.bar(x + width/2, closer_to_holdout, width, color=colors, hatch='////', 
                        label='Closer to Holdout Data', edgecolor='black', linewidth=0.5)
        
        # Add labels, title, and legend
        plt.xlabel('Synthetic Data Model', fontsize=12)
        plt.ylabel('Percentage of Synthetic Rows', fontsize=12)
        plt.title('Distribution of Proximity to Training vs. Holdout Data', fontsize=14)
        plt.xticks(x, datasets)
        plt.legend(loc='best')
        
        # Add percentage annotations on bars
        for bar in bars1:
            height = bar.get_height()
            if height > 0.05:  # Only show if there's enough room
                plt.text(
                    bar.get_x() + bar.get_width()/2,
                    height / 2,
                    f'{height:.1%}',
                    ha='center',
                    va='center',
                    fontsize=10,
                    color='black'
                )
        
        for bar in bars2:
            height = bar.get_height()
            if height > 0.05:  # Only show if there's enough room
                plt.text(
                    bar.get_x() + bar.get_width()/2,
                    height / 2,
                    f'{height:.1%}',
                    ha='center',
                    va='center',
                    fontsize=10,
                    color='black'
                )
        
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save the plot in each format
        for fmt in plot_formats:
            filename = os.path.join(output_dir, f"overfitting_protection_comparison.{fmt}")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            
        plt.close()
        
        # Also create the stacked bar chart (but with consistent coloring + hatching)
        plt.figure(figsize=(12, 8))
        
        # Create the stacked bars - use alpha to make colors lighter
        # and distinct hatching patterns to differentiate
        bars1 = plt.bar(datasets, closer_to_training, color=colors, hatch='', 
                        label='Closer to Training Data', edgecolor='black', linewidth=0.5)
        bars2 = plt.bar(datasets, closer_to_holdout, bottom=closer_to_training, color=colors,   alpha=0.7, hatch='////', label='Closer to Holdout Data', edgecolor='black', linewidth=0.5)
        
        # Add labels, title, and legend
        plt.xlabel('Synthetic Data Model', fontsize=12)
        plt.ylabel('Percentage of Synthetic Rows', fontsize=12)
        plt.title('Distribution of Proximity to Training vs. Holdout Data', fontsize=14)
        # plt.xticks(rotation=45, ha='right')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
        
        # Add percentage annotations on bars
        for i, dataset in enumerate(datasets):
            # Add percentage for training
            training_pct = closer_to_training.iloc[i]
            if training_pct > 0.05:  # Only show if there's enough room
                plt.text(
                    i,
                    training_pct / 2,
                    f'{training_pct:.1%}',
                    ha='center',
                    va='center',
                    fontsize=10,
                    color='black'
                )
            
            # Add percentage for holdout
            holdout_pct = closer_to_holdout.iloc[i]
            if holdout_pct > 0.05:  # Only show if there's enough room
                plt.text(
                    i,
                    training_pct + holdout_pct / 2,
                    f'{holdout_pct:.1%}',
                    ha='center',
                    va='center',
                    fontsize=10,
                    color='black'
                )
        
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save the plot in each format
        for fmt in plot_formats:
            filename = os.path.join(output_dir, f"overfitting_protection_percentages.{fmt}")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            
        plt.close()
        
        # Create a bar chart of the percentage closer to holdout (ideal is 50%)
        plt.figure(figsize=(10, 6))
        
        # Plot bars - using the same colors as before
        bars = plt.bar(datasets, closer_to_holdout, color=colors)
        
        # Add a horizontal line at y=0.5 (ideal value)
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Ideal Balance (50%)')
        
        # Add labels and title
        plt.xlabel('Synthetic Data Model', fontsize=12)
        plt.ylabel('Percentage of Rows Closer to Holdout', fontsize=12)
        plt.title('Percentage of Synthetic Data Closer to Holdout Set', fontsize=14)
        # plt.xticks(rotation=45, ha='right')
        plt.legend()
        
        # Add value annotations on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.01,
                f'{height:.1%}',
                ha='center', 
                va='bottom',
                fontsize=10
            )
        
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save the plot in each format
        for fmt in plot_formats:
            filename = os.path.join(output_dir, f"overfitting_protection_holdout_percentage.{fmt}")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            
        plt.close()


def plot_privacy_comparison(results_df, output_dir, plot_formats):
    """
    Create comparison plots between privacy metrics
    
    Args:
        results_df (pd.DataFrame): Results dataframe with both metrics
        output_dir (str): Output directory
        plot_formats (list): List of output formats
    """
    if 'BaselineProtectionScore' in results_df.columns and 'OverfittingProtectionScore' in results_df.columns:
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        
        # Get custom colors
        colors = get_custom_color_palette(len(results_df))
        
        # Create the scatter plot
        for i, (_, row) in enumerate(results_df.iterrows()):
            plt.scatter(
                row['BaselineProtectionScore'], 
                row['OverfittingProtectionScore'],
                s=100,
                color=colors[i],
                label=row['Dataset']
            )
            
            # Add dataset name as text label
            if row['Dataset'] == "REaLTabFormer":
                plt.annotate(
                    row['Dataset'],
                    (row['BaselineProtectionScore'], row['OverfittingProtectionScore']),
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=10,
                    ha='right'
                )
            else:
                plt.annotate(
                    row['Dataset'],
                    (row['BaselineProtectionScore'], row['OverfittingProtectionScore']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=10
                )
        
        # Add reference lines at 0.5
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Add a diagonal line y=x
        min_val = min(results_df['BaselineProtectionScore'].min(), results_df['OverfittingProtectionScore'].min())
        max_val = max(results_df['BaselineProtectionScore'].max(), results_df['OverfittingProtectionScore'].max())
        padding = (max_val - min_val) * 0.1
        plt.plot(
            [min_val - padding, max_val + padding], 
            [min_val - padding, max_val + padding], 
            color='red', linestyle='--', alpha=0.5, label='Equal Scores'
        )
        
        # Add quadrant labels
        plt.text(0.25, 0.75, "Good Overfitting Protection,\nPoor Baseline Protection", 
                 ha='center', va='center', fontsize=10, alpha=0.7)
        plt.text(0.75, 0.75, "Good Privacy Overall", 
                 ha='center', va='center', fontsize=10, alpha=0.7)
        plt.text(0.25, 0.25, "Poor Privacy Overall", 
                 ha='center', va='center', fontsize=10, alpha=0.7)
        plt.text(0.75, 0.25, "Good Baseline Protection,\nPoor Overfitting Protection", 
                 ha='center', va='center', fontsize=10, alpha=0.7)
        
        # Add labels and title
        plt.xlabel('Baseline Protection Score', fontsize=12)
        plt.ylabel('Overfitting Protection Score', fontsize=12)
        plt.title('Privacy Protection Metrics Comparison', fontsize=14)
        plt.legend(title='Synthetic Data Model', loc='upper right')
        
        # Set limits with padding
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Save the plot in each format
        for fmt in plot_formats:
            filename = os.path.join(output_dir, f"privacy_metrics_comparison.{fmt}")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            
        plt.close()
        
        # Create radar chart
        plt.figure(figsize=(10, 10), facecolor='white')
        
        # Number of variables
        categories = ['Baseline Protection', 'Overfitting Protection']
        N = len(categories)
        
        # Create angles for each category
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Create subplot with polar projection
        ax = plt.subplot(111, polar=True)
        
        # Draw category labels
        plt.xticks(angles[:-1], categories, fontsize=12)
        
        # Draw y-axis labels
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=10)
        plt.ylim(0, 1)
        
        # Get custom colors
        colors = get_custom_color_palette(len(results_df))
        
        # Plot each dataset
        for i, (_, row) in enumerate(results_df.iterrows()):
            # Create values array
            values = [row['BaselineProtectionScore'], row['OverfittingProtectionScore']]
            values += values[:1]  # Close the loop
            
            # Plot values
            ax.plot(angles, values, linewidth=2, linestyle='solid', color=colors[i], label=row['Dataset'])
            ax.fill(angles, values, color=colors[i], alpha=0.1)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), title="Synthetic Data Model")
        
        # Add title
        plt.title('Privacy Protection Metrics Comparison', y=1.1, fontsize=14)
        
        # Save the plot in each format
        for fmt in plot_formats:
            filename = os.path.join(output_dir, f"privacy_metrics_radar.{fmt}")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            
        plt.close()
        
        # Create heatmap
        plt.figure(figsize=(10, 6))
        
        # Prepare data for heatmap
        heatmap_data = results_df.set_index('Dataset')[['BaselineProtectionScore', 'OverfittingProtectionScore']]
        heatmap_data.columns = ['Baseline Protection', 'Overfitting Protection']
        
        # Create heatmap
        sns.heatmap(
            heatmap_data, 
            annot=True, 
            cmap='YlGnBu', 
            vmin=0, 
            vmax=1,
            fmt='.2f',
            linewidths=0.5,
            cbar_kws={'label': 'Privacy Score (higher is better)'}
        )
        
        # Add title
        plt.title('Privacy Protection Scores Heatmap', fontsize=14)
        plt.tight_layout()
        
        # Save the plot in each format
        for fmt in plot_formats:
            filename = os.path.join(output_dir, f"privacy_metrics_heatmap.{fmt}")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            
        plt.close()


def main():
    """Main function to generate plots from CSV results"""
    # Parse command line arguments
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Set output directory if not specified
    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, "plots")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set plot style
    sns.set_style(args.style)
    
    # Set up plot types
    plot_types = args.plot_types
    if "all" in plot_types:
        plot_types = ["scores", "details", "comparison"]
    
    # Look for results files
    baseline_path = os.path.join(args.results_dir, 'baseline_protection_results.csv')
    overfitting_path = os.path.join(args.results_dir, 'overfitting_protection_results.csv')
    combined_path = os.path.join(args.results_dir, 'privacy_results.csv')
    
    # Load results, prioritizing combined file if it exists
    results_df = None
    
    if os.path.exists(combined_path):
        print(f"Loading combined privacy results from {combined_path}")
        results_df = pd.read_csv(combined_path)
    else:
        # If we have both individual files, merge them
        if os.path.exists(baseline_path) and os.path.exists(overfitting_path):
            print(f"Loading and combining separate results files")
            baseline_df = pd.read_csv(baseline_path)
            overfitting_df = pd.read_csv(overfitting_path)
            
            # Merge the two dataframes on Dataset column
            results_df = pd.merge(baseline_df, overfitting_df, on='Dataset', suffixes=('_baseline', '_overfitting'))
            
            # Rename columns for consistency
            results_df = results_df.rename(columns={
                'Score_baseline': 'BaselineProtectionScore',
                'Score_overfitting': 'OverfittingProtectionScore'
            })
        elif os.path.exists(baseline_path):
            print(f"Loading baseline protection results from {baseline_path}")
            results_df = pd.read_csv(baseline_path)
        elif os.path.exists(overfitting_path):
            print(f"Loading overfitting protection results from {overfitting_path}")
            results_df = pd.read_csv(overfitting_path)
        else:
            print(f"No results files found in {args.results_dir}")
            return
    
    # Generate plots
    if results_df is not None:
        if "scores" in plot_types:
            print("Generating privacy score plots...")
            plot_privacy_scores(results_df, args.output_dir, args.plot_formats)
        
        if "details" in plot_types:
            print("Generating detailed plots...")
            
            if 'SyntheticMedianDCR' in results_df.columns:
                print("  Generating baseline protection details...")
                plot_baseline_details(results_df, args.output_dir, args.plot_formats)
            
            if 'CloserToTraining' in results_df.columns:
                print("  Generating overfitting protection details...")
                plot_overfitting_details(results_df, args.output_dir, args.plot_formats)
        
        if "comparison" in plot_types and 'BaselineProtectionScore' in results_df.columns and 'OverfittingProtectionScore' in results_df.columns:
            print("Generating comparison plots...")
            plot_privacy_comparison(results_df, args.output_dir, args.plot_formats)
    
    print(f"All plots generated and saved to {args.output_dir}")


if __name__ == "__main__":
    main()

# Example usage:
# python scripts/plot_privacy.py --results_dir results/privacy --output_dir results/privacy/plots --plot_types all