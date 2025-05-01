#!/usr/bin/env python
"""
Generate plots from Train-Synthetic-Test-Real (TSTR) evaluation results.

This script reads previously generated evaluation results from CSV files
and creates visualizations without needing to rerun the evaluation process.
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
        description="Generate plots from TSTR evaluation results"
    )
    
    # Required arguments
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
        "--targets",
        type=str,
        nargs="+",
        default=None,
        help="Target variables to plot (defaults to all available in the data)"
    )
    parser.add_argument(
        "--prediction_modes",
        type=str,
        nargs="+",
        default=None,
        help="Prediction modes to plot (defaults to all available in the data)"
    )
    parser.add_argument(
        "--plot_types",
        type=str,
        nargs="+",
        default=["performance", "utility", "feature_importance"],
        choices=["performance", "utility", "feature_importance"],
        help="Types of plots to generate"
    )
    parser.add_argument(
        "--top_n_features",
        type=int,
        default=15,
        help="Number of top features to show in feature importance plots"
    )
    
    return parser

def get_custom_color_palette(num_colors, exclude_real=False):
    """
    Get a custom color palette with visually appealing colors
    
    Args:
        num_colors (int): Number of colors needed
        exclude_real (bool): Whether to reserve a special color for 'Real' dataset
        
    Returns:
        list: List of colors in RGB tuples
    """
    # Define a visually appealing color palette
    # These are more distinctive and appealing colors
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
    
    if exclude_real:
        # Define a distinct color for 'Real' dataset 
        # Deep blue
        real_color = (0.00, 0.27, 0.55)
        
        # Need one less color from the palette
        required_colors = num_colors - 1
        
        # If we need more colors than in our custom palette, extend it with a seaborn color palette
        if required_colors > len(custom_colors):
            additional_colors = sns.color_palette("tab20", required_colors)
            result_colors = custom_colors + additional_colors
            result_colors = result_colors[:required_colors]
        else:
            result_colors = custom_colors[:required_colors]
            
        # Return colors with the real color first
        return [real_color] + result_colors
    else:
        # If we need more colors than in our custom palette, extend it with a seaborn color palette
        if num_colors > len(custom_colors):
            additional_colors = sns.color_palette("tab20", num_colors)
            result_colors = custom_colors + additional_colors
            return result_colors[:num_colors]
        else:
            return custom_colors[:num_colors]


def create_feature_label_mapping():
    """
    Create a mapping between technical column names and human-readable labels
    
    Returns:
        dict: Mapping from column names to readable labels
    """
    return {
        # Categorical features
        'IATA_CARRIER_CODE': 'Carrier',
        'DEPARTURE_IATA_AIRPORT_CODE': 'Departure Airport',
        'ARRIVAL_IATA_AIRPORT_CODE': 'Arrival Airport',
        'AIRCRAFT_TYPE_IATA': 'Aircraft Type',
        'AIRPORT_PAIR': 'Airport Pair',
        
        # Numerical features
        'SCHEDULED_MONTH': 'Scheduled Month',
        'SCHEDULED_DAY': 'Scheduled Day',
        'SCHEDULED_HOUR': 'Scheduled Hour',
        'SCHEDULED_MINUTE': 'Scheduled Minute',
        'SCHEDULED_DURATION_MIN': 'Scheduled Duration',
        'ACTUAL_DURATION_MIN': 'Actual Duration',
        'DURATION_DIFF_MIN': 'Duration Difference',
        'DAY_OF_WEEK': 'Day of Week',
        'ARRIVAL_HOUR': 'Arrival Hour',
        'ARRIVAL_DAY': 'Arrival Day',
        
        # Target variables
        'DEPARTURE_DELAY_MIN': 'Departure Delay',
        'ARRIVAL_DELAY_MIN': 'Arrival Delay',
        'TURNAROUND_MIN': 'Turnaround Time'
    }


def get_readable_title(target, mode):
    """Get human-readable title for plots"""
    label_mapping = create_feature_label_mapping()
    readable_target = label_mapping.get(target, target.replace('_', ' ').title())
    mode_str = mode.title()
    return f"{readable_target} Prediction ({mode_str})"


def plot_performance_metrics(performance_df, target, mode, output_dir):
    """
    Plot performance metrics for the given target and prediction mode
    
    Args:
        performance_df (pd.DataFrame): DataFrame containing performance results
        target (str): Target variable
        mode (str): Prediction mode
        output_dir (str): Directory to save plots
    """
    # Filter data for the target and mode
    filtered_df = performance_df[(performance_df['Target'] == target) & 
                                 (performance_df['Mode'] == mode)]
    
    if filtered_df.empty:
        print(f"No performance data available for {target} with {mode} mode")
        return
    
    # Get readable title
    title_base = get_readable_title(target, mode)
    
    # Create a directory for the target and mode
    target_mode_dir = os.path.join(output_dir, f"{target.lower()}_{mode}")
    os.makedirs(target_mode_dir, exist_ok=True)
    
    # Plot each metric
    metrics = [('RMSE', 'Root Mean Squared Error (lower is better)'),
               ('MAE', 'Mean Absolute Error (lower is better)'),
               ('R2', 'R² Score (higher is better)')]
    
    for metric, metric_title in metrics:
        plt.figure(figsize=(14, 8))
        
        # Prepare data for grouped bar chart
        models = filtered_df['Model'].unique()
        datasets = filtered_df['Dataset'].unique()
        
        x = np.arange(len(models))
        width = 0.8 / len(datasets)
        
        # Get custom colors
        has_real = 'Real' in datasets
        colors = get_custom_color_palette(len(datasets), exclude_real=has_real)
        
        # Plot bars for each dataset
        for i, dataset in enumerate(datasets):
            dataset_data = filtered_df[filtered_df['Dataset'] == dataset]
            values = []
            
            for model in models:
                model_data = dataset_data[dataset_data['Model'] == model]
                if not model_data.empty:
                    values.append(model_data[metric].values[0])
                else:
                    values.append(0)
            
            # Draw the bar with custom color
            color_index = 0 if dataset == 'Real' and has_real else i
            plt.bar(x + i*width, values, width, label=dataset, color=colors[color_index])
        
        # Labels and title
        plt.xlabel('Model', fontsize=12)
        plt.ylabel(metric_title, fontsize=12)
        plt.title(f'{title_base} - {metric_title}', fontsize=14)
        
        plt.xticks(x + width*(len(datasets)-1)/2, models)
        plt.grid(axis='y', alpha=0.3)
        plt.legend(title='Dataset', title_fontsize=12, fontsize=10, loc='best')
        plt.tight_layout()
        
        # Add value annotations on bars
        for i, dataset in enumerate(datasets):
            dataset_data = filtered_df[filtered_df['Dataset'] == dataset]
            
            for j, model in enumerate(models):
                model_data = dataset_data[dataset_data['Model'] == model]
                if not model_data.empty:
                    value = model_data[metric].values[0]
                    if metric == 'R2':
                        # Format R2 with 2 decimal places
                        plt.text(x[j] + i*width, value + 0.01 * max(values), 
                                f'{value:.2f}', ha='center', va='bottom',
                                fontsize=9)
                    else:
                        # Format RMSE and MAE with 1 decimal place
                        plt.text(x[j] + i*width, value + 0.01 * max(values), 
                                f'{value:.1f}', ha='center', va='bottom',
                                fontsize=9)
        
        # Save the plot
        plt.savefig(os.path.join(target_mode_dir, f"{target.lower()}_{mode}_{metric.lower()}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    

def plot_utility_scores(utility_df, target, mode, output_dir):
    """
    Plot utility scores for the given target and prediction mode
    
    Args:
        utility_df (pd.DataFrame): DataFrame containing utility scores
        target (str): Target variable
        mode (str): Prediction mode
        output_dir (str): Directory to save plots
    """
    # Filter data for the target and mode
    filtered_df = utility_df[(utility_df['Target'] == target) & 
                             (utility_df['Mode'] == mode)]
    
    if filtered_df.empty:
        print(f"No utility data available for {target} with {mode} mode")
        return
    
    # Get readable title
    title_base = get_readable_title(target, mode)
    
    # Create a directory for the target and mode
    target_mode_dir = os.path.join(output_dir, f"{target.lower()}_{mode}")
    os.makedirs(target_mode_dir, exist_ok=True)
    
    # Get datasets and models in their original order - this ensures consistency
    datasets = filtered_df['Dataset'].unique()
    models = filtered_df['Model'].unique()
    
    # Use custom color palette
    colors = get_custom_color_palette(len(datasets))
    dataset_colors = dict(zip(datasets, colors))
    
    # Create enhanced heatmap of overall utility scores
    plt.figure(figsize=(12, 9))
    
    # Pivot the dataframe for the heatmap
    heatmap_data = filtered_df.pivot(index='Model', columns='Dataset', values='Overall Utility')
    
    # Reorder the columns to match the original dataset order
    heatmap_data = heatmap_data[datasets]
    
    # Alternative approach - extract values to a numpy array with correct order
    # Initialize a numpy array with zeros
    heatmap_values = np.zeros((len(models), len(datasets)))
    
    # Fill the numpy array with values from the pivot table
    for i, model in enumerate(models):
        if model in heatmap_data.index:
            for j, dataset in enumerate(datasets):
                if dataset in heatmap_data.columns and not pd.isna(heatmap_data.loc[model, dataset]):
                    heatmap_values[i, j] = heatmap_data.loc[model, dataset]
    
    # Create a new DataFrame from the numpy array with explicit types
    ordered_heatmap = pd.DataFrame(
        data=heatmap_values,
        index=models,
        columns=datasets
    )
    
    # Use a custom color map (blue to green)
    cmap = plt.cm.YlGnBu
    
    # Plot enhanced heatmap with consistently ordered data
    ax = sns.heatmap(
        ordered_heatmap, 
        annot=True, 
        cmap=cmap,
        vmin=0, 
        vmax=1, 
        fmt='.2f',
        cbar_kws={'label': 'Utility Score (higher is better)'},
        linewidths=0.5,
        linecolor='white',
        annot_kws={"size": 12}
    )
    
    # Improve heatmap appearance
    plt.title(f'{title_base} - Synthetic Data Utility Scores', fontsize=14, pad=20)
    ax.set_xlabel('Synthetic Dataset', fontsize=14, labelpad=10)
    ax.set_ylabel('Model', fontsize=14, labelpad=10)
    
    # Improve tick label size
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add a border around the heatmap
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
    
    plt.tight_layout()
    
    # Save the enhanced heatmap
    plt.savefig(os.path.join(target_mode_dir, f"{target.lower()}_{mode}_utility_heatmap.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Rest of the function remains the same...
    # Bar chart of average utility by dataset with custom colors
    plt.figure(figsize=(10, 7))
    
    # Get datasets in the original order
    ordered_datasets = datasets  # Use the same datasets variable defined earlier
    
    # Calculate average utility for each dataset while preserving order
    avg_utility = []
    for dataset in ordered_datasets:
        avg = filtered_df[filtered_df['Dataset'] == dataset]['Overall Utility'].mean()
        avg_utility.append({'Dataset': dataset, 'Overall Utility': avg})
    
    avg_utility_df = pd.DataFrame(avg_utility)
    
    # Plot with custom colors
    bars = plt.bar(
        avg_utility_df['Dataset'],
        avg_utility_df['Overall Utility'],
        color=[colors[i] for i in range(len(ordered_datasets))],
        width=0.6
    )
    
    # Enhance bar chart appearance
    plt.title(f'{title_base} - Average Synthetic Data Utility', fontsize=16)
    plt.xlabel('Synthetic Dataset', fontsize=14)
    plt.ylabel('Average Utility Score', fontsize=14)
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.02,
            f'{height:.2f}',
            ha='center', 
            fontsize=12
        )
    
    plt.tight_layout()
    
    # Save the enhanced bar chart
    plt.savefig(os.path.join(target_mode_dir, f"{target.lower()}_{mode}_avg_utility.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importances(importance_df, target, mode, output_dir, top_n=15):
    """Plot feature importances for the given target and prediction mode"""
    # Filter data for the target and mode
    filtered_df = importance_df[(importance_df['Target'] == target) & 
                               (importance_df['Mode'] == mode)]
    
    if filtered_df.empty:
        print(f"No feature importance data available for {target} with {mode} mode")
        return
    
    # Get readable title
    title_base = get_readable_title(target, mode)
    
    # Create a directory for feature importance plots
    importance_dir = os.path.join(output_dir, f"{target.lower()}_{mode}", "feature_importances")
    os.makedirs(importance_dir, exist_ok=True)
    
    # Get feature label mapping
    label_mapping = create_feature_label_mapping()
    
    # Get all models and datasets
    models = filtered_df['Model'].unique()
    datasets = filtered_df['Dataset'].unique()
    
    # Get custom colors for datasets
    has_real = 'Real' in datasets
    colors = get_custom_color_palette(len(datasets), exclude_real=has_real)
    
    # Plot feature importances for each model across datasets
    for model in models:
        model_df = filtered_df[filtered_df['Model'] == model]
        
        if model_df.empty:
            continue
        
        # Calculate average importance across datasets for each feature
        feature_avg = model_df.groupby('Feature')['Importance'].mean().reset_index()
        
        # Get top N features by average importance
        top_features = feature_avg.nlargest(top_n, 'Importance')['Feature'].tolist()
        
        # Create a comparison DataFrame with only top features
        comparison_df = pd.DataFrame(index=top_features)
        
        for i, dataset in enumerate(datasets):
            dataset_model_df = model_df[model_df['Dataset'] == dataset]
            if not dataset_model_df.empty:
                dataset_importances = dataset_model_df.set_index('Feature')['Importance']
                # Only include values for top features
                comparison_df[dataset] = dataset_importances.reindex(top_features, fill_value=0)
        
        # Sort by average importance
        comparison_df['avg'] = comparison_df.mean(axis=1)
        comparison_df = comparison_df.sort_values('avg', ascending=True).drop('avg', axis=1)
        
        # Plot comparison as horizontal bar chart
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Reverse the order of the columns (datasets) in the DataFrame
        comparison_df = comparison_df[comparison_df.columns[::-1]]

        # Reverse the order of the colors
        reversed_colors = colors[::-1]

        # Create the horizontal bar chart with reversed colors
        comparison_df.plot(
            kind='barh', 
            ax=ax,
            color=[reversed_colors[i % len(reversed_colors)] for i in range(len(comparison_df.columns))],
            width=0.8
        )
        
        # Apply label mapping to y-axis labels
        readable_labels = []
        for feature in comparison_df.index:
            readable = feature
            for key in label_mapping:
                if feature == key:
                    readable = label_mapping[key]
                    break
                elif feature.startswith(key + '_'):
                    # Handle one-hot encoded features
                    category_value = feature[len(key)+1:]
                    readable = f"{label_mapping.get(key, key)}: {category_value}"
                    break
            readable_labels.append(readable)
        
        # Set new y-tick labels with improved style
        ax.set_yticklabels(readable_labels, fontsize=10)
        
        # Set consistent x-axis limit
        max_importance = comparison_df.values.max()
        ax.set_xlim(0, max_importance * 1.1)  # Add 10% margin
        
        # Add title and labels
        fig_title = f'Feature Importance - {model}\n{title_base}'
        plt.title(fig_title, fontsize=14)
        plt.xlabel('Normalized Importance', fontsize=12)
        
        # Reverse the legend order
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], title='Dataset', loc='lower right', fontsize=10)
        
        # Add grid for readability
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(importance_dir, f'{target.lower()}_{mode}_{model.replace(" ", "_")}_feature_comparison.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # ==== NEW CODE: Create average feature importance across all models ====
    # Calculate average importance for each feature across all models and datasets
    avg_importance = filtered_df.groupby('Feature')['Importance'].mean().reset_index()
    
    # Get top N features by average importance across all models
    top_features_all_models = avg_importance.nlargest(top_n, 'Importance')
    
    # Create dataset-specific averages for these top features
    all_models_comparison = pd.DataFrame(index=top_features_all_models['Feature'].tolist())
    
    for dataset in datasets:
        dataset_df = filtered_df[filtered_df['Dataset'] == dataset]
        if not dataset_df.empty:
            # Calculate average importance across all models for this dataset
            dataset_avg = dataset_df.groupby('Feature')['Importance'].mean()
            # Add to the comparison dataframe
            all_models_comparison[dataset] = dataset_avg.reindex(all_models_comparison.index, fill_value=0)
    
    # Sort by average importance across all datasets
    all_models_comparison['avg'] = all_models_comparison.mean(axis=1)
    all_models_comparison = all_models_comparison.sort_values('avg', ascending=True).drop('avg', axis=1)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Reverse the order of the columns (datasets) in the DataFrame
    all_models_comparison = all_models_comparison[all_models_comparison.columns[::-1]]
    
    # Create the horizontal bar chart
    all_models_comparison.plot(
        kind='barh', 
        ax=ax,
        color=[reversed_colors[i % len(reversed_colors)] for i in range(len(all_models_comparison.columns))],
        width=0.8
    )
    
    # Apply label mapping to y-axis labels
    readable_labels = []
    for feature in all_models_comparison.index:
        readable = feature
        for key in label_mapping:
            if feature == key:
                readable = label_mapping[key]
                break
            elif feature.startswith(key + '_'):
                # Handle one-hot encoded features
                category_value = feature[len(key)+1:]
                readable = f"{label_mapping.get(key, key)}: {category_value}"
                break
        readable_labels.append(readable)
    
    # Set new y-tick labels with improved style
    ax.set_yticklabels(readable_labels, fontsize=10)
    
    # Set consistent x-axis limit
    max_importance = all_models_comparison.values.max()
    ax.set_xlim(0, max_importance * 1.1)  # Add 10% margin
    
    # Add title and labels
    fig_title = f'Average Feature Importance Across All Models\n{title_base}'
    plt.title(fig_title, fontsize=14)
    plt.xlabel('Normalized Importance', fontsize=12)
    
    # Reverse the legend order
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], title='Dataset', loc='lower right', fontsize=10)
    
    # Add grid for readability
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(importance_dir, f'{target.lower()}_{mode}_all_models_feature_comparison.png'), 
                dpi=300, bbox_inches='tight')
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
    
    print(f"Reading evaluation results from {args.results_dir}")
    
    # Set plot style for consistent appearance
    plt.style.use('default')  # Reset to default style
    # Set custom style elements
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],
        'font.weight': 'normal',  # Use normal weight instead of bold
        'axes.facecolor': '#f8f8f8',  # Light gray background
        'figure.facecolor': 'white',
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.labelweight': 'normal',  # Normal weight for axis labels
        'axes.titleweight': 'normal',  # Normal weight for titles
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14
    })
    
    # Load results from CSV files
    performance_path = os.path.join(args.results_dir, 'all_performance_results.csv')
    utility_path = os.path.join(args.results_dir, 'all_utility_results.csv')
    
    if not os.path.exists(performance_path):
        raise FileNotFoundError(f"Performance results file not found at {performance_path}")
    
    if not os.path.exists(utility_path):
        raise FileNotFoundError(f"Utility results file not found at {utility_path}")
    
    performance_df = pd.read_csv(performance_path)
    utility_df = pd.read_csv(utility_path)
    
    # Load feature importance data
    importance_df = None
    if 'feature_importance' in args.plot_types:
        # Check for the consolidated all_feature_importances.csv file
        importance_path = os.path.join(args.results_dir, 'all_feature_importances.csv')
        if os.path.exists(importance_path):
            print(f"Loading feature importance data from {importance_path}")
            importance_df = pd.read_csv(importance_path)
        else:
            # Check if there's a directory with feature importances as fallback
            importance_dir = os.path.join(args.results_dir, 'feature_importances')
            if os.path.isdir(importance_dir):
                # Find all CSV files in the directory
                importance_files = [f for f in os.listdir(importance_dir) if f.endswith('.csv')]
                
                if importance_files:
                    # Load and concatenate all feature importance files
                    importance_dfs = []
                    for file in importance_files:
                        file_path = os.path.join(importance_dir, file)
                        importance_dfs.append(pd.read_csv(file_path))
                    
                    importance_df = pd.concat(importance_dfs, ignore_index=True)
                else:
                    print("No feature importance CSV files found. Skipping feature importance plots.")
            else:
                print("Feature importance data not found. Skipping feature importance plots.")
    
    # Get unique targets and modes from the data
    targets = args.targets or performance_df['Target'].unique()
    modes = args.prediction_modes or performance_df['Mode'].unique()
    
    # Generate plots for each target and mode
    for target in targets:
        for mode in modes:
            # Skip tactical mode for departure delay since it doesn't make sense
            if target == 'DEPARTURE_DELAY_MIN' and mode == 'tactical':
                print(f"Skipping {target} with {mode} mode (not applicable)")
                continue
            
            print(f"Generating plots for {target} ({mode} mode)")
            
            # Performance plots
            if 'performance' in args.plot_types:
                print(f"  Generating performance plots")
                plot_performance_metrics(performance_df, target, mode, args.output_dir)
            
            # Utility plots
            if 'utility' in args.plot_types:
                print(f"  Generating utility plots")
                plot_utility_scores(utility_df, target, mode, args.output_dir)
            
            # Feature importance plots
            if 'feature_importance' in args.plot_types and importance_df is not None:
                print(f"  Generating feature importance plots")
                plot_feature_importances(importance_df, target, mode, args.output_dir, args.top_n_features)
    
    print(f"All plots generated and saved to {args.output_dir}")


if __name__ == "__main__":
    main()

# Example usage:
# python scripts/generate_tstr_plots.py --results_dir tstr_results --output_dir tstr_plots