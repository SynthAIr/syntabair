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

# Default figure format (will be overridden by CLI argument)
FIG_FORMAT = "png"

# Font size scaling factor for academic papers
FONT_SCALE = 1.0  # No scaling
# Note: Value labels use smaller scale (8-10pt base) for less clutter


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
        default=["performance", "utility", "feature_importance", "feature_alignment"],
        choices=["performance", "utility", "feature_importance", "feature_alignment"],
        help="Types of plots to generate"
    )
    parser.add_argument(
        "--top_n_features",
        type=int,
        default=15,
        help="Number of top features to show in feature importance plots"
    )
    parser.add_argument(
        "--format",
        dest="fig_format",
        type=str,
        choices=["png", "pdf", "jpg"],
        default="pdf",
        help="Format of the saved figures"
    )
    parser.add_argument(
        "--font_scale",
        type=float,
        default=1.5,
        help="Font size scaling factor (default: 1.5 for academic papers, 1.0 for standard). Value labels use ~60%% of this scale."
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
    custom_colors = [
        (0.00, 0.63, 0.64),  # Teal
        (0.75, 0.12, 0.24),  # Crimson
        (0.85, 0.37, 0.01),  # Dark orange
        (0.13, 0.55, 0.13),  # Forest green
        (0.80, 0.52, 0.25),  # Ochre
        (0.55, 0.00, 0.26),  # Burgundy
        (0.00, 0.21, 0.41),  # Navy
        (0.64, 0.08, 0.18),  # Dark red
        (0.00, 0.39, 0.25),  # Dark green
        (0.55, 0.27, 0.07),  # Brown
        (0.28, 0.24, 0.55)   # Dark slate
    ]
    
    if exclude_real:
        real_color = (0.00, 0.27, 0.55)  # Deep blue
        required_colors = num_colors - 1
        if required_colors > len(custom_colors):
            additional_colors = sns.color_palette("tab20", required_colors)
            result_colors = custom_colors + additional_colors
            result_colors = result_colors[:required_colors]
        else:
            result_colors = custom_colors[:required_colors]
        return [real_color] + result_colors
    else:
        if num_colors > len(custom_colors):
            additional_colors = sns.color_palette("tab20", num_colors)
            result_colors = custom_colors + additional_colors
            return result_colors[:num_colors]
        else:
            return custom_colors[:num_colors]


def create_model_name_mapping():
    """Mapping between long model names and shorter display names"""
    return {
        'Gradient Boosting': 'GradBoost',
        'Random Forest': 'RandomForest',
        'Linear Regression': 'LinearReg',
        'Support Vector Machine': 'SVM',
        'Neural Network': 'NN',
        'XGBoost': 'XGBoost',
        'LightGBM': 'LightGBM',
        'Decision Tree': 'DecisionTree',
        'K-Nearest Neighbors': 'KNN',
        'Logistic Regression': 'Logistic'
    }


def get_short_model_name(model_name):
    """Get shortened model name for display"""
    mapping = create_model_name_mapping()
    return mapping.get(model_name, model_name)


def create_feature_label_mapping():
    """Mapping between technical column names and human-readable labels"""
    return {
        'IATA_CARRIER_CODE': 'Carrier',
        'DEPARTURE_IATA_AIRPORT_CODE': 'Departure Airport',
        'ARRIVAL_IATA_AIRPORT_CODE': 'Arrival Airport',
        'AIRCRAFT_TYPE_IATA': 'Aircraft Type',
        'AIRPORT_PAIR': 'Airport Pair',
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
    """Plot performance metrics for the given target and prediction mode"""
    filtered_df = performance_df[
        (performance_df['Target'] == target) & 
        (performance_df['Mode'] == mode)
    ]
    if filtered_df.empty:
        print(f"No performance data available for {target} with {mode} mode")
        return
    
    title_base = get_readable_title(target, mode)
    target_mode_dir = os.path.join(output_dir, f"{target.lower()}_{mode}")
    os.makedirs(target_mode_dir, exist_ok=True)
    
    metrics = [
        ('RMSE', 'Root Mean Squared Error'),
        ('MAE', 'Mean Absolute Error'),
        ('R2', 'R² Score')
    ]
    
    for metric, metric_title in metrics:
        plt.figure(figsize=(14, 8))
        models = filtered_df['Model'].unique()
        # Get short model names for display
        short_models = [get_short_model_name(model) for model in models]
        datasets = filtered_df['Dataset'].unique()
        x = np.arange(len(models))
        width = 0.8 / len(datasets)
        has_real = 'Real' in datasets
        colors = get_custom_color_palette(len(datasets), exclude_real=has_real)
        
        for i, dataset in enumerate(datasets):
            dataset_data = filtered_df[filtered_df['Dataset'] == dataset]
            values = []
            for model in models:
                model_data = dataset_data[dataset_data['Model'] == model]
                values.append(model_data[metric].values[0] if not model_data.empty else 0)
            color_index = 0 if (dataset == 'Real' and has_real) else i
            plt.bar(x + i*width, values, width, label=dataset, color=colors[color_index])
        
        plt.xlabel('Model', fontsize=18 * FONT_SCALE)
        plt.ylabel(metric_title, fontsize=18 * FONT_SCALE)
        # Use short model names and rotate if needed
        plt.xticks(x + width*(len(datasets)-1)/2, short_models, fontsize=12 * FONT_SCALE, rotation=45, ha='right')
        plt.yticks(fontsize=14 * FONT_SCALE)
        plt.grid(axis='y', alpha=0.3)
        plt.legend(title='Dataset', title_fontsize=16 * FONT_SCALE, fontsize=14 * FONT_SCALE, loc='best')
        
        # Add value labels with smaller font size
        for i, dataset in enumerate(datasets):
            dataset_data = filtered_df[filtered_df['Dataset'] == dataset]
            for j, model in enumerate(models):
                model_data = dataset_data[dataset_data['Model'] == model]
                if not model_data.empty:
                    value = model_data[metric].values[0]
                    fmt = f'{value:.2f}' if metric == 'R2' else f'{value:.1f}'
                    plt.text(
                        x[j] + i*width,
                        value + 0.01 * max(values),
                        fmt,
                        ha='center',
                        va='bottom',
                        fontsize=8 * FONT_SCALE,  # Smaller font size for values
                        rotation=90 if len(fmt) > 4 else 0  # Rotate long values
                    )
        
        plt.tight_layout()
        
        # Save the plot in the requested format
        plt.savefig(
            os.path.join(
                target_mode_dir,
                f"{target.lower()}_{mode}_{metric.lower()}.{FIG_FORMAT}"
            ),
            format=FIG_FORMAT,
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()


def plot_utility_scores(utility_df, target, mode, output_dir):
    """Plot utility scores for the given target and prediction mode"""
    filtered_df = utility_df[
        (utility_df['Target'] == target) & 
        (utility_df['Mode'] == mode)
    ]
    if filtered_df.empty:
        print(f"No utility data available for {target} with {mode} mode")
        return
    
    title_base = get_readable_title(target, mode)
    target_mode_dir = os.path.join(output_dir, f"{target.lower()}_{mode}")
    os.makedirs(target_mode_dir, exist_ok=True)
    
    datasets = filtered_df['Dataset'].unique()
    models = filtered_df['Model'].unique()
    # Get short model names for heatmap display
    short_models = [get_short_model_name(model) for model in models]
    colors = get_custom_color_palette(len(datasets))
    
    # Heatmap of overall utility scores
    plt.figure(figsize=(12, 9))
    heatmap_data = filtered_df.pivot(index='Model', columns='Dataset', values='Overall Utility')
    heatmap_data = heatmap_data[datasets]
    heatmap_values = heatmap_data.values
    ordered_heatmap = pd.DataFrame(heatmap_values, index=short_models, columns=datasets)
    cmap = plt.cm.YlGnBu
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
        annot_kws={"size": 10 * FONT_SCALE}  # Smaller font size for heatmap values
    )
    ax.set_xlabel('Synthetic Dataset', fontsize=18 * FONT_SCALE, labelpad=10)
    ax.set_ylabel('Model', fontsize=18 * FONT_SCALE, labelpad=10)
    plt.xticks(fontsize=16 * FONT_SCALE)
    plt.yticks(fontsize=16 * FONT_SCALE)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14 * FONT_SCALE)
    cbar.ax.yaxis.label.set_size(16 * FONT_SCALE)
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            target_mode_dir,
            f"{target.lower()}_{mode}_utility_heatmap.{FIG_FORMAT}"
        ),
        format=FIG_FORMAT,
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()
    
    # Bar chart of average utility
    plt.figure(figsize=(10, 7))
    avg_utility = [
        {'Dataset': ds, 'Overall Utility': filtered_df[filtered_df['Dataset'] == ds]['Overall Utility'].mean()}
        for ds in datasets
    ]
    avg_df = pd.DataFrame(avg_utility)
    bars = plt.bar(
        avg_df['Dataset'],
        avg_df['Overall Utility'],
        color=[colors[i] for i in range(len(datasets))],
        width=0.4
    )
    plt.xlabel('Synthetic Dataset', fontsize=18 * FONT_SCALE)
    plt.ylabel('Average Utility Score', fontsize=18 * FONT_SCALE)
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(fontsize=16 * FONT_SCALE)
    plt.yticks(fontsize=16 * FONT_SCALE)
    # Add value labels with smaller font size
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.02,
            f'{height:.2f}',
            ha='center',
            fontsize=10 * FONT_SCALE  # Smaller font size for values
        )
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            target_mode_dir,
            f"{target.lower()}_{mode}_avg_utility.{FIG_FORMAT}"
        ),
        format=FIG_FORMAT,
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()


def plot_feature_importances(importance_df, target, mode, output_dir, top_n=15):
    """Plot feature importances for the given target and prediction mode"""
    filtered_df = importance_df[
        (importance_df['Target'] == target) & 
        (importance_df['Mode'] == mode)
    ]
    if filtered_df.empty:
        print(f"No feature importance data available for {target} with {mode} mode")
        return
    
    title_base = get_readable_title(target, mode)
    importance_dir = os.path.join(output_dir, f"{target.lower()}_{mode}", "feature_importances")
    os.makedirs(importance_dir, exist_ok=True)
    
    label_mapping = create_feature_label_mapping()
    models = filtered_df['Model'].unique()
    datasets = filtered_df['Dataset'].unique()
    has_real = 'Real' in datasets
    colors = get_custom_color_palette(len(datasets), exclude_real=has_real)
    
    # Per-model feature comparisons
    for model in models:
        model_df = filtered_df[filtered_df['Model'] == model]
        if model_df.empty:
            continue
        
        feature_avg = model_df.groupby('Feature')['Importance'].mean().reset_index()
        top_features = feature_avg.nlargest(top_n, 'Importance')['Feature'].tolist()
        comp_df = pd.DataFrame(index=top_features)
        for ds in datasets:
            ds_df = model_df[model_df['Dataset'] == ds]
            comp_df[ds] = ds_df.set_index('Feature')['Importance'].reindex(top_features, fill_value=0)
        
        comp_df['avg'] = comp_df.mean(axis=1)
        comp_df = comp_df.sort_values('avg', ascending=True).drop('avg', axis=1)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        reversed_colors = colors[::-1]
        comp_df[comp_df.columns[::-1]].plot(
            kind='barh',
            ax=ax,
            color=[reversed_colors[i % len(reversed_colors)] for i in range(len(comp_df.columns))],
            width=0.8
        )
        
        readable_labels = []
        for feat in comp_df.index:
            label = feat
            if feat in label_mapping:
                label = label_mapping[feat]
            elif any(feat.startswith(k + '_') for k in label_mapping):
                for k in label_mapping:
                    if feat.startswith(k + '_'):
                        cat = feat[len(k) + 1:]
                        label = f"{label_mapping[k]}: {cat}"
                        break
            readable_labels.append(label)
        ax.set_yticklabels(readable_labels, fontsize=14 * FONT_SCALE)
        ax.set_xlim(0, comp_df.values.max() * 1.1)
        plt.xlabel('Normalized Importance', fontsize=18 * FONT_SCALE)
        plt.xticks(fontsize=14 * FONT_SCALE)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], title='Dataset', loc='lower right', 
                  fontsize=14 * FONT_SCALE, title_fontsize=16 * FONT_SCALE)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        # Use short model name in filename
        short_model_name = get_short_model_name(model)
        plt.savefig(
            os.path.join(
                importance_dir,
                f"{target.lower()}_{mode}_{short_model_name.replace(' ', '_')}_feature_comparison.{FIG_FORMAT}"
            ),
            format=FIG_FORMAT,
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
    
    # Average across all models
    avg_imp = filtered_df.groupby('Feature')['Importance'].mean().reset_index()
    top_all = avg_imp.nlargest(top_n, 'Importance')['Feature']
    all_comp = pd.DataFrame(index=top_all)
    for ds in datasets:
        ds_avg = filtered_df[filtered_df['Dataset'] == ds].groupby('Feature')['Importance'].mean()
        all_comp[ds] = ds_avg.reindex(all_comp.index, fill_value=0)
    all_comp['avg'] = all_comp.mean(axis=1)
    all_comp = all_comp.sort_values('avg', ascending=True).drop('avg', axis=1)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    all_comp[all_comp.columns[::-1]].plot(
        kind='barh',
        ax=ax,
        color=[reversed_colors[i % len(reversed_colors)] for i in range(len(all_comp.columns))],
        width=0.8
    )
    readable_labels = []
    for feat in all_comp.index:
        label = feat
        if feat in label_mapping:
            label = label_mapping[feat]
        elif any(feat.startswith(k + '_') for k in label_mapping):
            for k in label_mapping:
                if feat.startswith(k + '_'):
                    cat = feat[len(k) + 1:]
                    label = f"{label_mapping[k]}: {cat}"
                    break
        readable_labels.append(label)
    ax.set_yticklabels(readable_labels, fontsize=14 * FONT_SCALE)
    ax.set_xlim(0, all_comp.values.max() * 1.1)
    plt.xlabel('Normalized Importance', fontsize=18 * FONT_SCALE)
    plt.xticks(fontsize=14 * FONT_SCALE)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], title='Dataset', loc='lower right', 
              fontsize=14 * FONT_SCALE, title_fontsize=16 * FONT_SCALE)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            importance_dir,
            f"{target.lower()}_{mode}_all_models_feature_comparison.{FIG_FORMAT}"
        ),
        format=FIG_FORMAT,
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()


def calculate_feature_importance_alignment(importance_df, target, mode, output_dir):
    """Calculate and plot feature importance alignment scores between real and synthetic data."""
    filtered_df = importance_df[
        (importance_df['Target'] == target) & 
        (importance_df['Mode'] == mode)
    ]
    if filtered_df.empty:
        print(f"No feature importance data available for {target} with {mode} mode")
        return None
    
    title_base = get_readable_title(target, mode)
    alignment_dir = os.path.join(output_dir, f"{target.lower()}_{mode}")
    os.makedirs(alignment_dir, exist_ok=True)
    
    models = filtered_df['Model'].unique()
    # Get short model names for alignment plots
    short_models = [get_short_model_name(model) for model in models]
    datasets = filtered_df['Dataset'].unique()
    if 'Real' not in datasets:
        print(f"Real dataset not found for {target} with {mode}")
        return None
    synthetic_datasets = [d for d in datasets if d != 'Real']
    if not synthetic_datasets:
        print(f"No synthetic datasets for {target} with {mode}")
        return None
    
    alignment_scores = []
    for model in models:
        model_df = filtered_df[filtered_df['Model'] == model]
        if model_df.empty:
            continue
        real_vec = model_df[model_df['Dataset'] == 'Real'].set_index('Feature')['Importance']
        for ds in synthetic_datasets:
            synth_vec = model_df[model_df['Dataset'] == ds].set_index('Feature')['Importance']
            all_feats = list(set(real_vec.index) | set(synth_vec.index))
            rv = np.array([real_vec.get(f, 0) for f in all_feats])
            sv = np.array([synth_vec.get(f, 0) for f in all_feats])
            cosine_sim = (
                np.dot(rv, sv) / (np.linalg.norm(rv) * np.linalg.norm(sv))
                if np.linalg.norm(rv) * np.linalg.norm(sv) > 0 else 0
            )
            corr = (
                np.corrcoef(rv, sv)[0, 1]
                if len(all_feats) > 1 else 0
            )
            if np.isnan(corr):
                corr = 0
            alignment_scores.append({
                'Target': target,
                'Mode': mode,
                'Model': model,
                'Dataset': ds,
                'CosineSimScore': cosine_sim,
                'CorrelationScore': corr,
                'NumFeatures': len(all_feats)
            })
    
    alignment_df = pd.DataFrame(alignment_scores)
    if alignment_df.empty:
        print(f"Could not calculate alignment for {target} with {mode}")
        return None
    
    avg_scores = alignment_df.groupby('Dataset')[['CosineSimScore', 'CorrelationScore']].mean().reset_index()
    
    # Heatmap of cosine similarity with short model names
    plt.figure(figsize=(12, 8))
    cos_heat = alignment_df.pivot(index='Model', columns='Dataset', values='CosineSimScore')
    cos_heat = cos_heat[synthetic_datasets]
    # Replace model names with short versions for the heatmap
    cos_heat.index = [get_short_model_name(model) for model in cos_heat.index]
    ax = sns.heatmap(
        cos_heat,
        annot=True,
        cmap=plt.cm.viridis,
        vmin=0,
        vmax=1,
        fmt='.2f',
        cbar_kws={'label': 'Cosine Similarity Score (higher is better)'},
        linewidths=0.5,
        linecolor='white',
        annot_kws={"size": 10 * FONT_SCALE}  # Smaller font size for heatmap values
    )
    plt.xlabel('Synthetic Dataset', fontsize=18 * FONT_SCALE)
    plt.ylabel('Model', fontsize=18 * FONT_SCALE)
    plt.xticks(fontsize=16 * FONT_SCALE)
    plt.yticks(fontsize=16 * FONT_SCALE)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14 * FONT_SCALE)
    cbar.ax.yaxis.label.set_size(16 * FONT_SCALE)
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            alignment_dir,
            f"{target.lower()}_{mode}_cosine_alignment_heatmap.{FIG_FORMAT}"
        ),
        format=FIG_FORMAT,
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()
    
    # Bar chart of average cosine similarity
    ordered_avg = pd.concat([
        avg_scores[avg_scores['Dataset'] == ds]
        for ds in synthetic_datasets
    ], ignore_index=True)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        ordered_avg['Dataset'],
        ordered_avg['CosineSimScore'],
        color=get_custom_color_palette(len(synthetic_datasets)),
        width=0.4
    )
    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            h + 0.02,
            f'{h:.2f}',
            ha='center',
            fontsize=16 * FONT_SCALE
        )
    plt.xlabel('Synthetic Dataset', fontsize=18 * FONT_SCALE)
    plt.ylabel('Cosine Similarity Score', fontsize=18 * FONT_SCALE)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(fontsize=16 * FONT_SCALE)
    plt.yticks(fontsize=16 * FONT_SCALE)
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            alignment_dir,
            f"{target.lower()}_{mode}_avg_alignment_score.{FIG_FORMAT}"
        ),
        format=FIG_FORMAT,
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()
    
    return alignment_df


def main():
    """Main function to generate plots from CSV results"""
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Override global figure format and font scale
    global FIG_FORMAT, FONT_SCALE
    FIG_FORMAT = args.fig_format
    FONT_SCALE = args.font_scale
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, "plots")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Reading evaluation results from {args.results_dir}")
    print(f"Using font scale factor: {FONT_SCALE}")
    
    # Set matplotlib parameters with scaled font sizes
    plt.style.use('default')  # Reset to default style
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],
        'font.weight': 'normal',
        'axes.facecolor': '#f8f8f8',
        'figure.facecolor': 'white',
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.labelweight': 'normal',
        'axes.titleweight': 'normal',
        'xtick.labelsize': 14 * FONT_SCALE,
        'ytick.labelsize': 14 * FONT_SCALE,
        'axes.labelsize': 18 * FONT_SCALE,
        'axes.titlesize': 20 * FONT_SCALE,
        'legend.fontsize': 14 * FONT_SCALE,
        'legend.title_fontsize': 16 * FONT_SCALE,
        'font.size': 14 * FONT_SCALE,
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'grid.linewidth': 0.8
    })
    
    performance_path = os.path.join(args.results_dir, 'all_performance_results.csv')
    utility_path = os.path.join(args.results_dir, 'all_utility_results.csv')
    if not os.path.exists(performance_path):
        raise FileNotFoundError(f"Performance results file not found at {performance_path}")
    if not os.path.exists(utility_path):
        raise FileNotFoundError(f"Utility results file not found at {utility_path}")
    
    performance_df = pd.read_csv(performance_path)
    utility_df = pd.read_csv(utility_path)
    
    importance_df = None
    if 'feature_importance' in args.plot_types:
        importance_path = os.path.join(args.results_dir, 'all_feature_importances.csv')
        if os.path.exists(importance_path):
            print(f"Loading feature importance data from {importance_path}")
            importance_df = pd.read_csv(importance_path)
        else:
            importance_dir = os.path.join(args.results_dir, 'feature_importances')
            if os.path.isdir(importance_dir):
                files = [f for f in os.listdir(importance_dir) if f.endswith('.csv')]
                if files:
                    dfs = [pd.read_csv(os.path.join(importance_dir, f)) for f in files]
                    importance_df = pd.concat(dfs, ignore_index=True)
                else:
                    print("No feature importance CSV files found. Skipping.")
            else:
                print("Feature importance data not found. Skipping.")
    
    targets = args.targets or performance_df['Target'].unique()
    modes = args.prediction_modes or performance_df['Mode'].unique()
    
    for target in targets:
        for mode in modes:
            if target == 'DEPARTURE_DELAY_MIN' and mode == 'tactical':
                print(f"Skipping {target} with {mode} mode (not applicable)")
                continue
            print(f"Generating plots for {target} ({mode} mode)")
            if 'performance' in args.plot_types:
                plot_performance_metrics(performance_df, target, mode, args.output_dir)
            if 'utility' in args.plot_types:
                plot_utility_scores(utility_df, target, mode, args.output_dir)
            if 'feature_importance' in args.plot_types and importance_df is not None:
                plot_feature_importances(importance_df, target, mode, args.output_dir, args.top_n_features)
            if 'feature_alignment' in args.plot_types and importance_df is not None:
                calculate_feature_importance_alignment(importance_df, target, mode, args.output_dir)
    
    print(f"All plots generated and saved to {args.output_dir}")


if __name__ == "__main__":
    main()


# Example usage:
# python scripts/generate_tstr_plots.py --results_dir tstr_results --output_dir tstr_plots --font_scale 1.5
# For even larger fonts for presentations: --font_scale 2.0
# For standard size (original): --font_scale 1.0