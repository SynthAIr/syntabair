import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
import umap.umap_ as umap
import hdbscan
from matplotlib.colors import ListedColormap

from syntabair.preprocessing import preprocess_flight_data_for_prediction


# Configuration constants
DEFAULT_OUTPUT_DIR = Path("./embeddings/anomaly_detection")
DEFAULT_RANDOM_SEED = 42
DEFAULT_SAMPLE_SIZE = 100_000
DEFAULT_UMAP_MIN_DIST = 0.1
DEFAULT_UMAP_N_NEIGHBORS = 15
DEFAULT_CONTAMINATION = 0.05
DEFAULT_MIN_CLUSTER_SIZE = 2200
DEFAULT_HDBSCAN_MIN_SAMPLES = 100


def get_custom_color_palette(num_colors: int) -> List[Tuple[float, float, float]]:
    """Get a custom color palette for visualizations."""
    base_palette = [
        (0.00, 0.63, 0.64),  # Teal
        (0.75, 0.12, 0.24),  # Crimson
        (0.85, 0.37, 0.01),  # Dark orange
        (0.13, 0.55, 0.13),  # Forest green
        (0.00, 0.21, 0.41),  # Navy
        (0.55, 0.00, 0.26),  # Burgundy
        (0.80, 0.52, 0.25),  # Ochre
        (0.64, 0.08, 0.18),  # Dark red
        (0.00, 0.39, 0.25),  # Dark green
        (0.55, 0.27, 0.07),  # Brown
        (0.28, 0.24, 0.55),  # Dark slate
    ]
    if num_colors <= len(base_palette):
        return base_palette[:num_colors]
    # Fallback to matplotlib's tab20 for extra colors
    extra = sns.color_palette("tab20", num_colors - len(base_palette))
    return base_palette + extra


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare anomaly detection using Isolation Forest and HDBSCAN"
    )
    parser.add_argument("--embeddings", type=Path, required=True,
                        help="Path to TabSyn embeddings file (e.g., train_z.npy)")
    parser.add_argument("--data", type=Path, required=True,
                        help="Path to flight data CSV file")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to save outputs (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
                        help=f"Number of records to sample (default: {DEFAULT_SAMPLE_SIZE})")
    parser.add_argument("--min-dist", type=float, default=DEFAULT_UMAP_MIN_DIST,
                        help=f"UMAP min_dist parameter (default: {DEFAULT_UMAP_MIN_DIST})")
    parser.add_argument("--n-neighbors", type=int, default=DEFAULT_UMAP_N_NEIGHBORS,
                        help=f"UMAP n_neighbors parameter (default: {DEFAULT_UMAP_N_NEIGHBORS})")
    parser.add_argument("--contamination", type=float, default=DEFAULT_CONTAMINATION,
                        help=f"Expected proportion of anomalies for Isolation Forest (default: {DEFAULT_CONTAMINATION})")
    parser.add_argument("--min-cluster-size", type=int, default=DEFAULT_MIN_CLUSTER_SIZE,
                        help=f"Minimum cluster size for HDBSCAN (default: {DEFAULT_MIN_CLUSTER_SIZE})")
    parser.add_argument("--min-samples", type=int, default=DEFAULT_HDBSCAN_MIN_SAMPLES,
                        help=f"Min samples for HDBSCAN (default: {DEFAULT_HDBSCAN_MIN_SAMPLES})")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED,
                        help=f"Random seed (default: {DEFAULT_RANDOM_SEED})")
    return parser.parse_args()


def load_data(
    embeddings_path: Path,
    data_path: Path,
    sample_size: Optional[int],
    seed: int
) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, Union[np.ndarray, List[str]]]]:
    """Load embeddings and flight data with optional sampling."""
    logging.info(f"Loading embeddings from %s", embeddings_path)
    embeddings = np.load(embeddings_path)
    if embeddings.ndim == 3:
        logging.info(f"Reshaping embeddings from {embeddings.shape}")
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
        logging.info(f"to {embeddings.shape}")

    logging.info("Loading flight data from %s", data_path)
    df = pd.read_csv(data_path)
    df = preprocess_flight_data_for_prediction(df)

    if len(df) != len(embeddings):
        logging.warning(
            "Data length mismatch: flights=%d, embeddings=%d",
            len(df), len(embeddings)
        )
        n = min(len(df), len(embeddings))
        df, embeddings = df.iloc[:n], embeddings[:n]

    if sample_size and sample_size < len(embeddings):
        np.random.seed(seed)
        idx = np.random.choice(len(embeddings), sample_size, replace=False)
        embeddings = embeddings[idx]
        df = df.iloc[idx].reset_index(drop=True)
        logging.info("Sampled %d records", sample_size)
    
    # Extract metadata
    metadata = {}
    
    # Identify carrier column
    carrier_cols = [col for col in df.columns if 'CARRIER' in col or 'AIRLINE' in col]
    if carrier_cols:
        carrier_col = carrier_cols[0]
        metadata['Carrier'] = df[carrier_col].values
        logging.info(f"Using '{carrier_col}' as carrier column")
    else:
        logging.warning("No carrier column found")
        metadata['Carrier'] = np.array(['Unknown'] * len(df))
    
    # Identify departure airport column
    dep_airport_cols = [col for col in df.columns if 'DEPART' in col and 'AIRPORT' in col]
    if dep_airport_cols:
        dep_airport_col = dep_airport_cols[0]
        metadata['Departure_Airport'] = df[dep_airport_col].values
        logging.info(f"Using '{dep_airport_col}' as departure airport column")
    else:
        logging.warning("No departure airport column found")
        metadata['Departure_Airport'] = np.array(['Unknown'] * len(df))
    
    # Identify arrival airport column
    arr_airport_cols = [col for col in df.columns if 'ARRIV' in col and 'AIRPORT' in col]
    if arr_airport_cols:
        arr_airport_col = arr_airport_cols[0]
        metadata['Arrival_Airport'] = df[arr_airport_col].values
        logging.info(f"Using '{arr_airport_col}' as arrival airport column")
    else:
        logging.warning("No arrival airport column found")
        metadata['Arrival_Airport'] = np.array(['Unknown'] * len(df))
    
    # Identify aircraft column
    aircraft_cols = [col for col in df.columns if 'AIRCRAFT' in col or 'EQUIP' in col]
    if aircraft_cols:
        aircraft_col = aircraft_cols[0]
        metadata['Aircraft'] = df[aircraft_col].values
        logging.info(f"Using '{aircraft_col}' as aircraft column")
    else:
        logging.warning("No aircraft column found")
        metadata['Aircraft'] = np.array(['Unknown'] * len(df))
    
    return embeddings, df, metadata


def apply_umap(
    embeddings: np.ndarray, 
    min_dist: float = DEFAULT_UMAP_MIN_DIST, 
    n_neighbors: int = DEFAULT_UMAP_N_NEIGHBORS, 
    seed: int = DEFAULT_RANDOM_SEED
) -> np.ndarray:
    """Apply UMAP for dimensionality reduction."""
    logging.info(f"Applying UMAP with min_dist={min_dist}, n_neighbors={n_neighbors}")
    try:
        reducer = umap.UMAP(
            n_components=2,
            random_state=seed,
            min_dist=min_dist,
            n_neighbors=n_neighbors
        )
        reduced = reducer.fit_transform(embeddings)
        logging.info("UMAP dimensionality reduction completed")
        return reduced
    except Exception as e:
        logging.error(f"Error applying UMAP: {e}")
        raise


def apply_isolation_forest(
    embeddings: np.ndarray,
    contamination: float = DEFAULT_CONTAMINATION,
    seed: int = DEFAULT_RANDOM_SEED
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Isolation Forest for anomaly detection."""
    logging.info(f"Applying Isolation Forest with contamination={contamination}")
    
    # Create and fit the model
    clf = IsolationForest(
        contamination=contamination, 
        random_state=seed,
        n_estimators=100,
        max_samples='auto',
        n_jobs=-1
    )
    
    # Fit the model and get predictions (-1 for anomalies, 1 for normal)
    anomaly_scores = clf.fit_predict(embeddings)
    
    # Convert to binary classification (True for anomalies, False for normal)
    is_anomaly = anomaly_scores == -1
    anomaly_count = np.sum(is_anomaly)
    anomaly_percentage = 100 * anomaly_count / len(is_anomaly)
    
    logging.info(f"Isolation Forest detected {anomaly_count} anomalies ({anomaly_percentage:.2f}% of total)")
    
    # Get decision function scores (negative means more anomalous)
    anomaly_decision = clf.decision_function(embeddings)
    
    return is_anomaly, anomaly_decision


def apply_hdbscan(
    embeddings: np.ndarray, 
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: int = DEFAULT_HDBSCAN_MIN_SAMPLES
) -> Tuple[np.ndarray, hdbscan.HDBSCAN]:
    """Apply HDBSCAN for density-based clustering."""
    logging.info(f"Applying HDBSCAN with min_cluster_size={min_cluster_size}, min_samples={min_samples}")
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        gen_min_span_tree=True
    )
    clusters = clusterer.fit_predict(embeddings)
    
    # Points labeled as -1 are considered noise/anomalies in HDBSCAN
    is_noise = clusters == -1
    n_noise = np.sum(is_noise)
    noise_percentage = 100 * n_noise / len(clusters)
    
    logging.info(f"HDBSCAN detected {n_noise} noise points ({noise_percentage:.2f}% of total)")
    
    return is_noise, clusterer


def compare_anomaly_detection(
    embeddings: np.ndarray,
    flight_data: pd.DataFrame,
    metadata: Dict[str, Union[np.ndarray, List[str]]],
    output_dir: Path = None,
    min_dist: float = DEFAULT_UMAP_MIN_DIST,
    n_neighbors: int = DEFAULT_UMAP_N_NEIGHBORS,
    contamination: float = DEFAULT_CONTAMINATION,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: int = DEFAULT_HDBSCAN_MIN_SAMPLES,
    seed: int = DEFAULT_RANDOM_SEED
) -> Path:
    """
    Compare anomaly detection using Isolation Forest and HDBSCAN.
    
    Args:
        embeddings (np.ndarray): Embedding vectors
        flight_data (pd.DataFrame): Flight data
        metadata (dict): Additional metadata
        output_dir (Path, optional): Directory to save the figure and data
        min_dist (float): UMAP min_dist parameter
        n_neighbors (int): UMAP n_neighbors parameter
        contamination (float): Expected proportion of anomalies for Isolation Forest
        min_cluster_size (int): Minimum cluster size for HDBSCAN
        min_samples (int): Min samples for HDBSCAN
        seed (int): Random seed
        
    Returns:
        Path: Path to the saved visualization
    """
    if output_dir is None:
        output_dir = Path(DEFAULT_OUTPUT_DIR)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "anomaly_comparison.png"
    
    logging.info("Starting anomaly detection comparison")
    
    # Apply UMAP for dimensionality reduction
    umap_reduced = apply_umap(embeddings, min_dist, n_neighbors, seed)
    
    # Apply Isolation Forest
    iforest_anomalies, iforest_scores = apply_isolation_forest(embeddings, contamination, seed)
    
    # Apply HDBSCAN on the UMAP-reduced embeddings
    # Note: HDBSCAN works better in lower-dimensional space for density estimation
    hdbscan_anomalies, _ = apply_hdbscan(umap_reduced, min_cluster_size, min_samples)
    
    # Compare results
    both_anomalies = np.logical_and(iforest_anomalies, hdbscan_anomalies)
    only_iforest = np.logical_and(iforest_anomalies, ~hdbscan_anomalies)
    only_hdbscan = np.logical_and(~iforest_anomalies, hdbscan_anomalies)
    normal = np.logical_and(~iforest_anomalies, ~hdbscan_anomalies)
    
    # Count agreement/disagreement between methods
    n_both = np.sum(both_anomalies)
    n_only_iforest = np.sum(only_iforest)
    n_only_hdbscan = np.sum(only_hdbscan)
    n_normal = np.sum(normal)
    
    total = n_both + n_only_iforest + n_only_hdbscan + n_normal
    percent_both = 100 * n_both / total
    percent_only_iforest = 100 * n_only_iforest / total
    percent_only_hdbscan = 100 * n_only_hdbscan / total
    percent_normal = 100 * n_normal / total
    
    logging.info(f"Agreement between methods:")
    logging.info(f"  Both methods: {n_both} points ({percent_both:.2f}%)")
    logging.info(f"  Only Isolation Forest: {n_only_iforest} points ({percent_only_iforest:.2f}%)")
    logging.info(f"  Only HDBSCAN: {n_only_hdbscan} points ({percent_only_hdbscan:.2f}%)")
    logging.info(f"  Normal in both: {n_normal} points ({percent_normal:.2f}%)")
    
    # Create a categorical variable for the combined results
    # 0: Normal in both, 1: Only IForest, 2: Only HDBSCAN, 3: Both anomaly
    combined_results = np.zeros(len(embeddings), dtype=int)
    combined_results[only_iforest] = 1
    combined_results[only_hdbscan] = 2
    combined_results[both_anomalies] = 3
    
    # Create dataframe for visualization
    df_plot = pd.DataFrame({
        'UMAP1': umap_reduced[:, 0],
        'UMAP2': umap_reduced[:, 1],
        'IForest_Anomaly': iforest_anomalies,
        'IForest_Score': iforest_scores,
        'HDBSCAN_Anomaly': hdbscan_anomalies,
        'Combined_Result': combined_results,
        'Carrier': metadata['Carrier'],
        'Departure': metadata['Departure_Airport'],
        'Arrival': metadata['Arrival_Airport'],
        'Aircraft': metadata['Aircraft'],
        'Delay': flight_data['DEPARTURE_DELAY_MIN'],
        'Duration': flight_data['SCHEDULED_DURATION_MIN']
    })
    
    # Add turnaround if available
    if 'TURNAROUND_MIN' in flight_data.columns and flight_data['TURNAROUND_MIN'].notna().any():
        df_plot['Turnaround'] = flight_data['TURNAROUND_MIN']
    
    # Create visualization
    plt.figure(figsize=(20, 16))
    
    # Create a 2x2 grid for the visualizations
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))
    fig.patch.set_facecolor('white')  # Set figure background to white
    
    # Get custom colors
    colors = get_custom_color_palette(4)
    normal_color = (0.7, 0.7, 0.7)  # Grey for normal points
    iforest_color = colors[0]  # Teal for Isolation Forest
    hdbscan_color = colors[1]  # Crimson for HDBSCAN
    both_color = colors[2]  # Dark orange for both
    
    # 1. Isolation Forest Results (top left)
    ax1 = axs[0, 0]
    ax1.set_facecolor('none')
    
    # Plot normal points first (small and semi-transparent)
    ax1.scatter(
        df_plot.loc[~iforest_anomalies, 'UMAP1'],
        df_plot.loc[~iforest_anomalies, 'UMAP2'],
        color=normal_color,
        alpha=0.3,
        s=20,
        edgecolor='none',
        label='Normal'
    )
    
    # Plot anomalies (larger and more opaque)
    scatter = ax1.scatter(
        df_plot.loc[iforest_anomalies, 'UMAP1'],
        df_plot.loc[iforest_anomalies, 'UMAP2'],
        c=df_plot.loc[iforest_anomalies, 'IForest_Score'],
        cmap='coolwarm_r',  # Reversed coolwarm - red for more anomalous
        alpha=0.8,
        s=60,
        edgecolor='white',
        linewidth=0.3,
        label='Anomalies'
    )
    
    ax1.set_title('Isolation Forest Anomalies', fontsize=16, fontweight='bold')
    ax1.set_xlabel('UMAP Dimension 1', fontsize=13, fontweight='bold')
    ax1.set_ylabel('UMAP Dimension 2', fontsize=13, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=11)
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax1)
    cbar.set_label('Anomaly Score (lower = more anomalous)', fontsize=10)
    
    # Add legend
    ax1.legend(loc='upper right', fontsize=12)
    
    # Add border
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_color('lightgray')
        spine.set_linewidth(1)
    
    # 2. HDBSCAN Results (top right)
    ax2 = axs[0, 1]
    ax2.set_facecolor('none')
    
    # Plot non-noise points first
    ax2.scatter(
        df_plot.loc[~hdbscan_anomalies, 'UMAP1'],
        df_plot.loc[~hdbscan_anomalies, 'UMAP2'],
        color=normal_color,
        alpha=0.3,
        s=20,
        edgecolor='none',
        label='Normal'
    )
    
    # Plot noise points (anomalies)
    ax2.scatter(
        df_plot.loc[hdbscan_anomalies, 'UMAP1'],
        df_plot.loc[hdbscan_anomalies, 'UMAP2'],
        color=hdbscan_color,
        alpha=0.8,
        s=60,
        edgecolor='white',
        linewidth=0.3,
        label='Noise (Anomalies)'
    )
    
    ax2.set_title('HDBSCAN Anomalies (Noise Points)', fontsize=16, fontweight='bold')
    ax2.set_xlabel('UMAP Dimension 1', fontsize=13, fontweight='bold')
    ax2.set_ylabel('UMAP Dimension 2', fontsize=13, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=11)
    ax2.legend(loc='upper right', fontsize=12)
    
    # Add border
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_color('lightgray')
        spine.set_linewidth(1)
    
    # 3. Combined Results (bottom left)
    ax3 = axs[1, 0]
    ax3.set_facecolor('none')
    
    # Create a custom colormap for the combined results
    category_colors = [normal_color, iforest_color, hdbscan_color, both_color]
    cmap = ListedColormap(category_colors)
    
    # Plot all points with colors based on the combined results
    scatter3 = ax3.scatter(
        df_plot['UMAP1'],
        df_plot['UMAP2'],
        c=df_plot['Combined_Result'],
        cmap=cmap,
        alpha=0.7,
        s=50,
        edgecolor='white',
        linewidth=0.3
    )
    
    ax3.set_title('Combined Anomaly Detection Results', fontsize=16, fontweight='bold')
    ax3.set_xlabel('UMAP Dimension 1', fontsize=13, fontweight='bold')
    ax3.set_ylabel('UMAP Dimension 2', fontsize=13, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.tick_params(axis='both', which='major', labelsize=11)
    
    # Create a custom legend for the combined results
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=normal_color, label='Normal in Both'),
        Patch(facecolor=iforest_color, label='Only Isolation Forest'),
        Patch(facecolor=hdbscan_color, label='Only HDBSCAN'),
        Patch(facecolor=both_color, label='Both Methods')
    ]
    ax3.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # Add border
    for spine in ax3.spines.values():
        spine.set_visible(True)
        spine.set_color('lightgray')
        spine.set_linewidth(1)
    
    # 4. Venn diagram of agreement (bottom right)
    ax4 = axs[1, 1]
    ax4.set_facecolor('none')
    
    # Create a simple statistics summary
    stats_text = (
        f"Anomaly Detection Comparison\n\n"
        f"Total points: {total:,}\n\n"
        f"Identified as anomalies by both methods: {n_both:,} ({percent_both:.1f}%)\n\n"
        f"Identified only by Isolation Forest: {n_only_iforest:,} ({percent_only_iforest:.1f}%)\n\n"
        f"Identified only by HDBSCAN: {n_only_hdbscan:,} ({percent_only_hdbscan:.1f}%)\n\n"
        f"Normal in both methods: {n_normal:,} ({percent_normal:.1f}%)\n\n"
        f"Isolation Forest anomaly rate: {(n_both + n_only_iforest)/total*100:.1f}%\n\n"
        f"HDBSCAN anomaly rate: {(n_both + n_only_hdbscan)/total*100:.1f}%\n\n"
        f"Agreement rate: {(n_both + n_normal)/total*100:.1f}%"
    )
    
    # Add text box
    ax4.text(
        0.5, 0.5, stats_text,
        ha='center', va='center',
        fontsize=14,
        bbox=dict(
            boxstyle='round,pad=1', 
            facecolor='white', 
            edgecolor='lightgray', 
            alpha=0.9
        ),
        transform=ax4.transAxes
    )
    
    ax4.set_title('Agreement Between Methods', fontsize=16, fontweight='bold')
    ax4.axis('off')  # Hide axes
    
    # Add main title
    plt.suptitle(
        'Comparison of Anomaly Detection Methods: Isolation Forest vs. HDBSCAN',
        fontsize=20, fontweight='bold', y=0.98
    )
    
    # Improve layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save with high resolution
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    
    logging.info(f"Saved anomaly comparison visualization to {output_path}")
    
    # Save the combined results as CSV
    csv_path = output_path.with_suffix('.csv')
    df_plot.to_csv(csv_path, index=False)
    
    # Save agreement statistics
    # Convert NumPy integers to Python integers to make them JSON serializable
    stats = {
        'Total_Points': int(total),
        'Both_Anomaly': int(n_both),
        'Only_IForest': int(n_only_iforest),
        'Only_HDBSCAN': int(n_only_hdbscan),
        'Normal_Both': int(n_normal),
        'Percent_Both_Anomaly': float(percent_both),
        'Percent_Only_IForest': float(percent_only_iforest),
        'Percent_Only_HDBSCAN': float(percent_only_hdbscan),
        'Percent_Normal_Both': float(percent_normal),
        'IForest_Anomaly_Rate': float((n_both + n_only_iforest)/total*100),
        'HDBSCAN_Anomaly_Rate': float((n_both + n_only_hdbscan)/total*100),
        'Agreement_Rate': float((n_both + n_normal)/total*100)
    }
    
    stats_path = output_dir / "anomaly_comparison_stats.json"
    import json
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
    
    # Save as CSV for easier analysis
    stats_csv_path = output_dir / "anomaly_comparison_stats.csv"
    pd.DataFrame([stats]).to_csv(stats_csv_path, index=False)
    
    logging.info(f"Saved comparison data to {csv_path} and statistics to {stats_path}")
    
    return output_path


def main():
    """Main function to orchestrate the anomaly detection comparison."""
    setup_logging()
    args = parse_args()
    
    # Load data
    embeddings, flight_data, metadata = load_data(
        args.embeddings, args.data, args.sample_size, args.seed
    )
    
    # Run anomaly comparison
    compare_anomaly_detection(
        embeddings=embeddings,
        flight_data=flight_data,
        metadata=metadata,
        output_dir=args.output_dir,
        min_dist=args.min_dist,
        n_neighbors=args.n_neighbors,
        contamination=args.contamination,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        seed=args.seed
    )
    
    logging.info("Anomaly detection comparison complete")


if __name__ == "__main__":
    main()