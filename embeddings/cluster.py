import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import umap.umap_ as umap
import hdbscan
import seaborn as sns  # For color palette fallback

from syntabair.preprocessing import preprocess_flight_data_for_prediction


# Add the custom color palette function
def get_custom_color_palette(num_colors: int) -> List[Tuple[float, float, float]]:
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


# Configuration constants
DEFAULT_OUTPUT_DIR = Path("./embeddings/cluster_analysis")
DEFAULT_RANDOM_SEED = 42
DEFAULT_SAMPLE_SIZE = 100_000
DEFAULT_UMAP_MIN_DIST = 0.1
DEFAULT_UMAP_N_NEIGHBORS = 15
DEFAULT_MAX_CLUSTERS = 10
DEFAULT_MIN_CLUSTER_SIZE = 2200
DEFAULT_HDBSCAN_MIN_SAMPLES = 100


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate TabSyn embedding cluster analysis visualization"
    )
    # Existing arguments
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
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED,
                        help=f"Random seed (default: {DEFAULT_RANDOM_SEED})")
    parser.add_argument("--max-clusters", type=int, default=DEFAULT_MAX_CLUSTERS,
                        help=f"Maximum number of clusters to consider (default: {DEFAULT_MAX_CLUSTERS})")
    
    # Add new HDBSCAN parameters
    parser.add_argument("--min-cluster-size", type=int, default=DEFAULT_MIN_CLUSTER_SIZE,
                        help=f"Minimum cluster size for HDBSCAN (default: {DEFAULT_MIN_CLUSTER_SIZE})")
    parser.add_argument("--min-samples", type=int, default=DEFAULT_HDBSCAN_MIN_SAMPLES,
                        help=f"Min samples for HDBSCAN (default: {DEFAULT_HDBSCAN_MIN_SAMPLES})")
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


def apply_umap(embeddings: np.ndarray, min_dist: float, n_neighbors: int, seed: int) -> np.ndarray:
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


def embedding_cluster_analysis(
    embeddings: np.ndarray,
    flight_data: pd.DataFrame,
    metadata: Dict[str, Union[np.ndarray, List[str]]],
    output_dir: Path = None,
    umap_reduced: np.ndarray = None,
    min_dist: float = DEFAULT_UMAP_MIN_DIST,
    n_neighbors: int = DEFAULT_UMAP_N_NEIGHBORS,
    max_clusters: int = DEFAULT_MAX_CLUSTERS,
    seed: int = DEFAULT_RANDOM_SEED
) -> Path:
    """
    Use clustering to identify and analyze patterns in the embeddings.
    
    Args:
        embeddings (np.ndarray): Embedding vectors
        flight_data (pd.DataFrame): Flight data
        metadata (dict): Additional metadata
        output_dir (Path, optional): Directory to save the figure and data
        umap_reduced (np.ndarray, optional): Pre-computed UMAP reduction
        min_dist (float): UMAP min_dist parameter
        n_neighbors (int): UMAP n_neighbors parameter
        max_clusters (int): Maximum number of clusters to consider
        seed (int): Random seed
        
    Returns:
        Path: Path to the saved visualization
    """
    if output_dir is None:
        output_dir = Path(DEFAULT_OUTPUT_DIR)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "embedding_clusters.png"
    
    logging.info("Starting embedding cluster analysis")
    
    # Apply UMAP if not provided
    if umap_reduced is None:
        reduced = apply_umap(embeddings, min_dist, n_neighbors, seed)
    else:
        reduced = umap_reduced
    
    # Find optimal number of clusters
    # optimal_k = find_optimal_clusters(embeddings, max_clusters, seed)
    optimal_k = 7
    
    # Apply k-means with optimal k
    logging.info(f"Applying KMeans with {optimal_k} clusters")
    kmeans = KMeans(n_clusters=optimal_k, random_state=seed, n_init=10)
    clusters = kmeans.fit_predict(embeddings)
    cluster_labels = clusters.copy()
    
    # Create dataframe for plotting
    df_plot = pd.DataFrame({
        'UMAP1': reduced[:, 0],
        'UMAP2': reduced[:, 1],
        'Cluster': [f'Cluster {c+1}' for c in clusters],
        'Carrier': metadata['Carrier'],
        'Departure': metadata['Departure_Airport'],
        'Arrival': metadata['Arrival_Airport'],
        'Aircraft': metadata['Aircraft'],
        'Delay': flight_data['DEPARTURE_DELAY_MIN'],
        'Duration': flight_data['SCHEDULED_DURATION_MIN'],
    })
    
    # Add turnaround if available
    if 'TURNAROUND_MIN' in flight_data.columns and flight_data['TURNAROUND_MIN'].notna().any():
        df_plot['Turnaround'] = flight_data['TURNAROUND_MIN']
    
    # LAYOUT CHANGE: Use 1x3 grid instead of 2x2
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1])
    
    # Create three axes for the plots
    ax1 = fig.add_subplot(gs[0, 0])  # Main cluster plot
    ax2 = fig.add_subplot(gs[0, 1])  # Carrier distribution
    ax3 = fig.add_subplot(gs[0, 2])  # Delay/Turnaround

    # Use the custom color palette
    colors = get_custom_color_palette(optimal_k)
    
    # 1. Main cluster plot - IMPROVED VERSION
    # Remove the background color
    ax1.set_facecolor('none')
    
    # Create a dict of cluster dataframes for separate plotting
    cluster_dfs = {c: df_plot[df_plot['Cluster'] == f'Cluster {c+1}'] for c in range(optimal_k)}
    
    # Plot each cluster separately for better control
    for i in range(optimal_k):
        cluster_name = f'Cluster {i+1}'
        cluster_df = cluster_dfs[i]
        ax1.scatter(
            cluster_df['UMAP1'],
            cluster_df['UMAP2'],
            color=colors[i % len(colors)],
            alpha=0.8,
            s=65,
            edgecolor='white',
            linewidth=0.3,
            label=cluster_name
        )
    
    # Add titles and labels with improved typography
    ax1.set_title('Flight Embedding Clusters', fontsize=16, fontweight='bold')
    ax1.set_xlabel('UMAP Dimension 1', fontsize=13, fontweight='bold')
    ax1.set_ylabel('UMAP Dimension 2', fontsize=13, fontweight='bold')
    
    # Add grid but make it subtle
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Improve tick labels
    ax1.tick_params(axis='both', which='major', labelsize=11)
    
    # Add better legend
    legend = ax1.legend(
        title="Clusters",
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        facecolor='white',
        edgecolor='lightgray',
        fontsize=11,
        title_fontsize=13
    )
    
    # Add a border around the plot
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_color('lightgray')
        spine.set_linewidth(1)
    
    # 2. Cluster characteristics - Carrier distribution
    carrier_counts = df_plot.groupby(['Cluster', 'Carrier']).size().unstack(fill_value=0)
    carrier_props = carrier_counts.div(carrier_counts.sum(axis=1), axis=0)
    
    # Keep only top 5 carriers for readability
    top_carriers = carrier_counts.sum().nlargest(5).index
    carrier_props = carrier_props[top_carriers]
    
    # Remove background
    ax2.set_facecolor('none')
    
    # Use custom colors from palette for carrier distribution 
    carrier_colors = get_custom_color_palette(len(top_carriers))
    
    carrier_props.plot(
        kind='bar', 
        stacked=True, 
        color=carrier_colors,
        ax=ax2,
        width=0.8
    )
    
    ax2.set_title('Carrier Distribution by Cluster', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Cluster', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Proportion', fontsize=13, fontweight='bold')
    ax2.legend(title='Carrier', fontsize=10, title_fontsize=12)
    
    # 3. Cluster characteristics - Average delay and turnaround
    has_turnaround = 'Turnaround' in df_plot.columns and df_plot['Turnaround'].notna().any()
    
    # Remove background
    ax3.set_facecolor('none')
    
    if has_turnaround:
        cluster_stats = df_plot.groupby('Cluster').agg({
            'Delay': ['mean', 'std'],
            'Duration': ['mean', 'std'],
            'Turnaround': ['mean', 'std']
        })
        
        # Create bar chart for average delay and turnaround
        cluster_means = pd.DataFrame({
            'Delay': cluster_stats['Delay']['mean'],
            'Turnaround': cluster_stats['Turnaround']['mean']
        })
        
        # Use colors from custom palette
        delay_turnaround_colors = [colors[0], colors[2]]
        
        cluster_means.plot(
            kind='bar',
            ax=ax3,
            color=delay_turnaround_colors,
            edgecolor='white',
            width=0.7
        )
        
        ax3.set_title('Average Delay and Turnaround by Cluster', fontsize=16, fontweight='bold')
        ax3.set_xlabel('Cluster', fontsize=13, fontweight='bold')
        ax3.set_ylabel('Minutes', fontsize=13, fontweight='bold')
        
        # Add horizontal line at zero for delay reference
        ax3.axhline(y=0, color=colors[1], linestyle='--', alpha=0.7)
        
    else:
        # If no turnaround data, just show delay
        cluster_stats = df_plot.groupby('Cluster').agg({
            'Delay': ['mean', 'std'],
            'Duration': ['mean', 'std']
        })
        
        # Create bar chart for average delay
        cluster_stats['Delay']['mean'].plot(
            kind='bar',
            yerr=cluster_stats['Delay']['std'],
            ax=ax3,
            color=colors[0],
            edgecolor='white',
            width=0.7
        )
        
        ax3.set_title('Average Delay by Cluster', fontsize=16, fontweight='bold')
        ax3.set_xlabel('Cluster', fontsize=13, fontweight='bold')
        ax3.set_ylabel('Average Delay (minutes)', fontsize=13, fontweight='bold')
        
        # Add horizontal line at zero
        ax3.axhline(y=0, color=colors[1], linestyle='--', alpha=0.7)
    
    # 4. Route analysis preparation
    df_plot['Route'] = df_plot['Departure'] + '-' + df_plot['Arrival']
    
    # Get top routes by cluster
    top_routes_by_cluster = {}
    for cluster in df_plot['Cluster'].unique():
        cluster_routes = df_plot[df_plot['Cluster'] == cluster]['Route'].value_counts().head(3)
        top_routes_by_cluster[cluster] = cluster_routes
    
    # Create summary table instead of text
    # Prepare data for the table
    table_data = []
    clusters = sorted(df_plot['Cluster'].unique())
    
    for cluster in clusters:
        routes = top_routes_by_cluster[cluster]
        top_route = routes.index[0] if not routes.empty else "N/A"
        top_route_count = routes.iloc[0] if not routes.empty else 0
        
        top_aircraft = df_plot[df_plot['Cluster'] == cluster]['Aircraft'].value_counts().head(1)
        top_aircraft_type = top_aircraft.index[0] if not top_aircraft.empty else "N/A"
        top_aircraft_count = top_aircraft.iloc[0] if not top_aircraft.empty else 0
        
        avg_delay = cluster_stats.loc[cluster, ('Delay', 'mean')]
        avg_duration = cluster_stats.loc[cluster, ('Duration', 'mean')]
        
        if has_turnaround:
            avg_turnaround = cluster_stats.loc[cluster, ('Turnaround', 'mean')]
            row = [cluster, top_route, f"{top_route_count:,}", top_aircraft_type, 
                   f"{top_aircraft_count:,}", f"{avg_delay:.1f}", f"{avg_duration:.1f}", f"{avg_turnaround:.1f}"]
        else:
            row = [cluster, top_route, f"{top_route_count:,}", top_aircraft_type, 
                   f"{top_aircraft_count:,}", f"{avg_delay:.1f}", f"{avg_duration:.1f}"]
        
        table_data.append(row)
    
    # Create table in the bottom row that spans all columns
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis('off')
    ax_table.set_facecolor('none')
    
    # Define column headers
    if has_turnaround:
        columns = ['Cluster', 'Top Route', 'Count', 'Top Aircraft', 'Count', 
                   'Avg Delay (min)', 'Avg Duration (min)', 'Avg Turnaround (min)']
    else:
        columns = ['Cluster', 'Top Route', 'Count', 'Top Aircraft', 'Count', 
                   'Avg Delay (min)', 'Avg Duration (min)']
    
    # Create table with better formatting
    table = ax_table.table(
        cellText=table_data,
        colLabels=columns,
        loc='center',
        cellLoc='center',
        colColours=['#f0f0f0'] * len(columns),
        bbox=[0.05, 0.0, 0.9, 0.95]  # [left, bottom, width, height]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    
    # Adjust column widths
    table.auto_set_column_width([0, 1, 2, 3, 4, 5, 6, 7] if has_turnaround else [0, 1, 2, 3, 4, 5, 6])
    
    # Style cells
    for i in range(len(table_data) + 1):  # +1 for header row
        for j in range(len(columns)):
            cell = table[(i, j)]
            
            if i == 0:  # Header row
                cell.set_text_props(weight='bold', color='black')
                cell.set_facecolor('#d9d9d9')
            else:
                if j == 0:  # Cluster column
                    cell.set_text_props(weight='bold')
                    cluster_idx = int(table_data[i-1][0].split()[1]) - 1
                    # Convert RGB tuple to RGBA tuple with transparency
                    cluster_color = colors[cluster_idx % len(colors)]
                    cluster_color_rgba = (cluster_color[0], cluster_color[1], cluster_color[2], 0.25)
                    cell.set_facecolor(cluster_color_rgba)
                
                # Alternate row colors for better readability
                if i % 2 == 0 and j > 0:
                    cell.set_facecolor('#f5f5f5')
    
    # Add main title
    fig.patch.set_facecolor('white')  # Set figure background to white
    plt.suptitle(f'Analysis of {optimal_k} KMeans Embedding Clusters', 
                fontsize=20, y=0.98, fontweight='bold')
    
    # Improve layout with more space
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save with higher resolution
    plt.savefig(output_path, bbox_inches='tight', dpi=400, facecolor='white')
    plt.close()
    
    logging.info(f"Saved embedding cluster visualization to {output_path}")
    
    # Save data as CSVs
    # 1. Main cluster data
    csv_path = output_path.with_suffix('.csv')
    df_plot.to_csv(csv_path, index=False)
    
    # 2. Cluster statistics
    stats_csv_path = output_dir / "embedding_clusters_stats.csv"
    cluster_stats.to_csv(stats_csv_path)
    
    # 3. Route distribution
    route_data = pd.DataFrame()
    for cluster in df_plot['Cluster'].unique():
        cluster_routes = df_plot[df_plot['Cluster'] == cluster]['Route'].value_counts().head(10)
        cluster_route_df = pd.DataFrame(cluster_routes).reset_index()
        cluster_route_df.columns = ['Route', 'Count']
        cluster_route_df['Cluster'] = cluster
        route_data = pd.concat([route_data, cluster_route_df])
    
    routes_csv_path = output_dir / "embedding_clusters_routes.csv"
    route_data.to_csv(routes_csv_path, index=False)
    
    # 4. Cluster centroids in embedding space
    cluster_centroids = pd.DataFrame(kmeans.cluster_centers_)
    cluster_centroids.index = [f'Cluster {i+1}' for i in range(optimal_k)]
    cluster_centroids.columns = [f'Dim_{i}' for i in range(cluster_centroids.shape[1])]
    
    centroids_csv_path = output_dir / "embedding_clusters_centroids.csv"
    cluster_centroids.to_csv(centroids_csv_path)
    
    logging.info(f"Saved cluster data to {csv_path}, {stats_csv_path}, {routes_csv_path}, and {centroids_csv_path}")
    
    # Generate additional plot: 3D PCA of clusters for better visualization if embedding dimension > 3
    if embeddings.shape[1] > 3:
        logging.info("Generating 3D PCA visualization of clusters")
        
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(embeddings)
        
        pca_df = pd.DataFrame({
            'PC1': pca_result[:, 0],
            'PC2': pca_result[:, 1],
            'PC3': pca_result[:, 2],
            'Cluster': [f'Cluster {c+1}' for c in cluster_labels]
        })
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        fig.patch.set_facecolor('white')  # Set figure background to white
        ax.set_facecolor('none')  # Remove background
        
        for i, cluster in enumerate(sorted(pca_df['Cluster'].unique())):
            cluster_data = pca_df[pca_df['Cluster'] == cluster]
            ax.scatter(
                cluster_data['PC1'],
                cluster_data['PC2'],
                cluster_data['PC3'],
                color=colors[i % len(colors)],
                label=cluster,
                alpha=0.7,
                s=30
            )
        
        ax.set_title('3D PCA Visualization of Flight Embedding Clusters', fontsize=16)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})', fontsize=12)
        ax.legend(title='Cluster', fontsize=10)
        plt.tight_layout()
        
        pca_path = output_dir / "embedding_clusters_3d_pca.png"
        plt.savefig(pca_path, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
        
        # Save PCA data
        pca_csv_path = pca_path.with_suffix('.csv')
        pca_df.to_csv(pca_csv_path, index=False)
        
        logging.info(f"Saved 3D PCA visualization to {pca_path} and data to {pca_csv_path}")
    
    return output_path

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
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)  # -1 represents noise in HDBSCAN
    n_noise = list(clusters).count(-1)
    noise_percentage = (n_noise / len(clusters) * 100) if len(clusters) > 0 else 0
    logging.info(f"HDBSCAN found {n_clusters} clusters and {n_noise} noise points ({noise_percentage:.2f}%)")
    return clusters, clusterer

def hdbscan_cluster_analysis(
    embeddings: np.ndarray,
    flight_data: pd.DataFrame,
    metadata: Dict[str, Union[np.ndarray, List[str]]],
    output_dir: Path = None,
    umap_reduced: np.ndarray = None,
    min_dist: float = DEFAULT_UMAP_MIN_DIST,
    n_neighbors: int = DEFAULT_UMAP_N_NEIGHBORS,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: int = DEFAULT_HDBSCAN_MIN_SAMPLES,
    seed: int = DEFAULT_RANDOM_SEED
) -> Path:
    """
    Use HDBSCAN clustering to identify and analyze patterns in the embeddings.
    """
    if output_dir is None:
        output_dir = Path(DEFAULT_OUTPUT_DIR)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "hdbscan_clusters.png"
    
    logging.info("Starting HDBSCAN embedding cluster analysis")
    
    # Apply UMAP if not provided
    if umap_reduced is None:
        reduced = apply_umap(embeddings, min_dist, n_neighbors, seed)
    else:
        reduced = umap_reduced
    
    # Apply HDBSCAN
    clusters, clusterer = apply_hdbscan(reduced, min_cluster_size, min_samples)
    
    # Prepare cluster labels
    cluster_labels = []
    for c in clusters:
        if c == -1:
            cluster_labels.append('Noise')
        else:
            cluster_labels.append(f'Cluster {c+1}')
    
    # Create dataframe for plotting
    df_plot = pd.DataFrame({
        'UMAP1': reduced[:, 0],
        'UMAP2': reduced[:, 1],
        'Cluster': cluster_labels,
        'Carrier': metadata['Carrier'],
        'Departure': metadata['Departure_Airport'],
        'Arrival': metadata['Arrival_Airport'],
        'Aircraft': metadata['Aircraft'],
        'Delay': flight_data['DEPARTURE_DELAY_MIN'],
        'Duration': flight_data['SCHEDULED_DURATION_MIN'],
    })
    
    # Add turnaround if available
    if 'TURNAROUND_MIN' in flight_data.columns and flight_data['TURNAROUND_MIN'].notna().any():
        df_plot['Turnaround'] = flight_data['TURNAROUND_MIN']
    
    # Create figure
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1])
    fig.patch.set_facecolor('white')  # Set figure background to white
    
    # Create three axes for the plots
    ax1 = fig.add_subplot(gs[0, 0])  # Main cluster plot
    ax2 = fig.add_subplot(gs[0, 1])  # Carrier distribution
    ax3 = fig.add_subplot(gs[0, 2])  # Delay/Turnaround

    # Use custom color palette
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    colors = get_custom_color_palette(max(n_clusters, 1))
    
    # Add a specific color for noise
    noise_color = (0.7, 0.7, 0.7)  # Grey for noise
    
    # 1. Main cluster plot - remove background
    ax1.set_facecolor('none')
    
    # Plot noise points first
    noise_df = df_plot[df_plot['Cluster'] == 'Noise']
    if not noise_df.empty:
        ax1.scatter(
            noise_df['UMAP1'],
            noise_df['UMAP2'],
            color=noise_color,
            alpha=0.3,
            s=30,
            edgecolor='white',
            linewidth=0.2,
            label='Noise'
        )
    
    # Plot each cluster separately
    unique_clusters = [c for c in df_plot['Cluster'].unique() if c != 'Noise']
    for i, cluster_name in enumerate(sorted(unique_clusters)):
        cluster_df = df_plot[df_plot['Cluster'] == cluster_name]
        ax1.scatter(
            cluster_df['UMAP1'],
            cluster_df['UMAP2'],
            color=colors[i % len(colors)],
            alpha=0.8,
            s=65,
            edgecolor='white',
            linewidth=0.3,
            label=cluster_name
        )
    
    # Add titles and labels
    ax1.set_title('HDBSCAN Embedding Clusters', fontsize=16, fontweight='bold')
    ax1.set_xlabel('UMAP Dimension 1', fontsize=13, fontweight='bold')
    ax1.set_ylabel('UMAP Dimension 2', fontsize=13, fontweight='bold')
    
    # Add grid but make it subtle
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Improve tick labels
    ax1.tick_params(axis='both', which='major', labelsize=11)
    
    # Add better legend
    legend = ax1.legend(
        title="Clusters",
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        facecolor='white',
        edgecolor='lightgray',
        fontsize=11,
        title_fontsize=13
    )
    
    # Add a border around the plot
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_color('lightgray')
        spine.set_linewidth(1)
    
    # 2. Cluster characteristics - Carrier distribution (excluding noise)
    ax2.set_facecolor('none')
    df_no_noise = df_plot[df_plot['Cluster'] != 'Noise']
    
    if not df_no_noise.empty and len(df_no_noise['Cluster'].unique()) > 0:
        carrier_counts = df_no_noise.groupby(['Cluster', 'Carrier']).size().unstack(fill_value=0)
        carrier_props = carrier_counts.div(carrier_counts.sum(axis=1), axis=0)
        
        # Keep only top 5 carriers for readability
        top_carriers = carrier_counts.sum().nlargest(5).index
        carrier_props = carrier_props[top_carriers]
        
        # Use colors from custom palette for carriers
        carrier_colors = get_custom_color_palette(len(top_carriers))
        
        carrier_props.plot(
            kind='bar', 
            stacked=True, 
            color=carrier_colors,
            ax=ax2,
            width=0.8
        )
        
        ax2.set_title('Carrier Distribution by Cluster', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Cluster', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Proportion', fontsize=13, fontweight='bold')
        ax2.legend(title='Carrier', fontsize=10, title_fontsize=12)
    else:
        ax2.text(0.5, 0.5, 'No clusters detected', horizontalalignment='center',
                verticalalignment='center', transform=ax2.transAxes, fontsize=14)
    
    # 3. Cluster characteristics - Average delay and turnaround (excluding noise)
    ax3.set_facecolor('none')
    has_turnaround = 'Turnaround' in df_plot.columns and df_plot['Turnaround'].notna().any()
    
    if not df_no_noise.empty and len(df_no_noise['Cluster'].unique()) > 0:
        if has_turnaround:
            cluster_stats = df_no_noise.groupby('Cluster').agg({
                'Delay': ['mean', 'std'],
                'Duration': ['mean', 'std'],
                'Turnaround': ['mean', 'std']
            })
            
            # Create bar chart for average delay and turnaround
            cluster_means = pd.DataFrame({
                'Delay': cluster_stats['Delay']['mean'],
                'Turnaround': cluster_stats['Turnaround']['mean']
            })
            
            # Use colors from custom palette
            delay_turnaround_colors = [colors[0], colors[2]]
            
            cluster_means.plot(
                kind='bar',
                ax=ax3,
                color=delay_turnaround_colors,
                edgecolor='white',
                width=0.7
            )
            
            ax3.set_title('Average Delay and Turnaround by Cluster', fontsize=16, fontweight='bold')
            ax3.set_xlabel('Cluster', fontsize=13, fontweight='bold')
            ax3.set_ylabel('Minutes', fontsize=13, fontweight='bold')
            
            # Add horizontal line at zero for delay reference
            ax3.axhline(y=0, color=colors[1], linestyle='--', alpha=0.7)
            
        else:
            # If no turnaround data, just show delay
            cluster_stats = df_no_noise.groupby('Cluster').agg({
                'Delay': ['mean', 'std'],
                'Duration': ['mean', 'std']
            })
            
            # Create bar chart for average delay
            cluster_stats['Delay']['mean'].plot(
                kind='bar',
                yerr=cluster_stats['Delay']['std'],
                ax=ax3,
                color=colors[0],
                edgecolor='white',
                width=0.7
            )
            
            ax3.set_title('Average Delay by Cluster', fontsize=16, fontweight='bold')
            ax3.set_xlabel('Cluster', fontsize=13, fontweight='bold')
            ax3.set_ylabel('Average Delay (minutes)', fontsize=13, fontweight='bold')
            
            # Add horizontal line at zero
            ax3.axhline(y=0, color=colors[1], linestyle='--', alpha=0.7)
    else:
        ax3.text(0.5, 0.5, 'No clusters detected', horizontalalignment='center',
                verticalalignment='center', transform=ax3.transAxes, fontsize=14)
    
    # 4. Route analysis preparation
    df_plot['Route'] = df_plot['Departure'] + '-' + df_plot['Arrival']
    
    # Get top routes by cluster, excluding noise
    top_routes_by_cluster = {}
    for cluster in [c for c in df_plot['Cluster'].unique() if c != 'Noise']:
        cluster_routes = df_plot[df_plot['Cluster'] == cluster]['Route'].value_counts().head(3)
        top_routes_by_cluster[cluster] = cluster_routes
    
    # Get statistics on noise points
    noise_count = list(clusters).count(-1)
    noise_percentage = noise_count / len(clusters) * 100
    
    # Create summary table
    table_data = []
    
    # Add noise row if there are noise points
    if noise_count > 0:
        noise_routes = df_plot[df_plot['Cluster'] == 'Noise']['Route'].value_counts().head(1)
        noise_route = noise_routes.index[0] if not noise_routes.empty else "N/A"
        noise_route_count = noise_routes.iloc[0] if not noise_routes.empty else 0
        
        noise_aircraft = df_plot[df_plot['Cluster'] == 'Noise']['Aircraft'].value_counts().head(1)
        noise_aircraft_type = noise_aircraft.index[0] if not noise_aircraft.empty else "N/A"
        noise_aircraft_count = noise_aircraft.iloc[0] if not noise_aircraft.empty else 0
        
        noise_stats = df_plot[df_plot['Cluster'] == 'Noise'].agg({
            'Delay': 'mean',
            'Duration': 'mean'
        })
        
        avg_delay = noise_stats['Delay']
        avg_duration = noise_stats['Duration']
        
        if has_turnaround:
            avg_turnaround = df_plot[df_plot['Cluster'] == 'Noise']['Turnaround'].mean()
            noise_row = ['Noise', noise_route, f"{noise_route_count:,}", noise_aircraft_type, 
                       f"{noise_aircraft_count:,}", f"{avg_delay:.1f}", f"{avg_duration:.1f}", f"{avg_turnaround:.1f}"]
        else:
            noise_row = ['Noise', noise_route, f"{noise_route_count:,}", noise_aircraft_type, 
                       f"{noise_aircraft_count:,}", f"{avg_delay:.1f}", f"{avg_duration:.1f}"]
        
        table_data.append(noise_row)
    
    # Add regular cluster rows
    for cluster in sorted(unique_clusters):
        routes = top_routes_by_cluster.get(cluster, pd.Series())
        top_route = routes.index[0] if not routes.empty else "N/A"
        top_route_count = routes.iloc[0] if not routes.empty else 0
        
        top_aircraft = df_plot[df_plot['Cluster'] == cluster]['Aircraft'].value_counts().head(1)
        top_aircraft_type = top_aircraft.index[0] if not top_aircraft.empty else "N/A"
        top_aircraft_count = top_aircraft.iloc[0] if not top_aircraft.empty else 0
        
        if not df_no_noise.empty and cluster in cluster_stats.index:
            avg_delay = cluster_stats.loc[cluster, ('Delay', 'mean')]
            avg_duration = cluster_stats.loc[cluster, ('Duration', 'mean')]
            
            if has_turnaround:
                avg_turnaround = cluster_stats.loc[cluster, ('Turnaround', 'mean')]
                row = [cluster, top_route, f"{top_route_count:,}", top_aircraft_type, 
                       f"{top_aircraft_count:,}", f"{avg_delay:.1f}", f"{avg_duration:.1f}", f"{avg_turnaround:.1f}"]
            else:
                row = [cluster, top_route, f"{top_route_count:,}", top_aircraft_type, 
                       f"{top_aircraft_count:,}", f"{avg_delay:.1f}", f"{avg_duration:.1f}"]
            
            table_data.append(row)
    
    # Create table in the bottom row that spans all columns
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis('off')
    ax_table.set_facecolor('none')
    
    # Define column headers
    if has_turnaround:
        columns = ['Cluster', 'Top Route', 'Count', 'Top Aircraft', 'Count', 
                   'Avg Delay (min)', 'Avg Duration (min)', 'Avg Turnaround (min)']
    else:
        columns = ['Cluster', 'Top Route', 'Count', 'Top Aircraft', 'Count', 
                   'Avg Delay (min)', 'Avg Duration (min)']
    
    # Create table with better formatting
    if table_data:
        table = ax_table.table(
            cellText=table_data,
            colLabels=columns,
            loc='center',
            cellLoc='center',
            colColours=['#f0f0f0'] * len(columns),
            bbox=[0.05, 0.0, 0.9, 0.95]  # [left, bottom, width, height]
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        
        # Adjust column widths
        table.auto_set_column_width([0, 1, 2, 3, 4, 5, 6, 7] if has_turnaround else [0, 1, 2, 3, 4, 5, 6])
        
        # Style cells
        for i in range(len(table_data) + 1):  # +1 for header row
            for j in range(len(columns)):
                cell = table[(i, j)]
                
                if i == 0:  # Header row
                    cell.set_text_props(weight='bold', color='black')
                    cell.set_facecolor('#d9d9d9')
                else:
                    if j == 0:  # Cluster column
                        cell.set_text_props(weight='bold')
                        # Special color for noise
                        if table_data[i-1][0] == 'Noise':
                            cell.set_facecolor((noise_color[0], noise_color[1], noise_color[2], 0.25))
                        else:
                            cluster_idx = int(table_data[i-1][0].split()[1]) - 1
                            cluster_color = colors[cluster_idx % len(colors)]
                            cell.set_facecolor((cluster_color[0], cluster_color[1], cluster_color[2], 0.25))
                    
                    # Alternate row colors for better readability
                    if i % 2 == 0 and j > 0:
                        cell.set_facecolor('#f5f5f5')
    else:
        ax_table.text(0.5, 0.5, 'No clusters detected by HDBSCAN', horizontalalignment='center',
                    verticalalignment='center', fontsize=14, fontweight='bold')
    
    # Add main title and info about noise
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    plt.suptitle(f'Analysis of {n_clusters} HDBSCAN Embedding Clusters ({noise_percentage:.1f}% noise points)', 
                fontsize=20, y=0.98, fontweight='bold')
    
    # Improve layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save with higher resolution
    plt.savefig(output_path, bbox_inches='tight', dpi=400, facecolor='white')
    plt.close()
    
    logging.info(f"Saved HDBSCAN embedding visualization to {output_path}")
    
    # Save data as CSVs
    csv_path = output_path.with_suffix('.csv')
    df_plot.to_csv(csv_path, index=False)
    
    # Save HDBSCAN parameters
    hdbscan_params = {
        'min_cluster_size': min_cluster_size,
        'min_samples': min_samples,
        'num_clusters': n_clusters,
        'noise_points': noise_count,
        'noise_percentage': noise_percentage
    }
    
    params_csv_path = output_dir / "hdbscan_params.csv"
    pd.DataFrame([hdbscan_params]).to_csv(params_csv_path, index=False)
    
    logging.info(f"Saved HDBSCAN cluster data to {csv_path} and {params_csv_path}")
    
    return output_path

def main():
    """Main function to orchestrate the embedding cluster analysis process."""
    setup_logging()
    args = parse_args()
    
    # Load data
    embeddings, flight_data, metadata = load_data(
        args.embeddings, args.data, args.sample_size, args.seed
    )
    
    # Apply UMAP once to reuse for both analyses
    umap_reduced = apply_umap(embeddings, args.min_dist, args.n_neighbors, args.seed)
    
    # Run KMeans embedding cluster analysis
    embedding_cluster_analysis(
        embeddings=embeddings,
        flight_data=flight_data,
        metadata=metadata,
        output_dir=args.output_dir,
        umap_reduced=umap_reduced,  # Pass the precomputed UMAP reduction
        min_dist=args.min_dist,
        n_neighbors=args.n_neighbors,
        max_clusters=args.max_clusters,
        seed=args.seed
    )
    
    # Run HDBSCAN embedding cluster analysis
    hdbscan_cluster_analysis(
        embeddings=embeddings,
        flight_data=flight_data,
        metadata=metadata,
        output_dir=args.output_dir,
        umap_reduced=umap_reduced,  # Reuse the UMAP reduction
        min_dist=args.min_dist,
        n_neighbors=args.n_neighbors,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        seed=args.seed
    )
    
    logging.info("Embedding cluster analyses complete (KMeans and HDBSCAN)")

if __name__ == "__main__":
    main()