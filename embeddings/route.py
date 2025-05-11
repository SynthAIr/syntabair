import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from syntabair.preprocessing import preprocess_flight_data_for_prediction


# Configuration constants
DEFAULT_OUTPUT_DIR = Path("./embeddings/route")
DEFAULT_RANDOM_SEED = 42
DEFAULT_SAMPLE_SIZE = 2_000_000
DEFAULT_TOP_ROUTES = 6
DEFAULT_PERPLEXITY = 30
DEFAULT_TSNE_ITERATIONS = 1000


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate TabSyn embedding visualizations comparing flight routes"
    )
    parser.add_argument("--embeddings", type=Path, required=True,
                        help="Path to TabSyn embeddings file (e.g., train_z.npy)")
    parser.add_argument("--data", type=Path, required=True,
                        help="Path to flight data CSV file")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to save outputs (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
                        help=f"Number of records to sample (default: {DEFAULT_SAMPLE_SIZE})")
    parser.add_argument("--top-routes", type=int, default=DEFAULT_TOP_ROUTES,
                        help=f"Number of top routes to visualize (default: {DEFAULT_TOP_ROUTES})")
    parser.add_argument("--perplexity", type=int, default=DEFAULT_PERPLEXITY,
                        help=f"t-SNE perplexity parameter (default: {DEFAULT_PERPLEXITY})")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED,
                        help=f"Random seed (default: {DEFAULT_RANDOM_SEED})")
    return parser.parse_args()


def load_data(
    embeddings_path: Path,
    data_path: Path,
    sample_size: Optional[int],
    seed: int
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load embeddings and flight data with optional sampling."""
    logging.info(f"Loading embeddings from %s", embeddings_path)
    embeddings = np.load(embeddings_path)
    if embeddings.ndim == 3:
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
        logging.info("Reshaped embeddings to %s", embeddings.shape)

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
    
    # Create route feature from departure and arrival airports
    if 'DEPARTURE_IATA_AIRPORT_CODE' in df.columns and 'ARRIVAL_IATA_AIRPORT_CODE' in df.columns:
        df['Route'] = df['DEPARTURE_IATA_AIRPORT_CODE'] + '-' + df['ARRIVAL_IATA_AIRPORT_CODE']
    else:
        # Fallback for different column names
        departure_col = [col for col in df.columns if 'DEPART' in col and 'AIRPORT' in col][0]
        arrival_col = [col for col in df.columns if 'ARRIV' in col and 'AIRPORT' in col][0]
        df['Route'] = df[departure_col] + '-' + df[arrival_col]
        logging.info(f"Created Route from {departure_col} and {arrival_col}")
    
    return embeddings, df


def get_top_routes(
    df: pd.DataFrame,
    top_n: int
) -> List[str]:
    """Identify the top N routes by frequency."""
    route_counts = df['Route'].value_counts()
    top_routes = route_counts.head(top_n).index.tolist()
    logging.info(f"Top {top_n} routes identified with frequencies: %s", 
                route_counts.head(top_n).to_dict())
    return top_routes


def create_route_visualizations(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    top_routes: List[str],
    perplexity: int,
    seed: int
) -> Tuple[np.ndarray, pd.DataFrame, Dict]:
    """Apply dimensionality reduction and prepare route statistics."""
    # Filter data for top routes
    route_mask = df['Route'].isin(top_routes)
    filtered_embeddings = embeddings[route_mask]
    filtered_df = df.loc[route_mask].reset_index(drop=True)
    
    logging.info(f"Filtered to {len(filtered_df)} flights across {len(top_routes)} routes")
    
    # Apply t-SNE for visualization
    actual_perplexity = min(perplexity, len(filtered_embeddings)-1)
    logging.info(f"Applying t-SNE with perplexity={actual_perplexity}")
    tsne = TSNE(
        n_components=2, 
        random_state=seed, 
        perplexity=actual_perplexity,
        max_iter=DEFAULT_TSNE_ITERATIONS
    )
    reduced = tsne.fit_transform(filtered_embeddings)
    logging.info("t-SNE dimensionality reduction completed")
    
    # Prepare route statistics
    route_stats = calculate_route_statistics(filtered_df, top_routes)
    
    return reduced, filtered_df, route_stats


def calculate_route_statistics(
    df: pd.DataFrame,
    top_routes: List[str]
) -> Dict:
    """Calculate statistics for each route."""
    route_stats = {}
    
    for route in top_routes:
        route_df = df[df['Route'] == route]
        
        stats = {
            'Count': len(route_df),
            'Avg_Delay': route_df['DEPARTURE_DELAY_MIN'].mean(),
            'Avg_Duration': route_df['SCHEDULED_DURATION_MIN'].mean(),
            'Pct_On_Time': 100 * (route_df['DEPARTURE_DELAY_MIN'] <= 15).mean(),
            'Pct_Early': 100 * (route_df['DEPARTURE_DELAY_MIN'] < 0).mean()
        }
        
        # Add turnaround stats if available
        if 'TURNAROUND_MIN' in df.columns:
            stats['Avg_Turnaround'] = route_df['TURNAROUND_MIN'].mean()
        
        route_stats[route] = stats
    
    logging.info(f"Calculated statistics for {len(route_stats)} routes")
    return route_stats


def draw_and_save(
    reduced: np.ndarray, 
    df: pd.DataFrame,
    top_routes: List[str],
    route_stats: Dict,
    output_dir: Path
) -> None:
    """Create visualization plots and save results to files."""
    logging.info("Creating route embedding comparison visualization")
    
    # Set up figure with subplots
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    axs = axs.flatten()

    # Compute global bounds from the full reduced embedding
    x_min, x_max = reduced[:, 0].min(), reduced[:, 0].max()
    y_min, y_max = reduced[:, 1].min(), reduced[:, 1].max()
    
    # Main plot - All routes together
    for i, route in enumerate(top_routes):
        route_indices = df[df['Route'] == route].index
        if len(route_indices) > 0:  # Make sure we have data for this route
            axs[0].scatter(
                reduced[route_indices, 0],
                reduced[route_indices, 1],
                label=route,
                alpha=0.7,
                s=50,
                edgecolor='none'
            )
    
    axs[0].set_title('Route Comparison in TabSyn Embedding Space', fontsize=14)
    axs[0].set_xlim(x_min, x_max)
    axs[0].set_ylim(y_min, y_max)
    axs[0].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axs[0].set_ylabel('t-SNE Dimension 2', fontsize=12)
    axs[0].legend(title='Route', fontsize=10)
    
    # Individual route plots with delay coloring
    for i, route in enumerate(top_routes[:5], 1):  # Plot first 5 routes individually
        route_df = df[df['Route'] == route]
        route_indices = route_df.index
        
        if len(route_indices) > 0:  # Make sure we have data for this route
            # Color by delay
            delays = route_df['DEPARTURE_DELAY_MIN']
            scatter = axs[i].scatter(
                reduced[route_indices, 0],
                reduced[route_indices, 1],
                c=delays,
                cmap='coolwarm',
                vmin=-30,
                vmax=60,
                alpha=0.8,
                s=50,
                edgecolor='none'
            )
            
            # Add titles and labels
            axs[i].set_title(f'Route: {route}', fontsize=14)
            axs[i].set_xlim(x_min, x_max)
            axs[i].set_ylim(y_min, y_max)
            axs[i].set_xlabel('t-SNE Dimension 1', fontsize=12)
            axs[i].set_ylabel('t-SNE Dimension 2', fontsize=12)
            
            # Add colorbar
            cbar = fig.colorbar(scatter, ax=axs[i])
            cbar.set_label('Delay (minutes)', fontsize=10)
            
            # Add route statistics
            stats = route_stats[route]
            stats_text = (
                f"Flights: {stats['Count']}\n"
                f"Avg Delay: {stats['Avg_Delay']:.1f} min\n"
                f"Avg Duration: {stats['Avg_Duration']:.1f} min\n"
            )
            
            # Add turnaround stats if available
            if 'Avg_Turnaround' in stats:
                stats_text += f"Avg Turnaround: {stats['Avg_Turnaround']:.1f} min\n"
            
            stats_text += f"On-Time %: {stats['Pct_On_Time']:.1f}%"
            
            axs[i].text(
                0.05, 0.95,
                stats_text,
                transform=axs[i].transAxes,
                fontsize=10,
                va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8)
            )
    
    # Add main title
    plt.suptitle('Route-Specific TabSyn Embedding Patterns and Delay Characteristics', fontsize=18)
    
    # Improve layout
    plt.tight_layout(rect=[0, 0.01, 1, 0.95])
    
    # Save visualization
    img_path = output_dir / "route_embedding_comparison.png"
    csv_base = img_path.stem
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info("Saved visualization to %s", img_path)
    
    # Save data as CSV
    # Create dataframe with route information and t-SNE coordinates
    tsne_data = []
    for i, row in df.iterrows():
        data_point = {
            'Route': row['Route'],
            'TSNE1': reduced[i, 0],
            'TSNE2': reduced[i, 1],
            'Delay': row['DEPARTURE_DELAY_MIN'],
            'Duration': row['SCHEDULED_DURATION_MIN']
        }
        
        # Add hour if available
        if 'SCHEDULED_HOUR' in df.columns:
            data_point['Hour'] = row['SCHEDULED_HOUR']
        
        # Add turnaround if available
        if 'TURNAROUND_MIN' in df.columns:
            data_point['Turnaround'] = row['TURNAROUND_MIN']
        
        tsne_data.append(data_point)
    
    route_df = pd.DataFrame(tsne_data)
    route_csv = output_dir / f"{csv_base}.csv"
    route_df.to_csv(route_csv, index=False)
    
    # Create a summary dataframe with route statistics
    stats_df = pd.DataFrame.from_dict(route_stats, orient='index')
    stats_df.index.name = 'Route'
    stats_csv = output_dir / f"{csv_base}_stats.csv"
    stats_df.to_csv(stats_csv)
    
    logging.info("Saved data to %s and %s", route_csv, stats_csv)


def main():
    """Main function to orchestrate the route embedding visualization process."""
    setup_logging()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    embeddings, df = load_data(
        args.embeddings, args.data, args.sample_size, args.seed
    )
    
    # Get top routes
    top_routes = get_top_routes(df, args.top_routes)
    
    # Create visualizations
    reduced, filtered_df, route_stats = create_route_visualizations(
        embeddings, df, top_routes, args.perplexity, args.seed
    )
    
    # Draw and save results
    draw_and_save(
        reduced, filtered_df, top_routes, route_stats, args.output_dir
    )
    
    logging.info("Route embedding visualization complete")


if __name__ == "__main__":
    main()