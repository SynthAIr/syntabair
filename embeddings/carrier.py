import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import umap.umap_ as umap

from syntabair.preprocessing import preprocess_flight_data_for_prediction


# Configuration constants
DEFAULT_OUTPUT_DIR = Path("./embeddings/carrier_comparison")
DEFAULT_RANDOM_SEED = 42
DEFAULT_SAMPLE_SIZE = 200_000
DEFAULT_TOP_CARRIERS = 6
DEFAULT_UMAP_MIN_DIST = 0.1
DEFAULT_UMAP_N_NEIGHBORS = 15


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate TabSyn carrier operation comparison visualization"
    )
    parser.add_argument("--embeddings", type=Path, required=True,
                        help="Path to TabSyn embeddings file (e.g., train_z.npy)")
    parser.add_argument("--data", type=Path, required=True,
                        help="Path to flight data CSV file")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to save outputs (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
                        help=f"Number of records to sample (default: {DEFAULT_SAMPLE_SIZE})")
    parser.add_argument("--top-carriers", type=int, default=DEFAULT_TOP_CARRIERS,
                        help=f"Number of top carriers to visualize (default: {DEFAULT_TOP_CARRIERS})")
    parser.add_argument("--min-dist", type=float, default=DEFAULT_UMAP_MIN_DIST,
                        help=f"UMAP min_dist parameter (default: {DEFAULT_UMAP_MIN_DIST})")
    parser.add_argument("--n-neighbors", type=int, default=DEFAULT_UMAP_N_NEIGHBORS,
                        help=f"UMAP n_neighbors parameter (default: {DEFAULT_UMAP_N_NEIGHBORS})")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED,
                        help=f"Random seed (default: {DEFAULT_RANDOM_SEED})")
    return parser.parse_args()


# Custom color palette function
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
    import seaborn as sns  # lazy import
    extra = sns.color_palette("tab20", num_colors - len(base_palette))
    return base_palette + extra


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
    
    return embeddings, df


def identify_carrier_column(df: pd.DataFrame) -> str:
    """Identify the carrier column in the dataframe."""
    carrier_col_candidates = [
        'CARRIER', 'CARRIER_CODE', 'AIRLINE', 'AIRLINE_CODE', 
        'OPERATING_CARRIER', 'MARKETING_CARRIER'
    ]
    
    for col in carrier_col_candidates:
        if col in df.columns:
            return col
    
    # Try to find a suitable column if standard names are not found
    for col in df.columns:
        if 'CARRIER' in col or 'AIRLINE' in col:
            return col
    
    raise ValueError("No carrier column found in the data")


def get_top_carriers(
    df: pd.DataFrame,
    carrier_col: str,
    top_n: int
) -> List[str]:
    """Identify the top N carriers by flight count."""
    carrier_counts = df[carrier_col].value_counts()
    top_carriers = carrier_counts.head(top_n).index.tolist()
    logging.info(f"Top {top_n} carriers identified with frequencies: %s", 
                carrier_counts.head(top_n).to_dict())
    return top_carriers


def prepare_carrier_data(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    carrier_col: str,
    top_carriers: List[str]
) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """Filter data to include only top carriers."""
    # Filter data for top carriers
    carrier_mask = df[carrier_col].isin(top_carriers)
    filtered_embeddings = embeddings[carrier_mask]
    filtered_df = df.loc[carrier_mask].reset_index(drop=True)
    filtered_carriers = filtered_df[carrier_col].tolist()
    
    logging.info(f"Filtered to {len(filtered_df)} flights across {len(top_carriers)} carriers")
    return filtered_embeddings, filtered_df, filtered_carriers


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


def calculate_carrier_metrics(
    df: pd.DataFrame,
    carrier_col: str,
    top_carriers: List[str]
) -> Dict[str, Dict]:
    """Calculate operational metrics for each carrier."""
    carrier_metrics = {}
    for carrier in top_carriers:
        carrier_mask = df[carrier_col] == carrier
        carrier_data = df[carrier_mask]
        
        metrics = {
            'Delay': carrier_data['DEPARTURE_DELAY_MIN'].mean(),
            'Duration': carrier_data['SCHEDULED_DURATION_MIN'].mean(),
            'On-Time %': 100 * (carrier_data['DEPARTURE_DELAY_MIN'] <= 15).mean(),
            'Early %': 100 * (carrier_data['DEPARTURE_DELAY_MIN'] < 0).mean(),
        }
        
        # Add peak hour if hour data is available
        if 'SCHEDULED_HOUR' in df.columns:
            hour_counts = carrier_data['SCHEDULED_HOUR'].value_counts()
            if not hour_counts.empty:
                metrics['Peak Hour'] = hour_counts.idxmax()
            else:
                metrics['Peak Hour'] = None
        
        # Add turnaround if available
        if 'TURNAROUND_MIN' in df.columns:
            metrics['Turnaround'] = carrier_data['TURNAROUND_MIN'].mean()
        
        carrier_metrics[carrier] = metrics
    
    logging.info(f"Calculated metrics for {len(carrier_metrics)} carriers")
    return carrier_metrics


def create_carrier_visualization(
    reduced: np.ndarray,
    df: pd.DataFrame,
    carrier_col: str,
    top_carriers: List[str],
    carrier_metrics: Dict[str, Dict],
    output_dir: Path
) -> None:
    """Create the carrier operation comparison visualization with UMAP and density contours."""
    # Set up figure with 2x3 subplots
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    axs = axs.flatten()
    
    # Get custom color palette
    colors = get_custom_color_palette(len(top_carriers))
    
    # Calculate global x and y limits for consistency
    x_min, x_max = reduced[:, 0].min(), reduced[:, 0].max()
    y_min, y_max = reduced[:, 1].min(), reduced[:, 1].max()
    
    # Add some padding to the limits
    x_padding = (x_max - x_min) * 0.05
    y_padding = (y_max - y_min) * 0.05
    
    x_min -= x_padding
    x_max += x_padding
    y_min -= y_padding
    y_max += y_padding
    
    # Create a DataFrame with carrier and reduced dimensions
    carrier_data = pd.DataFrame({
        'Carrier': df[carrier_col],
        'UMAP1': reduced[:, 0],
        'UMAP2': reduced[:, 1],
        'Delay': df['DEPARTURE_DELAY_MIN']  # Add delay for coloring
    })
    
    # For each carrier, create a scatter plot with density contours
    for i, carrier in enumerate(top_carriers):
        if i >= len(axs):  # Skip if we have more carriers than subplots
            continue
            
        ax = axs[i]
        carrier_mask = carrier_data['Carrier'] == carrier
        carrier_subset = carrier_data[carrier_mask]
        
        # Skip if not enough data points
        if len(carrier_subset) <= 10:
            ax.text(0.5, 0.5, f"Not enough data for {carrier}", 
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            continue
        
        # Get carrier color
        color = colors[i % len(colors)]
        
        # Create a diverging colormap for delay (green=early, yellow=on-time, red=delayed)
        # Using a custom normalization to center the colormap at 0 delay
        from matplotlib.colors import Normalize
        
        # Create a normalization that puts 0 at the center
        # Typical delay range might be -15 (early) to +60 (delayed)
        norm = Normalize(vmin=-15, vmax=60)
        
        scatter = ax.scatter(
            carrier_subset['UMAP1'],
            carrier_subset['UMAP2'],
            c=carrier_subset['Delay'],
            cmap='RdYlGn_r',  # Red-Yellow-Green (reversed so red=delayed)
            norm=norm,
            alpha=0.5,
            s=30,
            edgecolor='none'
        )
        
        # Add colorbar if it's the last subplot or top-right subplot
        if i == len(top_carriers) - 1 or i == 2:
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label('Departure Delay (minutes)', fontsize=10)
        
        # Calculate and plot centroid
        centroid_x = carrier_subset['UMAP1'].mean()
        centroid_y = carrier_subset['UMAP2'].mean()
        ax.scatter(
            centroid_x, centroid_y,
            c='black',
            s=200,
            marker='X',
            edgecolor='white',
            linewidth=1.5,
            zorder=10,
            label='Centroid'
        )
        
        # Add density contours
        try:
            # Calculate KDE for density estimation
            xy = np.vstack([carrier_subset['UMAP1'], carrier_subset['UMAP2']])
            kde = gaussian_kde(xy, bw_method='scott')
            
            # Create a grid for contour plotting
            xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            z = kde(positions).reshape(xx.shape)
            
            # Plot contours - adjusted for RGB tuples from custom palette
            contour = ax.contour(xx, yy, z, 
                                levels=5, 
                                colors=[(color[0], color[1], color[2], level) for level in np.linspace(0.5, 1, 5)],
                                linewidths=2)
        except Exception as e:
            logging.warning(f"Could not create density contours for carrier {carrier}: {e}")
        
        # Add titles 
        ax.set_title(f'Carrier: {carrier}', fontsize=14)
        
        # Only add x-axis labels to bottom row subplots
        if i >= len(axs) - 3:  # Bottom row (assuming 2x3 grid)
            ax.set_xlabel('UMAP Dimension 1', fontsize=10)
        else:
            ax.set_xticklabels([])
            
        # Only add y-axis labels to leftmost column subplots
        if i % 3 == 0:  # Left column (assuming 2x3 grid)
            ax.set_ylabel('UMAP Dimension 2', fontsize=10)
        else:
            ax.set_yticklabels([])
        
        # Calculate operational statistics
        stats_text = ""
        
        # Average delay
        avg_delay = carrier_metrics[carrier]['Delay']
        stats_text += f"Avg Delay: {avg_delay:.1f} min\n"
        
        # Average duration
        avg_duration = carrier_metrics[carrier]['Duration']
        stats_text += f"Avg Duration: {avg_duration:.1f} min\n"
        
        # Average turnaround if available
        if 'Turnaround' in carrier_metrics[carrier]:
            avg_turnaround = carrier_metrics[carrier]['Turnaround']
            stats_text += f"Avg Turnaround: {avg_turnaround:.1f} min\n"
        
        # On-time percentage
        on_time_pct = carrier_metrics[carrier]['On-Time %']
        stats_text += f"On-Time: {on_time_pct:.1f}%\n"
        
        # Early percentage
        early_pct = carrier_metrics[carrier]['Early %']
        stats_text += f"Early: {early_pct:.1f}%\n"
        
        # Number of flights
        n_flights = sum(carrier_mask)
        stats_text += f"Flights: {n_flights}"
        
        # Add text box with stats
        ax.text(
            0.05, 0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8)
        )
        
        # Set consistent limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.6)
    
    # Add comparison table as the last subplot if there are fewer carriers than subplots
    remaining_subplots = len(axs) - len(top_carriers)
    if remaining_subplots > 0 and remaining_subplots < len(axs):
        for i in range(len(top_carriers), len(axs)):
            axs[i].axis('off')
        
        # Create comparison table in the last subplot
        axs[-1].axis('off')
        
        # Create comparison table
        table_data = []
        header = ['Carrier', 'Avg Delay', 'Avg Duration', 'On-Time %', 'Early %']
        
        if 'Peak Hour' in carrier_metrics[top_carriers[0]]:
            header.append('Peak Hour')
        
        if 'Turnaround' in carrier_metrics[top_carriers[0]]:
            header.insert(3, 'Avg Turnaround')
        
        for carrier in top_carriers:
            row = [
                carrier,
                f"{carrier_metrics[carrier]['Delay']:.1f}",
                f"{carrier_metrics[carrier]['Duration']:.1f}"
            ]
            
            if 'Turnaround' in carrier_metrics[carrier]:
                row.append(f"{carrier_metrics[carrier]['Turnaround']:.1f}")
            
            row.extend([
                f"{carrier_metrics[carrier]['On-Time %']:.1f}%",
                f"{carrier_metrics[carrier]['Early %']:.1f}%"
            ])
            
            if 'Peak Hour' in carrier_metrics[carrier]:
                row.append(f"{carrier_metrics[carrier]['Peak Hour']}")
            
            table_data.append(row)
        
        # Create the table
        table = axs[-1].table(
            cellText=table_data,
            colLabels=header,
            loc='center',
            cellLoc='center'
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        table.auto_set_column_width(col=list(range(len(header))))
        
        # Add a title to the table
        axs[-1].set_title('Carrier Comparison', fontsize=14)
    
    # Add main title
    plt.suptitle('Comparison of Carrier Operations Using TabSyn Embeddings', fontsize=18)
    
    # Improve layout
    plt.tight_layout(rect=[0, 0.01, 1, 0.95])
    
    # Save figure
    output_path = output_dir / "carrier_comparison_umap.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    logging.info(f"Carrier comparison visualization saved to {output_path}")
    
    # Create a combined visualization with density contours only (no points)
    plt.figure(figsize=(14, 10))
    
    # Plot density contours for each carrier on the same plot
    for i, carrier in enumerate(top_carriers):
        carrier_mask = carrier_data['Carrier'] == carrier
        carrier_subset = carrier_data[carrier_mask]
        
        if len(carrier_subset) > 10:
            try:
                # Calculate KDE
                xy = np.vstack([carrier_subset['UMAP1'], carrier_subset['UMAP2']])
                kde = gaussian_kde(xy, bw_method='scott')
                
                # Create grid
                xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                z = kde(positions).reshape(xx.shape)
                
                # Get carrier color
                color = colors[i % len(colors)]
                
                # Plot contours with different line styles for better distinction
                linestyles = ['-', '--', '-.', ':']
                contour = plt.contour(xx, yy, z, 
                                     levels=3,
                                     colors=[color],
                                     linewidths=3,
                                     linestyles=[linestyles[i % len(linestyles)]],
                                     alpha=0.8)
                
                # Add label to the contours
                plt.clabel(contour, inline=1, fontsize=10)
                
                # Add to legend
                plt.plot([], [], color=color, linestyle=linestyles[i % len(linestyles)], 
                         linewidth=3, label=carrier)
                
                # Plot centroid
                centroid_x = carrier_subset['UMAP1'].mean()
                centroid_y = carrier_subset['UMAP2'].mean()
                plt.scatter(
                    centroid_x, centroid_y,
                    c=[color],
                    s=200,
                    marker='X',
                    edgecolor='white',
                    linewidth=1.5,
                    zorder=10
                )
                
            except Exception as e:
                logging.warning(f"Could not create density contours for carrier {carrier}: {e}")
    
    # Add titles and labels
    plt.title('Comparative Carrier Operations - Density Contours Only', fontsize=18)
    plt.xlabel('UMAP Dimension 1', fontsize=14)
    plt.ylabel('UMAP Dimension 2', fontsize=14)
    
    # Set consistent limits
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add legend
    plt.legend(title='Carrier', title_fontsize=14, fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save the combined visualization
    combined_path = output_dir / "carrier_contours_combined.png"
    plt.savefig(combined_path, bbox_inches='tight', dpi=300)
    plt.close()
    logging.info(f"Combined contour visualization saved to {combined_path}")
    
    # Save data as CSV
    # Full dataset with UMAP coordinates
    output_df = df.copy()
    output_df['UMAP1'] = reduced[:, 0]
    output_df['UMAP2'] = reduced[:, 1]
    
    csv_path = output_dir / "carrier_comparison_umap.csv"
    output_df.to_csv(csv_path, index=False)
    logging.info(f"Data saved to {csv_path}")
    
    # Generate a carrier metrics summary
    metrics_df = pd.DataFrame.from_dict(carrier_metrics, orient='index')
    metrics_df.index.name = 'Carrier'
    
    # Add centroid coordinates to the metrics
    centroids = carrier_data.groupby('Carrier').agg({
        'UMAP1': 'mean',
        'UMAP2': 'mean'
    }).reset_index()
    centroids.columns = ['Carrier', 'Centroid_X', 'Centroid_Y']
    
    # Merge with metrics
    metrics_df = metrics_df.reset_index()
    metrics_df = pd.merge(metrics_df, centroids, on='Carrier')
    
    # Save carrier metrics summary
    metrics_csv = output_dir / "carrier_metrics_umap.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    logging.info(f"Carrier metrics saved to {metrics_csv}")


def main():
    """Main function to orchestrate the carrier comparison visualization process."""
    setup_logging()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    embeddings, df = load_data(
        args.embeddings, args.data, args.sample_size, args.seed
    )
    
    # Identify carrier column
    carrier_col = identify_carrier_column(df)
    logging.info(f"Using '{carrier_col}' as the carrier column")
    
    # Get top carriers
    top_carriers = get_top_carriers(df, carrier_col, args.top_carriers)
    
    # Filter data for top carriers
    filtered_embeddings, filtered_df, filtered_carriers = prepare_carrier_data(
        embeddings, df, carrier_col, top_carriers
    )
    
    # Apply UMAP for dimensionality reduction (instead of PCA)
    reduced = apply_umap(filtered_embeddings, args.min_dist, args.n_neighbors, args.seed)
    
    # Calculate carrier metrics
    carrier_metrics = calculate_carrier_metrics(filtered_df, carrier_col, top_carriers)
    
    # Create and save visualization
    create_carrier_visualization(
        reduced, filtered_df, carrier_col, top_carriers, carrier_metrics, args.output_dir
    )
    
    logging.info("Carrier operation comparison visualization complete")


if __name__ == "__main__":
    main()