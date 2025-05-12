import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple, List  # Added List type

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import umap.umap_ as umap


from syntabair.preprocessing import preprocess_flight_data_for_prediction


# Configuration constants
DEFAULT_OUTPUT_DIR = Path("./embeddings/seasonal_patterns")
DEFAULT_RANDOM_SEED = 42
DEFAULT_SAMPLE_SIZE = 100_000
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
        description="Generate TabSyn seasonal pattern embeddings visualization"
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


def validate_month_data(df: pd.DataFrame) -> None:
    """Validate that the dataframe contains month information."""
    if 'SCHEDULED_MONTH' not in df.columns:
        # Try to find a suitable alternative column
        month_cols = [col for col in df.columns if 'MONTH' in col]
        if month_cols:
            df['SCHEDULED_MONTH'] = df[month_cols[0]]
            logging.info(f"Using {month_cols[0]} as the month column")
        else:
            raise ValueError("No month column found in the data. Need 'SCHEDULED_MONTH' or similar column.")
    
    # Ensure month is in the correct range (1-12)
    if not all(df['SCHEDULED_MONTH'].between(1, 12)):
        logging.warning("Month values outside the expected range (1-12)")
        # Try to fix by converting to int and modulo 12
        df['SCHEDULED_MONTH'] = df['SCHEDULED_MONTH'].astype(int) % 12
        df.loc[df['SCHEDULED_MONTH'] == 0, 'SCHEDULED_MONTH'] = 12
        logging.info("Adjusted month values to be in range 1-12")


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

def create_seasonal_visualization(
    reduced: np.ndarray,
    df: pd.DataFrame,
    output_dir: Path
) -> None:
    """Create small multiples visualization with one plot per month, centroids and density contours."""
    # Create month categories
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    month_data = pd.DataFrame({
        'Month': df['SCHEDULED_MONTH'],
        'Month_Name': df['SCHEDULED_MONTH'].map(lambda x: month_names[int(x)-1]),
        'UMAP1': reduced[:, 0],
        'UMAP2': reduced[:, 1]
    })
    
    # Get the unique months in the data
    available_months = sorted(month_data['Month'].unique())
    # Filter out months with less than 10 points
    month_counts = month_data['Month'].value_counts()
    available_months = [month for month in available_months if month_counts[month] > 10]
    
    # Create a grid of subplots (2x2 for 4 months)
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), sharex=True, sharey=True)
    axes = axes.flatten()
    
    # Calculate global x and y limits to keep consistent across all plots
    x_min, x_max = month_data['UMAP1'].min(), month_data['UMAP1'].max()
    y_min, y_max = month_data['UMAP2'].min(), month_data['UMAP2'].max()
    
    # Add some padding to the limits
    x_padding = (x_max - x_min) * 0.05
    y_padding = (y_max - y_min) * 0.05
    
    x_min -= x_padding
    x_max += x_padding
    y_min -= y_padding
    y_max += y_padding
    
    # Use custom color palette for months instead of HSV
    colors = get_custom_color_palette(12)
    
    from scipy.stats import gaussian_kde
    
    # Plot each available month in its own subplot
    for i, month in enumerate(available_months):
        ax = axes[i]
        month_mask = month_data['Month'] == month
        month_subset = month_data[month_mask]
        
        if len(month_subset) > 0:
            # Get month index (0-11) and corresponding color
            month_idx = month - 1
            color = colors[month_idx]
            
            # Plot scatter points with alpha for transparency
            ax.scatter(
                month_subset['UMAP1'],
                month_subset['UMAP2'],
                c=[color],
                alpha=0.3,
                s=30,
                edgecolor='none'
            )
            
            # Calculate and plot centroid
            centroid_x = month_subset['UMAP1'].mean()
            centroid_y = month_subset['UMAP2'].mean()
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
            
            # Add density contours if we have enough points
            if len(month_subset) > 10:
                try:
                    # Calculate KDE for density estimation
                    xy = np.vstack([month_subset['UMAP1'], month_subset['UMAP2']])
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
                    logging.warning(f"Could not create density contours for month {month}: {e}")
            
            # Set title and limits
            ax.set_title(f"{month_names[month_idx]}", fontsize=16)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Only add x and y labels to bottom and left plots
            if i >= 2:  # Bottom row
                ax.set_xlabel('UMAP Dimension 1', fontsize=12)
            if i % 2 == 0:  # Left column
                ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    
    # Add main title
    fig.suptitle('Seasonal Patterns in Flight Embeddings', fontsize=20, y=0.98)
    
    # # Add text describing the approach
    # fig.text(0.5, 0.01, 
    #          'Each plot shows the distribution of flight embeddings for a single month.\n'
    #          'Centroids (X) indicate the mean position. Contour lines show density of points.',
    #          ha='center', fontsize=12)
    
    # Improve layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.07)
    
    # Save figure
    output_path = output_dir / "seasonal_patterns_small_multiples.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    logging.info(f"Small multiples visualization saved to {output_path}")
    
    # Create a combined visualization with density contours only (no points)
    plt.figure(figsize=(14, 10))
    
    # Plot density contours for each month on the same plot
    for i, month in enumerate(available_months):
        month_mask = month_data['Month'] == month
        month_subset = month_data[month_mask]
        
        if len(month_subset) > 10:
            try:
                # Calculate KDE
                xy = np.vstack([month_subset['UMAP1'], month_subset['UMAP2']])
                kde = gaussian_kde(xy, bw_method='scott')
                
                # Create grid
                xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                z = kde(positions).reshape(xx.shape)
                
                # Get month color and name
                month_idx = month - 1
                color = colors[month_idx]
                
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
                         linewidth=3, label=month_names[month_idx])
                
                # Plot centroid
                centroid_x = month_subset['UMAP1'].mean()
                centroid_y = month_subset['UMAP2'].mean()
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
                logging.warning(f"Could not create density contours for month {month}: {e}")
    
    # Add titles and labels
    plt.title('Comparative Seasonal Patterns - Density Contours Only', fontsize=18)
    plt.xlabel('UMAP Dimension 1', fontsize=14)
    plt.ylabel('UMAP Dimension 2', fontsize=14)
    
    # Set consistent limits
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add legend
    plt.legend(title='Month', title_fontsize=14, fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save the combined visualization
    combined_path = output_dir / "seasonal_patterns_contours_combined.png"
    plt.savefig(combined_path, bbox_inches='tight', dpi=300)
    plt.close()
    logging.info(f"Combined contour visualization saved to {combined_path}")
    
    # Save data as CSV
    csv_path = output_dir / "seasonal_patterns.csv"
    month_data.to_csv(csv_path, index=False)
    logging.info(f"Data saved to {csv_path}")
    
    # Generate a month distribution summary
    month_summary = month_data.groupby(['Month', 'Month_Name']).size().reset_index(name='Count')
    month_summary['Percentage'] = 100 * month_summary['Count'] / month_summary['Count'].sum()
    
    # Add centroid coordinates to the summary
    centroids = month_data.groupby('Month').agg({
        'UMAP1': 'mean',
        'UMAP2': 'mean'
    }).reset_index()
    centroids.columns = ['Month', 'Centroid_X', 'Centroid_Y']
    
    month_summary = pd.merge(month_summary, centroids, on='Month')
    
    # Save month distribution summary
    summary_path = output_dir / "month_distribution.csv"
    month_summary.to_csv(summary_path, index=False)
    logging.info(f"Month distribution summary saved to {summary_path}")


def main():
    """Main function to orchestrate the seasonal pattern visualization process."""
    setup_logging()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    embeddings, df = load_data(
        args.embeddings, args.data, args.sample_size, args.seed
    )
    
    # Validate month data
    validate_month_data(df)
    
    # Apply UMAP for dimensionality reduction
    reduced = apply_umap(embeddings, args.min_dist, args.n_neighbors, args.seed)
    
    # Create and save visualization
    create_seasonal_visualization(reduced, df, args.output_dir)
    
    logging.info("Seasonal pattern visualization complete")
    


if __name__ == "__main__":
    main()