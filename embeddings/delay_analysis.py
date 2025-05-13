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
DEFAULT_OUTPUT_DIR = Path("./embeddings/delay_patterns")
DEFAULT_RANDOM_SEED = 42
DEFAULT_SAMPLE_SIZE = 200_000
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
        description="Generate TabSyn delay pattern analysis visualization"
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


def calculate_delay_category_metrics(
    df: pd.DataFrame
) -> Dict[str, Dict]:
    """Calculate operational metrics for each delay category."""
    delay_metrics = {}
    
    for category in df['Delay_Category'].unique():
        category_mask = df['Delay_Category'] == category
        category_data = df[category_mask]
        
        metrics = {
            'Count': len(category_data),
            'Avg Delay': category_data['DEPARTURE_DELAY_MIN'].mean(),
            'Median Delay': category_data['DEPARTURE_DELAY_MIN'].median(),
            'Duration': category_data['SCHEDULED_DURATION_MIN'].mean() if 'SCHEDULED_DURATION_MIN' in category_data.columns else None
        }
        
        # Add peak hour if available
        if 'SCHEDULED_HOUR' in df.columns:
            hour_counts = category_data['SCHEDULED_HOUR'].value_counts()
            if not hour_counts.empty:
                metrics['Peak Hour'] = hour_counts.idxmax()
            else:
                metrics['Peak Hour'] = None
        
        # Add turnaround if available
        if 'TURNAROUND_MIN' in df.columns:
            metrics['Turnaround'] = category_data['TURNAROUND_MIN'].mean()
        
        delay_metrics[category] = metrics
    
    logging.info(f"Calculated metrics for {len(delay_metrics)} delay categories")
    return delay_metrics


def delay_pattern_analysis(
    embeddings: np.ndarray,
    flight_data: pd.DataFrame,
    output_dir: Path = None,
    umap_reduced: np.ndarray = None,
    min_dist: float = DEFAULT_UMAP_MIN_DIST,
    n_neighbors: int = DEFAULT_UMAP_N_NEIGHBORS,
    seed: int = DEFAULT_RANDOM_SEED
) -> Path:
    """
    Analyze patterns in flight delays as captured by the embeddings.
    
    Args:
        embeddings (np.ndarray): Embedding vectors
        flight_data (pd.DataFrame): Flight data
        output_dir (Path, optional): Directory to save the figure and data
        umap_reduced (np.ndarray, optional): Pre-computed UMAP reduction
        min_dist (float): UMAP min_dist parameter
        n_neighbors (int): UMAP n_neighbors parameter
        seed (int): Random seed
        
    Returns:
        Path: Path to the saved visualization
    """
    if output_dir is None:
        output_dir = Path(DEFAULT_OUTPUT_DIR)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "delay_patterns_subplots.png"
    
    logging.info("Starting delay pattern analysis with separate subplots")
    
    # Apply UMAP if not provided
    if umap_reduced is None:
        reduced = apply_umap(embeddings, min_dist, n_neighbors, seed)
    else:
        reduced = umap_reduced
    
    # Create delay categories
    if 'Delay_Category' not in flight_data.columns:
        bins = [-float('inf'), -15, 0, 15, 30, 60, float('inf')]
        labels = ['Very Early', 'Early', 'On Time', 'Delayed', 'Very Delayed', 'Extremely Delayed']
        
        flight_data['Delay_Category'] = pd.cut(
            flight_data['DEPARTURE_DELAY_MIN'],
            bins=bins,
            labels=labels
        )
    
    # Define delay category colors
    delay_colors = {
        'Very Early': '#1a9850',      # Dark green
        'Early': '#91cf60',           # Light green
        'On Time': '#d9ef8b',         # Yellow-green
        'Delayed': '#fee08b',         # Light orange
        'Very Delayed': '#fc8d59',    # Orange
        'Extremely Delayed': '#d73027' # Red
    }
    
    # Get unique delay categories and their order
    delay_categories = flight_data['Delay_Category'].unique().tolist()

    # the delay_categories should have the following ordering ['Very Early', 'Early', 'On Time', 'Delayed', 'Very Delayed', 'Extremely Delayed']
    delay_categories_order = ['Very Early', 'Early', 'On Time', 'Delayed', 'Very Delayed', 'Extremely Delayed']
    delay_categories = [cat for cat in delay_categories_order if cat in delay_categories]
    
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
    
    # Calculate metrics for each delay category
    delay_metrics = calculate_delay_category_metrics(flight_data)
    
    # ---------------------------------------------------------------------------------
    # Part 1: Create individual subplots for each delay category with density contours
    # ---------------------------------------------------------------------------------
    
    # Determine grid dimensions for subplots
    n_categories = len(delay_categories)
    if n_categories <= 3:
        n_rows, n_cols = 1, n_categories
    elif n_categories <= 6:
        n_rows, n_cols = 2, 3
    else:
        n_rows, n_cols = 3, 3  # Maximum 9 categories
    
    # Create a DataFrame with category and reduced dimensions
    category_data = pd.DataFrame({
        'Category': flight_data['Delay_Category'],
        'UMAP1': reduced[:, 0],
        'UMAP2': reduced[:, 1],
        'Delay': flight_data['DEPARTURE_DELAY_MIN']  # Add raw delay for reference
    })
    
    # Set up figure for individual subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
    
    # Flatten axes if multiple rows or columns
    if n_rows > 1 or n_cols > 1:
        axs = axs.flatten()
    else:
        axs = [axs]  # Make it iterable for single plot case
    
    # Plot each delay category in a separate subplot
    for i, category in enumerate(delay_categories):
        if i >= len(axs):  # Skip if we have more categories than subplots
            logging.warning(f"Too many categories, skipping {category}")
            continue
            
        ax = axs[i]
        category_mask = category_data['Category'] == category
        category_subset = category_data[category_mask]
        
        # Skip if not enough data points
        if len(category_subset) <= 10:
            ax.text(0.5, 0.5, f"Not enough data for {category}", 
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            continue
        
        # Get category color
        color = delay_colors.get(category, '#1f77b4')  # Default to blue if not found
        
        # Create scatter plot with points colored by actual delay value
        scatter = ax.scatter(
            category_subset['UMAP1'],
            category_subset['UMAP2'],
            c=category_subset['Delay'],
            cmap='coolwarm',
            vmin=-30,
            vmax=60,
            alpha=0.5,
            s=30,
            edgecolor='none'
        )
        
        # Add colorbar
        if i == len(delay_categories) - 1 or i == (n_cols - 1):
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label('Departure Delay (minutes)', fontsize=10)
        
        # Calculate and plot centroid
        centroid_x = category_subset['UMAP1'].mean()
        centroid_y = category_subset['UMAP2'].mean()
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
            xy = np.vstack([category_subset['UMAP1'], category_subset['UMAP2']])
            kde = gaussian_kde(xy, bw_method='scott')
            
            # Create a grid for contour plotting
            xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            z = kde(positions).reshape(xx.shape)
            
            # Convert color to RGB tuple for contour levels
            if isinstance(color, str):
                import matplotlib.colors as mcolors
                color_rgb = mcolors.to_rgb(color)
            else:
                color_rgb = color  # Already an RGB tuple
                
            # Plot contours with different alpha levels
            contour = ax.contour(xx, yy, z, 
                                levels=5, 
                                colors=[(color_rgb[0], color_rgb[1], color_rgb[2], level) 
                                        for level in np.linspace(0.5, 1, 5)],
                                linewidths=2)
        except Exception as e:
            logging.warning(f"Could not create density contours for category {category}: {e}")
        
        # Add title
        ax.set_title(f'Delay Category: {category}', fontsize=14)
        
        # Only add x-axis labels to bottom row subplots
        if i >= len(axs) - n_cols:  # Bottom row
            ax.set_xlabel('UMAP Dimension 1', fontsize=10)
        else:
            ax.set_xticklabels([])
            
        # Only add y-axis labels to leftmost column subplots
        if i % n_cols == 0:  # Left column
            ax.set_ylabel('UMAP Dimension 2', fontsize=10)
        else:
            ax.set_yticklabels([])
        
        # Calculate statistics for text box
        stats_text = ""
        
        # # Number of flights
        # n_flights = sum(category_mask)
        # pct_flights = 100 * n_flights / len(category_data)
        # stats_text += f"Flights: {n_flights} ({pct_flights:.1f}%)\n"
        
        # Average and median delay
        avg_delay = delay_metrics[category]['Avg Delay']
        median_delay = delay_metrics[category]['Median Delay']
        stats_text += f"Avg Delay: {avg_delay:.1f} min\n"
        stats_text += f"Median Delay: {median_delay:.1f} min\n"
        
        # Average duration if available
        if 'Duration' in delay_metrics[category] and delay_metrics[category]['Duration'] is not None:
            avg_duration = delay_metrics[category]['Duration']
            stats_text += f"Avg Duration: {avg_duration:.1f} min\n"
        
        # Average turnaround if available
        if 'Turnaround' in delay_metrics[category]:
            avg_turnaround = delay_metrics[category]['Turnaround']
            stats_text += f"Avg Turnaround: {avg_turnaround:.1f} min\n"
        
        # Peak hour if available
        if 'Peak Hour' in delay_metrics[category] and delay_metrics[category]['Peak Hour'] is not None:
            peak_hour = delay_metrics[category]['Peak Hour']
            stats_text += f"Peak Hour: {peak_hour}\n"
        
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
    
    # Hide empty subplots if there are more subplot slots than categories
    for i in range(len(delay_categories), len(axs)):
        axs[i].axis('off')
    
    # Add summary comparison table in the last subplot if there's an empty subplot
    if len(delay_categories) < len(axs):
        # Create comparison table
        table_data = []
        header = ['Category', 'Count', 'Avg Delay', 'Median Delay']
        
        if 'Duration' in delay_metrics[delay_categories[0]] and delay_metrics[delay_categories[0]]['Duration'] is not None:
            header.append('Avg Duration')
        
        if 'Turnaround' in delay_metrics[delay_categories[0]]:
            header.append('Avg Turnaround')
            
        if 'Peak Hour' in delay_metrics[delay_categories[0]]:
            header.append('Peak Hour')
        
        for category in delay_categories:
            row = [
                category,
                f"{delay_metrics[category]['Count']}",
                f"{delay_metrics[category]['Avg Delay']:.1f}",
                f"{delay_metrics[category]['Median Delay']:.1f}"
            ]
            
            if 'Duration' in delay_metrics[category] and delay_metrics[category]['Duration'] is not None:
                row.append(f"{delay_metrics[category]['Duration']:.1f}")
            
            if 'Turnaround' in delay_metrics[category]:
                row.append(f"{delay_metrics[category]['Turnaround']:.1f}")
                
            if 'Peak Hour' in delay_metrics[category] and delay_metrics[category]['Peak Hour'] is not None:
                row.append(f"{delay_metrics[category]['Peak Hour']}")
            
            table_data.append(row)
        
        # Use the last subplot for the table
        axs[-1].axis('off')
        
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
        axs[-1].set_title('Delay Category Comparison', fontsize=14)
    
    # Add main title
    plt.suptitle('Flight Delay Categories Analysis with TabSyn Embeddings', fontsize=18)
    
    # Improve layout
    plt.tight_layout(rect=[0, 0.01, 1, 0.95])
    
    # Save figure
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    logging.info(f"Saved delay pattern subplots visualization to {output_path}")
    
    # ---------------------------------------------------------------------------------
    # Part 2: Create a combined visualization with density contours only (no points)
    # ---------------------------------------------------------------------------------
    plt.figure(figsize=(14, 10))
    
    # Plot density contours for each category on the same plot
    for i, category in enumerate(delay_categories):
        category_mask = category_data['Category'] == category
        category_subset = category_data[category_mask]
        
        if len(category_subset) > 10:
            try:
                # Calculate KDE
                xy = np.vstack([category_subset['UMAP1'], category_subset['UMAP2']])
                kde = gaussian_kde(xy, bw_method='scott')
                
                # Create grid
                xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                z = kde(positions).reshape(xx.shape)
                
                # Get category color
                color = delay_colors.get(category, '#1f77b4')  # Default to blue if not found
                
                # Convert color to RGB tuple if it's a string
                if isinstance(color, str):
                    import matplotlib.colors as mcolors
                    color = mcolors.to_rgb(color)
                
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
                         linewidth=3, label=category)
                
                # Plot centroid
                centroid_x = category_subset['UMAP1'].mean()
                centroid_y = category_subset['UMAP2'].mean()
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
                logging.warning(f"Could not create density contours for category {category}: {e}")
    
    # Add titles and labels
    plt.title('Comparative Delay Categories - Density Contours Only', fontsize=18)
    plt.xlabel('UMAP Dimension 1', fontsize=14)
    plt.ylabel('UMAP Dimension 2', fontsize=14)
    
    # Set consistent limits
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add legend
    plt.legend(title='Delay Category', title_fontsize=14, fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save the combined visualization
    combined_path = output_dir / "delay_contours_combined.png"
    plt.savefig(combined_path, bbox_inches='tight', dpi=300)
    plt.close()
    logging.info(f"Combined contour visualization saved to {combined_path}")
    
    # ---------------------------------------------------------------------------------
    # Part 3: Save additional data files
    # ---------------------------------------------------------------------------------
    
    # Save main dataset with UMAP coordinates
    output_df = flight_data.copy()
    output_df['UMAP1'] = reduced[:, 0]
    output_df['UMAP2'] = reduced[:, 1]
    
    csv_path = output_dir / "delay_patterns_subplots.csv"
    output_df.to_csv(csv_path, index=False)
    
    # Generate a delay metrics summary
    metrics_df = pd.DataFrame.from_dict(delay_metrics, orient='index')
    metrics_df.index.name = 'Delay_Category'
    
    # Add centroid coordinates to the metrics
    centroids = category_data.groupby('Category').agg({
        'UMAP1': 'mean',
        'UMAP2': 'mean'
    }).reset_index()
    centroids.columns = ['Delay_Category', 'Centroid_X', 'Centroid_Y']
    
    # Merge with metrics
    metrics_df = metrics_df.reset_index()
    metrics_df = pd.merge(metrics_df, centroids, on='Delay_Category')
    
    # Save delay metrics summary
    metrics_csv = output_dir / "delay_metrics_summary.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    
    logging.info(f"Saved delay pattern data to {csv_path} and {metrics_csv}")
    
    return output_path

def create_delay_summary_figure(
    flight_data: pd.DataFrame,
    output_dir: Path,
    seed: int = DEFAULT_RANDOM_SEED
) -> Path:
    """
    Create a dedicated figure with three key delay visualizations in a single row:
    1. Distribution of departure delays
    2. Average delay by hour
    3. Turnaround vs departure delay
    
    Args:
        flight_data (pd.DataFrame): Preprocessed flight data
        output_dir (Path): Directory to save the output figure
        seed (int): Random seed for reproducibility
        
    Returns:
        Path: Path to the saved figure
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "delay_summary.png"
    
    logging.info("Creating delay summary figure with three subplots in a row")
    
    # Set up figure with 1x3 subplots (all in a single row)
    # fig, axs = plt.subplots(1, 3, figsize=(16, 7))
    n_rows, n_cols = 1, 3
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
    
    # 1. Delay distribution (histogram)
    axs[0].hist(
        flight_data['DEPARTURE_DELAY_MIN'],
        bins=30,
        range=(-60, 120),
        color='skyblue',
        edgecolor='black',
        alpha=0.7
    )
    
    axs[0].set_title('Distribution of Departure Delays', fontsize=14)
    axs[0].set_xlabel('Delay (minutes)', fontsize=12)
    axs[0].set_ylabel('Frequency', fontsize=12)
    
    # Add vertical line at zero
    axs[0].axvline(x=0, color='red', linestyle='--')
    
    # Add grid
    axs[0].grid(True, linestyle='--', alpha=0.6)
    
    # 2. Average delay by hour (bar chart)
    # Calculate average delay by hour
    hour_groups = flight_data.groupby('SCHEDULED_HOUR')
    avg_delay_by_hour = hour_groups['DEPARTURE_DELAY_MIN'].mean()
    
    # Create hour axis
    hours = sorted(flight_data['SCHEDULED_HOUR'].unique())
    
    axs[1].bar(
        hours,
        avg_delay_by_hour[hours],
        color='skyblue',
        alpha=0.7,
        width=0.7
    )
    
    axs[1].set_title('Average Delay by Hour of Day', fontsize=14)
    axs[1].set_xlabel('Hour of Day', fontsize=12)
    axs[1].set_ylabel('Average Delay (minutes)', fontsize=12)
    axs[1].set_xticks(hours[::2])  # Show every other hour for clarity
    
    # Add horizontal line at zero
    axs[1].axhline(y=0, color='red', linestyle='--')
    
    # Add grid
    axs[1].grid(True, linestyle='--', alpha=0.6)
    
    # 3. Turnaround vs departure delay (scatter plot)
    if 'TURNAROUND_MIN' in flight_data.columns and flight_data['TURNAROUND_MIN'].notna().any():
        # Plot turnaround vs delay
        scatter = axs[2].scatter(
            flight_data['TURNAROUND_MIN'],
            flight_data['DEPARTURE_DELAY_MIN'],
            alpha=0.5,
            s=20,
            c=flight_data['DEPARTURE_DELAY_MIN'],
            cmap='coolwarm',
            vmin=-30,
            vmax=60
        )
        
        # Add regression line
        from scipy import stats
        
        # Filter out extreme values for better regression fit
        valid_mask = (
            (flight_data['TURNAROUND_MIN'] < 300) &
            (flight_data['DEPARTURE_DELAY_MIN'] > -60) &
            (flight_data['DEPARTURE_DELAY_MIN'] < 120)
        )
        
        if sum(valid_mask) > 10:  # Only compute regression if enough data points
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                flight_data.loc[valid_mask, 'TURNAROUND_MIN'],
                flight_data.loc[valid_mask, 'DEPARTURE_DELAY_MIN']
            )
            
            x_range = np.linspace(0, flight_data['TURNAROUND_MIN'].quantile(0.95), 100)
            axs[2].plot(
                x_range,
                intercept + slope * x_range,
                'r-',
                linewidth=2,
                label=f'Slope: {slope:.3f}, R²: {r_value**2:.3f}'
            )
            
            axs[2].legend(fontsize=10)
        
        axs[2].set_title('Turnaround Time vs Departure Delay', fontsize=14)
        axs[2].set_xlabel('Turnaround Time (minutes)', fontsize=12)
        axs[2].set_ylabel('Departure Delay (minutes)', fontsize=12)
        
        # Add horizontal line at zero
        axs[2].axhline(y=0, color='black', linestyle='--')
        
        # Limit view to reasonable values
        axs[2].set_xlim(0, min(300, flight_data['TURNAROUND_MIN'].quantile(0.999)))
        axs[2].set_ylim(-30, min(120, flight_data['DEPARTURE_DELAY_MIN'].quantile(0.999)))
        
        # Add grid
        axs[2].grid(True, linestyle='--', alpha=0.6)
        
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=axs[2])
        cbar.set_label('Departure Delay (minutes)', fontsize=10)
    else:
        # Calculate percentage of delayed flights by hour (delay > 15 min) if turnaround data not available
        delayed_by_hour = {}
        for hour, group in hour_groups:
            delayed_by_hour[hour] = 100 * (group['DEPARTURE_DELAY_MIN'] > 15).mean()
        delayed_by_hour = pd.Series(delayed_by_hour)
        
        # Plot percentage of delayed flights by hour
        axs[2].bar(
            hours,
            delayed_by_hour[hours],
            color='orange',
            alpha=0.7,
            width=0.7
        )
        
        axs[2].set_title('Percentage of Delayed Flights by Hour', fontsize=14)
        axs[2].set_xlabel('Hour of Day', fontsize=12)
        axs[2].set_ylabel('Percentage of Flights Delayed (>15 min)', fontsize=12)
        axs[2].set_xticks(hours[::2])  # Show every other hour for clarity
        
        # Add grid
        axs[2].grid(True, linestyle='--', alpha=0.6)
    
    # Add main title
    plt.suptitle('Flight Delay Summary', fontsize=18)
    
    # Improve layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save figure
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    logging.info(f"Saved delay summary figure to {output_path}")
    
    return output_path

def main():
    """Main function to orchestrate the delay pattern analysis process."""
    setup_logging()
    args = parse_args()
    
    # Load data
    embeddings, flight_data = load_data(
        args.embeddings, args.data, args.sample_size, args.seed
    )
    
    # Run delay pattern analysis
    delay_pattern_analysis(
        embeddings=embeddings,
        flight_data=flight_data,
        output_dir=args.output_dir,
        min_dist=args.min_dist,
        n_neighbors=args.n_neighbors,
        seed=args.seed
    )
    
    # Create dedicated delay summary figure
    create_delay_summary_figure(
        flight_data=flight_data,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    logging.info("Delay pattern analysis complete")

if __name__ == "__main__":
    main()