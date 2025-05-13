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
DEFAULT_OUTPUT_DIR = Path("./embeddings/turnaround_analysis")
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
        description="Generate TabSyn turnaround time analysis visualization"
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


def check_turnaround_data(flight_data: pd.DataFrame) -> bool:
    """Check if turnaround data is available in the dataset."""
    if 'TURNAROUND_MIN' not in flight_data.columns:
        logging.warning("Turnaround time data not available in the dataset.")
        return False
    
    if not flight_data['TURNAROUND_MIN'].notna().any():
        logging.warning("Turnaround time data contains only NaN values.")
        return False
    
    logging.info(f"Found {flight_data['TURNAROUND_MIN'].notna().sum()} valid turnaround time records.")
    return True


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


def calculate_turnaround_category_metrics(
    df: pd.DataFrame
) -> Dict[str, Dict]:
    """Calculate operational metrics for each turnaround category."""
    turnaround_metrics = {}
    
    for category in df['Turnaround_Category'].unique():
        category_mask = df['Turnaround_Category'] == category
        category_data = df[category_mask]
        
        metrics = {
            'Count': len(category_data),
            'Avg Turnaround': category_data['TURNAROUND_MIN'].mean(),
            'Median Turnaround': category_data['TURNAROUND_MIN'].median(),
            'Duration': category_data['SCHEDULED_DURATION_MIN'].mean() if 'SCHEDULED_DURATION_MIN' in category_data.columns else None
        }
        
        # Add peak hour if available
        if 'SCHEDULED_HOUR' in df.columns:
            hour_counts = category_data['SCHEDULED_HOUR'].value_counts()
            if not hour_counts.empty:
                metrics['Peak Hour'] = hour_counts.idxmax()
            else:
                metrics['Peak Hour'] = None
        
        # Add delay if available
        if 'DEPARTURE_DELAY_MIN' in df.columns:
            metrics['Avg Delay'] = category_data['DEPARTURE_DELAY_MIN'].mean()
            metrics['On-Time %'] = 100 * (category_data['DEPARTURE_DELAY_MIN'] <= 15).mean()
        
        turnaround_metrics[category] = metrics
    
    logging.info(f"Calculated metrics for {len(turnaround_metrics)} turnaround categories")
    return turnaround_metrics


def turnaround_analysis(
    embeddings: np.ndarray,
    flight_data: pd.DataFrame,
    output_dir: Path = None,
    umap_reduced: np.ndarray = None,
    min_dist: float = DEFAULT_UMAP_MIN_DIST,
    n_neighbors: int = DEFAULT_UMAP_N_NEIGHBORS,
    seed: int = DEFAULT_RANDOM_SEED
) -> Optional[Path]:
    """
    Analyze patterns in turnaround times as captured by the embeddings.
    
    Args:
        embeddings (np.ndarray): Embedding vectors
        flight_data (pd.DataFrame): Flight data
        output_dir (Path, optional): Directory to save the figure and data
        umap_reduced (np.ndarray, optional): Pre-computed UMAP reduction
        min_dist (float): UMAP min_dist parameter
        n_neighbors (int): UMAP n_neighbors parameter
        seed (int): Random seed
        
    Returns:
        Optional[Path]: Path to the saved visualization if successful, None otherwise
    """
    if output_dir is None:
        output_dir = Path(DEFAULT_OUTPUT_DIR)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "turnaround_patterns_subplots.png"
    
    logging.info("Starting turnaround time analysis with separate subplots")
    
    # Check if turnaround data is available
    if not check_turnaround_data(flight_data):
        logging.error("Cannot perform turnaround analysis without turnaround data.")
        return None
    
    # Apply UMAP if not provided
    if umap_reduced is None:
        reduced = apply_umap(embeddings, min_dist, n_neighbors, seed)
    else:
        reduced = umap_reduced
    
    # Create turnaround categories if not already there
    if 'Turnaround_Category' not in flight_data.columns:
        bins = [0, 30, 60, 90, 120, float('inf')]
        labels = ['Very Short', 'Short', 'Medium', 'Long', 'Very Long']
        
        flight_data['Turnaround_Category'] = pd.cut(
            flight_data['TURNAROUND_MIN'],
            bins=bins,
            labels=labels
        )
    
    # Define turnaround category colors
    turnaround_colors = {
        'Very Short': '#1a9850',    # Dark green
        'Short': '#91cf60',         # Light green
        'Medium': '#ffffbf',        # Yellow
        'Long': '#fc8d59',          # Orange
        'Very Long': '#d73027'      # Red
    }
    
    # Get unique turnaround categories and their order
    turnaround_categories = flight_data['Turnaround_Category'].unique().tolist()

    # The turnaround_categories should have the following ordering
    turnaround_categories_order = ['Very Short', 'Short', 'Medium', 'Long', 'Very Long']
    turnaround_categories = [cat for cat in turnaround_categories_order if cat in turnaround_categories]
    
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
    
    # Calculate metrics for each turnaround category
    turnaround_metrics = calculate_turnaround_category_metrics(flight_data)
    
    # ---------------------------------------------------------------------------------
    # Part 1: Create individual subplots for each turnaround category with density contours
    # ---------------------------------------------------------------------------------
    
    # Determine grid dimensions for subplots
    n_categories = len(turnaround_categories)
    if n_categories <= 3:
        n_rows, n_cols = 1, n_categories
    elif n_categories <= 6:
        n_rows, n_cols = 2, 3
    else:
        n_rows, n_cols = 3, 3  # Maximum 9 categories
    
    # Create a DataFrame with category and reduced dimensions
    category_data = pd.DataFrame({
        'Category': flight_data['Turnaround_Category'],
        'UMAP1': reduced[:, 0],
        'UMAP2': reduced[:, 1],
        'Turnaround': flight_data['TURNAROUND_MIN'],  # Add raw turnaround for reference
        'Delay': flight_data['DEPARTURE_DELAY_MIN'] if 'DEPARTURE_DELAY_MIN' in flight_data.columns else np.nan
    })
    
    # Set up figure for individual subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
    
    # Flatten axes if multiple rows or columns
    if n_rows > 1 or n_cols > 1:
        axs = axs.flatten()
    else:
        axs = [axs]  # Make it iterable for single plot case
    
    # Plot each turnaround category in a separate subplot
    for i, category in enumerate(turnaround_categories):
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
        color = turnaround_colors.get(category, '#1f77b4')  # Default to blue if not found
        
        # Create scatter plot with points colored by actual turnaround value
        color_values = category_subset['Delay'] if 'Delay' in category_subset and not category_subset['Delay'].isna().all() else category_subset['Turnaround']
        scatter = ax.scatter(
            category_subset['UMAP1'],
            category_subset['UMAP2'],
            c=color_values,
            cmap='coolwarm',
            vmin=color_values.quantile(0.05),
            vmax=color_values.quantile(0.95),
            alpha=0.5,
            s=30,
            edgecolor='none'
        )
        
        # Add colorbar
        if i == len(turnaround_categories) - 1 or i == (n_cols - 1):
            cbar = fig.colorbar(scatter, ax=ax)
            cbar_label = 'Departure Delay (minutes)' if 'Delay' in category_subset and not category_subset['Delay'].isna().all() else 'Turnaround Time (minutes)'
            cbar.set_label(cbar_label, fontsize=10)
        
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
        ax.set_title(f'Turnaround Category: {category}', fontsize=14)
        
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
        
        # Average and median turnaround
        avg_turnaround = turnaround_metrics[category]['Avg Turnaround']
        median_turnaround = turnaround_metrics[category]['Median Turnaround']
        stats_text += f"Avg Turnaround: {avg_turnaround:.1f} min\n"
        stats_text += f"Median Turnaround: {median_turnaround:.1f} min\n"
        
        # Average duration if available
        if 'Duration' in turnaround_metrics[category] and turnaround_metrics[category]['Duration'] is not None:
            avg_duration = turnaround_metrics[category]['Duration']
            stats_text += f"Avg Duration: {avg_duration:.1f} min\n"
        
        # Average delay if available
        if 'Avg Delay' in turnaround_metrics[category]:
            avg_delay = turnaround_metrics[category]['Avg Delay']
            stats_text += f"Avg Delay: {avg_delay:.1f} min\n"
        
        # On-time percentage if available
        if 'On-Time %' in turnaround_metrics[category]:
            on_time_pct = turnaround_metrics[category]['On-Time %']
            stats_text += f"On-Time %: {on_time_pct:.1f}%\n"
        
        # Peak hour if available
        if 'Peak Hour' in turnaround_metrics[category] and turnaround_metrics[category]['Peak Hour'] is not None:
            peak_hour = turnaround_metrics[category]['Peak Hour']
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
    for i in range(len(turnaround_categories), len(axs)):
        axs[i].axis('off')
    
    # Add main title
    plt.suptitle('Flight Turnaround Categories Analysis with TabSyn Embeddings', fontsize=18)
    
    # Improve layout
    plt.tight_layout(rect=[0, 0.01, 1, 0.95])
    
    # Save figure
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    logging.info(f"Saved turnaround pattern subplots visualization to {output_path}")
    
    # ---------------------------------------------------------------------------------
    # Part 2: Create a combined visualization with density contours only (no points)
    # ---------------------------------------------------------------------------------
    plt.figure(figsize=(14, 10))
    
    # Plot density contours for each category on the same plot
    for i, category in enumerate(turnaround_categories):
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
                color = turnaround_colors.get(category, '#1f77b4')  # Default to blue if not found
                
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
    plt.title('Comparative Turnaround Categories - Density Contours', fontsize=18)
    plt.xlabel('UMAP Dimension 1', fontsize=14)
    plt.ylabel('UMAP Dimension 2', fontsize=14)
    
    # Set consistent limits
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add legend
    plt.legend(title='Turnaround Category', title_fontsize=14, fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save the combined visualization
    combined_path = output_dir / "turnaround_contours_combined.png"
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
    
    csv_path = output_dir / "turnaround_patterns_data.csv"
    output_df.to_csv(csv_path, index=False)
    
    # Generate a turnaround metrics summary
    metrics_df = pd.DataFrame.from_dict(turnaround_metrics, orient='index')
    metrics_df.index.name = 'Turnaround_Category'
    
    # Add centroid coordinates to the metrics
    centroids = category_data.groupby('Category').agg({
        'UMAP1': 'mean',
        'UMAP2': 'mean'
    }).reset_index()
    centroids.columns = ['Turnaround_Category', 'Centroid_X', 'Centroid_Y']
    
    # Merge with metrics
    metrics_df = metrics_df.reset_index()
    metrics_df = pd.merge(metrics_df, centroids, on='Turnaround_Category')
    
    # Save turnaround metrics summary
    metrics_csv = output_dir / "turnaround_metrics_summary.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    
    logging.info(f"Saved turnaround pattern data to {csv_path} and {metrics_csv}")
    
    return output_path


def create_turnaround_summary_figure(
    flight_data: pd.DataFrame,
    output_dir: Path,
    seed: int = DEFAULT_RANDOM_SEED
) -> Path:
    """
    Create a dedicated figure with three key turnaround visualizations in a single row:
    1. Distribution of turnaround times
    2. Average turnaround by hour
    3. Turnaround vs departure delay
    
    Args:
        flight_data (pd.DataFrame): Preprocessed flight data
        output_dir (Path): Directory to save the output figure
        seed (int): Random seed for reproducibility
        
    Returns:
        Path: Path to the saved figure
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "turnaround_summary.png"
    
    logging.info("Creating turnaround summary figure with three subplots in a row")
    
    # Set up figure with 1x3 subplots (all in a single row)
    n_rows, n_cols = 1, 3
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
    
    # 1. Turnaround distribution (histogram)
    axs[0].hist(
        flight_data['TURNAROUND_MIN'],
        bins=30,
        range=(0, flight_data['TURNAROUND_MIN'].quantile(0.95)),
        color='skyblue',
        edgecolor='black',
        alpha=0.7
    )
    
    axs[0].set_title('Distribution of Turnaround Times', fontsize=14)
    axs[0].set_xlabel('Turnaround (minutes)', fontsize=12)
    axs[0].set_ylabel('Frequency', fontsize=12)
    
    # Add grid
    axs[0].grid(True, linestyle='--', alpha=0.6)
    
    # 2. Average turnaround by hour (bar chart)
    # Calculate average turnaround by hour
    hour_groups = flight_data.groupby('SCHEDULED_HOUR')
    avg_turnaround_by_hour = hour_groups['TURNAROUND_MIN'].mean()
    
    # Create hour axis
    hours = sorted(flight_data['SCHEDULED_HOUR'].unique())
    
    axs[1].bar(
        hours,
        avg_turnaround_by_hour[hours],
        color='lightgreen',
        alpha=0.7,
        width=0.7
    )
    
    axs[1].set_title('Average Turnaround Time by Hour of Day', fontsize=14)
    axs[1].set_xlabel('Hour of Day', fontsize=12)
    axs[1].set_ylabel('Average Turnaround (minutes)', fontsize=12)
    axs[1].set_xticks(hours[::2])  # Show every other hour for clarity
    
    # Add grid
    axs[1].grid(True, linestyle='--', alpha=0.6)
    
    # 3. Turnaround vs departure delay (scatter plot)
    if 'DEPARTURE_DELAY_MIN' in flight_data.columns and flight_data['DEPARTURE_DELAY_MIN'].notna().any():
        # Plot turnaround vs delay
        scatter = axs[2].scatter(
            flight_data['TURNAROUND_MIN'],
            flight_data['DEPARTURE_DELAY_MIN'],
            alpha=0.5,
            s=20,
            c=flight_data['SCHEDULED_HOUR'],
            cmap='viridis'
        )
        
        # Add regression line
        from scipy import stats
        
        # Filter out extreme values for better regression fit
        valid_mask = (
            (flight_data['TURNAROUND_MIN'] < flight_data['TURNAROUND_MIN'].quantile(0.99)) &
            (flight_data['DEPARTURE_DELAY_MIN'] > flight_data['DEPARTURE_DELAY_MIN'].quantile(0.01)) &
            (flight_data['DEPARTURE_DELAY_MIN'] < flight_data['DEPARTURE_DELAY_MIN'].quantile(0.99))
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
        axs[2].set_xlim(0, flight_data['TURNAROUND_MIN'].quantile(0.95))
        axs[2].set_ylim(flight_data['DEPARTURE_DELAY_MIN'].quantile(0.01), 
                        flight_data['DEPARTURE_DELAY_MIN'].quantile(0.99))
        
        # Add grid
        axs[2].grid(True, linestyle='--', alpha=0.6)
        
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=axs[2])
        cbar.set_label('Hour of Day', fontsize=10)
    else:
        # If no delay data, show on-time percentage by turnaround category
        if 'Turnaround_Category' in flight_data.columns:
            # Calculate on-time percentage by category (assuming on-time means less than 15 min delay)
            on_time_by_category = {}
            for category in flight_data['Turnaround_Category'].unique():
                subset = flight_data[flight_data['Turnaround_Category'] == category]
                on_time_by_category[category] = 100 * subset['ON_TIME'].mean() if 'ON_TIME' in subset.columns else 0
            
            categories = list(on_time_by_category.keys())
            on_time_values = [on_time_by_category[cat] for cat in categories]
            
            # Plot on-time percentage by category
            axs[2].bar(
                categories,
                on_time_values,
                color='orange',
                alpha=0.7,
                width=0.7
            )
            
            axs[2].set_title('On-Time Performance by Turnaround Category', fontsize=14)
            axs[2].set_xlabel('Turnaround Category', fontsize=12)
            axs[2].set_ylabel('On-Time Percentage', fontsize=12)
            
            # Rotate x-tick labels for better readability
            plt.setp(axs[2].get_xticklabels(), rotation=45, ha='right')
            
            # Add grid
            axs[2].grid(True, linestyle='--', alpha=0.6)
        else:
            # If no categorical data either, show turnaround by aircraft type or carrier
            if 'CARRIER' in flight_data.columns:
                # Calculate average turnaround by carrier
                carrier_turnaround = flight_data.groupby('CARRIER')['TURNAROUND_MIN'].mean().sort_values(ascending=False)
                
                # Take top 10 carriers if there are many
                if len(carrier_turnaround) > 10:
                    carrier_turnaround = carrier_turnaround.head(10)
                
                # Plot average turnaround by carrier
                axs[2].bar(
                    carrier_turnaround.index,
                    carrier_turnaround.values,
                    color='purple',
                    alpha=0.7,
                    width=0.7
                )
                
                axs[2].set_title('Average Turnaround by Carrier', fontsize=14)
                axs[2].set_xlabel('Carrier', fontsize=12)
                axs[2].set_ylabel('Average Turnaround (minutes)', fontsize=12)
                
                # Rotate x-tick labels for better readability
                plt.setp(axs[2].get_xticklabels(), rotation=45, ha='right')
                
                # Add grid
                axs[2].grid(True, linestyle='--', alpha=0.6)
            else:
                # Placeholder if no relevant data is available
                axs[2].text(0.5, 0.5, "Insufficient data for additional analysis", 
                           ha='center', va='center', fontsize=14, transform=axs[2].transAxes)
                axs[2].axis('off')
    
    # Add main title
    plt.suptitle('Flight Turnaround Time Summary', fontsize=18)
    
    # Improve layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save figure
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    logging.info(f"Saved turnaround summary figure to {output_path}")
    
    return output_path


def main():
    """Main function to orchestrate the turnaround analysis process."""
    setup_logging()
    args = parse_args()
    
    # Load data
    embeddings, flight_data = load_data(
        args.embeddings, args.data, args.sample_size, args.seed
    )
    
    # Run turnaround analysis with the updated approach
    result = turnaround_analysis(
        embeddings=embeddings,
        flight_data=flight_data,
        output_dir=args.output_dir,
        min_dist=args.min_dist,
        n_neighbors=args.n_neighbors,
        seed=args.seed
    )
    
    if result:
        # Create dedicated turnaround summary figure
        create_turnaround_summary_figure(
            flight_data=flight_data,
            output_dir=args.output_dir,
            seed=args.seed
        )
        
        logging.info("Turnaround analysis complete")
    else:
        logging.warning("Turnaround analysis could not be completed due to missing data")


if __name__ == "__main__":
    main()