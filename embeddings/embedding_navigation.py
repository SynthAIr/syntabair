import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn.linear_model import LinearRegression

from syntabair.preprocessing import preprocess_flight_data_for_prediction


# Configuration constants
DEFAULT_OUTPUT_DIR = Path("./embeddings/embedding_navigation")
DEFAULT_RANDOM_SEED = 42
DEFAULT_SAMPLE_SIZE = 100_000


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate TabSyn embedding space navigation visualizations"
    )
    parser.add_argument("--embeddings", type=Path, required=True,
                        help="Path to TabSyn embeddings file (e.g., train_z.npy)")
    parser.add_argument("--data", type=Path, required=True,
                        help="Path to flight data CSV file")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to save outputs (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
                        help=f"Number of records to sample (default: {DEFAULT_SAMPLE_SIZE})")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED,
                        help=f"Random seed (default: {DEFAULT_RANDOM_SEED})")
    return parser.parse_args()


def load_data(
    embeddings_path: Path,
    data_path: Path,
    sample_size: Optional[int],
    seed: int
) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, np.ndarray]]:
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


def create_embedding_navigation_visualization(
    embeddings: np.ndarray,
    flight_data: pd.DataFrame,
    output_dir: Path
) -> None:
    """Create visualization of feature directions in embedding space."""
    logging.info("Creating embedding space navigation visualization")
    output_path = output_dir / "embedding_navigation.png"

    logging.info("Applying UMAP for dimensionality reduction")
    reducer = umap.UMAP(n_components=2, random_state=42)
    reduced = reducer.fit_transform(embeddings)
    
    # Set up figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 14))
    
    # Create dataframe for analysis
    df = pd.DataFrame({
        'Dimension 1': reduced[:, 0],
        'Dimension 2': reduced[:, 1],
        'Delay': flight_data['DEPARTURE_DELAY_MIN'],
        'Duration': flight_data['SCHEDULED_DURATION_MIN'],
        'Hour': flight_data['SCHEDULED_HOUR'],
    })
    
    # Add turnaround if available
    has_turnaround = False
    if 'TURNAROUND_MIN' in flight_data.columns and flight_data['TURNAROUND_MIN'].notna().any():
        df['Turnaround'] = flight_data['TURNAROUND_MIN']
        has_turnaround = True
    
    logging.info("Fitting linear models for feature directions")
    
    # 1. Delay direction in embedding space
    X = reduced
    y = flight_data['DEPARTURE_DELAY_MIN'].values
    
    delay_model = LinearRegression()
    delay_model.fit(X, y)
    
    # Create a grid for plotting the delay direction
    x_min, x_max = reduced[:, 0].min() - 1, reduced[:, 0].max() + 1
    y_min, y_max = reduced[:, 1].min() - 1, reduced[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    # Predict delay for each point in the grid
    Z = delay_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot delay contours
    contour = axs[0, 0].contourf(xx, yy, Z, cmap='coolwarm', alpha=0.5)
    
    # Plot original points colored by delay
    scatter = axs[0, 0].scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=flight_data['DEPARTURE_DELAY_MIN'],
        cmap='coolwarm',
        vmin=-30,
        vmax=60,
        alpha=0.7,
        s=30,
        edgecolor='none'
    )
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=axs[0, 0])
    cbar.set_label('Delay (minutes)', fontsize=10)
    
    # Show the direction of increasing delay
    arrow_start = (reduced[:, 0].mean(), reduced[:, 1].mean())
    arrow_direction = delay_model.coef_ / np.linalg.norm(delay_model.coef_) * 2
    
    axs[0, 0].arrow(
        arrow_start[0], arrow_start[1],
        arrow_direction[0], arrow_direction[1],
        head_width=0.3,
        head_length=0.3,
        fc='black',
        ec='black',
        linewidth=2
    )
    
    # Label the direction
    axs[0, 0].text(
        arrow_start[0] + arrow_direction[0] * 1.2,
        arrow_start[1] + arrow_direction[1] * 1.2,
        'Increasing Delay',
        fontsize=12,
        ha='center',
        va='center',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8)
    )
    
    axs[0, 0].set_title('Delay Direction in TabSyn Embedding Space', fontsize=14)
    axs[0, 0].set_xlabel('UMAP1', fontsize=12)
    axs[0, 0].set_ylabel('UMAP2', fontsize=12)
    
    # 2. Hour of day direction
    y_hour = flight_data['SCHEDULED_HOUR'].values
    
    hour_model = LinearRegression()
    hour_model.fit(X, y_hour)
    
    Z_hour = hour_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_hour = Z_hour.reshape(xx.shape)
    
    contour_hour = axs[0, 1].contourf(xx, yy, Z_hour, cmap='viridis', alpha=0.5)
    
    scatter_hour = axs[0, 1].scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=flight_data['SCHEDULED_HOUR'],
        cmap='viridis',
        alpha=0.7,
        s=30,
        edgecolor='none'
    )
    
    cbar_hour = fig.colorbar(scatter_hour, ax=axs[0, 1])
    cbar_hour.set_label('Hour of Day', fontsize=10)
    
    arrow_direction_hour = hour_model.coef_ / np.linalg.norm(hour_model.coef_) * 2
    
    axs[0, 1].arrow(
        arrow_start[0], arrow_start[1],
        arrow_direction_hour[0], arrow_direction_hour[1],
        head_width=0.3,
        head_length=0.3,
        fc='black',
        ec='black',
        linewidth=2
    )
    
    axs[0, 1].text(
        arrow_start[0] + arrow_direction_hour[0] * 1.2,
        arrow_start[1] + arrow_direction_hour[1] * 1.2,
        'Later Hours',
        fontsize=12,
        ha='center',
        va='center',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8)
    )
    
    axs[0, 1].set_title('Hour of Day Direction in TabSyn Embedding Space', fontsize=14)
    axs[0, 1].set_xlabel('UMAP1', fontsize=12)
    axs[0, 1].set_ylabel('UMAP2', fontsize=12)
    
    # 3. Duration direction
    y_duration = flight_data['SCHEDULED_DURATION_MIN'].values
    
    duration_model = LinearRegression()
    duration_model.fit(X, y_duration)
    
    Z_duration = duration_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_duration = Z_duration.reshape(xx.shape)
    
    contour_duration = axs[1, 0].contourf(xx, yy, Z_duration, cmap='plasma', alpha=0.5)
    
    scatter_duration = axs[1, 0].scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=flight_data['SCHEDULED_DURATION_MIN'],
        cmap='plasma',
        alpha=0.7,
        s=30,
        edgecolor='none'
    )
    
    cbar_duration = fig.colorbar(scatter_duration, ax=axs[1, 0])
    cbar_duration.set_label('Duration (minutes)', fontsize=10)
    
    arrow_direction_duration = duration_model.coef_ / np.linalg.norm(duration_model.coef_) * 2
    
    axs[1, 0].arrow(
        arrow_start[0], arrow_start[1],
        arrow_direction_duration[0], arrow_direction_duration[1],
        head_width=0.3,
        head_length=0.3,
        fc='black',
        ec='black',
        linewidth=2
    )
    
    axs[1, 0].text(
        arrow_start[0] + arrow_direction_duration[0] * 1.2,
        arrow_start[1] + arrow_direction_duration[1] * 1.2,
        'Longer Duration',
        fontsize=12,
        ha='center',
        va='center',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8)
    )
    
    axs[1, 0].set_title('Flight Duration Direction in TabSyn Embedding Space', fontsize=14)
    axs[1, 0].set_xlabel('UMAP1', fontsize=12)
    axs[1, 0].set_ylabel('UMAP2', fontsize=12)
    
    # 4. Turnaround direction (if available) or combined directions
    if has_turnaround:
        # Turnaround direction
        logging.info("Adding turnaround direction plot")
        y_turnaround = flight_data['TURNAROUND_MIN'].values
        
        turnaround_model = LinearRegression()
        turnaround_model.fit(X, y_turnaround)
        
        Z_turnaround = turnaround_model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z_turnaround = Z_turnaround.reshape(xx.shape)
        
        contour_turnaround = axs[1, 1].contourf(xx, yy, Z_turnaround, cmap='Oranges', alpha=0.5)
        
        scatter_turnaround = axs[1, 1].scatter(
            reduced[:, 0],
            reduced[:, 1],
            c=flight_data['TURNAROUND_MIN'],
            cmap='Oranges',
            alpha=0.7,
            s=30,
            edgecolor='none'
        )
        
        cbar_turnaround = fig.colorbar(scatter_turnaround, ax=axs[1, 1])
        cbar_turnaround.set_label('Turnaround Time (minutes)', fontsize=10)
        
        arrow_direction_turnaround = turnaround_model.coef_ / np.linalg.norm(turnaround_model.coef_) * 2
        
        axs[1, 1].arrow(
            arrow_start[0], arrow_start[1],
            arrow_direction_turnaround[0], arrow_direction_turnaround[1],
            head_width=0.3,
            head_length=0.3,
            fc='black',
            ec='black',
            linewidth=2
        )
        
        axs[1, 1].text(
            arrow_start[0] + arrow_direction_turnaround[0] * 1.2,
            arrow_start[1] + arrow_direction_turnaround[1] * 1.2,
            'Longer Turnaround',
            fontsize=12,
            ha='center',
            va='center',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8)
        )
        
        axs[1, 1].set_title('Turnaround Time Direction in TabSyn Embedding Space', fontsize=14)
        axs[1, 1].set_xlabel('UMAP1', fontsize=12)
        axs[1, 1].set_ylabel('UMAP2', fontsize=12)
    else:
        # Combined direction plot
        logging.info("Creating combined direction plot")
        axs[1, 1].scatter(
            reduced[:, 0],
            reduced[:, 1],
            c='lightgray',
            alpha=0.3,
            s=20,
            edgecolor='none'
        )
        
        # Add all direction arrows
        arrow_scale = 2.5
        
        # Delay arrow
        axs[1, 1].arrow(
            arrow_start[0], arrow_start[1],
            arrow_direction[0] * arrow_scale, arrow_direction[1] * arrow_scale,
            head_width=0.3,
            head_length=0.3,
            fc='red',
            ec='red',
            linewidth=2
        )
        
        axs[1, 1].text(
            arrow_start[0] + arrow_direction[0] * arrow_scale * 1.2,
            arrow_start[1] + arrow_direction[1] * arrow_scale * 1.2,
            'Delay',
            fontsize=12,
            ha='center',
            va='center',
            color='red',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8)
        )
        
        # Hour arrow
        axs[1, 1].arrow(
            arrow_start[0], arrow_start[1],
            arrow_direction_hour[0] * arrow_scale, arrow_direction_hour[1] * arrow_scale,
            head_width=0.3,
            head_length=0.3,
            fc='green',
            ec='green',
            linewidth=2
        )
        
        axs[1, 1].text(
            arrow_start[0] + arrow_direction_hour[0] * arrow_scale * 1.2,
            arrow_start[1] + arrow_direction_hour[1] * arrow_scale * 1.2,
            'Hour',
            fontsize=12,
            ha='center',
            va='center',
            color='green',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8)
        )
        
        # Duration arrow
        axs[1, 1].arrow(
            arrow_start[0], arrow_start[1],
            arrow_direction_duration[0] * arrow_scale, arrow_direction_duration[1] * arrow_scale,
            head_width=0.3,
            head_length=0.3,
            fc='blue',
            ec='blue',
            linewidth=2
        )
        
        axs[1, 1].text(
            arrow_start[0] + arrow_direction_duration[0] * arrow_scale * 1.2,
            arrow_start[1] + arrow_direction_duration[1] * arrow_scale * 1.2,
            'Duration',
            fontsize=12,
            ha='center',
            va='center',
            color='blue',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8)
        )
        
        axs[1, 1].set_title('Combined Feature Directions in TabSyn Embedding Space', fontsize=14)
        axs[1, 1].set_xlabel('Dimension 1', fontsize=12)
        axs[1, 1].set_ylabel('Dimension 2', fontsize=12)
    
    # Add main title
    plt.suptitle('Navigating the TabSyn Embedding Space', fontsize=18)
    
    # Improve layout
    plt.tight_layout(rect=[0, 0.01, 1, 0.95])
    
    # Save figure
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    logging.info(f"Saved visualization to {output_path}")
    
    # Save data as CSV
    # Main dataset with embeddings and features
    df_to_save = pd.DataFrame({
        'Dimension 1': reduced[:, 0],
        'Dimension 2': reduced[:, 1],
        'Delay': flight_data['DEPARTURE_DELAY_MIN'],
        'Hour': flight_data['SCHEDULED_HOUR'],
        'Duration': flight_data['SCHEDULED_DURATION_MIN']
    })
    
    if has_turnaround:
        df_to_save['Turnaround'] = flight_data['TURNAROUND_MIN']
    
    csv_path = output_path.with_suffix('.csv')
    df_to_save.to_csv(csv_path, index=False)
    logging.info(f"Saved data to {csv_path}")
    
    # Save direction models
    direction_data = {
        'Feature': ['Delay', 'Hour', 'Duration'],
        'Dimension_1_Direction': [
            delay_model.coef_[0], 
            hour_model.coef_[0], 
            duration_model.coef_[0]
        ],
        'Dimension_2_Direction': [
            delay_model.coef_[1], 
            hour_model.coef_[1], 
            duration_model.coef_[1]
        ],
        'Intercept': [
            delay_model.intercept_,
            hour_model.intercept_,
            duration_model.intercept_
        ]
    }
    
    if has_turnaround:
        direction_data['Feature'].append('Turnaround')
        direction_data['Dimension_1_Direction'].append(turnaround_model.coef_[0])
        direction_data['Dimension_2_Direction'].append(turnaround_model.coef_[1])
        direction_data['Intercept'].append(turnaround_model.intercept_)
    
    direction_df = pd.DataFrame(direction_data)
    dir_csv_path = output_path.with_name(f"{output_path.stem}_directions.csv")
    direction_df.to_csv(dir_csv_path, index=False)
    logging.info(f"Saved direction data to {dir_csv_path}")


def main():
    """Main function to orchestrate the embedding space navigation visualization process."""
    setup_logging()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    embeddings, flight_data= load_data(
        args.embeddings, args.data, args.sample_size, args.seed
    )
    
    # Create visualizations
    create_embedding_navigation_visualization(
        embeddings, flight_data, args.output_dir
    )
    
    logging.info("Embedding space navigation visualization complete")


if __name__ == "__main__":
    main()