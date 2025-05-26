
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from syntabair.preprocessing import preprocess_flight_data_for_prediction


# Configuration constants
DEFAULT_OUTPUT_DIR = Path("./embeddings/pca_interpretation")
DEFAULT_RANDOM_SEED = 42
DEFAULT_SAMPLE_SIZE = 100_000
DEFAULT_N_COMPONENTS = 6


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate TabSyn PCA component interpretation visualization"
    )
    parser.add_argument("--embeddings", type=Path, required=True,
                        help="Path to TabSyn embeddings file (e.g., train_z.npy)")
    parser.add_argument("--data", type=Path, required=True,
                        help="Path to flight data CSV file")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to save outputs (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
                        help=f"Number of records to sample (default: {DEFAULT_SAMPLE_SIZE})")
    parser.add_argument("--n-components", type=int, default=DEFAULT_N_COMPONENTS,
                        help=f"Number of PCA components to analyze (default: {DEFAULT_N_COMPONENTS})")
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


def preprocess_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Helper to handle embeddings of different shapes."""
    if len(embeddings.shape) == 3:
        logging.info(f"Reshaping embeddings from {embeddings.shape}")
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
        logging.info(f"to {embeddings.shape}")
    return embeddings


def apply_pca(embeddings: np.ndarray, n_components: int, seed: int) -> Tuple[np.ndarray, PCA]:
    """Apply PCA to reduce dimensions of embeddings."""
    logging.info(f"Applying PCA with {n_components} components")
    pca = PCA(n_components=n_components, random_state=seed)
    pca_result = pca.fit_transform(embeddings)
    logging.info(f"PCA explained variance ratios: {pca.explained_variance_ratio_}")
    return pca_result, pca


def get_feature_names_and_labels(flight_data: pd.DataFrame) -> Tuple[List[str], Dict[str, str]]:
    """Get feature names and their readable labels."""
    # Define standard feature names to look for
    features = [
        'SCHEDULED_MONTH', 'SCHEDULED_DAY', 'SCHEDULED_HOUR', 'SCHEDULED_MINUTE',
        'SCHEDULED_DURATION_MIN', 'ACTUAL_DURATION_MIN', 'DEPARTURE_DELAY_MIN'
    ]
    
    # Add turnaround if available
    if 'TURNAROUND_MIN' in flight_data.columns:
        features.append('TURNAROUND_MIN')
    
    # For readable labels
    feature_labels = {
        'SCHEDULED_MONTH': 'Month',
        'SCHEDULED_DAY': 'Day',
        'SCHEDULED_HOUR': 'Hour',
        'SCHEDULED_MINUTE': 'Minute',
        'SCHEDULED_DURATION_MIN': 'Sched Duration',
        'ACTUAL_DURATION_MIN': 'Actual Duration',
        'DEPARTURE_DELAY_MIN': 'Delay',
        'TURNAROUND_MIN': 'Turnaround'
    }
    
    # Filter to only include features that exist in the flight data
    available_features = [f for f in features if f in flight_data.columns]
    
    logging.info(f"Using features: {available_features}")
    return available_features, feature_labels


def calculate_component_correlations(
    pca_result: np.ndarray,
    flight_data: pd.DataFrame,
    features: List[str],
    feature_labels: Dict[str, str],
    n_components: int
) -> List[Dict[str, float]]:
    """Calculate correlation between each component and each feature."""
    logging.info("Calculating correlations between PCA components and flight features")
    component_correlations = []
    
    for i in range(n_components):
        correlations = {}
        for feature in features:
            if feature in flight_data.columns:
                corr = np.corrcoef(pca_result[:, i], flight_data[feature])[0, 1]
                correlations[feature_labels[feature]] = corr
        component_correlations.append(correlations)
    
    return component_correlations


def create_pca_interpretation_visualization(
    component_correlations: List[Dict[str, float]],
    pca: PCA,
    n_components: int,
    output_dir: Path
) -> Path:
    """Create visualization of PCA component interpretations."""
    logging.info("Creating PCA component interpretation visualization")
    output_path = output_dir / "pca_interpretation.png"
    
    # Create figure with n_components subplots
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    axs = axs.flatten()
    
    # For each component, create a correlation plot
    for i in range(min(n_components, len(axs))):
        # Sort correlations for better visualization
        sorted_correlations = sorted(
            component_correlations[i].items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        features_sorted = [x[0] for x in sorted_correlations]
        correlations_sorted = [x[1] for x in sorted_correlations]
        
        # Create bar chart
        bars = axs[i].barh(
            features_sorted,
            correlations_sorted,
            color=[plt.cm.RdBu(0.5 * (x + 1)) for x in correlations_sorted],
            edgecolor='gray'
        )
        
        # Add component variance explained
        var_explained = pca.explained_variance_ratio_[i]
        
        # Add titles and labels
        axs[i].set_title(f'PC{i+1} ({var_explained:.1%} variance)', fontsize=14)
        axs[i].set_xlabel('Correlation', fontsize=12)
        axs[i].set_xlim(-1, 1)
        axs[i].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Add correlation values as text
        for j, v in enumerate(correlations_sorted):
            axs[i].text(
                v + (0.05 if v >= 0 else -0.05),
                j,
                f'{v:.2f}',
                va='center',
                ha='left' if v >= 0 else 'right',
                fontsize=9
            )
    
    # Hide unused subplots
    for i in range(n_components, len(axs)):
        axs[i].set_visible(False)
    
    # Add main title
    plt.suptitle('Interpretation of Principal Components in TabSyn Embedding Space', fontsize=18)
    
    # Improve layout
    plt.tight_layout(rect=[0, 0.01, 1, 0.95])
    
    # Save figure
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    logging.info(f"PCA component interpretation visualization saved to {output_path}")
    return output_path


def save_pca_data(
    pca_result: np.ndarray,
    flight_data: pd.DataFrame,
    features: List[str],
    component_correlations: List[Dict[str, float]],
    pca: PCA,
    n_components: int,
    output_dir: Path,
    output_path: Path
) -> None:
    """Save PCA data to CSV files."""
    # Create a correlation matrix dataframe
    component_names = [f'PC{i+1}' for i in range(n_components)]
    corr_data = {name: {} for name in component_names}
    
    for i, comp_name in enumerate(component_names):
        for feature, value in component_correlations[i].items():
            corr_data[comp_name][feature] = value
    
    corr_df = pd.DataFrame(corr_data)
    
    # Add variance explained
    corr_df.loc['Variance_Explained'] = pca.explained_variance_ratio_
    
    # Save correlation matrix
    corr_csv = output_path.with_suffix('.csv')
    corr_df.to_csv(corr_csv)
    logging.info(f"Correlation matrix saved to {corr_csv}")
    
    # Save PCA components
    pca_df = pd.DataFrame(
        pca_result,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    # Add original features for reference
    for feature in features:
        if feature in flight_data.columns:
            pca_df[feature] = flight_data[feature].values
    
    components_csv = output_path.with_name(f"{output_path.stem}_components.csv")
    pca_df.to_csv(components_csv, index=False)
    logging.info(f"PCA components saved to {components_csv}")
    
    # Save PCA loadings (i.e., the components themselves)
    loadings_df = pd.DataFrame(
        pca.components_,
        index=[f'PC{i+1}' for i in range(n_components)]
    )
    
    # If the embeddings have named dimensions, use those; otherwise use generic names
    loadings_df.columns = [f'Dim_{i}' for i in range(loadings_df.shape[1])]
    
    loadings_csv = output_dir / "pca_loadings.csv"
    loadings_df.to_csv(loadings_csv)
    logging.info(f"PCA loadings saved to {loadings_csv}")


def pca_component_interpretation(
    embeddings: np.ndarray,
    flight_data: pd.DataFrame,
    output_dir: Path = None,
    n_components: int = DEFAULT_N_COMPONENTS,
    seed: int = DEFAULT_RANDOM_SEED
) -> Path:
    """
    Interpret and visualize what each principal component represents.
    
    Args:
        embeddings (np.ndarray): Embedding vectors
        flight_data (pd.DataFrame): Flight data
        output_dir (Path, optional): Path to save the figure
        n_components (int): Number of PCA components to analyze
        seed (int): Random seed for reproducibility
        
    Returns:
        Path: Path to the saved visualization
    """
    if output_dir is None:
        output_dir = Path(DEFAULT_OUTPUT_DIR)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Preprocess embeddings
    embeddings = preprocess_embeddings(embeddings)
    
    # Apply PCA to reduce dimensions
    pca_result, pca = apply_pca(embeddings, n_components, seed)
    
    # Get feature names and labels
    features, feature_labels = get_feature_names_and_labels(flight_data)
    
    # Calculate correlation between each component and each feature
    component_correlations = calculate_component_correlations(
        pca_result, flight_data, features, feature_labels, n_components
    )
    
    # Create and save visualization
    output_path = create_pca_interpretation_visualization(
        component_correlations, pca, n_components, output_dir
    )
    
    # Save data to CSV files
    save_pca_data(
        pca_result, flight_data, features, component_correlations, 
        pca, n_components, output_dir, output_path
    )
    
    return output_path


def main():
    """Main function to orchestrate the PCA component interpretation visualization process."""
    setup_logging()
    args = parse_args()
    
    # Load data
    embeddings, flight_data = load_data(
        args.embeddings, args.data, args.sample_size, args.seed
    )
    
    # Run PCA component interpretation
    pca_component_interpretation(
        embeddings=embeddings,
        flight_data=flight_data,
        output_dir=args.output_dir,
        n_components=args.n_components,
        seed=args.seed
    )
    
    logging.info("PCA component interpretation complete")


if __name__ == "__main__":
    main()