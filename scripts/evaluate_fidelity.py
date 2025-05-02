#!/usr/bin/env python
"""
Evaluate synthetic flight data using fidelity metrics.

This script evaluates the fidelity of synthetic data by comparing its statistical properties
to real data using several categories of metrics:
1. Statistical tests (KS test, Chi-squared)
2. Distribution metrics (KL divergence)
3. Correlation metrics (Pearson, Spearman, correlation matrix distance)
4. Likelihood-based metrics (BN Likelihood, GM Log Likelihood)
5. Detection-based metrics (Logistic Detection, SVC Detection)

All results are saved to consolidated CSV files for later analysis and visualization.
"""

import os
import argparse
import pandas as pd
import numpy as np
from syntabair.preprocessing import preprocess_flight_data_for_prediction
from syntabair.evaluation.fidelity import (
    # Statistical metrics
    KSComplement,
    CSTest,
    # Distribution metrics
    ContinuousKLDivergence,
    DiscreteKLDivergence,
    # Correlation metrics
    PearsonCorrelation,
    SpearmanCorrelation,
    KendallCorrelation,
    CorrelationMatrixDistance,
    MixedTypeCorrelation,
    # Likelihood metrics
    BNLogLikelihood,
    GMLogLikelihood,
    # Detection metrics
    LogisticDetection,
)

# Metadata defining column types for the metrics
FLIGHT_DATA_METADATA = {
    "columns": {
        "IATA_CARRIER_CODE": {"sdtype": "categorical"},
        "DEPARTURE_IATA_AIRPORT_CODE": {"sdtype": "categorical"},
        "ARRIVAL_IATA_AIRPORT_CODE": {"sdtype": "categorical"},
        "AIRCRAFT_TYPE_IATA": {"sdtype": "categorical"},
        "SCHEDULED_MONTH": {"sdtype": "numerical"},
        "SCHEDULED_DAY": {"sdtype": "numerical"},
        "SCHEDULED_HOUR": {"sdtype": "numerical"},
        "SCHEDULED_MINUTE": {"sdtype": "numerical"},
        "DAY_OF_WEEK": {"sdtype": "categorical"},
        "SCHEDULED_DURATION_MIN": {"sdtype": "numerical"},
        "ACTUAL_DURATION_MIN": {"sdtype": "numerical"},
        "DURATION_DIFF_MIN": {"sdtype": "numerical"},
        "DEPARTURE_DELAY_MIN": {"sdtype": "numerical"},
        "ARRIVAL_DELAY_MIN": {"sdtype": "numerical"},
        "TURNAROUND_MIN": {"sdtype": "numerical"},
    }
}


def setup_argparse():
    """Set up command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Evaluate synthetic flight data fidelity using statistical metrics"
    )
    
    # Required arguments
    parser.add_argument(
        "--real_data_path",
        type=str,
        required=True,
        help="Path to the real data CSV file"
    )
    
    # Synthetic data arguments
    parser.add_argument(
        "--synthetic_paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths to synthetic datasets CSV files"
    )
    parser.add_argument(
        "--synthetic_names",
        type=str,
        nargs="+",
        required=True,
        help="Names for synthetic datasets in results (must match number of paths)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/fidelity",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["statistical", "correlation", "distribution", "likelihood", "detection"],
        choices=["statistical", "correlation", "distribution", "likelihood", "detection", "all"],
        help="Categories of fidelity metrics to evaluate"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    return parser


def evaluate_statistical_metrics(real_data, synthetic_data, metadata=FLIGHT_DATA_METADATA):
    """
    Evaluate statistical test metrics.
    
    Args:
        real_data (pd.DataFrame): Real dataset
        synthetic_data (pd.DataFrame): Synthetic dataset
        metadata (dict, optional): Metadata about the datasets
        
    Returns:
        dict: Dictionary with metric results
    """
    results = {}
    
    # KS Complement for continuous columns

    ks_score = KSComplement.compute(real_data, synthetic_data, metadata)
    results["KSComplement"] = ks_score
    
    # Get breakdown by column
    ks_breakdown = KSComplement.compute_breakdown(real_data, synthetic_data, metadata)
    for col, col_result in ks_breakdown.items():
        if 'score' in col_result:
            results[f"KSComplement_{col}"] = col_result['score']

    
    # Chi-Squared test for categorical columns

    cs_score = CSTest.compute(real_data, synthetic_data, metadata)
    results["CSTest"] = cs_score
    
    # Get breakdown by column
    cs_breakdown = CSTest.compute_breakdown(real_data, synthetic_data, metadata)
    for col, col_result in cs_breakdown.items():
        if 'score' in col_result:
            results[f"CSTest_{col}"] = col_result['score']

    return results


def evaluate_correlation_metrics(real_data, synthetic_data, metadata=FLIGHT_DATA_METADATA):
    """
    Evaluate correlation preservation metrics.
    
    Args:
        real_data (pd.DataFrame): Real dataset
        synthetic_data (pd.DataFrame): Synthetic dataset
        metadata (dict, optional): Metadata about the datasets
        
    Returns:
        dict: Dictionary with metric results
    """
    results = {}
    
    # Pearson correlation
    pearson_score = PearsonCorrelation.compute(real_data, synthetic_data, metadata)
    results["PearsonCorrelation"] = pearson_score
    
    # Get breakdown
    pearson_breakdown = PearsonCorrelation.compute_breakdown(real_data, synthetic_data, metadata)
    # Store only up to 10 pair scores to avoid cluttering results
    count = 0
    for col_pair, col_result in pearson_breakdown.items():
        if 'score' in col_result and count < 10 and not isinstance(col_pair, str):
            pair_name = f"{col_pair[0]}_{col_pair[1]}"
            results[f"PearsonCorrelation_{pair_name}"] = col_result['score']
            count += 1
        
    # Spearman correlation
    spearman_score = SpearmanCorrelation.compute(real_data, synthetic_data, metadata)
    results["SpearmanCorrelation"] = spearman_score

        
    # Kendall correlation
    kendall_score = KendallCorrelation.compute(real_data, synthetic_data, metadata)
    results["KendallCorrelation"] = kendall_score

    
    # Correlation matrix distance
    cmd_score = CorrelationMatrixDistance.compute(real_data, synthetic_data, metadata)
    results["CorrelationMatrixDistance"] = cmd_score

    
    # Mixed-type correlation
    mixed_score = MixedTypeCorrelation.compute(real_data, synthetic_data, metadata)
    results["MixedTypeCorrelation"] = mixed_score
    
    # Get breakdown
    mixed_breakdown = MixedTypeCorrelation.compute_breakdown(real_data, synthetic_data, metadata)
    for key, value in mixed_breakdown.items():
        if key != 'score':
            results[f"MixedTypeCorrelation_{key}"] = value

    return results

def evaluate_distribution_metrics(real_data, synthetic_data, metadata=FLIGHT_DATA_METADATA):
    """
    Evaluate distribution-based metrics.
    
    Args:
        real_data (pd.DataFrame): Real dataset
        synthetic_data (pd.DataFrame): Synthetic dataset
        metadata (dict, optional): Metadata about the datasets
        
    Returns:
        dict: Dictionary with metric results
    """
    results = {}
    
    # Continuous KL divergence
    cont_kl_score = ContinuousKLDivergence.compute(real_data, synthetic_data, metadata)
    results["ContinuousKLDivergence"] = cont_kl_score
    
    # Get breakdown by column pairs
    cont_kl_breakdown = ContinuousKLDivergence.compute_breakdown(real_data, synthetic_data, metadata)
    # Store only the first 10 pairs to avoid too much data
    count = 0
    for col_pair, col_result in cont_kl_breakdown.items():
        if 'score' in col_result and count < 10:
            pair_name = f"{col_pair[0]}_{col_pair[1]}"
            results[f"ContinuousKLDivergence_{pair_name}"] = col_result['score']
            count += 1

    # Discrete KL divergence
    disc_kl_score = DiscreteKLDivergence.compute(real_data, synthetic_data, metadata)
    results["DiscreteKLDivergence"] = disc_kl_score
    
    # Get breakdown by column pairs
    disc_kl_breakdown = DiscreteKLDivergence.compute_breakdown(real_data, synthetic_data, metadata)
    # Store only the first 10 pairs to avoid too much data
    count = 0
    for col_pair, col_result in disc_kl_breakdown.items():
        if 'score' in col_result and count < 10:
            pair_name = f"{col_pair[0]}_{col_pair[1]}"
            results[f"DiscreteKLDivergence_{pair_name}"] = col_result['score']
            count += 1

    return results


def evaluate_likelihood_metrics(real_data, synthetic_data, metadata=FLIGHT_DATA_METADATA):
    """
    Evaluate likelihood-based metrics.
    
    Args:
        real_data (pd.DataFrame): Real dataset
        synthetic_data (pd.DataFrame): Synthetic dataset
        metadata (dict, optional): Metadata about the datasets
        
    Returns:
        dict: Dictionary with metric results
    """
    results = {}
    
    # BN Log Likelihood
    bn_log_score = BNLogLikelihood.compute(real_data, synthetic_data, metadata)
    results["BNLogLikelihood"] = bn_log_score

    # Gaussian Mixture Log Likelihood
    gm_log_score = GMLogLikelihood.compute(real_data, synthetic_data, metadata)
    results["GMLogLikelihood"] = gm_log_score

    return results


def evaluate_detection_metrics(real_data, synthetic_data, metadata=FLIGHT_DATA_METADATA):
    """
    Evaluate detection-based metrics.
    
    Args:
        real_data (pd.DataFrame): Real dataset
        synthetic_data (pd.DataFrame): Synthetic dataset
        metadata (dict, optional): Metadata about the datasets
        
    Returns:
        dict: Dictionary with metric results
    """
    import warnings
    from sklearn.exceptions import ConvergenceWarning

    # Filter out the specific ConvergenceWarning
    warnings.filterwarnings('ignore', category=ConvergenceWarning)

    results = {}
    
    # Logistic Regression Detection
    lr_score = LogisticDetection.compute(real_data, synthetic_data, metadata)
    results["LogisticDetection"] = lr_score
    print(f"  LogisticDetection score: {lr_score:.4f}")
    
    # # SVC Detection
    # svc_score = SVCDetection.compute(real_data, synthetic_data, metadata)
    # results["SVCDetection"] = svc_score

    return results


def main():
    # Parse command line arguments
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Validate arguments
    if len(args.synthetic_paths) != len(args.synthetic_names):
        raise ValueError("Number of synthetic paths must match number of synthetic names")
    
    # Set random seed for reproducibility
    np.random.seed(args.random_seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading and preprocessing datasets...")
    
    # Load real dataset
    real_data = pd.read_csv(args.real_data_path)
    
    # Load synthetic datasets
    synthetic_datasets = {}
    for path, name in zip(args.synthetic_paths, args.synthetic_names):
        print(f"  Loading {name} dataset from {path}")
        synthetic_datasets[name] = pd.read_csv(path)
    
    # Preprocess datasets
    print("Preprocessing datasets...")
    real_data_processed = preprocess_flight_data_for_prediction(real_data)
    
    synthetic_processed = {}
    for name, data in synthetic_datasets.items():
        print(f"  Preprocessing {name} dataset")
        synthetic_processed[name] = preprocess_flight_data_for_prediction(data)
    
    # Prepare to store all evaluation results
    all_results = []
    
    # Set up metrics to evaluate
    metric_categories = args.metrics
    if "all" in metric_categories:
        metric_categories = ["statistical", "correlation", "distribution", "likelihood", "detection"]
    
    # Evaluate each synthetic dataset
    for name, synthetic_data in synthetic_processed.items():
        print(f"\n=== EVALUATING {name} ===")
        
        dataset_results = {"Dataset": name}
        
        # Statistical metrics
        if "statistical" in metric_categories:
            print("  Evaluating statistical metrics...")
            stat_results = evaluate_statistical_metrics(real_data_processed, synthetic_data)
            dataset_results.update(stat_results)
        
        # Correlation metrics
        if "correlation" in metric_categories:
            print("  Evaluating correlation metrics...")
            corr_results = evaluate_correlation_metrics(real_data_processed, synthetic_data)
            dataset_results.update(corr_results)
        
        # Distribution metrics
        if "distribution" in metric_categories:
            print("  Evaluating distribution metrics...")
            dist_results = evaluate_distribution_metrics(real_data_processed, synthetic_data)
            dataset_results.update(dist_results)
        
        # Likelihood metrics
        if "likelihood" in metric_categories:
            print("  Evaluating likelihood metrics...")
            like_results = evaluate_likelihood_metrics(
                real_data_processed, 
                synthetic_data, 
            )
            dataset_results.update(like_results)
        
        # Detection metrics
        if "detection" in metric_categories:
            print("  Evaluating detection metrics...")
            detect_results = evaluate_detection_metrics(real_data_processed, synthetic_data)
            dataset_results.update(detect_results)
        
        all_results.append(dataset_results)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save to CSV
    results_path = os.path.join(args.output_dir, "fidelity_results.csv")
    results_df.to_csv(results_path, index=False)
    
    print(f"\nEvaluation complete. Results saved to {results_path}")
    print("To generate plots from these results, use the generate_fidelity_plots.py script.")


if __name__ == "__main__":
    main()

# Example usage:
# python scripts/evaluate_fidelity.py --real_data_path data/real/train.csv --synthetic_paths data/synthetic/copula.csv data/synthetic/tvae.csv data/synthetic/ctgan.csv data/synthetic/rtf.csv --synthetic_names GaussianCopula TVAE CTGAN RTF --metrics all --output_dir results/fidelity
