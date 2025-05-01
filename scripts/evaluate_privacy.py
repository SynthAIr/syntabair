#!/usr/bin/env python
"""
Evaluate synthetic flight data using privacy metrics.

This script evaluates the privacy of synthetic data by comparing it to real data using
DCRBaselineProtection and DCROverfittingProtection metrics. DCRBaselineProtection measures
how well the synthetic data protects against member inference attacks compared to random data,
while DCROverfittingProtection measures whether the synthetic data has been overfit to the 
training data.
"""

import os
import argparse
import pandas as pd
import numpy as np
from syntabair.preprocessing import preprocess_flight_data_for_prediction
from syntabair.evaluation.privacy import DCRBaselineProtection, DCROverfittingProtection

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
        description="Evaluate synthetic flight data privacy using DCR metrics"
    )
    
    # Required arguments
    parser.add_argument(
        "--real_train_path",
        type=str,
        required=True,
        help="Path to the real training data CSV file"
    )
    parser.add_argument(
        "--real_validation_path",
        type=str,
        required=True,
        help="Path to the real validation data CSV file (for overfitting protection)"
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
        default="results/privacy",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["baseline", "overfitting", "all"],
        choices=["baseline", "overfitting", "all"],
        help="Privacy metrics to evaluate"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100000,
        help="Number of rows to sample from each dataset (for faster evaluation)"
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=1,
        help="Number of iterations when using subsampling (higher for more stable results)"
    )
    
    return parser


def evaluate_baseline_protection(real_data, synthetic_data, metadata, sample_size=None, num_iterations=1):
    """
    Evaluate baseline protection using DCRBaselineProtection metric.
    
    Args:
        real_data (pd.DataFrame): Real dataset
        synthetic_data (pd.DataFrame): Synthetic dataset
        metadata (dict): Metadata about the datasets
        sample_size (int, optional): Number of rows to sample
        num_iterations (int): Number of iterations for sampling
        
    Returns:
        dict: Dictionary with metric results
    """
    print(f"  Computing DCRBaselineProtection with sample_size={sample_size}, iterations={num_iterations}")
    result = DCRBaselineProtection.compute_breakdown(
        real_data=real_data,
        synthetic_data=synthetic_data,
        metadata=metadata,
        num_rows_subsample=sample_size,
        num_iterations=num_iterations
    )
    
    return result


def evaluate_overfitting_protection(real_training_data, synthetic_data, real_validation_data, 
                                   metadata, sample_size=None, num_iterations=1):
    """
    Evaluate overfitting protection using DCROverfittingProtection metric.
    
    Args:
        real_training_data (pd.DataFrame): Real training dataset
        synthetic_data (pd.DataFrame): Synthetic dataset
        real_validation_data (pd.DataFrame): Real validation dataset
        metadata (dict): Metadata about the datasets
        sample_size (int, optional): Number of rows to sample
        num_iterations (int): Number of iterations for sampling
        
    Returns:
        dict: Dictionary with metric results
    """
    print(f"  Computing DCROverfittingProtection with sample_size={sample_size}, iterations={num_iterations}")
    result = DCROverfittingProtection.compute_breakdown(
        real_training_data=real_training_data,
        synthetic_data=synthetic_data,
        real_validation_data=real_validation_data,
        metadata=metadata,
        num_rows_subsample=sample_size,
        num_iterations=num_iterations
    )
    
    return result


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
    
    # Load real datasets
    real_train = pd.read_csv(args.real_train_path)
    real_validation = pd.read_csv(args.real_validation_path)
    
    # Load synthetic datasets
    synthetic_datasets = {}
    for path, name in zip(args.synthetic_paths, args.synthetic_names):
        print(f"  Loading {name} dataset from {path}")
        synthetic_datasets[name] = pd.read_csv(path)
    
    # Preprocess datasets
    print("Preprocessing datasets...")
    real_train_processed = preprocess_flight_data_for_prediction(real_train)
    real_validation_processed = preprocess_flight_data_for_prediction(real_validation)
    
    synthetic_processed = {}
    for name, data in synthetic_datasets.items():
        print(f"  Preprocessing {name} dataset")
        synthetic_processed[name] = preprocess_flight_data_for_prediction(data)
    
    # Determine which metrics to evaluate
    metrics_to_evaluate = args.metrics
    if "all" in metrics_to_evaluate:
        metrics_to_evaluate = ["baseline", "overfitting"]
    
    # Prepare to store all evaluation results
    baseline_results = []
    overfitting_results = []
    
    # Evaluate each synthetic dataset
    for name, synthetic_data in synthetic_processed.items():
        print(f"\n=== EVALUATING {name} ===")
        
        # Baseline protection
        if "baseline" in metrics_to_evaluate:
            print("  Evaluating baseline protection...")
            baseline_result = evaluate_baseline_protection(
                real_data=real_train_processed,
                synthetic_data=synthetic_data,
                metadata=FLIGHT_DATA_METADATA,
                sample_size=args.sample_size,
                num_iterations=args.num_iterations
            )
            
            # Add dataset name to results
            baseline_result_entry = {
                'Dataset': name,
                'Score': baseline_result['score'],
                'SyntheticMedianDCR': baseline_result['median_DCR_to_real_data']['synthetic_data'],
                'RandomMedianDCR': baseline_result['median_DCR_to_real_data']['random_data_baseline']
            }
            baseline_results.append(baseline_result_entry)
            
            print(f"  Baseline Protection Score: {baseline_result['score']:.4f}")
        
        # Overfitting protection
        if "overfitting" in metrics_to_evaluate:
            print("  Evaluating overfitting protection...")
            overfitting_result = evaluate_overfitting_protection(
                real_training_data=real_train_processed,
                synthetic_data=synthetic_data,
                real_validation_data=real_validation_processed,
                metadata=FLIGHT_DATA_METADATA,
                sample_size=args.sample_size,
                num_iterations=args.num_iterations
            )
            
            # Add dataset name to results
            overfitting_result_entry = {
                'Dataset': name,
                'Score': overfitting_result['score'],
                'CloserToTraining': overfitting_result['synthetic_data_percentages']['closer_to_training'],
                'CloserToHoldout': overfitting_result['synthetic_data_percentages']['closer_to_holdout']
            }
            overfitting_results.append(overfitting_result_entry)
            
            print(f"  Overfitting Protection Score: {overfitting_result['score']:.4f}")
    
    # Convert results to DataFrames
    if baseline_results:
        baseline_df = pd.DataFrame(baseline_results)
        baseline_path = os.path.join(args.output_dir, "baseline_protection_results.csv")
        baseline_df.to_csv(baseline_path, index=False)
        print(f"\nBaseline protection results saved to {baseline_path}")
    
    if overfitting_results:
        overfitting_df = pd.DataFrame(overfitting_results)
        overfitting_path = os.path.join(args.output_dir, "overfitting_protection_results.csv")
        overfitting_df.to_csv(overfitting_path, index=False)
        print(f"Overfitting protection results saved to {overfitting_path}")
    
    # Save combined results
    if baseline_results and overfitting_results:
        combined_results = []
        for b_entry in baseline_results:
            for o_entry in overfitting_results:
                if b_entry['Dataset'] == o_entry['Dataset']:
                    combined_results.append({
                        'Dataset': b_entry['Dataset'],
                        'BaselineProtectionScore': b_entry['Score'],
                        'OverfittingProtectionScore': o_entry['Score'],
                        'SyntheticMedianDCR': b_entry['SyntheticMedianDCR'],
                        'RandomMedianDCR': b_entry['RandomMedianDCR'],
                        'CloserToTraining': o_entry['CloserToTraining'],
                        'CloserToHoldout': o_entry['CloserToHoldout']
                    })
        
        combined_df = pd.DataFrame(combined_results)
        combined_path = os.path.join(args.output_dir, "privacy_results.csv")
        combined_df.to_csv(combined_path, index=False)
        print(f"Combined privacy results saved to {combined_path}")
    
    print("\nEvaluation complete. To generate plots from these results, use the plot_privacy.py script.")


if __name__ == "__main__":
    main()

# Example usage:
# python scripts/evaluate_privacy.py --real_train_path data/real/train.csv --real_validation_path data/real/test.csv --synthetic_paths data/synthetic/copula.csv data/synthetic/tvae.csv data/synthetic/ctgan.csv --synthetic_names GaussianCopula TVAE CTGAN --output_dir results/privacy --metrics all --sample_size 100000 --num_iterations 3