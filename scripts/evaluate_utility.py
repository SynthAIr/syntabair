#!/usr/bin/env python
"""
Evaluate synthetic flight data using Train-Synthetic-Test-Real (TSTR) methodology.

This script evaluates the utility of synthetic data by training models on synthetic data
and testing them on real data, with both pre-tactical and tactical prediction modes.
All results are saved to consolidated CSV files for later analysis and visualization.
"""

import os
import argparse
import pandas as pd
import numpy as np
from syntabair.preprocessing import preprocess_flight_data_for_prediction
from syntabair.evaluation.utility import (
    evaluate_tstr,
    calculate_utility_scores,
    extract_feature_importances_target_encoding
)


def setup_argparse():
    """Set up command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Evaluate synthetic flight data using TSTR methodology"
    )
    
    # Required arguments
    parser.add_argument(
        "--real_train_path",
        type=str,
        required=True,
        help="Path to the real training data CSV file"
    )
    parser.add_argument(
        "--real_test_path",
        type=str,
        required=True,
        help="Path to the real test data CSV file"
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
        default="results/utility",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--targets",
        type=str,
        nargs="+",
        default=["DEPARTURE_DELAY_MIN", "ARRIVAL_DELAY_MIN", "TURNAROUND_MIN"],
        choices=["DEPARTURE_DELAY_MIN", "ARRIVAL_DELAY_MIN", "TURNAROUND_MIN"],
        help="Target variables to evaluate"
    )
    parser.add_argument(
        "--prediction_modes",
        type=str,
        nargs="+",
        default=["pre-tactical", "tactical"],
        choices=["pre-tactical", "tactical"],
        help="Prediction modes to evaluate (pre-tactical: before departure, tactical: real-time)"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    return parser


def convert_results_to_dataframe(results, target, prediction_mode):
    """
    Convert nested results dictionary to a flat DataFrame.
    
    Args:
        results (dict): Nested dictionary of evaluation results
        target (str): Target variable
        prediction_mode (str): Prediction mode
        
    Returns:
        pd.DataFrame: Flat DataFrame of results
    """
    rows = []
    
    for dataset_name, dataset_results in results.items():
        for model_name, model_results in dataset_results.items():
            row = {
                'Target': target,
                'Mode': prediction_mode,
                'Dataset': dataset_name,
                'Model': model_name,
                'RMSE': model_results['rmse'],
                'MAE': model_results['mae'],
                'R2': model_results['r2']
            }
            rows.append(row)
    
    return pd.DataFrame(rows)


def extract_feature_importances_dataframe(importances, target, prediction_mode):
    """
    Extract feature importances to a flat DataFrame.
    
    Args:
        importances (dict): Nested dictionary of feature importances
        target (str): Target variable
        prediction_mode (str): Prediction mode
        
    Returns:
        pd.DataFrame: Flat DataFrame of feature importances
    """
    rows = []
    
    for dataset_name, dataset_importances in importances.items():
        for model_name, feature_importances in dataset_importances.items():
            for feature, importance in feature_importances.items():
                row = {
                    'Target': target,
                    'Mode': prediction_mode,
                    'Dataset': dataset_name,
                    'Model': model_name,
                    'Feature': feature,
                    'Importance': importance
                }
                rows.append(row)
    
    return pd.DataFrame(rows)


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
    real_test = pd.read_csv(args.real_test_path)
    
    # Load synthetic datasets
    synthetic_datasets = {}
    for path, name in zip(args.synthetic_paths, args.synthetic_names):
        print(f"  Loading {name} dataset from {path}")
        synthetic_datasets[name] = pd.read_csv(path)
    
    # Preprocess datasets
    print("Preprocessing datasets...")
    real_train_processed = preprocess_flight_data_for_prediction(real_train)
    real_test_processed = preprocess_flight_data_for_prediction(real_test)
    
    synthetic_processed = {}
    for name, data in synthetic_datasets.items():
        print(f"  Preprocessing {name} dataset")
        synthetic_processed[name] = preprocess_flight_data_for_prediction(data)
    
    # Prepare train datasets for evaluation
    train_datasets = {"Real": real_train_processed}
    train_datasets.update({name: data for name, data in synthetic_processed.items()})
    
    # Create DataFrames to store all results
    all_performance_results = []
    all_utility_results = []
    all_feature_importances = []
    all_metadata = []
    
    for target in args.targets:
        for mode in args.prediction_modes:
            # Skip tactical mode for departure delay since it doesn't make sense
            if target == 'DEPARTURE_DELAY_MIN' and mode == 'tactical':
                print(f"\nSkipping {target} with {mode} mode (not applicable)")
                continue
            
            print(f"\n=== {target.replace('_', ' ')} PREDICTION ({mode.upper()} MODE) ===")
            
            # Run TSTR evaluation
            results, cat_features, num_features = evaluate_tstr(
                target=target,
                train_datasets=train_datasets,
                test_dataset=real_test_processed,
                prediction_mode=mode
            )
            
            # Save metadata about the features used for this target and mode
            all_metadata.append({
                'Target': target,
                'Mode': mode,
                'CategoryFeatures': ','.join(cat_features),
                'NumericFeatures': ','.join(num_features)
            })
            
            # Convert results to DataFrame and append to all results
            performance_df = convert_results_to_dataframe(results, target, mode)
            all_performance_results.append(performance_df)
            
            # Calculate utility scores
            utility_df = calculate_utility_scores(results)
            # Add target and mode to utility DataFrame
            utility_df['Target'] = target
            utility_df['Mode'] = mode
            all_utility_results.append(utility_df)
            
            # Extract feature importances
            importances = extract_feature_importances_target_encoding(results, cat_features, num_features)
            importance_df = extract_feature_importances_dataframe(importances, target, mode)
            all_feature_importances.append(importance_df)
            
            print(f"  Completed evaluation for {target} ({mode} mode)")
    
    # Combine all results into single DataFrames
    all_performance_df = pd.concat(all_performance_results, ignore_index=True)
    all_utility_df = pd.concat(all_utility_results, ignore_index=True)
    all_importances_df = pd.concat(all_feature_importances, ignore_index=True)
    metadata_df = pd.DataFrame(all_metadata)
    
    # Save consolidated results to CSV files
    all_performance_df.to_csv(os.path.join(args.output_dir, 'all_performance_results.csv'), index=False)
    all_utility_df.to_csv(os.path.join(args.output_dir, 'all_utility_results.csv'), index=False)
    all_importances_df.to_csv(os.path.join(args.output_dir, 'all_feature_importances.csv'), index=False)
    metadata_df.to_csv(os.path.join(args.output_dir, 'features_metadata.csv'), index=False)
    
    print(f"\nEvaluation complete. All results saved to {args.output_dir}")
    print(f"Performance results: {os.path.join(args.output_dir, 'all_performance_results.csv')}")
    print(f"Utility results: {os.path.join(args.output_dir, 'all_utility_results.csv')}")
    print(f"Feature importance results: {os.path.join(args.output_dir, 'all_feature_importances.csv')}")
    print(f"Features metadata: {os.path.join(args.output_dir, 'features_metadata.csv')}")
    print("\nTo generate plots from these results, use the generate_tstr_plots.py script.")


if __name__ == "__main__":
    main()

# Example usage:
# python scripts/evaluate_tstr.py --real_train_path data/real/train.csv --real_test_path data/real/test.csv --synthetic_paths data/synthetic/ctgan.csv data/synthetic/tvae.csv data/synthetic/rtf.csv --synthetic_names CTGAN TVAE RTF --output_dir tstr_results --targets DEPARTURE_DELAY_MIN ARRIVAL_DELAY_MIN TURNAROUND_MIN --prediction_modes pre-tactical tactical --random_seed 42