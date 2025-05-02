#!/usr/bin/env python
"""
Generate synthetic data using a trained TabSyn model.

Usage:
    python generate_with_tabsyn.py --model_dir models/flight_model --output_path synthetic/flights_samples.csv
"""

import os
import json
import argparse
import pandas as pd
from pathlib import Path

from syntabair.preprocessing import reconstruct_original_format
from syntabair.generators import TabSyn


def detect_data_type(model_dir):
    """
    Detect the data type from the training parameters.
    
    Args:
        model_dir: Path to the model directory
        
    Returns:
        str: Data type ('flights' or 'generic')
    """
    params_path = os.path.join(model_dir, "training_params.json")
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params = json.load(f)
        return params.get("data_type", "generic")
    
    # If no training params, try to detect based on columns
    info_path = os.path.join(model_dir, "data", "info.json")
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            info = json.load(f)
        flight_columns = ["IATA_CARRIER_CODE", "DEPARTURE_IATA_AIRPORT_CODE", "ARRIVAL_IATA_AIRPORT_CODE"]
        column_names = info.get("column_names", [])
        if all(col in column_names for col in flight_columns):
            return "flights"
    
    return "generic"  # Default to generic if we can't detect


def load_model(model_dir):
    """
    Load a trained TabSyn model from the model directory.
    
    Args:
        model_dir: Path to the model directory
        
    Returns:
        TabSyn: Loaded TabSyn model
    """
    model_path = os.path.join(model_dir, "tabsyn_model.pkl")
    
    if not os.path.exists(model_path):
        # Try to find any .pkl file in the directory
        pkl_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
        if pkl_files:
            model_path = os.path.join(model_dir, pkl_files[0])
        else:
            raise FileNotFoundError(f"No model file found in {model_dir}")
    
    print(f"Loading TabSyn model from {model_path}...")
    return TabSyn.load(model_path)


def generate_synthetic_data(
    model_dir,
    output_path,
    num_samples=None,
    steps=50,
    reference_data_path=None,
    reconstruction_year=2019,
):
    """
    Generate synthetic tabular data using a trained TabSyn model.
    
    Args:
        model_dir: Path to the model directory
        output_path: Path to save the generated synthetic data
        num_samples: Number of samples to generate (if None, matches training data size)
        steps: Number of sampling steps for the diffusion model
        reference_data_path: Path to reference data for determining number of samples
        reconstruction_year: Year to use for date reconstruction (for flight data)
        
    Returns:
        pd.DataFrame: Generated synthetic data
    """
    # Ensure model_dir is a Path object
    model_dir = Path(model_dir)
    
    # Determine number of samples if reference is provided
    if reference_data_path is not None:
        print(f"Loading reference data from {reference_data_path}...")
        reference_data = pd.read_csv(reference_data_path)
        num_samples = len(reference_data)
        print(f"Setting number of samples to match reference data: {num_samples}")
    
    # Load the trained model
    tabsyn = load_model(model_dir)
    
    # Generate synthetic data
    print(f"Generating{' '+str(num_samples) if num_samples else ''} synthetic samples using {steps} steps...")
    synthetic_data = tabsyn.sample(n_samples=num_samples, steps=steps)
    


    output_data = reconstruct_original_format(synthetic_data, default_year=reconstruction_year)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save generated data
    print(f"Saving synthetic data to {output_path}...")
    output_data.to_csv(output_path, index=False)
    
    
    return output_data


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data using a trained TabSyn model")
    
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing the trained TabSyn model"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the generated synthetic data (CSV file)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to generate (default: matches training data size)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of sampling steps for the diffusion model"
    )
    parser.add_argument(
        "--reference_data_path",
        type=str,
        default=None,
        help="Path to reference data for determining number of samples (overrides --num_samples)"
    )
    parser.add_argument(
        "--reconstruction_year",
        type=int,
        default=2019,
        help="Year to use for date reconstruction (for flight data)"
    )
    
    args = parser.parse_args()
    
    # If output_path is not specified, create one based on model_dir
    if args.output_path is None:
        model_name = os.path.basename(os.path.normpath(args.model_dir))
        args.output_path = f"synthetic_{model_name}.csv"
    
    synthetic_data = generate_synthetic_data(
        model_dir=args.model_dir,
        output_path=args.output_path,
        num_samples=args.num_samples,
        steps=args.steps,
        reference_data_path=args.reference_data_path,
        reconstruction_year=args.reconstruction_year
    )
    
    print(f"Generation complete. {len(synthetic_data)} samples saved to: {args.output_path}")


if __name__ == "__main__":
    main()

# Example usage for flight data:
# python generate_with_tabsyn.py --model_dir models/flight_model --output_path synthetic/flights_samples.csv
#
# Example usage with reference data to match sample size:
# python generate_with_tabsyn.py --model_dir models/adult_model --reference_data_path data/adult/test.csv