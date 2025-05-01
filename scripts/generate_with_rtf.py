#!/usr/bin/env python
"""
Generate synthetic flight data using a trained REaLTabFormer model.

This script loads a trained REaLTabFormer model, generates synthetic flight data,
and saves the generated data to a specified directory.
"""

import os
import argparse
import pandas as pd
from syntabair.preprocessing import reconstruct_original_format
from syntabair.generators import REaLTabFormer


def generate_synthetic_data(
    model_path,
    output_path,
    num_samples=1000,
    reference_data_path=None,
    device="cuda"
):
    """
    Generate synthetic flight data using a trained REaLTabFormer model.
    
    Parameters:
    -----------
    model_path : str
        Path to the trained REaLTabFormer model (directory containing experiment ID)
    output_path : str
        Path to save the generated synthetic data (CSV file)
    num_samples : int
        Number of samples to generate
    reference_data_path : str or None
        Path to reference data for determining number of samples (if None, use num_samples)
    device : str
        Device to use for generation ("cuda" or "cpu")
    
    Returns:
    --------
    pd.DataFrame
        Generated synthetic data
    """
    # Determine number of samples
    if reference_data_path is not None:
        print(f"Loading reference data from {reference_data_path}...")
        reference_data = pd.read_csv(reference_data_path)
        num_samples = len(reference_data)
        print(f"Setting number of samples to match reference data: {num_samples}")
    
    # Load the trained model
    print(f"Loading REaLTabFormer model from {model_path}...")
    rtf_model = REaLTabFormer.load_from_dir(path=model_path)
    
    # Generate synthetic data
    print(f"Generating {num_samples} synthetic samples on {device}...")
    synthetic_data = rtf_model.sample(n_samples=num_samples, device=device)
    
    # Reconstruct original format
    print("Converting generated data to original format...")
    reconstructed_data = reconstruct_original_format(synthetic_data)
    
    # Save generated data
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving synthetic data to {output_path}...")
    reconstructed_data.to_csv(output_path, index=False)
    
    return reconstructed_data


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic flight data using a trained REaLTabFormer model")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained REaLTabFormer model (including experiment ID directory)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the generated synthetic data (CSV file)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--reference_data_path",
        type=str,
        default=None,
        help="Path to reference data for determining number of samples (overrides --num_samples)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for generation"
    )
    
    args = parser.parse_args()
    
    synthetic_data = generate_synthetic_data(
        model_path=args.model_path,
        output_path=args.output_path,
        num_samples=args.num_samples,
        reference_data_path=args.reference_data_path,
        device=args.device
    )
    
    print(f"Generation complete. {len(synthetic_data)} samples saved to: {args.output_path}")


if __name__ == "__main__":
    main()

# example usage:
# python scripts/generate_with_rtf.py --model_path /path/to/model --output_path /path/to/output.csv --reference_data_path /path/to/reference.csv --num_samples 1000 --device cuda