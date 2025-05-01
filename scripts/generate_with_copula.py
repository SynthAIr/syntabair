#!/usr/bin/env python
"""
Generate synthetic flight data using a Gaussian Copula model.

This script fits a Gaussian Multivariate Copula model on real training data,
generates synthetic flight data, and saves the generated data to a specified directory.
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from syntabair.preprocessing import preprocess_flight_data, reconstruct_original_format
from syntabair.generators.copula import GaussianMultivariate


def train_and_generate(
    train_data_path,
    output_path,
    random_state=42,
    batch_size=10000,
    max_samples=None
):
    """
    Train a Gaussian Copula model and generate synthetic flight data.
    
    Parameters:
    -----------
    train_data_path : str
        Path to the CSV file containing real training data
    output_path : str
        Path to save the generated synthetic data (CSV file)
    random_state : int
        Random seed for reproducibility
    batch_size : int
        Number of samples to generate in each batch
    max_samples : int
        Maximum number of samples to generate (default: same as training data)
    
    Returns:
    --------
    pd.DataFrame
        Generated synthetic data
    """
    # Set random seed
    np.random.seed(random_state)
    
    # Load the training data
    print(f"Loading training data from {train_data_path}...")
    train_data = pd.read_csv(train_data_path)
    
    # Preprocess data
    print("Preprocessing data...")
    preprocessed_data = preprocess_flight_data(train_data)
    
    # Determine number of samples to generate
    if max_samples is None:
        num_samples = len(preprocessed_data)
    else:
        num_samples = min(max_samples, len(preprocessed_data))
    
    print(f"Will generate {num_samples} samples in batches of {batch_size}")
    
    # Define categorical columns (these need special handling)
    categorical_columns = [
        'IATA_CARRIER_CODE',
        'DEPARTURE_IATA_AIRPORT_CODE',
        'ARRIVAL_IATA_AIRPORT_CODE',
        'AIRCRAFT_TYPE_IATA',
        'SCHEDULED_MONTH',
        'SCHEDULED_DAY',
        'SCHEDULED_HOUR',
        'SCHEDULED_MINUTE'
    ]
    
    # Separate categorical and numerical columns
    numerical_columns = [col for col in preprocessed_data.columns if col not in categorical_columns]
    
    # Create encoders for categorical data
    print("Encoding categorical data...")
    encoders = {}
    encoded_categorical_data = pd.DataFrame(index=preprocessed_data.index)
    
    for col in categorical_columns:
        # Use OrdinalEncoder for each categorical column
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        encoded_vals = encoder.fit_transform(preprocessed_data[[col]])
        encoded_categorical_data[col] = encoded_vals.flatten()  # Flatten to 1D
        encoders[col] = encoder
    
    # Combine encoded categorical data with numerical data
    combined_data = pd.concat([
        encoded_categorical_data,
        preprocessed_data[numerical_columns]
    ], axis=1)
    
    # Initialize and train the model
    print("Training Gaussian Copula model...")
    copula = GaussianMultivariate(random_state=random_state)
    copula.fit(combined_data)
    
    # Generate synthetic data in batches
    print(f"Generating {num_samples} synthetic samples in batches...")
    
    all_reconstructed_data = []
    remaining_samples = num_samples
    
    while remaining_samples > 0:
        current_batch_size = min(batch_size, remaining_samples)
        print(f"Generating batch of {current_batch_size} samples...")
        
        # Generate batch of synthetic data
        synthetic_combined_data = copula.sample(current_batch_size)
        
        # Decode categorical data
        synthetic_data = pd.DataFrame(index=range(current_batch_size))
        
        for col in categorical_columns:
            # Round to nearest integer and ensure valid index by clipping
            col_data = synthetic_combined_data[col].round().astype(int)
            max_val = len(encoders[col].categories_[0]) - 1
            col_data = np.clip(col_data, 0, max_val)
            
            # Convert back to original categories - ensure we get a 1D array
            decoded_vals = encoders[col].inverse_transform(col_data.values.reshape(-1, 1))
            synthetic_data[col] = decoded_vals.flatten()  # Flatten the 2D array to 1D
        
        # Add numerical columns directly
        for col in numerical_columns:
            synthetic_data[col] = synthetic_combined_data[col]
        
        # Reconstruct original format
        print("Converting batch to original format...")
        reconstructed_batch = reconstruct_original_format(synthetic_data)
        all_reconstructed_data.append(reconstructed_batch)
        
        # Update remaining samples
        remaining_samples -= current_batch_size
        print(f"Remaining samples: {remaining_samples}")
    
    # Combine all batches
    print("Combining all batches...")
    reconstructed_data = pd.concat(all_reconstructed_data, ignore_index=True)
    
    # Save generated data
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving synthetic data to {output_path}...")
    reconstructed_data.to_csv(output_path, index=False)
    
    return reconstructed_data


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic flight data using a Gaussian Copula model")
    
    parser.add_argument(
        "--train_data_path",
        type=str,
        required=True,
        help="Path to the CSV file containing real training data"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the generated synthetic data (CSV file)"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10000,
        help="Number of samples to generate in each batch"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to generate (default: same as training data)"
    )
    
    args = parser.parse_args()
    
    synthetic_data = train_and_generate(
        train_data_path=args.train_data_path,
        output_path=args.output_path,
        random_state=args.random_state,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )
    
    print(f"Generation complete. {len(synthetic_data)} samples saved to: {args.output_path}")


if __name__ == "__main__":
    main()

# Example usage:
# python scripts/generate_with_copula.py --train_data_path data/real/train.csv --output_path data/synthetic/copula.csv