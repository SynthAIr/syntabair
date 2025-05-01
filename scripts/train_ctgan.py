#!/usr/bin/env python
"""
Train a CTGAN model on flight data.

This script loads flight data, preprocesses it, trains a CTGAN model,
and saves the trained model to a specified directory.
"""

import os
import argparse
import pandas as pd
from syntabair.preprocessing import preprocess_flight_data
from syntabair.generators import CTGAN


def train_ctgan(
    data_path,
    model_path,
    batch_size=500,
    epochs=1000,
    sample_size=None,
    random_state=42,
    device="cuda",
):
    """
    Train a CTGAN model on flight data.
    
    Parameters:
    -----------
    data_path : str
        Path to the CSV file containing flight data
    model_path : str
        Path to save the trained model (including filename)
    batch_size : int
        Batch size for training
    epochs : int
        Number of training epochs
    sample_size : int or None
        Number of rows to sample from the data (None uses all data)
    random_state : int
        Random seed for reproducibility
    device : str
        Device to use for training ("cuda" or "cpu")
    force_retrain : bool
        Whether to retrain if a model already exists at model_path
    
    Returns:
    --------
    CTGAN
        Trained CTGAN model
    """
 
    # Create output directory if it doesn't exist
    model_dir = os.path.dirname(model_path)
    os.makedirs(model_dir, exist_ok=True)
    
    # Load and optionally sample data
    print(f"Loading data from {data_path}...")
    real_data = pd.read_csv(data_path)
    
    if sample_size is not None and sample_size < len(real_data):
        print(f"Sampling {sample_size} rows from dataset...")
        real_data = real_data.sample(n=sample_size, random_state=random_state)
    
    # Preprocess data
    print("Preprocessing data...")
    preprocessed_data = preprocess_flight_data(real_data)
    
    # Define the discrete columns
    discrete_columns = [
        'IATA_CARRIER_CODE',
        'DEPARTURE_IATA_AIRPORT_CODE',
        'ARRIVAL_IATA_AIRPORT_CODE',
        'AIRCRAFT_TYPE_IATA',
        'SCHEDULED_MONTH',
        'SCHEDULED_DAY',
        'SCHEDULED_HOUR',
        'SCHEDULED_MINUTE'
    ]
    
    # Initialize and train model
    print(f"Initializing CTGAN model with batch_size={batch_size}, epochs={epochs}...")
    ctgan = CTGAN(
        batch_size=batch_size,
        epochs=epochs,
        verbose=True,
        cuda=(device == "cuda")
    )
    
    print(f"Training model for {epochs} epochs on {device}...")
    ctgan.fit(preprocessed_data, discrete_columns)
    
    # Save model
    print(f"Saving model to {model_path}...")
    ctgan.save(model_path)
    
    return ctgan


def main():
    parser = argparse.ArgumentParser(description="Train CTGAN model on flight data")
    
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the CSV file containing flight data"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/ctgan.pkl",
        help="Path to save the trained model (including filename)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=500,
        help="Batch size for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of rows to sample from the data (default: use all data)"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for training"
    )
    
    args = parser.parse_args()
    
    
    model = train_ctgan(
        data_path=args.data_path,
        model_path=args.model_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        sample_size=args.sample_size,
        random_state=args.random_state,
        device=args.device,
    )
    
    print(f"Training complete. Model saved to: {args.model_path}")


if __name__ == "__main__":
    main()

# example usage:
# python scripts/train_ctgan.py --data_path data/flight_data.csv --model_path models/ctgan.pkl --batch_size 500 --epochs 1000 --sample_size 10000 --random_state 42 --device cuda