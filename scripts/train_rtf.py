#!/usr/bin/env python
"""
Train a REaLTabFormer model on flight data.

This script loads flight data, preprocesses it, trains a REaLTabFormer model,
and saves the trained model to a specified directory.
"""

import os
import argparse
import pandas as pd
from syntabair.preprocessing import preprocess_flight_data
from syntabair.generators import REaLTabFormer


def train_realtabformer(
    data_path,
    model_dir,
    epochs=50,
    batch_size=64,
    sample_size=None,
    random_state=42,
    device="cuda"
):
    """
    Train a REaLTabFormer model on flight data.
    
    Parameters:
    -----------
    data_path : str
        Path to the CSV file containing flight data
    model_dir : str
        Directory to save the trained model
    epochs : int
        Number of training epochs
    sample_size : int or None
        Number of rows to sample from the data (None uses all data)
    random_state : int
        Random seed for reproducibility
    device : str
        Device to use for training ("cuda" or "cpu")
    
    Returns:
    --------
    str
        Path to the saved model
    """
    # Create output directory if it doesn't exist
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
    
    # Initialize model
    print("Initializing REaLTabFormer model...")
    rtf_model = REaLTabFormer(
        model_type="tabular",
        gradient_accumulation_steps=4,
        logging_steps=100,
        epochs=epochs,
        batch_size=batch_size,
        report_to="none",  # Disable wandb and other reporting
        random_state=random_state
    )
    
    # Train model
    print(f"Training model for {epochs} epochs on {device}...")
    rtf_model.fit(df=preprocessed_data, device=device)
    
    # Save model
    print(f"Saving model to {model_dir}...")
    rtf_model.save(path=model_dir)
    
    # Get the experiment ID for the saved model
    experiment_id = rtf_model.experiment_id
    model_path = os.path.join(model_dir, experiment_id)
    print(f"Model saved with experiment ID: {experiment_id}")
    
    return model_path


def main():
    parser = argparse.ArgumentParser(description="Train REaLTabFormer model on flight data")
    
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the CSV file containing flight data"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Directory to save the trained model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    # Add argument for batch size
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training"
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
    
    model_path = train_realtabformer(
        data_path=args.data_path,
        model_dir=args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sample_size=args.sample_size,
        random_state=args.random_state,
        device=args.device
    )
    
    print(f"Training complete. Model saved to: {model_path}")


if __name__ == "__main__":
    main()

# example usage:
# python scripts/train_rtf.py --data_path data/flight_data.csv --model_dir models/rtf_model --epochs 100 --batch_size 32 --sample_size 10000 --random_state 123 --device cuda