#!/usr/bin/env python
"""
TabSyn training script for flight data - handles preprocessing and training in one step.

Usage:
    python train_tabsyn.py --train_path data/real/train.csv --test_path data/real/test.csv --model_dir models/flight_model
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from syntabair.generators import TabSyn
from syntabair.preprocessing import preprocess_flight_data


def get_column_name_mapping(data_df, num_col_idx, cat_col_idx, target_col_idx, column_names=None):
    """
    Create mappings between column indices and names.
    
    Args:
        data_df: DataFrame containing the data
        num_col_idx: Indices of numerical columns
        cat_col_idx: Indices of categorical columns
        target_col_idx: Indices of target columns
        column_names: Optional list of column names
        
    Returns:
        tuple: Mappings between indices and names
    """
    if not column_names:
        column_names = np.array(data_df.columns.tolist())
    
    idx_mapping = {}
    curr_num_idx = 0
    curr_cat_idx = len(num_col_idx)
    curr_target_idx = curr_cat_idx + len(cat_col_idx)

    for idx in range(len(column_names)):
        if idx in num_col_idx:
            idx_mapping[int(idx)] = curr_num_idx
            curr_num_idx += 1
        elif idx in cat_col_idx:
            idx_mapping[int(idx)] = curr_cat_idx
            curr_cat_idx += 1
        else:
            idx_mapping[int(idx)] = curr_target_idx
            curr_target_idx += 1

    inverse_idx_mapping = {}
    for k, v in idx_mapping.items():
        inverse_idx_mapping[int(v)] = k
        
    idx_name_mapping = {}
    for i in range(len(column_names)):
        idx_name_mapping[int(i)] = column_names[i]

    return idx_mapping, inverse_idx_mapping, idx_name_mapping


def split_and_save(df, split, save_dir, num_idx, cat_idx, tgt_idx):
    """
    Split data into features and target, and save as numpy arrays.
    
    Args:
        df: DataFrame to split
        split: Split name ('train' or 'test')
        save_dir: Directory to save files
        num_idx: Indices of numerical columns
        cat_idx: Indices of categorical columns
        tgt_idx: Indices of target columns
    """
    X_num = df.iloc[:, num_idx].astype(np.float32).to_numpy()
    X_cat = df.iloc[:, cat_idx].astype(str).to_numpy()
    y = df.iloc[:, tgt_idx].astype(np.float32).to_numpy().reshape(-1, 1)

    np.save(save_dir / f"X_num_{split}.npy", X_num)
    np.save(save_dir / f"X_cat_{split}.npy", X_cat)
    np.save(save_dir / f"y_{split}.npy", y)


def preprocess_flight_dataset(train_path, model_dir):
    """
    Preprocess flight data for TabSyn and save in the model directory.
    
    Args:
        train_path: Path to training data
        test_path: Path to test data
        model_dir: Directory to save model and processed data
        
    Returns:
        str: Path to the processed dataset directory
    """
    # Create data directory within model_dir
    data_dir = Path(model_dir) / "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Default categorical columns for flight data
    cat_columns = [
        "IATA_CARRIER_CODE",
        "DEPARTURE_IATA_AIRPORT_CODE",
        "ARRIVAL_IATA_AIRPORT_CODE",
        "AIRCRAFT_TYPE_IATA",
    ]
    
    # Default target column for flight data
    target_column = "DEPARTURE_DELAY_MIN"
    
    # Load and preprocess data
    raw_train = pd.read_csv(train_path)
    
    train_df = preprocess_flight_data(raw_train)

    
    
    # Get column indices
    column_order = train_df.columns.tolist()
    
    num_idx = [i for i, c in enumerate(column_order)
               if c not in cat_columns + [target_column]]
    cat_idx = [column_order.index(c) for c in cat_columns]
    tgt_idx = [column_order.index(target_column)]
    
    # Save as numpy arrays
    split_and_save(train_df, "train", data_dir, num_idx, cat_idx, tgt_idx)

    
    # Create and save info.json
    idx_map, inv_map, idx_name_map = get_column_name_mapping(
        train_df, num_idx, cat_idx, tgt_idx, column_order)
    
    # Get dataset name from model directory
    dataset_name = os.path.basename(os.path.normpath(model_dir))
    
    info = {
        "name": dataset_name,
        "task_type": "regression",  # Flight delay is regression
        "header": "infer",
        "column_names": column_order,
        "num_col_idx": num_idx,
        "cat_col_idx": cat_idx,
        "target_col_idx": tgt_idx,
        "file_type": "csv",
        # Raw CSVs for reference
        "data_path": str(data_dir / "train.csv"),
        "test_path": str(data_dir / "test.csv"),
        # Helper mappings for TabSyn
        "idx_mapping": idx_map,
        "inverse_idx_mapping": inv_map,
        "idx_name_mapping": idx_name_map,
        "train_size": len(train_df),
    }
    
    with open(data_dir / "info.json", "w") as f:
        json.dump(info, f, indent=4)
    
    print(f"✅ Flight dataset processed and saved to {data_dir}")
    return str(data_dir)


def train_tabsyn(
    train_path,
    model_dir,
    vae_epochs=10,
    diffusion_epochs=10,
    embedding_dim=4,
    vae_lr=1e-3,
    diffusion_lr=1e-4,
    batch_size=4096,
    max_beta=1e-2,
    min_beta=1e-5,
    beta_decay=0.7,
    device="cuda",
    verbose=True,
):
    """
    Preprocess flight data and train a TabSyn model.
    
    Args:
        train_path: Path to the training data CSV
        test_path: Path to the test data CSV
        model_dir: Directory to save the model and data
        vae_epochs: Number of epochs to train the VAE component
        diffusion_epochs: Number of epochs to train the diffusion component
        embedding_dim: Dimension of the embedding space
        vae_lr: Learning rate for VAE training
        diffusion_lr: Learning rate for diffusion model training
        batch_size: Batch size for training
        max_beta: Maximum beta value for KL annealing
        min_beta: Minimum beta value for KL annealing
        beta_decay: Decay factor for beta annealing
        device: Device to use for training ("cuda" or "cpu")
        verbose: Whether to print verbose output
    
    Returns:
        TabSyn: Trained TabSyn model
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Step 1: Preprocess the data
    print(f"Preprocessing flight data...")
    data_dir = preprocess_flight_dataset(
        train_path=train_path,
        model_dir=model_dir
    )
    
    # Step 2: Initialize TabSyn model
    print(f"Initializing TabSyn model...")
    vae_factor = 32  
    vae_layers = 2   
    n_haed = 1      # Number of heads for the transformer
    tabsyn = TabSyn(
        embedding_dim=embedding_dim,
        vae_factor=vae_factor,
        vae_layers=vae_layers,
        vae_lr=vae_lr,
        max_beta=max_beta,
        min_beta=min_beta,
        beta_decay=beta_decay,
        diffusion_lr=diffusion_lr,
        vae_epochs=vae_epochs,
        diffusion_epochs=diffusion_epochs,
        batch_size=batch_size,
        device=device,
        verbose=verbose
    )
    
    # Step 3: Train the model
    print(f"Training TabSyn model...")
    tabsyn.fit(data_dir, task_type="regression")
    
    # Step 4: Save the model
    model_path = os.path.join(model_dir, "tabsyn_model.pkl")
    print(f"Saving model to {model_path}...")
    tabsyn.save(model_path)
    
    # Save training parameters for reference
    params = {
        "vae_epochs": vae_epochs,
        "diffusion_epochs": diffusion_epochs,
        "embedding_dim": embedding_dim,
        "vae_lr": vae_lr,
        "diffusion_lr": diffusion_lr,
        "batch_size": batch_size,
        "max_beta": max_beta,
        "min_beta": min_beta,
        "beta_decay": beta_decay,
        "device": device,
        "vae_layers": vae_layers,
        "vae_factor": vae_factor,
        "n_head": n_haed,
    }
    
    with open(os.path.join(model_dir, "training_params.json"), "w") as f:
        json.dump(params, f, indent=4)
    
    print(f"Training complete. Model saved to: {model_path}")
    return tabsyn


def main():
    parser = argparse.ArgumentParser(description="Preprocess flight data and train a TabSyn model")
    
    # Data input paths
    parser.add_argument(
        "--train_path",
        type=str,
        required=True,
        help="Path to the training CSV file"
    )
    
    # Model output directory
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory to save the model and processed data"
    )
    
    # Training parameters
    parser.add_argument(
        "--vae_epochs",
        type=int,
        default=200,
        help="Number of epochs to train the VAE component"
    )
    parser.add_argument(
        "--diffusion_epochs",
        type=int,
        default=1000,
        help="Number of epochs to train the diffusion component"
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=4,
        help="Dimension of the embedding space"
    )
    parser.add_argument(
        "--vae_lr",
        type=float,
        default=1e-3,
        help="Learning rate for VAE training"
    )
    parser.add_argument(
        "--diffusion_lr",
        type=float,
        default=3e-4,
        help="Learning rate for diffusion model training"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8192,
        help="Batch size for training"
    )
    parser.add_argument(
        "--max_beta",
        type=float,
        default=1e-2,
        help="Maximum beta value for KL annealing"
    )
    parser.add_argument(
        "--min_beta",
        type=float,
        default=1e-5,
        help="Minimum beta value for KL annealing"
    )
    parser.add_argument(
        "--beta_decay",
        type=float,
        default=0.7,
        help="Decay factor for beta annealing"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for training"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print verbose output"
    )
    
    args = parser.parse_args()
    
    # Train the model
    train_tabsyn(
        train_path=args.train_path,
        model_dir=args.model_dir,
        vae_epochs=args.vae_epochs,
        diffusion_epochs=args.diffusion_epochs,
        embedding_dim=args.embedding_dim,
        vae_lr=args.vae_lr,
        diffusion_lr=args.diffusion_lr,
        batch_size=args.batch_size,
        max_beta=args.max_beta,
        min_beta=args.min_beta,
        beta_decay=args.beta_decay,
        device=args.device,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

# Example usage:
# python train_tabsyn.py --train_path data/real/train.csv --test_path data/real/test.csv --model_dir models/flight_model