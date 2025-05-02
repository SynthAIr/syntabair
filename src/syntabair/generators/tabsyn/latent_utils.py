"""Latent space utilities for TabSyn."""

import os
import json
import numpy as np
import pandas as pd
import torch
from .data_utils import preprocess
from .vae.model import Decoder_model

def get_input_train(args):
    """
    Get training input for diffusion model from VAE output.
    
    Args:
        args: Arguments object with dataname and device
        
    Returns:
        tuple: Training embeddings and metadata
    """
    dataname = args.dataname

    # Get paths
    curr_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = f'data/{dataname}'
    
    # Load dataset info
    with open(f'{dataset_dir}/info.json', 'r') as f:
        info = json.load(f)

    # Get checkpoint directory
    ckpt_dir = f'{curr_dir}/ckpt/{dataname}/'
    
    # Load VAE embeddings
    embedding_save_path = f'{curr_dir}/tabsyn/vae/ckpt/{dataname}/train_z.npy'
    train_z = torch.tensor(np.load(embedding_save_path)).float()

    # Remove the first token (CLS token) and reshape
    train_z = train_z[:, 1:, :]
    B, num_tokens, token_dim = train_z.size()
    in_dim = num_tokens * token_dim
    
    train_z = train_z.view(B, in_dim)

    return train_z, curr_dir, dataset_dir, ckpt_dir, info


def get_input_generate(model_dir):
    """
    Get input for generation from trained model.
    
    Args:
        dataname: Name of the dataset or path to model directory
        
    Returns:
        tuple: Training embeddings and metadata
    """
  
    dataset_dir = os.path.join(model_dir, 'data')
    ckpt_dir = os.path.join(model_dir, 'ckpt')

    # Load dataset info
    with open(f'{dataset_dir}/info.json', 'r') as f:
        info = json.load(f)

    task_type = info['task_type']

    # Get dataset with inverse transforms
    _, _, categories, d_numerical, num_inverse, cat_inverse = preprocess(
        dataset_dir, task_type=task_type, inverse=True
    )

    # Load VAE embeddings
    embedding_save_path = os.path.join(ckpt_dir, 'train_z.npy')
    train_z = torch.tensor(np.load(embedding_save_path)).float()

    # Remove the first token (CLS token) and reshape
    train_z = train_z[:, 1:, :]
    B, num_tokens, token_dim = train_z.size()
    in_dim = num_tokens * token_dim
    train_z = train_z.view(B, in_dim)
    
    # Load training parameters
    params_path = os.path.join(model_dir, "training_params.json")
    with open(params_path, 'r') as f:
        params = json.load(f)

    # Load decoder
    pre_decoder = Decoder_model(
        params.get("vae_layers", 2),           # Default 2 if not found
        d_numerical, 
        categories, 
        params.get("embedding_dim", 4),        # Default 4 if not found
        n_head=params.get("n_head", 1),        # Default 1 if not found
        factor=params.get("vae_factor", 32)    # Default 32 if not found
    )
    # pre_decoder = Decoder_model(3, d_numerical, categories, 8, n_head=1, factor=64)
    decoder_save_path = os.path.join(ckpt_dir, 'decoder.pt')
    pre_decoder.load_state_dict(torch.load(decoder_save_path, map_location=torch.device('cpu')))

    # Update info with decoder and token_dim
    info['pre_decoder'] = pre_decoder
    info['token_dim'] = token_dim

    return train_z, None, dataset_dir, ckpt_dir, info, num_inverse, cat_inverse


@torch.no_grad()
def split_num_cat_target(syn_data, info, num_inverse, cat_inverse, device):
    """
    Split synthetic data into numerical, categorical, and target components.
    
    Args:
        syn_data: Synthetic data from diffusion model
        info: Dataset info
        num_inverse: Numerical inverse transform function
        cat_inverse: Categorical inverse transform function
        device: Computation device
        
    Returns:
        tuple: Numerical features, categorical features, and target
    """
    task_type = info['task_type']

    # Get column indices
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    # Count features
    n_num_feat = len(num_col_idx)
    n_cat_feat = len(cat_col_idx)

    if task_type == 'regression':
        n_num_feat += len(target_col_idx)
    else:
        n_cat_feat += len(target_col_idx)

    # Get decoder and token dimension
    pre_decoder = info['pre_decoder']
    token_dim = info['token_dim']
    
    # Make sure the decoder is on the correct device
    for param in pre_decoder.parameters():
        param_device = param.device
        if str(param_device) != str(device):
            pre_decoder = pre_decoder.to(device)
            break

    # Reshape data for decoder
    syn_data = syn_data.reshape(syn_data.shape[0], -1, token_dim)
    
    # Move data to the same device as the decoder
    syn_data_tensor = torch.tensor(syn_data, device=device)
    
    # Decode synthetic data 
    norm_input = pre_decoder(syn_data_tensor)
    x_hat_num, x_hat_cat = norm_input

    # Process categorical predictions
    syn_cat = []
    for pred in x_hat_cat:
        syn_cat.append(pred.argmax(dim=-1))

    # Convert to numpy
    syn_num = x_hat_num.cpu().numpy()
    syn_cat = torch.stack(syn_cat).t().cpu().numpy()

    # Apply inverse transforms
    syn_num = num_inverse(syn_num)
    syn_cat = cat_inverse(syn_cat)

    # Extract target based on task type
    if info['task_type'] == 'regression':
        syn_target = syn_num[:, :len(target_col_idx)]
        syn_num = syn_num[:, len(target_col_idx):]
    else:
        syn_target = syn_cat[:, :len(target_col_idx)]
        syn_cat = syn_cat[:, len(target_col_idx):]

    return syn_num, syn_cat, syn_target

def recover_data(syn_num, syn_cat, syn_target, info):
    """
    Recover data into original format with column ordering.
    
    Args:
        syn_num: Synthetic numerical features
        syn_cat: Synthetic categorical features
        syn_target: Synthetic target
        info: Dataset info
        
    Returns:
        pandas.DataFrame: Recovered data
    """
    # Get column indices
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    # Get index mapping
    idx_mapping = info['idx_mapping']
    idx_mapping = {int(key): value for key, value in idx_mapping.items()}

    # Create dataframe
    syn_df = pd.DataFrame()

    # Fill in data based on task type
    if info['task_type'] == 'regression':
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]] 
            elif i in set(cat_col_idx):
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            else:
                syn_df[i] = syn_target[:, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)]
    else:
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]]
            elif i in set(cat_col_idx):
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            else:
                syn_df[i] = syn_target[:, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)]

    return syn_df