"""Data utility functions for TabSyn."""

import numpy as np
import os
import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    """PyTorch Dataset for tabular data with numerical and categorical features."""
    
    def __init__(self, X_num, X_cat):
        """
        Initialize TabularDataset.
        
        Args:
            X_num (torch.Tensor): Numerical features
            X_cat (torch.Tensor): Categorical features
        """
        # Make sure data stays on CPU for DataLoader workers
        self.X_num = X_num.cpu() if isinstance(X_num, torch.Tensor) else X_num
        self.X_cat = X_cat.cpu() if isinstance(X_cat, torch.Tensor) else X_cat

    def __getitem__(self, index):
        """Get item at index."""
        this_num = self.X_num[index]
        this_cat = self.X_cat[index]
        sample = (this_num, this_cat)
        return sample

    def __len__(self):
        """Return dataset length."""
        return self.X_num.shape[0]

def preprocess(dataset_path, task_type='regression', inverse=False, cat_encoding=None, concat=True):
    """
    Preprocess data for TabSyn.
    
    Args:
        dataset_path (str): Path to dataset directory
        task_type (str): Type of task ('regression' or 'classification')
        inverse (bool): Whether to return inverse transforms
        cat_encoding (str, optional): Categorical encoding method
        concat (bool): Whether to concatenate target to features
        
    Returns:
        tuple: Processed data components
    """
    from . import transform_module  # Import here to avoid circular imports
    
    T_dict = {}
    T_dict['normalization'] = "quantile"
    T_dict['num_nan_policy'] = 'mean'
    T_dict['cat_nan_policy'] = None
    T_dict['cat_min_frequency'] = None
    T_dict['cat_encoding'] = cat_encoding
    T_dict['y_policy'] = "default"

    T = transform_module.Transformations(**T_dict)

    dataset = make_dataset(
        data_path=dataset_path,
        T=T,
        task_type=task_type,
        change_val=False,
        concat=concat
    )

    if cat_encoding is None:
        X_num = dataset.X_num
        X_cat = dataset.X_cat

        X_train_num, X_test_num = X_num['train'], X_num['test']
        X_train_cat, X_test_cat = X_cat['train'], X_cat['test']
        
        categories = get_categories(X_train_cat)
        d_numerical = X_train_num.shape[1]

        X_num = (X_train_num, X_test_num)
        X_cat = (X_train_cat, X_test_cat)

        if inverse:
            num_inverse = dataset.num_transform.inverse_transform
            cat_inverse = dataset.cat_transform.inverse_transform

            return X_num, X_cat, categories, d_numerical, num_inverse, cat_inverse
        else:
            return X_num, X_cat, categories, d_numerical
    else:
        return dataset


def concat_y_to_X(X, y):
    """
    Concatenate target y to features X.
    
    Args:
        X (numpy.ndarray): Feature matrix or None
        y (numpy.ndarray): Target vector
        
    Returns:
        numpy.ndarray: Concatenated matrix
    """
    if X is None:
        return y.reshape(-1, 1)
    return np.concatenate([y.reshape(-1, 1), X], axis=1)


def make_dataset(data_path, T, task_type, change_val=False, concat=True):
    """
    Create dataset from raw files.
    
    Args:
        data_path (str): Path to data directory
        T: Transformations object
        task_type (str): Type of task
        change_val (bool): Whether to change validation set
        concat (bool): Whether to concatenate y to X
        
    Returns:
        object: Dataset object
    """
    from . import transform_module  # Import here to avoid circular imports
    
    # For classification tasks
    if task_type == 'binclass' or task_type == 'multiclass':
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')) else None
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) else None
        y = {} if os.path.exists(os.path.join(data_path, 'y_train.npy')) else None

        for split in ['train', 'test']:
            X_num_t, X_cat_t, y_t = read_pure_data(data_path, split)
            if X_num is not None:
                X_num[split] = X_num_t
            if X_cat is not None:
                if concat:
                    X_cat_t = concat_y_to_X(X_cat_t, y_t)
                X_cat[split] = X_cat_t  
            if y is not None:
                y[split] = y_t
    else:
        # For regression tasks
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')) else None
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) else None
        y = {} if os.path.exists(os.path.join(data_path, 'y_train.npy')) else None

        for split in ['train', 'test']:
            X_num_t, X_cat_t, y_t = read_pure_data(data_path, split)

            if X_num is not None:
                if concat:
                    X_num_t = concat_y_to_X(X_num_t, y_t)
                X_num[split] = X_num_t
            if X_cat is not None:
                X_cat[split] = X_cat_t
            if y is not None:
                y[split] = y_t

    # Load dataset info
    import json
    with open(os.path.join(data_path, 'info.json'), 'r') as f:
        info = json.load(f)

    # Create and transform dataset
    D = transform_module.Dataset(
        X_num,
        X_cat,
        y,
        y_info={},
        task_type=transform_module.TaskType(info['task_type']),
        n_classes=info.get('n_classes')
    )

    if change_val:
        D = transform_module.change_val(D)

    return transform_module.transform_dataset(D, T, None)


def get_categories(X_train_cat):
    """
    Get number of categories for each categorical feature.
    
    Args:
        X_train_cat (numpy.ndarray): Categorical features
        
    Returns:
        list: Number of categories for each feature
    """
    return (
        None
        if X_train_cat is None
        else [
            len(set(X_train_cat[:, i]))
            for i in range(X_train_cat.shape[1])
        ]
    )


def read_pure_data(path, split='train'):
    """
    Read raw data files.
    
    Args:
        path (str): Data directory path
        split (str): Data split ('train' or 'test')
        
    Returns:
        tuple: Numerical features, categorical features, and target
    """
    y = np.load(os.path.join(path, f'y_{split}.npy'), allow_pickle=True)
    X_num = None
    X_cat = None
    if os.path.exists(os.path.join(path, f'X_num_{split}.npy')):
        X_num = np.load(os.path.join(path, f'X_num_{split}.npy'), allow_pickle=True)
    if os.path.exists(os.path.join(path, f'X_cat_{split}.npy')):
        X_cat = np.load(os.path.join(path, f'X_cat_{split}.npy'), allow_pickle=True)

    return X_num, X_cat, y