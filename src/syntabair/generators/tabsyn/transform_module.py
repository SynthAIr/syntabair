"""Transform module for data preprocessing in TabSyn."""

import enum
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler, OrdinalEncoder
from sklearn.pipeline import make_pipeline


class TaskType(enum.Enum):
    """Enum for task types."""
    BINCLASS = 'binclass'
    MULTICLASS = 'multiclass'
    REGRESSION = 'regression'
    
    def __str__(self) -> str:
        return self.value


@dataclass
class Dataset:
    """Dataset class for TabSyn."""
    X_num: Optional[Dict[str, np.ndarray]]
    X_cat: Optional[Dict[str, np.ndarray]]
    y: Dict[str, np.ndarray]
    y_info: Dict[str, Any]
    task_type: TaskType
    n_classes: Optional[int]
    
    @property
    def is_regression(self) -> bool:
        """Check if task is regression."""
        return self.task_type == TaskType.REGRESSION
    
    @property
    def n_num_features(self) -> int:
        """Get number of numerical features."""
        return 0 if self.X_num is None else self.X_num['train'].shape[1]
    
    @property
    def n_cat_features(self) -> int:
        """Get number of categorical features."""
        return 0 if self.X_cat is None else self.X_cat['train'].shape[1]


@dataclass
class Transformations:
    """Configuration for data transformations."""
    normalization: Optional[str] = None
    num_nan_policy: Optional[str] = None
    cat_nan_policy: Optional[str] = None
    cat_min_frequency: Optional[float] = None
    cat_encoding: Optional[str] = None
    y_policy: Optional[str] = 'default'
    seed: int = 0


def change_val(dataset: Dataset, val_size: float = 0.2):
    """
    Change validation set in the dataset.
    
    Args:
        dataset: Dataset object
        val_size: Size of validation set
        
    Returns:
        Dataset: Updated dataset
    """
    from sklearn.model_selection import train_test_split
    
    # Should be done before transformations
    if 'val' not in dataset.y:
        dataset.y['val'] = np.empty((0, dataset.y['train'].shape[1]) 
                                    if dataset.y['train'].ndim > 1 
                                    else (0,))
    
    y = np.concatenate([dataset.y['train'], dataset.y['val']], axis=0)
    
    ixs = np.arange(y.shape[0])
    if dataset.is_regression:
        train_ixs, val_ixs = train_test_split(ixs, test_size=val_size, random_state=777)
    else:
        train_ixs, val_ixs = train_test_split(ixs, test_size=val_size, random_state=777, stratify=y)
    
    dataset.y['train'] = y[train_ixs]
    dataset.y['val'] = y[val_ixs]
    
    if dataset.X_num is not None:
        X_num = np.concatenate([dataset.X_num['train'], dataset.X_num['val']], axis=0)
        dataset.X_num['train'] = X_num[train_ixs]
        dataset.X_num['val'] = X_num[val_ixs]
    
    if dataset.X_cat is not None:
        X_cat = np.concatenate([dataset.X_cat['train'], dataset.X_cat['val']], axis=0)
        dataset.X_cat['train'] = X_cat[train_ixs]
        dataset.X_cat['val'] = X_cat[val_ixs]
    
    return dataset


def num_process_nans(dataset: Dataset, policy: Optional[str]):
    """
    Process NaN values in numerical features.
    
    Args:
        dataset: Dataset object
        policy: NaN handling policy
        
    Returns:
        Dataset: Processed dataset
    """
    assert dataset.X_num is not None
    nan_masks = {k: np.isnan(v) for k, v in dataset.X_num.items()}
    if not any(x.any() for x in nan_masks.values()):
        print('No NaNs in numerical features, skipping')
        return dataset
    
    assert policy is not None
    if policy == 'mean':
        new_values = np.nanmean(dataset.X_num['train'], axis=0)
        X_num = {k: v.copy() for k, v in dataset.X_num.items()}
        for k, v in X_num.items():
            num_nan_indices = np.where(nan_masks[k])
            v[num_nan_indices] = np.take(new_values, num_nan_indices[1])
        dataset.X_num = X_num
    else:
        raise ValueError(f"Unknown policy: {policy}")
    
    return dataset


def normalize(X: Dict[str, np.ndarray], normalization: str, seed: int, return_normalizer: bool = False):
    """
    Normalize numerical features.
    
    Args:
        X: Dictionary of numerical features by split
        normalization: Normalization method
        seed: Random seed
        return_normalizer: Whether to return the normalizer
        
    Returns:
        Dict or tuple: Normalized features (and normalizer if requested)
    """
    X_train = X['train']
    
    if normalization == 'standard':
        normalizer = StandardScaler()
    elif normalization == 'minmax':
        normalizer = MinMaxScaler()
    elif normalization == 'quantile':
        normalizer = QuantileTransformer(
            output_distribution='normal',
            n_quantiles=max(min(X['train'].shape[0] // 30, 1000), 10),
            subsample=int(1e9),
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown normalization: {normalization}")
    
    normalizer.fit(X_train)
    
    if return_normalizer:
        return {k: normalizer.transform(v) for k, v in X.items()}, normalizer
    
    return {k: normalizer.transform(v) for k, v in X.items()}


def cat_encode(X: Dict[str, np.ndarray], return_encoder: bool = False):
    """
    Encode categorical features.
    
    Args:
        X: Dictionary of categorical features by split
        return_encoder: Whether to return the encoder
        
    Returns:
        tuple: Encoded features, is_numerical flag, and encoder if requested
    """
    # Map strings to 0-based ranges
    unknown_value = np.iinfo('int64').max - 3
    oe = OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=unknown_value,
        dtype='int64',
    ).fit(X['train'])
    
    encoder = make_pipeline(oe)
    encoder.fit(X['train'])
    
    X_encoded = {k: encoder.transform(v) for k, v in X.items()}
    max_values = X_encoded['train'].max(axis=0)
    
    for part in X_encoded.keys():
        if part == 'train':
            continue
        for column_idx in range(X_encoded[part].shape[1]):
            mask = X_encoded[part][:, column_idx] == unknown_value
            X_encoded[part][mask, column_idx] = max_values[column_idx] + 1
    
    if return_encoder:
        return X_encoded, False, encoder
    
    return X_encoded, False


def build_target(y: Dict[str, np.ndarray], policy: Optional[str], task_type: TaskType):
    """
    Process target variable.
    
    Args:
        y: Dictionary of target values by split
        policy: Processing policy
        task_type: Type of task
        
    Returns:
        tuple: Processed target and info
    """
    info: Dict[str, Any] = {'policy': policy}
    
    if policy is None:
        return y, info
    elif policy == 'default':
        if task_type == TaskType.REGRESSION:
            mean, std = float(y['train'].mean()), float(y['train'].std())
            y = {k: (v - mean) / std for k, v in y.items()}
            info['mean'] = mean
            info['std'] = std
    else:
        raise ValueError(f"Unknown policy: {policy}")
    
    return y, info


def transform_dataset(dataset: Dataset, transformations: Transformations, cache_dir=None, return_transforms: bool = False):
    """
    Apply transformations to dataset.
    
    Args:
        dataset: Dataset object
        transformations: Transformation configuration
        cache_dir: Cache directory (not used)
        return_transforms: Whether to return transformation objects
        
    Returns:
        Dataset: Transformed dataset
    """
    # Process numerical NaNs
    if dataset.X_num is not None:
        dataset = num_process_nans(dataset, transformations.num_nan_policy)
    
    num_transform = None
    cat_transform = None
    X_num = dataset.X_num
    
    # Normalize numerical features
    if X_num is not None and transformations.normalization is not None:
        X_num, num_transform = normalize(
            X_num,
            transformations.normalization,
            transformations.seed,
            return_normalizer=True
        )
    
    # Process categorical features
    if dataset.X_cat is None:
        X_cat = None
    else:
        X_cat = dataset.X_cat
        X_cat, is_num, cat_transform = cat_encode(
            X_cat,
            return_encoder=True
        )
        
        if is_num:
            X_num = (
                X_cat
                if X_num is None
                else {x: np.hstack([X_num[x], X_cat[x]]) for x in X_num}
            )
            X_cat = None
    
    # Process target
    y, y_info = build_target(dataset.y, transformations.y_policy, dataset.task_type)
    
    # Create new dataset
    new_dataset = Dataset(
        X_num=X_num,
        X_cat=X_cat,
        y=y,
        y_info=y_info,
        task_type=dataset.task_type,
        n_classes=dataset.n_classes
    )
    
    # Add transforms as attributes
    new_dataset.num_transform = num_transform
    new_dataset.cat_transform = cat_transform
    
    return new_dataset