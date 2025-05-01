"""
Train on Synthetic, Test on Real (TSTR) Evaluation Module.

This module contains functions for evaluating synthetic data by 
training models on synthetic data and testing them on real data.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from category_encoders import TargetEncoder


def train_and_evaluate_model(model_name, model, X_train, y_train, X_test, y_test, 
                           cat_features, num_features):
    """
    Train a model and evaluate its performance
    For CatBoost, use its native categorical feature handling
    For other models, use target encoding
    """
    if model_name == 'CatBoost':
        # CatBoost handles categorical features natively
        # Create a new DataFrame with all features
        X_train_all = X_train.copy()
        X_test_all = X_test.copy()
        
        # CatBoost requires categorical features to be of 'int' or 'str' type
        for col in cat_features:
            X_train_all[col] = X_train_all[col].astype('str')
            X_test_all[col] = X_test_all[col].astype('str')
        
        # Get indices of categorical features
        cat_features_indices = [X_train_all.columns.get_loc(col) for col in cat_features]
        
        # Train the model with categorical features specified
        model.fit(X_train_all, y_train, cat_features=cat_features_indices)
        
        # Make predictions
        y_pred = model.predict(X_test_all)
    else:
        # For other models, use target encoding and standardization
        # Target encoding for categorical features
        encoder = TargetEncoder()
        X_train_cat_encoded = encoder.fit_transform(X_train[cat_features], y_train)
        X_test_cat_encoded = encoder.transform(X_test[cat_features])
        
        # Scale numerical features
        scaler = StandardScaler()
        X_train_num_scaled = pd.DataFrame(scaler.fit_transform(X_train[num_features]), 
                                        columns=num_features)
        X_test_num_scaled = pd.DataFrame(scaler.transform(X_test[num_features]), 
                                       columns=num_features)
        
        # Combine categorical and numerical features
        X_train_processed = pd.concat([X_train_cat_encoded, X_train_num_scaled], axis=1)
        X_test_processed = pd.concat([X_test_cat_encoded, X_test_num_scaled], axis=1)
        
        # Train the model
        model.fit(X_train_processed, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_processed)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'model_name': model_name,
        'pipeline': model,  # Return the model rather than a pipeline
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': y_pred
    }


def evaluate_tstr(target, train_datasets, test_dataset, prediction_mode='pre-tactical'):
    """
    Perform Train-Synthetic-Test-Real (TSTR) evaluation
    
    Args:
        target (str): Target variable to predict ('DEPARTURE_DELAY_MIN', 'ARRIVAL_DELAY_MIN', or 'TURNAROUND_MIN')
        train_datasets (dict): Dictionary of datasets to train on (name -> dataframe)
        test_dataset (pd.DataFrame): Real test dataset
        prediction_mode (str): 'pre-tactical' (before departure) or 'tactical' (real-time with latest info)
    
    Returns:
        tuple: (results, cat_features, num_features) - Results dict and feature lists used
    """
    # Define features based on target and prediction mode
    if target == 'DEPARTURE_DELAY_MIN':
        # Only pre-tactical is applicable for departure delay predictions
        # since we're predicting the departure delay itself
        cat_features = [
            'IATA_CARRIER_CODE', 
            'DEPARTURE_IATA_AIRPORT_CODE', 
            'ARRIVAL_IATA_AIRPORT_CODE',
            'AIRCRAFT_TYPE_IATA',
            'DAY_OF_WEEK'  # Added for weekly patterns
        ]
        
        num_features = [
            'SCHEDULED_MONTH',
            'SCHEDULED_DAY',
            'SCHEDULED_HOUR',
            'SCHEDULED_MINUTE',
            'SCHEDULED_DURATION_MIN'
        ]
        
        # For departure delay, tactical mode doesn't make sense, so we enforce pre-tactical
        if prediction_mode == 'tactical':
            print("Warning: For DEPARTURE_DELAY_MIN, only pre-tactical mode is applicable. Using pre-tactical features.")
            prediction_mode = 'pre-tactical'
            
    elif target == 'ARRIVAL_DELAY_MIN':
        cat_features = [
            'IATA_CARRIER_CODE', 
            'DEPARTURE_IATA_AIRPORT_CODE',
            'ARRIVAL_IATA_AIRPORT_CODE', 
            'AIRCRAFT_TYPE_IATA',
            'DAY_OF_WEEK'
        ]
        
        if prediction_mode == 'pre-tactical':
            # Pre-tactical: Don't use departure delay or actual flight info in features
            num_features = [
                'SCHEDULED_MONTH',
                'SCHEDULED_DAY',
                'SCHEDULED_HOUR',
                'SCHEDULED_MINUTE',
                'SCHEDULED_DURATION_MIN'
            ]
        else:  # tactical
            # Tactical: Include departure delay and duration difference
            num_features = [
                'SCHEDULED_MONTH',
                'SCHEDULED_DAY',
                'SCHEDULED_HOUR',
                'SCHEDULED_MINUTE',
                'SCHEDULED_DURATION_MIN',
                'DEPARTURE_DELAY_MIN',  # Available in real-time after departure
                # 'DURATION_DIFF_MIN'     # Can be partially estimated during flight (but not used here since it may leak info)
            ]
            
    elif target == 'TURNAROUND_MIN':
        cat_features = [
            'IATA_CARRIER_CODE',
            'DEPARTURE_IATA_AIRPORT_CODE',
            'AIRCRAFT_TYPE_IATA',
            'ARRIVAL_IATA_AIRPORT_CODE',
            'DAY_OF_WEEK'
        ]
        
        if prediction_mode == 'pre-tactical':
            # Pre-tactical: Only use scheduled information
            num_features = [
                'SCHEDULED_MONTH',
                'SCHEDULED_DAY',
                'SCHEDULED_HOUR',
                'SCHEDULED_MINUTE',
                'SCHEDULED_DURATION_MIN'
            ]
        else:  # tactical
            # Tactical: Include actual flight information
            num_features = [
                'SCHEDULED_MONTH',
                'SCHEDULED_DAY',
                'SCHEDULED_HOUR',
                'SCHEDULED_MINUTE',
                'SCHEDULED_DURATION_MIN',
                'ACTUAL_DURATION_MIN',  # Known after the flight
                'DURATION_DIFF_MIN',    # Difference between actual and scheduled
                'DEPARTURE_DELAY_MIN',  # Known after departure
                'ARRIVAL_DELAY_MIN'     # Known after arrival (for turnaround prediction)
            ]
    else:
        raise ValueError("Invalid target variable. Use 'DEPARTURE_DELAY_MIN', 'ARRIVAL_DELAY_MIN', or 'TURNAROUND_MIN'")
    
    # Define model factories instead of instances
    model_factories = {
        'Decision Tree': lambda: DecisionTreeRegressor(max_depth=10, min_samples_split=10),
        'Random Forest': lambda: RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=10, n_jobs=-1),
        'Gradient Boosting': lambda: GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, min_samples_split=10),
        'XGBoost': lambda: XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5),
        'CatBoost': lambda: CatBoostRegressor(iterations=100, learning_rate=0.1, depth=5, verbose=0)
    }
    
    # Prepare test data
    X_test = test_dataset[cat_features + num_features]
    y_test = test_dataset[target]
    
    # Results container
    results = {}
    
    # For each dataset, train each model and evaluate on test data
    for dataset_name, train_dataset in train_datasets.items():
        print(f"\nTraining models on {dataset_name} dataset for {target} prediction ({prediction_mode} mode)...")
        
        # Prepare training data
        X_train = train_dataset[cat_features + num_features]
        y_train = train_dataset[target]
        
        dataset_results = {}
        
        # Train and evaluate each model
        # for model_name, model in models.items():
        for model_name, model_factory in model_factories.items():
            # Create a fresh model instance each time
            model = model_factory()
            print(f"  Training {model_name}...")
            
            result = train_and_evaluate_model(
                model_name, model, X_train, y_train, X_test, y_test, 
                cat_features, num_features
            )
            
            print(f"  {model_name} - RMSE: {result['rmse']:.2f}, MAE: {result['mae']:.2f}, R²: {result['r2']:.4f}")
            
            dataset_results[model_name] = result
        
        results[dataset_name] = dataset_results
    
    return results, cat_features, num_features


def calculate_utility_scores(results):
    """
    Calculate utility scores for synthetic datasets.
    Utility score measures how close the performance of a model trained on synthetic data
    is to a model trained on real data.
    
    Args:
        results (dict): Results from evaluate_tstr function
        
    Returns:
        pd.DataFrame: Utility scores for each model and synthetic dataset
    """
    # Get list of datasets and models
    datasets = list(results.keys())
    synthetic_datasets = [d for d in datasets if d != 'Real']
    model_names = list(results[datasets[0]].keys())
    
    # Calculate utility scores
    utility_scores = []
    
    for model_name in model_names:
        real_rmse = results['Real'][model_name]['rmse']
        real_r2 = results['Real'][model_name]['r2']
        
        for dataset_name in synthetic_datasets:
            synth_rmse = results[dataset_name][model_name]['rmse']
            synth_r2 = results[dataset_name][model_name]['r2']
            
            # Calculate RMSE utility (lower is better, so we want synth_rmse close to real_rmse)
            # We take the ratio of real to synthetic, capped at 1.0
            rmse_utility = min(real_rmse / synth_rmse, 1.0) if synth_rmse > 0 else 0.0
            
            # Calculate R² utility (higher is better)
            # For negative R², we set utility to 0
            r2_utility = synth_r2 / real_r2 if real_r2 > 0 and synth_r2 > 0 else 0.0
            r2_utility = max(min(r2_utility, 1.0), 0.0)  # Cap between 0 and 1
            
            # Overall utility is average of RMSE and R² utilities
            overall_utility = (rmse_utility + r2_utility) / 2
            
            utility_scores.append({
                'Model': model_name,
                'Dataset': dataset_name,
                'RMSE Utility': rmse_utility,
                'R² Utility': r2_utility,
                'Overall Utility': overall_utility
            })
    
    utility_df = pd.DataFrame(utility_scores)
    return utility_df

def extract_feature_importances_target_encoding(results, cat_features, num_features):
    """Extract feature importances from models using target encoding"""
    feature_importances = {}
    
    for dataset_name, models in results.items():
        feature_importances[dataset_name] = {}
        
        for model_name, model_result in models.items():
            # Get the model object
            model = model_result['pipeline']
            feature_names = cat_features + num_features
            
            try:
                if model_name == 'CatBoost' and hasattr(model, 'get_feature_importance'):
                    # For CatBoost, get feature names and importances
                    importances = model.get_feature_importance()
                    # Ensure feature names are aligned with importance values
                    if hasattr(model, 'feature_names_'):
                        feature_names = model.feature_names_
                
                elif model_name == 'XGBoost':
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                    elif hasattr(model, 'get_booster'):
                        # Alternative for XGBoost: use the booster's feature importance
                        importance_dict = model.get_booster().get_score(importance_type='gain')
                        if hasattr(model, 'feature_names_'):
                            feature_names = model.feature_names_
                            importances = [importance_dict.get(f, 0) for f in feature_names]
                        else:
                            importances = np.zeros(len(feature_names))
                            for feature, importance in importance_dict.items():
                                if feature.startswith('f'):
                                    try:
                                        idx = int(feature[1:])
                                        if idx < len(importances):
                                            importances[idx] = importance
                                    except ValueError:
                                        pass
                    else:
                        print(f"Cannot extract feature importances for XGBoost model in {dataset_name}")
                        continue
                
                elif hasattr(model, 'feature_importances_'):
                    # For tree-based models (Decision Tree, Random Forest, etc.)
                    importances = model.feature_importances_
                
                else:
                    print(f"Skipping {model_name} for {dataset_name} - cannot extract feature importances")
                    continue
                
                # Verify lengths match
                if len(importances) != len(feature_names):
                    print(f"Warning: Feature importance length ({len(importances)}) doesn't match feature names length ({len(feature_names)}) for {model_name} in {dataset_name}")
                    # Use minimal length to avoid errors
                    min_len = min(len(importances), len(feature_names))
                    importances = importances[:min_len]
                    feature_names = feature_names[:min_len]
                
                # Normalize importances to sum to 1.0 for consistent comparison
                if np.sum(importances) > 0:
                    importances = importances / np.sum(importances)
                
                # Store feature importances as a pandas Series
                feature_importances[dataset_name][model_name] = pd.Series(
                    importances,
                    index=feature_names
                ).sort_values(ascending=False)
                
            except Exception as e:
                print(f"Error extracting feature importances for {model_name} in {dataset_name}: {str(e)}")
                continue
    
    return feature_importances