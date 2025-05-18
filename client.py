#!/usr/bin/env python3
import os
import argparse
import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from utils import load_and_preprocess_data, evaluate_model, save_model
from network import send_model

def tune_hyperparameters(X_train, y_train, n_estimators_range=None, max_depth_range=None, random_state=42):
    """
    Tune hyperparameters for Random Forest model using Grid Search.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators_range: Range of n_estimators to try
        max_depth_range: Range of max_depth to try
        random_state: Random seed
        
    Returns:
        Best parameters
    """
    if n_estimators_range is None:
        n_estimators_range = [10, 20, 50]
    if max_depth_range is None:
        max_depth_range = [5, 10, 15, None]
    
    print("\n===== Tuning hyperparameters =====")
    param_grid = {
        'n_estimators': n_estimators_range,
        'max_depth': max_depth_range,
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    # Use a small subset for faster tuning if dataset is large
    if len(X_train) > 1000:
        sample_indices = np.random.choice(len(X_train), 1000, replace=False)
        X_sample = X_train.iloc[sample_indices]
        y_sample = y_train.iloc[sample_indices]
    else:
        X_sample = X_train
        y_sample = y_train
    
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=random_state),
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    start_time = time.time()
    grid_search.fit(X_sample, y_sample)
    tuning_time = time.time() - start_time
    
    print(f"Hyperparameter tuning completed in {tuning_time:.2f} seconds")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best MSE: {-grid_search.best_score_:.4f}")
    
    return grid_search.best_params_

def train_local_model(data_path, n_estimators=10, max_depth=5, min_samples_split=2, 
                     min_samples_leaf=1, bootstrap=True, random_state=42, auto_tune=False):
    """
    Train a local Random Forest model.
    
    Args:
        data_path: Path to the dataset
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of the trees
        min_samples_split: Minimum samples required to split a node
        min_samples_leaf: Minimum samples required at a leaf node
        bootstrap: Whether to use bootstrap samples
        random_state: Random seed for reproducibility
        auto_tune: Whether to automatically tune hyperparameters
        
    Returns:
        Trained model, test data, and validation metrics
    """
    print("\n===== Loading and preprocessing data =====")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(
        data_path, test_size=0.2, random_state=random_state
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Tune hyperparameters if requested
    if auto_tune:
        best_params = tune_hyperparameters(X_train, y_train, random_state=random_state)
        n_estimators = best_params.get('n_estimators', n_estimators)
        max_depth = best_params.get('max_depth', max_depth)
        min_samples_split = best_params.get('min_samples_split', min_samples_split)
        min_samples_leaf = best_params.get('min_samples_leaf', min_samples_leaf)
        bootstrap = best_params.get('bootstrap', bootstrap)
    
    print("\n===== Training local Random Forest model =====")
    print(f"Parameters: n_estimators={n_estimators}, max_depth={max_depth}, "
          f"min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, "
          f"bootstrap={bootstrap}")
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        bootstrap=bootstrap,
        random_state=random_state
    )
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"Model trained in {training_time:.2f} seconds")
    
    print("\n===== Evaluating local model =====")
    metrics = evaluate_model(model, X_test, y_test, "Local Random Forest")
    
    # Feature importance analysis
    feature_importances = model.feature_importances_
    feature_names = X_train.columns
    
    # Print top 5 features
    sorted_idx = np.argsort(feature_importances)[::-1]
    print("\nTop 5 important features:")
    for i in range(min(5, len(feature_names))):
        idx = sorted_idx[i]
        print(f"{feature_names[idx]}: {feature_importances[idx]:.4f}")
    
    return model, X_test, y_test, metrics

def main():
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument("--data", type=str, default="/Users/sidharthmohan/sem6/dist_sys/fed/housing.csv",
                        help="Path to the dataset")
    parser.add_argument("--server", type=str, default="localhost",
                        help="Server hostname")
    parser.add_argument("--port", type=int, default=9999,
                        help="Server port")
    parser.add_argument("--n-estimators", type=int, default=10,
                        help="Number of trees in the forest")
    parser.add_argument("--max-depth", type=int, default=5,
                        help="Maximum depth of the trees")
    parser.add_argument("--min-samples-split", type=int, default=2,
                        help="Minimum samples required to split a node")
    parser.add_argument("--min-samples-leaf", type=int, default=1,
                        help="Minimum samples required at a leaf node")
    parser.add_argument("--bootstrap", type=bool, default=True,
                        help="Whether to use bootstrap samples")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--save-model", type=str, default="client_model.pkl",
                        help="Path to save the model")
    parser.add_argument("--auto-tune", action="store_true",
                        help="Automatically tune hyperparameters")
    
    args = parser.parse_args()
    
    print("===== Federated Learning Client (Computer B) =====")
    print(f"Dataset: {args.data}")
    print(f"Server: {args.server}:{args.port}")
    
    # Train local model
    model, X_test, y_test, metrics = train_local_model(
        args.data,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        bootstrap=args.bootstrap,
        random_state=args.random_state,
        auto_tune=args.auto_tune
    )
    
    # Save model locally
    save_model(model, args.save_model)
    
    # Send model to server
    print("\n===== Sending model to server =====")
    success = send_model(model, host=args.server, port=args.port)
    
    if success:
        print("Model successfully sent to server")
    else:
        print("Failed to send model to server")
    
    print("\n===== Client process completed =====")

if __name__ == "__main__":
    main() 