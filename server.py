#!/usr/bin/env python3
import os
import argparse
import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from utils import load_and_preprocess_data, evaluate_model, save_model, load_model
from network import receive_model, start_server
from federated_aggregator import FederatedRandomForest

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
    metrics = evaluate_model(model, X_test, y_test, "Local Random Forest (Server)")
    
    # Feature importance analysis
    feature_importances = model.feature_importances_
    feature_names = X_train.columns
    importance_df = {
        'Feature': feature_names,
        'Importance': feature_importances
    }
    
    # Print top 5 features
    sorted_idx = np.argsort(feature_importances)[::-1]
    print("\nTop 5 important features:")
    for i in range(min(5, len(feature_names))):
        idx = sorted_idx[i]
        print(f"{feature_names[idx]}: {feature_importances[idx]:.4f}")
    
    return model, X_test, y_test, metrics

def main():
    parser = argparse.ArgumentParser(description="Federated Learning Server")
    parser.add_argument("--data", type=str, default="/Users/sidharthmohan/sem6/dist_sys/fed/housing.csv",
                        help="Path to the dataset")
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
    parser.add_argument("--save-model", type=str, default="server_model.pkl",
                        help="Path to save the local model")
    parser.add_argument("--save-federated", type=str, default="federated_model.pkl",
                        help="Path to save the federated model")
    parser.add_argument("--num-clients", type=int, default=1,
                        help="Number of clients to wait for")
    parser.add_argument("--auto-tune", action="store_true",
                        help="Automatically tune hyperparameters")
    parser.add_argument("--use-weights", action="store_true",
                        help="Use model performance for weighting")
    
    args = parser.parse_args()
    
    print("===== Federated Learning Server (Computer A) =====")
    print(f"Dataset: {args.data}")
    print(f"Port: {args.port}")
    print(f"Waiting for {args.num_clients} client(s)")
    
    # Train local model
    local_model, X_test, y_test, local_metrics = train_local_model(
        args.data,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        bootstrap=args.bootstrap,
        random_state=args.random_state,
        auto_tune=args.auto_tune
    )
    
    # Save local model
    save_model(local_model, args.save_model)
    
    # Initialize federated aggregator
    federated_model = FederatedRandomForest()
    
    # Add local model with weight based on its performance (inverse MSE)
    local_weight = 1.0
    if args.use_weights:
        local_weight = 1.0 / max(local_metrics['mse'], 0.001)  # Avoid division by zero
    
    federated_model.add_model(local_model, weight=local_weight)
    
    # Start server to receive models from clients
    print("\n===== Starting server to receive client models =====")
    print(f"Waiting for {args.num_clients} client(s) to connect...")
    
    client_models = []
    client_metrics = []
    
    # Option 1: Use the receive_model function for a single client
    if args.num_clients == 1:
        client_model = receive_model(port=args.port)
        if client_model:
            client_models.append(client_model)
            
            # Evaluate client model on server test data
            print("\n===== Evaluating client model on server data =====")
            client_metrics.append(evaluate_model(client_model, X_test, y_test, "Client Model on Server Data"))
            
            # Add client model with weight based on its performance
            client_weight = 1.0
            if args.use_weights and client_metrics:
                client_weight = 1.0 / max(client_metrics[0]['mse'], 0.001)  # Avoid division by zero
            
            federated_model.add_model(client_model, weight=client_weight)
    # Option 2: Use the start_server function for multiple clients
    else:
        received_models = start_server(port=args.port)
        for i, model in enumerate(received_models):
            client_models.append(model)
            
            # Evaluate client model on server test data
            print(f"\n===== Evaluating client {i+1} model on server data =====")
            metrics = evaluate_model(model, X_test, y_test, f"Client {i+1} Model on Server Data")
            client_metrics.append(metrics)
            
            # Add client model with weight based on its performance
            client_weight = 1.0
            if args.use_weights:
                client_weight = 1.0 / max(metrics['mse'], 0.001)  # Avoid division by zero
            
            federated_model.add_model(model, weight=client_weight)
    
    # Create aggregated model
    print("\n===== Creating aggregated federated model =====")
    aggregated_model = federated_model.create_aggregated_model(base_model=local_model)
    
    # Save federated model
    save_model(aggregated_model, args.save_federated)
    
    # Evaluate federated model
    print("\n===== Evaluating federated model =====")
    federated_metrics = evaluate_model(aggregated_model, X_test, y_test, "Federated Random Forest")
    
    # Compare with local model
    print("\n===== Comparison: Local vs. Federated =====")
    local_pred = local_model.predict(X_test)
    local_mse = np.mean((local_pred - y_test) ** 2)
    
    federated_pred = aggregated_model.predict(X_test)
    federated_mse = np.mean((federated_pred - y_test) ** 2)
    
    improvement = (local_mse - federated_mse) / local_mse * 100
    print(f"Local model MSE: {local_mse:.4f}")
    print(f"Federated model MSE: {federated_mse:.4f}")
    print(f"Improvement: {improvement:.2f}%")
    
    # Feature importance analysis
    feature_importances = federated_model.feature_importances
    feature_names = X_test.columns
    
    # Print top 5 features from federated model
    print("\n===== Federated Model Feature Importance =====")
    if feature_importances is not None:
        sorted_idx = np.argsort(feature_importances)[::-1]
        print("Top 5 important features:")
        for i in range(min(5, len(feature_names))):
            idx = sorted_idx[i]
            print(f"{feature_names[idx]}: {feature_importances[idx]:.4f}")
    else:
        # Use the feature importances from the aggregated model directly
        feature_importances = aggregated_model.feature_importances_
        sorted_idx = np.argsort(feature_importances)[::-1]
        print("Top 5 important features:")
        for i in range(min(5, len(feature_names))):
            idx = sorted_idx[i]
            print(f"{feature_names[idx]}: {feature_importances[idx]:.4f}")
    
    print("\n===== Server process completed =====")

if __name__ == "__main__":
    main() 