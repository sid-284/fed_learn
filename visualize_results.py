#!/usr/bin/env python3
import os
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score

from utils import load_and_preprocess_data, load_model

def evaluate_model_detailed(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a model using multiple metrics.
    
    Args:
        model: Trained model with predict method
        X_test: Test features
        y_test: True target values
        model_name: Name of the model for printing
        
    Returns:
        Dictionary with evaluation metrics
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    
    print(f"{model_name} Evaluation:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R^2 Score: {r2:.4f}")
    print(f"Explained Variance Score: {evs:.4f}")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'evs': evs
    }

def visualize_feature_importance(model, feature_names, title="Feature Importance"):
    """
    Visualize feature importance.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: Names of features
        title: Title for the plot
    """
    # Get feature importances
    importances = model.feature_importances_
    
    # Sort feature importances
    indices = np.argsort(importances)[::-1]
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    plt.title(title)
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.xlim([-1, min(10, len(importances))])
    plt.tight_layout()
    
    return plt.gcf()

def visualize_comparison(server_model_path, client_model_path, federated_model_path, 
                         server_data_path, client_data_path):
    """
    Visualize the comparison between local and federated models.
    
    Args:
        server_model_path: Path to the server model
        client_model_path: Path to the client model
        federated_model_path: Path to the federated model
        server_data_path: Path to the server data
        client_data_path: Path to the client data
    """
    # Load models
    server_model = load_model(server_model_path)
    client_model = load_model(client_model_path)
    federated_model = load_model(federated_model_path)
    
    # Load data
    _, server_X_test, _, server_y_test = load_and_preprocess_data(server_data_path)
    _, client_X_test, _, client_y_test = load_and_preprocess_data(client_data_path)
    
    # Combine test data for global evaluation
    global_X_test = pd.concat([server_X_test, client_X_test])
    global_y_test = pd.concat([server_y_test, client_y_test])
    
    # Make predictions
    server_pred_server = server_model.predict(server_X_test)
    server_pred_client = server_model.predict(client_X_test)
    server_pred_global = server_model.predict(global_X_test)
    
    client_pred_server = client_model.predict(server_X_test)
    client_pred_client = client_model.predict(client_X_test)
    client_pred_global = client_model.predict(global_X_test)
    
    federated_pred_server = federated_model.predict(server_X_test)
    federated_pred_client = federated_model.predict(client_X_test)
    federated_pred_global = federated_model.predict(global_X_test)
    
    # Calculate detailed metrics
    print("\n===== Server Model Evaluation =====")
    server_metrics_server = evaluate_model_detailed(server_model, server_X_test, server_y_test, "Server Model on Server Data")
    server_metrics_client = evaluate_model_detailed(server_model, client_X_test, client_y_test, "Server Model on Client Data")
    server_metrics_global = evaluate_model_detailed(server_model, global_X_test, global_y_test, "Server Model on Global Data")
    
    print("\n===== Client Model Evaluation =====")
    client_metrics_server = evaluate_model_detailed(client_model, server_X_test, server_y_test, "Client Model on Server Data")
    client_metrics_client = evaluate_model_detailed(client_model, client_X_test, client_y_test, "Client Model on Client Data")
    client_metrics_global = evaluate_model_detailed(client_model, global_X_test, global_y_test, "Client Model on Global Data")
    
    print("\n===== Federated Model Evaluation =====")
    federated_metrics_server = evaluate_model_detailed(federated_model, server_X_test, server_y_test, "Federated Model on Server Data")
    federated_metrics_client = evaluate_model_detailed(federated_model, client_X_test, client_y_test, "Federated Model on Client Data")
    federated_metrics_global = evaluate_model_detailed(federated_model, global_X_test, global_y_test, "Federated Model on Global Data")
    
    # Print summary table
    print("\n===== Model Performance Summary (MSE) =====")
    print(f"{'Model':<20} {'Server Data':<15} {'Client Data':<15} {'Global Data':<15}")
    print("-" * 65)
    print(f"{'Server Model':<20} {server_metrics_server['mse']:<15.4f} {server_metrics_client['mse']:<15.4f} {server_metrics_global['mse']:<15.4f}")
    print(f"{'Client Model':<20} {client_metrics_server['mse']:<15.4f} {client_metrics_client['mse']:<15.4f} {client_metrics_global['mse']:<15.4f}")
    print(f"{'Federated Model':<20} {federated_metrics_server['mse']:<15.4f} {federated_metrics_client['mse']:<15.4f} {federated_metrics_global['mse']:<15.4f}")
    
    print("\n===== Model Performance Summary (R^2) =====")
    print(f"{'Model':<20} {'Server Data':<15} {'Client Data':<15} {'Global Data':<15}")
    print("-" * 65)
    print(f"{'Server Model':<20} {server_metrics_server['r2']:<15.4f} {server_metrics_client['r2']:<15.4f} {server_metrics_global['r2']:<15.4f}")
    print(f"{'Client Model':<20} {client_metrics_server['r2']:<15.4f} {client_metrics_client['r2']:<15.4f} {client_metrics_global['r2']:<15.4f}")
    print(f"{'Federated Model':<20} {federated_metrics_server['r2']:<15.4f} {federated_metrics_client['r2']:<15.4f} {federated_metrics_global['r2']:<15.4f}")
    
    # Create bar chart for MSE
    models = ['Server Model', 'Client Model', 'Federated Model']
    server_mse = [server_metrics_server['mse'], client_metrics_server['mse'], federated_metrics_server['mse']]
    client_mse = [server_metrics_client['mse'], client_metrics_client['mse'], federated_metrics_client['mse']]
    global_mse = [server_metrics_global['mse'], client_metrics_global['mse'], federated_metrics_global['mse']]
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width, server_mse, width, label='Server Data')
    rects2 = ax.bar(x, client_mse, width, label='Client Data')
    rects3 = ax.bar(x + width, global_mse, width, label='Global Data')
    
    ax.set_ylabel('Mean Squared Error')
    ax.set_title('Model Performance Comparison (MSE)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    # Add value labels
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    add_labels(rects1)
    add_labels(rects2)
    add_labels(rects3)
    
    fig.tight_layout()
    
    # Save the figure
    plt.savefig('model_comparison_mse.png')
    print("\nVisualization saved as 'model_comparison_mse.png'")
    
    # Create bar chart for R^2
    server_r2 = [server_metrics_server['r2'], client_metrics_server['r2'], federated_metrics_server['r2']]
    client_r2 = [server_metrics_client['r2'], client_metrics_client['r2'], federated_metrics_client['r2']]
    global_r2 = [server_metrics_global['r2'], client_metrics_global['r2'], federated_metrics_global['r2']]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width, server_r2, width, label='Server Data')
    rects2 = ax.bar(x, client_r2, width, label='Client Data')
    rects3 = ax.bar(x + width, global_r2, width, label='Global Data')
    
    ax.set_ylabel('R^2 Score')
    ax.set_title('Model Performance Comparison (R^2)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    add_labels(rects1)
    add_labels(rects2)
    add_labels(rects3)
    
    fig.tight_layout()
    
    # Save the figure
    plt.savefig('model_comparison_r2.png')
    print("Visualization saved as 'model_comparison_r2.png'")
    
    # Create a scatter plot of actual vs. predicted values for the global dataset
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Server model
    axes[0].scatter(global_y_test, server_pred_global, alpha=0.5)
    axes[0].plot([global_y_test.min(), global_y_test.max()], 
                [global_y_test.min(), global_y_test.max()], 
                'k--', lw=2)
    axes[0].set_xlabel('Actual')
    axes[0].set_ylabel('Predicted')
    axes[0].set_title(f'Server Model\nMSE: {server_metrics_global["mse"]:.2f}, R²: {server_metrics_global["r2"]:.2f}')
    
    # Client model
    axes[1].scatter(global_y_test, client_pred_global, alpha=0.5)
    axes[1].plot([global_y_test.min(), global_y_test.max()], 
                [global_y_test.min(), global_y_test.max()], 
                'k--', lw=2)
    axes[1].set_xlabel('Actual')
    axes[1].set_ylabel('Predicted')
    axes[1].set_title(f'Client Model\nMSE: {client_metrics_global["mse"]:.2f}, R²: {client_metrics_global["r2"]:.2f}')
    
    # Federated model
    axes[2].scatter(global_y_test, federated_pred_global, alpha=0.5)
    axes[2].plot([global_y_test.min(), global_y_test.max()], 
                [global_y_test.min(), global_y_test.max()], 
                'k--', lw=2)
    axes[2].set_xlabel('Actual')
    axes[2].set_ylabel('Predicted')
    axes[2].set_title(f'Federated Model\nMSE: {federated_metrics_global["mse"]:.2f}, R²: {federated_metrics_global["r2"]:.2f}')
    
    fig.suptitle('Actual vs. Predicted Values (Global Dataset)', fontsize=16)
    fig.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    # Save the figure
    plt.savefig('prediction_comparison.png')
    print("Visualization saved as 'prediction_comparison.png'")
    
    # Visualize feature importance
    feature_names = global_X_test.columns
    
    # Server model feature importance
    server_fig = visualize_feature_importance(server_model, feature_names, "Server Model Feature Importance")
    server_fig.savefig('server_feature_importance.png')
    print("Visualization saved as 'server_feature_importance.png'")
    
    # Client model feature importance
    client_fig = visualize_feature_importance(client_model, feature_names, "Client Model Feature Importance")
    client_fig.savefig('client_feature_importance.png')
    print("Visualization saved as 'client_feature_importance.png'")
    
    # Federated model feature importance
    federated_fig = visualize_feature_importance(federated_model, feature_names, "Federated Model Feature Importance")
    federated_fig.savefig('federated_feature_importance.png')
    print("Visualization saved as 'federated_feature_importance.png'")

def main():
    parser = argparse.ArgumentParser(description="Visualize federated learning results")
    parser.add_argument("--server-model", type=str, default="server_model.pkl",
                        help="Path to the server model")
    parser.add_argument("--client-model", type=str, default="client_model.pkl",
                        help="Path to the client model")
    parser.add_argument("--federated-model", type=str, default="federated_model.pkl",
                        help="Path to the federated model")
    parser.add_argument("--server-data", type=str, default="/Users/sidharthmohan/sem6/dist_sys/fed/split_data/housing_part_1.csv",
                        help="Path to the server data")
    parser.add_argument("--client-data", type=str, default="/Users/sidharthmohan/sem6/dist_sys/fed/split_data/housing_part_2.csv",
                        help="Path to the client data")
    
    args = parser.parse_args()
    
    print("===== Visualizing Federated Learning Results =====")
    visualize_comparison(
        args.server_model,
        args.client_model,
        args.federated_model,
        args.server_data,
        args.client_data
    )
    print("\nVisualization completed!")

if __name__ == "__main__":
    main() 