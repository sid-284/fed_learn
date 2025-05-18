import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

def load_and_preprocess_data(data_path, test_size=0.2, random_state=42):
    """
    Load and preprocess the housing dataset.
    
    Args:
        data_path: Path to the CSV file
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test: Split datasets
    """
    # Check if the file is the original dataset or a split part
    if 'housing_part' in data_path:
        # Split data files have headers
        data = pd.read_csv(data_path, sep=' ')
    else:
        # The original Boston Housing dataset doesn't have headers, so we'll add them
        column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
                      'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
        
        # Load the data
        data = pd.read_csv(data_path, header=None, delim_whitespace=True, names=column_names)
    
    # Split features and target
    X = data.drop('MEDV', axis=1)
    y = data['MEDV']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

def split_data_for_federated_learning(data_path, n_clients=2, test_size=0.2, random_state=42):
    """
    Split the dataset into multiple parts for federated learning.
    
    Args:
        data_path: Path to the CSV file
        n_clients: Number of clients (parts to split the data)
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        client_data: Dictionary with client data splits
        global_test_data: Test data for global evaluation
    """
    # Load the full dataset
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
                   'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    data = pd.read_csv(data_path, header=None, delim_whitespace=True, names=column_names)
    
    # Split features and target
    X = data.drop('MEDV', axis=1)
    y = data['MEDV']
    
    # First split into global train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Further split the training data for each client
    client_data = {}
    
    # Determine indices for each client
    train_indices = np.arange(len(X_train))
    np.random.seed(random_state)
    np.random.shuffle(train_indices)
    
    client_indices = np.array_split(train_indices, n_clients)
    
    for i in range(n_clients):
        indices = client_indices[i]
        client_X_train = X_train.iloc[indices]
        client_y_train = y_train.iloc[indices]
        
        # Each client gets their own train/test split
        client_X_train, client_X_test, client_y_train, client_y_test = train_test_split(
            client_X_train, client_y_train, test_size=test_size, random_state=random_state
        )
        
        client_data[f'client_{i+1}'] = {
            'X_train': client_X_train,
            'X_test': client_X_test,
            'y_train': client_y_train,
            'y_test': client_y_test
        }
    
    global_test_data = {
        'X_test': X_test,
        'y_test': y_test
    }
    
    return client_data, global_test_data

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a model using MSE and R^2 score.
    
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
    r2 = r2_score(y_test, y_pred)
    
    print(f"{model_name} Evaluation:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}")
    print(f"Root Mean Squared Error: {np.sqrt(mse):.4f}")
    
    return {
        'mse': mse,
        'r2': r2,
        'rmse': np.sqrt(mse)
    }

def save_model(model, filepath):
    """Save a model to disk using pickle"""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """Load a model from disk using pickle"""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filepath}")
    return model 