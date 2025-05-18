#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def split_dataset(data_path, output_dir, n_parts=2, random_state=42):
    """
    Split a dataset into multiple parts for federated learning simulation.
    
    Args:
        data_path: Path to the dataset
        output_dir: Directory to save the split datasets
        n_parts: Number of parts to split the dataset
        random_state: Random seed for reproducibility
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # The Boston Housing dataset doesn't have headers, so we'll add them
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
                   'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    
    # Load the data
    data = pd.read_csv(data_path, header=None, delim_whitespace=True, names=column_names)
    
    # Shuffle the data
    data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Split the data into n_parts
    data_parts = np.array_split(data, n_parts)
    
    # Save each part to a separate file with whitespace delimiter
    for i, part in enumerate(data_parts):
        output_path = os.path.join(output_dir, f"housing_part_{i+1}.csv")
        # Save with whitespace delimiter instead of comma
        part.to_csv(output_path, index=False, sep=' ')
        print(f"Part {i+1} saved to {output_path} with {len(part)} rows")

def main():
    parser = argparse.ArgumentParser(description="Split dataset for federated learning")
    parser.add_argument("--data", type=str, default="/Users/sidharthmohan/sem6/dist_sys/fed/housing.csv",
                        help="Path to the dataset")
    parser.add_argument("--output", type=str, default="/Users/sidharthmohan/sem6/dist_sys/fed/split_data",
                        help="Output directory")
    parser.add_argument("--n-parts", type=int, default=2,
                        help="Number of parts to split the dataset")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    print(f"Splitting dataset {args.data} into {args.n_parts} parts...")
    split_dataset(args.data, args.output, args.n_parts, args.random_state)
    print("Done!")

if __name__ == "__main__":
    main() 