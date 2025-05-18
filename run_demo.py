#!/usr/bin/env python3
import os
import argparse
import subprocess
import time
import threading
import signal
import sys

def run_command(command, name=None):
    """Run a command in a subprocess and capture its output"""
    print(f"Starting {name}: {' '.join(command)}")
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Print output in real-time
        for line in process.stdout:
            prefix = f"[{name}] " if name else ""
            print(f"{prefix}{line}", end="")
            
        process.wait()
        return process.returncode
    except Exception as e:
        print(f"Error running {name}: {e}")
        return 1

def split_data_thread(args):
    """Split the dataset into parts"""
    cmd = [
        sys.executable, "split_data.py",
        "--data", args.data,
        "--output", args.output_dir,
        "--n-parts", "2",
        "--random-state", str(args.random_state)
    ]
    return run_command(cmd, "Data Splitter")

def server_thread(args):
    """Run the server process"""
    server_data = os.path.join(args.output_dir, "housing_part_1.csv")
    cmd = [
        sys.executable, "server.py",
        "--data", server_data,
        "--port", str(args.port),
        "--n-estimators", str(args.n_estimators),
        "--max-depth", str(args.max_depth),
        "--min-samples-split", str(args.min_samples_split),
        "--min-samples-leaf", str(args.min_samples_leaf),
        "--bootstrap", str(args.bootstrap),
        "--random-state", str(args.random_state),
        "--num-clients", "1"
    ]
    
    # Add auto-tune flag if specified
    if args.auto_tune:
        cmd.append("--auto-tune")
    
    # Add use-weights flag if specified
    if args.use_weights:
        cmd.append("--use-weights")
    
    return run_command(cmd, "Server (Computer A)")

def client_thread(args):
    """Run the client process"""
    # Wait a bit for the server to start
    time.sleep(5)
    
    client_data = os.path.join(args.output_dir, "housing_part_2.csv")
    cmd = [
        sys.executable, "client.py",
        "--data", client_data,
        "--server", "localhost",
        "--port", str(args.port),
        "--n-estimators", str(args.n_estimators),
        "--max-depth", str(args.max_depth),
        "--min-samples-split", str(args.min_samples_split),
        "--min-samples-leaf", str(args.min_samples_leaf),
        "--bootstrap", str(args.bootstrap),
        "--random-state", str(args.random_state)
    ]
    
    # Add auto-tune flag if specified
    if args.auto_tune:
        cmd.append("--auto-tune")
    
    return run_command(cmd, "Client (Computer B)")

def main():
    parser = argparse.ArgumentParser(description="Run Federated Learning Demo")
    parser.add_argument("--data", type=str, default="/Users/sidharthmohan/sem6/dist_sys/fed/housing.csv",
                        help="Path to the dataset")
    parser.add_argument("--output-dir", type=str, default="/Users/sidharthmohan/sem6/dist_sys/fed/split_data",
                        help="Directory to store split data")
    parser.add_argument("--port", type=int, default=9999,
                        help="Port for communication")
    parser.add_argument("--n-estimators", type=int, default=50,
                        help="Number of trees in the forest")
    parser.add_argument("--max-depth", type=int, default=10,
                        help="Maximum depth of the trees")
    parser.add_argument("--min-samples-split", type=int, default=2,
                        help="Minimum samples required to split a node")
    parser.add_argument("--min-samples-leaf", type=int, default=1,
                        help="Minimum samples required at a leaf node")
    parser.add_argument("--bootstrap", type=bool, default=True,
                        help="Whether to use bootstrap samples")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--auto-tune", action="store_true",
                        help="Automatically tune hyperparameters")
    parser.add_argument("--use-weights", action="store_true",
                        help="Use model performance for weighting")
    
    args = parser.parse_args()
    
    print("===== Federated Learning Demo =====")
    print(f"Dataset: {args.data}")
    print(f"Output directory: {args.output_dir}")
    print(f"Port: {args.port}")
    print(f"Random Forest parameters: n_estimators={args.n_estimators}, max_depth={args.max_depth}, " +
          f"min_samples_split={args.min_samples_split}, min_samples_leaf={args.min_samples_leaf}, " +
          f"bootstrap={args.bootstrap}")
    print(f"Auto-tune: {args.auto_tune}")
    print(f"Use weights: {args.use_weights}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Split the dataset
    print("\n===== Step 1: Splitting the dataset =====")
    split_result = split_data_thread(args)
    if split_result != 0:
        print("Error splitting the dataset. Exiting.")
        return
    
    # Start server and client in separate threads
    print("\n===== Step 2: Starting server and client =====")
    server_thread_obj = threading.Thread(target=server_thread, args=(args,))
    client_thread_obj = threading.Thread(target=client_thread, args=(args,))
    
    try:
        server_thread_obj.start()
        client_thread_obj.start()
        
        # Wait for both threads to complete
        server_thread_obj.join()
        client_thread_obj.join()
        
        print("\n===== Demo completed =====")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted. Exiting...")
        sys.exit(0)

if __name__ == "__main__":
    main() 