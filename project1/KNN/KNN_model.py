import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

def load_data(data_path, train_size, val_size, test_size):
    """ Args:
            data_path: path to the dataset file (CSV format)
            train_size: proportion of the dataset to include in the train split
            val_size: proportion of the dataset to include in the validation split
            test_size: proportion of the dataset to include in the test split
        Returns:
            X_train, X_val, X_test: feature sets for training, validation, and testing
            y_train, y_val, y_test: labels for training, validation, and testing
    """
    random_state = 42  # fix the ran  for reproducibility
    df = pd.read_csv(data_path)
    y = df.iloc[:, -1].to_numpy()
    X = df.iloc[:, :-1].to_numpy()

    X_train, X_rem, y_train, y_rem = train_test_split(
        X, y, train_size=train_size, random_state=random_state, stratify=y
    )
    test_prop = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_rem, y_rem, test_size=test_prop, random_state=random_state, stratify=y_rem
    )

    #normalize the data
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_train = (X_train - X_min) / (X_max - X_min)
    X_val = (X_val - X_min) / (X_max - X_min)
    X_test = (X_test - X_min) / (X_max - X_min)
   
    return X_train, X_val, X_test, y_train, y_val, y_test

def MyKNN(X_train, y_train, X_pred, k=5, distance_metric='L2'):
    """ Args:
            X_train: feature set for training
            y_train: labels for training
            X_test: feature set for testing
            k: number of nearest neighbors to consider
        Returns:
            y_pred: predicted labels for the test set
    """
    num_test_samples = X_pred.shape[0]
    y_pred = np.zeros(num_test_samples)

    for i in range(num_test_samples):
        if distance_metric == 'L1':
            distances = np.linalg.norm(X_train - X_pred[i], ord=1, axis=1)
        else:
            distances = np.linalg.norm(X_train - X_pred[i], axis=1)
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = y_train[k_indices]
        unique, counts = np.unique(k_nearest_labels, return_counts=True)
        y_pred[i] = unique[np.argmax(counts)]
    return y_pred
    
def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def main():
    parser = argparse.ArgumentParser(description="KNN Classifier")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset file (CSV format)")
    parser.add_argument("--train_size", type=float, default=0.7, help="Proportion of the dataset to include in the train split")
    parser.add_argument("--val_size", type=float, default=0.15, help="Proportion of the dataset to include in the validation split")
    parser.add_argument("--test_size", type=float, default=0.15, help="Proportion of the dataset to include in the test split")
    parser.add_argument("--k", type=int, default=5, help="Number of nearest neighbors to consider")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save the results")
    parser.add_argument("--distance_metric", type=str, default="L2", choices=['L1', 'L2'], help="Distance metric to use (L1 or L2)")
    args = parser.parse_args()

    for file in os.listdir(args.data_path):
        if file.endswith('.csv'):
            data_file = os.path.join(args.data_path, file)
            print(f"Processing file: {data_file}")

            X_train, X_val, X_test, y_train, y_val, y_test = load_data(
                data_file, args.train_size, args.val_size, args.test_size
            )

            print("Training KNN classifier...")
            y_val_pred = MyKNN(X_train, y_train, X_val, k=args.k, distance_metric=args.distance_metric)
            val_accuracy = accuracy(y_val, y_val_pred)
            print(f"Validation Accuracy: {val_accuracy:.4f}")

            y_test_pred = MyKNN(X_train, y_train, X_test, k=args.k, distance_metric=args.distance_metric)
            test_accuracy = accuracy(y_test, y_test_pred)
            print(f"Test Accuracy: {test_accuracy:.4f}")

        if not os.path.exists(args.results_dir):
            os.makedirs(args.results_dir)

        results_filename = f"results_{file}.txt"
        results_filepath = os.path.join(args.results_dir, results_filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(results_filepath, 'a') as f:
            f.write(f"========================================\n")
            f.write(f"Results for: {file}\n{timestamp}\n")

            f.write("--- Hyperparameters ---\n")
            f.write(f"k: {args.k}\n")
            f.write(f"Distance Metric: {args.distance_metric}\n")
            
            f.write("--- Evaluation Metrics ---\n")
            f.write(f"Validation Accuracy: {val_accuracy:.4f}\n")
            f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
            f.write(f"========================================\n\n")
if __name__ == "__main__":
    main()