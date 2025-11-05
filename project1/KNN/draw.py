import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

def load_data(data_path, train_size=0.1, test_size=0.9):
    """ Args:
            data_path: path to the dataset file (CSV format)
            train_size: proportion of the dataset to include in the train split
            test_size: proportion of the dataset to include in the test split
        Returns:
            X_train, X_test: feature sets for training and testing
            y_train, y_test: labels for training and testing
    """
    random_state = 42  # fix the random state for reproducibility
    df = pd.read_csv(data_path)
    y = df.iloc[:, -1].to_numpy()
    X = df.iloc[:, :-1].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=random_state, stratify=y
    )
    #normalize the data
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_train = (X_train - X_min) / (X_max - X_min)
    X_test = (X_test - X_min) / (X_max - X_min)
    return X_train, X_test, y_train, y_test

def MyKNN(X_train, y_train, X_pred, k=5,  distance_metric='L2'):
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
    """ Args:
            y_true: true labels
            y_pred: predicted labels
        Returns:
            accuracy: accuracy of the predictions
    """
    return np.sum(y_true == y_pred) / len(y_true)

def draw_decision_boundary(X, y, model, k, title, save_path):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model(X, y, grid_points, k)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6, 10))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdYlBu)
    plt.title(title)
    plt.xlabel('Normalized Learning Time')
    plt.ylabel('Normalized Midterm Grade')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.grid()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="KNN Classifier")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset file (CSV format)")
    parser.add_argument("--train_size", type=float, default=0.7, help="Proportion of the dataset to include in the train split")
    parser.add_argument("--test_size", type=float, default=0.15, help="Proportion of the dataset to include in the test split")
    parser.add_argument("--k", type=int, default=5, help="Number of nearest neighbors to consider")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save the results")
    parser.add_argument("--distance_metric", type=str, default="L2", choices=['L1', 'L2'], help="Distance metric to use (L1 or L2)")
    args = parser.parse_args()

    for file in os.listdir(args.data_path):
        if file.endswith('.csv'):
            data_file = os.path.join(args.data_path, file)
            print(f"Processing file: {data_file}")

            X_train, X_test, y_train, y_test = load_data(
                data_file, args.train_size, args.test_size
            )
            print("Training KNN classifier...")
            y_decision = MyKNN(X_train, y_train, X_test, k=args.k)
            test_acc = accuracy(y_test, y_decision)
            print(f"Test Accuracy: {test_acc:.4f}")
            title = f"KNN Decision Boundary (k={args.k} on {file})"
            save_path = os.path.join(args.results_dir, f"knn_decision_boundary_{file}.png")
            draw_decision_boundary(X_train, y_train, MyKNN, args.k, title, save_path)

if __name__ == "__main__":
    main()