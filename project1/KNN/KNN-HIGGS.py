import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import argparse
from datetime import datetime
import time

def load_data(data_path, train_size):
    random_state = 42
    print("Loading and preprocessing data...")
    
    # 假设数据没有表头
    df = pd.read_csv(data_path, header=None)
    
    # 标签 (y) 应该是 0 或 1，特征 (X) 是其余列
    y_train = df.iloc[:train_size+1, 0].to_numpy() # 取前train_size作为训练集
    X_train = df.iloc[:train_size+1, 1:].to_numpy()

    X_test = df.iloc[495000:, 1:].to_numpy()  # 取后5000行作为测试集
    y_test = df.iloc[495000:, 0].to_numpy()


    # 标准化特征
    train_min = X_train.min(axis=0)
    train_max = X_train.max(axis=0)
    X_train = (X_train - train_min) / (train_max - train_min)
    X_test = (X_test - train_min) / (train_max - train_min)

    return X_train, X_test, y_train, y_test


def MyKNN(X_train, y_train, X_pred, k_list=[5],distance_metric='L1'):
    """ Args:
            X_train: feature set for training
            y_train: labels for training
            X_pred: feature set for prediction
            k_list: list of k values for nearest neighbors to consider
        Returns:
            y_preds: a dictionary of predicted labels for the test set for each k. 
                     {k1: y_pred1, k2: y_pred2, ...}
    """
    num_test_samples = X_pred.shape[0]
    # 初始化一个字典来存储不同k值的预测结果
    y_preds = {k: np.zeros(num_test_samples) for k in k_list}
    max_k = max(k_list)

    for i in range(num_test_samples):
        if distance_metric == 'L1':
            distances = np.linalg.norm(X_train - X_pred[i], ord=1, axis=1)
        elif distance_metric == 'L2':
            distances = np.linalg.norm(X_train - X_pred[i], ord=2, axis=1)
        k_indices = np.argsort(distances)[:max_k]
        k_nearest_labels = y_train[k_indices]
        for k in k_list:
            labels_for_k = k_nearest_labels[:k]
            unique, counts = np.unique(labels_for_k, return_counts=True)
            y_preds[k][i] = unique[np.argmax(counts)]
            
    return y_preds
    
def accuracy(y_true, y_preds_dict):
    """ Args:
            y_true: true labels
            y_preds_dict: A dictionary of predictions from MyKNN, {k: y_pred}
        Returns:
            accuracies: A dictionary of accuracies for each k, {k: accuracy}
    """
    accuracies = {}
    for k, y_pred in y_preds_dict.items():
        acc = np.sum(y_true == y_pred) / len(y_true)
        accuracies[k] = acc
    return accuracies

def main():
    parser = argparse.ArgumentParser(description="KNN Classifier")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset file (CSV format)")
    parser.add_argument("--train_size", type=int,  help="Proportion of the dataset to include in the train split")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save the results")
    parser.add_argument("--distance_metric", type=str, default="L1", choices=["L1", "L2"], help="Distance metric to use")
    args = parser.parse_args()

    k_values = [1, 3, 5, 7, 10, 15, 30, 50, 70, 100, 150, 300] # 要测试的k值列表

    for file in os.listdir(args.data_path):
        if file.endswith('.csv'):
            data_file = os.path.join(args.data_path, file)
            print(f"Processing file: {data_file}")

            X_train, X_test, y_train, y_test = load_data(
                data_file, args.train_size)

            print(f"Training KNN classifier for k in {k_values}...")
            start_time = time.perf_counter()
            y_test_preds_dict = MyKNN(X_train, y_train, X_test, k_list=k_values, distance_metric=args.distance_metric)
            end_time = time.perf_counter()
            train_time = end_time - start_time
            
            test_accuracies = accuracy(y_test, y_test_preds_dict)
            
            print(f"Total training time for all k's: {train_time:.4f} seconds")
            for k, acc in test_accuracies.items():
                print(f"k = {k}, Test Accuracy: {acc:.4f}")

            if not os.path.exists(args.results_dir):
                os.makedirs(args.results_dir)

            results_filename = f"results_{file}.txt"
            results_filepath = os.path.join(args.results_dir, results_filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            with open(results_filepath, 'a') as f:
                f.write(f"========================================\n")
                f.write(f"Results for: {file}\n{timestamp}\n")
                f.write(f"Total Train Time for all k's: {train_time:.4f} seconds\n")

                f.write("--- Hyperparameters ---\n")
                f.write(f"Train Size: {args.train_size}\n\n")
                f.write(f"Distance Metric: {args.distance_metric}\n\n")

                f.write("--- Evaluation Metrics ---\n")
                for k, acc in test_accuracies.items():
                    f.write(f"k = {k}, Test Accuracy: {acc:.4f}\n")
                f.write(f"========================================\n\n")

if __name__ == "__main__":
    main()