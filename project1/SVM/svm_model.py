import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

def load_data(data_path, train_size, val_size, test_size):
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
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_train = (X_train - X_min) / (X_max - X_min)
    X_val = (X_val - X_min) / (X_max - X_min)
    X_test = (X_test - X_min) / (X_max - X_min)
   
    return X_train, X_val, X_test, y_train, y_val, y_test

def MySVM(X_train, y_train, iterations=1000, lr=1e-5, penalty=1):
    """ Args:
            feature_num: number of features in the dataset
            X_train: feature set for training
            y_train: labels for training
            iterations: number of iterations for training
            lr: learning rate
            penalty: regularization penalty
        Returns:
            model: trained SVM model
    """
    # initialize w，b
    num_samples, num_features = X_train.shape
    w = np.zeros(num_features)
    b = 0
    loss= np.sum(penalty* np.maximum(0, 1 - y_train * (np.dot(X_train, w) + b))) + 0.5 * np.dot(w, w)
    loss_history = [loss]
    
    # 2. 用tqdm包裹循环，并添加描述
    for i in tqdm(range(iterations), desc="Training SVM"): 
        gradient_w = np.zeros(num_features)
        gradient_b = 0
        for idx, x_i in enumerate(X_train):
            condition = y_train[idx] * (np.dot(w, x_i) + b) >= 1
            if not condition:    
                gradient_w -= penalty * y_train[idx] *x_i
                gradient_b -= penalty * y_train[idx]
        gradient_w += w

        w -= gradient_w*lr 
        b -= gradient_b*lr

        if i % 20 == 0:
            loss= np.sum(penalty* np.maximum(0, 1 - y_train * (np.dot(X_train, w) + b))) + 0.5 * np.dot(w, w)
            loss_history.append(loss)
    return w, b, loss_history

def MySVM_eval(X, y, w, b):
    """ Args:
            X: feature set for prediction
            y: true labels for the feature set
            w: weights of the trained SVM model
            b: bias of the trained SVM model
        Returns:
            predictions: predicted labels for the input features
    """
    linear_output = np.dot(X, w) + b
    predictions = np.sign(linear_output)
    accuracy = np.mean(predictions == y)
    return accuracy

def main():
    parser = argparse.ArgumentParser(description="Train SVM model on dataset")
    parser.add_argument("--data_dir", type=str, default="/data1/zhouruiqi/ml_hw1/SVM/dataset", help="Directory containing dataset files")
    parser.add_argument('--lr', type=float, default=1e-5, help='学习率 (learning rate)')
    parser.add_argument('--iterations', type=int, default=1000, help='训练迭代次数')
    parser.add_argument('--penalty', type=float, default=1.0, help='SVM的惩罚系数 (penalty parameter)')
    parser.add_argument('--train_proportion', type=float, default=0.8, help='训练集占比 (training set proportion)')
    parser.add_argument('--val_proportion', type=float, default=0.1, help='验证集占比 (validation set proportion)')
    parser.add_argument('--test_proportion', type=float, default=0.1, help='测试集占比 (test set proportion)')
    parser.add_argument('--results_dir', type=str, default="/data1/zhouruiqi/ml_hw1/SVM/results", help='结果保存目录 (results directory)')

    args = parser.parse_args()
    data_dir=args.data_dir
    lr=args.lr
    iterations=args.iterations
    penalty=args.penalty
    train_proportion=args.train_proportion
    val_proportion=args.val_proportion
    test_proportion=args.test_proportion
    results_dir=args.results_dir

    for file in os.listdir(data_dir):
        data_path = os.path.join(data_dir, file)
        train_size = train_proportion
        val_size = val_proportion
        test_size = test_proportion

        X_train, X_val, X_test, y_train, y_val, y_test = load_data(data_path, train_size, val_size, test_size)

        # Train the SVM model
        print(f"Dataset: {file}")
        w, b, loss_history = MySVM(X_train=X_train, y_train=y_train, iterations=iterations, lr=lr, penalty=penalty)
        print("Trained SVM model parameters:")
        print("Weights:", w)
        print("Bias:", b)
        # Evaluate the model
        train_accuracy = MySVM_eval(X_train, y_train, w, b)
        val_accuracy = MySVM_eval(X_val, y_val, w, b)
        test_accuracy = MySVM_eval(X_test, y_test, w, b)

        print("Training accuracy:", train_accuracy)
        print("Validation accuracy:", val_accuracy)
        print("Test accuracy:", test_accuracy)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        results_filename = f"results_{file}.txt"
        results_filepath = os.path.join(results_dir, results_filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(results_filepath, 'a') as f:
            f.write(f"========================================\n")
            f.write(f"Results for: {file}\n{timestamp}\n")
            f.write(f"========================================\n\n")

            f.write("--- Hyperparameters ---\n")
            f.write(f"Learning Rate: {lr}\n")
            f.write(f"Iterations: {iterations}\n")
            f.write(f"Penalty (C): {penalty}\n\n")

            f.write("--- Trained Model Parameters ---\n")
            f.write(f"Weights (w): {w}\n")
            f.write(f"Bias (b): {b}\n\n")

            f.write("--- Evaluation Metrics ---\n")
            f.write(f"Training Accuracy: {train_accuracy:.4f}\n")
            f.write(f"Validation Accuracy: {val_accuracy:.4f}\n")
            f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
            f.write(f"========================================\n\n")

        iterations_recorded = range(0, args.iterations+1, 20)
        plt.figure()
        plt.plot(iterations_recorded, loss_history)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title(f"SVM Training Loss Curve - {file}")
        plt.grid(True)
        plt.savefig(f"{results_dir}/loss_curve_{file}.png")
        plt.close() # 关闭图形，释放内存

if __name__ == "__main__":
    main()