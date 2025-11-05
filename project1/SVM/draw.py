import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

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

def draw_decision_boundary(X, y, w, b, title, save_path):
    x_min, x_max = X[:, 0].min() - 0.05, X[:, 0].max() + 0.05
    y_min, y_max = X[:, 1].min() - 0.05, X[:, 1].max() + 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    Z = np.sign(np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b)
    Z = Z.reshape(xx.shape) 

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    
    # --- 修正后的代码：能够处理垂直线和水平线 ---
    
    # 创建一组x和y值用于绘制直线
    plot_x = np.linspace(xx.min(), xx.max(), 100)
    
    # 使用一个很小的阈值来判断是否为0
    epsilon = 1e-6

    # 检查w[1]是否接近于0 (垂直线)
    if abs(w[1]) < epsilon:
        if abs(w[0]) > epsilon:
            # 方程为 w[0]*x + b = 0  =>  x = -b / w[0]
            boundary_x = -b / w[0]
            plt.axvline(x=boundary_x, color='black', linestyle='-', linewidth=2, label='Decision Boundary')
            # 间隔线
            plt.axvline(x=(1 - b) / w[0], color='black', linestyle='--', linewidth=1, label='Margins')
            plt.axvline(x=(-1 - b) / w[0], color='black', linestyle='--', linewidth=1)
    # 检查w[0]是否接近于0 (水平线)
    elif abs(w[0]) < epsilon:
        if abs(w[1]) > epsilon:
            # 方程为 w[1]*y + b = 0  =>  y = -b / w[1]
            boundary_y = -b / w[1]
            plt.axhline(y=boundary_y, color='black', linestyle='-', linewidth=2, label='Decision Boundary')
            # 间隔线
            plt.axhline(y=(1 - b) / w[1], color='black', linestyle='--', linewidth=1, label='Margins')
            plt.axhline(y=(-1 - b) / w[1], color='black', linestyle='--', linewidth=1)
    # 一般情况 (斜线)
    else:
        # 方程为 y = (-w[0]*x - b) / w[1]
        plot_y = (-w[0] * plot_x - b) / w[1]
        plt.plot(plot_x, plot_y, color='black', linestyle='-', linewidth=2, label='Decision Boundary')
        
        # 方程为 y = (-w[0]*x - b + 1) / w[1]
        plot_y_margin_pos = (-w[0] * plot_x - b + 1) / w[1]
        plt.plot(plot_x, plot_y_margin_pos, color='black', linestyle='--', linewidth=1, label='Margins')
        
        # 方程为 y = (-w[0]*x - b - 1) / w[1]
        plot_y_margin_neg = (-w[0] * plot_x - b - 1) / w[1]
        plt.plot(plot_x, plot_y_margin_neg, color='black', linestyle='--', linewidth=1)


    # ------------------------------------

    plt.xlabel('Normalized learning time')
    plt.ylabel('Normalized midterm grade')
    plt.title(title)
    plt.legend()
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train SVM model on dataset")
    parser.add_argument("--data_dir", type=str, default="/data1/zhouruiqi/ml_hw1/SVM/draw_decision/dataset", help="Directory containing dataset files")
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


        print("Training accuracy:", train_accuracy)
        print("Validation accuracy:", val_accuracy)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        decision_boundary_path = os.path.join(results_dir, f"svm_decision_boundary_{file}.png")
        title = f"SVM Decision Boundary on {file}"
        draw_decision_boundary(X_train, y_train, w, b, title, decision_boundary_path)

if __name__ == "__main__":
    main()