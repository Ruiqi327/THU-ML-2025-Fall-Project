import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler 


def plot_checkpoint_accuracy(history, save_path):
    """
    绘制并保存在每个检查点的测试准确率曲线。
    Args:
        history (list): 一个包含 (epoch, accuracy) 元组的列表。
        save_path (str): 图片保存的完整路径。
    """
    if not history:
        print("没有检查点历史记录可供绘制。")
        return

    # 从历史记录中分离出epoch和accuracy
    epochs = [item[0] for item in history]
    accuracies = [item[1] for item in history]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accuracies, marker='o', linestyle='-', color='royalblue', label='Test Accuracy')
    
    # 在每个点上标注准确率数值
    for i, txt in enumerate(accuracies):
        plt.annotate(f"{txt:.4f}", (epochs[i], accuracies[i]), textcoords="offset points", xytext=(0,5), ha='center')

    plt.title('Test Accuracy at Checkpoints', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(epochs)  # 确保X轴只显示有记录的epoch
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    
    try:
        plt.savefig(save_path)
        print(f"检查点准确率曲线图已保存至: {save_path}")
    except Exception as e:
        print(f"保存曲线图失败: {e}")
    finally:
        plt.close()

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
    df = pd.read_csv(data_path, header=None)
    y = df.iloc[:, 0].replace(0, -1).to_numpy()
    X = df.iloc[:, 1:-1].to_numpy()

    X_train= X[0:train_size]
    y_train= y[0:train_size]
    X_val= X[train_size:train_size+val_size]
    y_val= y[train_size:train_size+val_size]
    X_test= X[495000:]
    y_test= y[495000:]

   #normalization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test, y_train, y_val, y_test

def MySVM(X_train, y_train, epochs=100, lr=1e-5, penalty=1, batch_size=64, X_test=None, y_test=None, results_dir=None):
    """ Args:
            X_train: feature set for training
            y_train: labels for training
            epochs: number of epochs for training
            lr: learning rate
            penalty: regularization penalty
            batch_size: size of the mini-batch for SGD
        Returns:
            model: trained SVM model
    """
    # initialize w，b
    num_samples, num_features = X_train.shape
    w = np.zeros(num_features)
    b = 0
    loss= np.mean(penalty* np.maximum(0, 1 - y_train * (np.dot(X_train, w) + b))) + 0.5 * np.dot(w, w)
    loss_history = [loss]
 
    iter_range = tqdm(range(epochs), desc="Training", ncols=80)
    test_acc_history = []
    for i in iter_range:
        # 在每个 epoch 开始时随机打乱数据
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        # Mini-batch SGD
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            X_batch = X_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]
            margins = y_batch * (np.dot(X_batch, w) + b)
            misclassified_mask = margins < 1
            
            grad_w_reg = w
            grad_w_hinge = -penalty * np.dot((X_batch[misclassified_mask]).T, y_batch[misclassified_mask]) / len(y_batch)
            grad_b_hinge =  penalty * np.sum(y_batch[misclassified_mask]) / len(y_batch)

            grad_w = grad_w_reg + grad_w_hinge
            grad_b = grad_b_hinge

            w -= lr * grad_w
            b -= lr * grad_b

        loss= np.mean(penalty* np.maximum(0, 1 - y_train * (np.dot(X_train, w) + b))) + 0.5 * np.dot(w, w)
        loss_history.append(loss)

        if (i+1) % 10 == 0:
            test_acc = 0
            # 1. 在测试集上评估
            if X_test is not None and y_test is not None:
                test_acc = MySVM_eval(X_test, y_test, w, b)
                test_acc_history.append((i + 1, test_acc))
                iter_range.set_postfix(loss=f"{loss:.4f}", test_acc=f"{test_acc:.4f}")
            else:
                iter_range.set_postfix(loss=f"{loss:.4f}")

            # 2. 将检查点参数追加写入到同一个txt文件 
            if results_dir:
                with open(results_dir, 'a') as f:
                    f.write(f"\n--- Checkpoint @ Epoch {i+1} ---\n")
                    f.write(f"Test Accuracy: {test_acc:.4f}\n")
                    # 将w和b转换为字符串写入，避免过长可以只写一部分或摘要
                    f.write(f"w: {np.array2string(w, max_line_width=200, threshold=100)}\n")
                    f.write(f"b: {b}\n")
        else:
            iter_range.set_postfix(loss=f"{loss:.4f}")


    return w, b, loss_history, test_acc_history

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
    parser.add_argument("--data_dir", type=str, default="/data1/zhouruiqi/ml_hw1/SVM-HIGGS/dataset/HIGGS_pca.csv", help="Directory containing dataset files")
    parser.add_argument('--lr', type=float, default=1e-5, help='学习率 (learning rate)')
    parser.add_argument('--epochs', type=int, default=1000, help='训练轮次 (epochs)')
    parser.add_argument('--penalty', type=float, default=1.0, help='SVM的惩罚系数 (penalty parameter)')
    parser.add_argument('--train_proportion', type=int, default=2000, help='训练集占比 (training set proportion)')
    parser.add_argument('--val_proportion', type=int, default=100, help='验证集占比 (validation set proportion)')
    parser.add_argument('--test_proportion', type=int, default=100, help='测试集占比 (test set proportion)')
    parser.add_argument('--results_dir', type=str, default="/data1/zhouruiqi/ml_hw1/SVM-HIGGS/results", help='结果保存目录 (results directory)')
    parser.add_argument('--batch_size', type=int, default=64, help='小批量梯度下降的批次大小 (batch size)')

    args = parser.parse_args()
    data_dir=args.data_dir
    lr=args.lr
    epochs=args.epochs
    penalty=args.penalty
    train_proportion=args.train_proportion
    val_proportion=args.val_proportion
    test_proportion=args.test_proportion
    results_dir=args.results_dir
    batch_size=args.batch_size

    data_path = data_dir
    file=os.path.basename(data_path)
    train_size = train_proportion
    val_size = val_proportion
    test_size = test_proportion

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_filename = f"results_{file}.txt"
    results_filepath = os.path.join(results_dir, results_filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(results_filepath, 'w') as f:
        f.write(f"========================================\n")
        f.write(f"Results for: {file}\n{timestamp}\n")
        f.write(f"========================================\n\n")
        f.write("--- Hyperparameters ---\n")
        f.write(f"Learning Rate: {lr}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Penalty (C): {penalty}\n")
        f.write(f"Batch Size: {batch_size}\n\n")

    X_train, X_val, X_test, y_train, y_val, y_test = load_data(data_path, train_size, val_size, test_size)

    # 2. 训练模型，此时MySVM会向已创建的文件中追加检查点
    w, b, loss_history, test_acc_history = MySVM(X_train=X_train, y_train=y_train, epochs=epochs, lr=lr, penalty=penalty, batch_size=batch_size, X_test=X_test, y_test=y_test, results_dir=results_filepath)

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
    with open(results_filepath, 'a') as f:
        f.write("\n--- Final Trained Model Parameters ---\n")
        f.write(f"Weights (w): {w}\n")
        f.write(f"Bias (b): {b}\n\n")

        f.write("--- Final Evaluation Metrics ---\n")
        f.write(f"Training Accuracy: {train_accuracy:.4f}\n")
        f.write(f"Validation Accuracy: {val_accuracy:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")

    plot_path = os.path.join(results_dir, "checkpoint_accuracy.png")
    plot_checkpoint_accuracy(test_acc_history, plot_path)

    # 绘制损失曲线
    epochs_recorded = range(0, epochs, 1)
    plt.figure(figsize=(10, 5))

    epochs_recorded = range(0, args.epochs + 1, 1)
    if len(epochs_recorded) > len(loss_history):
        epochs_recorded = epochs_recorded[:len(loss_history)]
    elif len(loss_history) > len(epochs_recorded):
        loss_history = loss_history[:len(epochs_recorded)]

    plt.figure()
    plt.plot(epochs_recorded, loss_history)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"SVM Training Loss Curve - {file}")
    plt.grid(True)
    plt.savefig(f"{results_dir}/loss_curve_{file}.png")
    plt.close() # 关闭图形，释放内存

if __name__ == "__main__":
    main()