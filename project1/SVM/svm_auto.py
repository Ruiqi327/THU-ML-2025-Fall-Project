import os
import argparse
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def load_data(data_path, train_size):
    df = pd.read_csv(data_path)
    y_train = df.iloc[0:train_size, 0].replace(0, -1).to_numpy()
    X_train = df.iloc[0:train_size, 1:].to_numpy()

    X_val = df.iloc[490000:495000, 1:].to_numpy()
    y_val = df.iloc[490000:495000, 0].replace(0, -1).to_numpy()

    X_test = df.iloc[495000:, 1:].to_numpy()
    y_test = df.iloc[495000:, 0].replace(0, -1).to_numpy()
    return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    parser = argparse.ArgumentParser(description='使用 scikit-learn 的 SVM 在 HIGGS 数据集上进行训练和测试。')
    parser.add_argument('--data_path', type=str, default='./HIGGS.csv', help='HIGGS 数据集文件的路径。')
    parser.add_argument('--C', type=float, default=1.0, help='SVM 的正则化参数 C。')
    parser.add_argument('--gamma', type=float, default='scale', help='RBF 核的系数 gamma ("scale" 或 "auto" 或一个浮点数)。')
    parser.add_argument('--train_size', type=int, help='训练集所占的比例（由于数据集很大，建议使用较小比例进行快速测试）。')

    args = parser.parse_args()

    # 创建结果目录和文件 
    results_dir = "auto_svm_results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"results_sklearn_svm_{timestamp}.txt"
    results_filepath = os.path.join(results_dir, results_filename)

    # 加载数据
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(
        args.data_path, train_size=args.train_size)

    # 初始化SVM模型
    # verbose=True 会在训练时打印进度信息
    print("\n正在初始化 scikit-learn SVM 模型...")
    model = SVC(C=args.C, kernel='rbf', gamma=args.gamma, verbose=True, random_state=42)
    
    # 训练模型
    print("开始训练模型... (这可能需要很长时间)")
    model.fit(X_train, y_train)
    print("模型训练完成。")

    # 评估模型
    print("正在评估模型...")
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    val_accuracy = accuracy_score(y_val, y_pred_val)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    # 打印并保存结果
    print("\n--- 评估结果 ---")
    print(f"训练集准确率: {train_accuracy:.4f}")
    print(f"验证集准确率: {val_accuracy:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")

    with open(results_filepath, 'w') as f:
        f.write(f"scikit-learn SVM 评估结果\n")
        f.write(f"时间: {timestamp}\n")
        f.write("="*30 + "\n\n")
        f.write("--- 超参数 ---\n")
        f.write(f"C: {args.C}\n")
        f.write(f"gamma: {args.gamma}\n")
        f.write("--- 最终准确率 ---\n")
        f.write(f"训练集准确率: {train_accuracy:.4f}\n")
        f.write(f"验证集准确率: {val_accuracy:.4f}\n")
        f.write(f"测试集准确率: {test_accuracy:.4f}\n")

    print(f"\n结果已保存至: {results_filepath}")

if __name__ == '__main__':
    main()