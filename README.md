# 机器学习课程项目（2025 Fall, THU）

## Project 1:从零实现KNN和SVM

### 简介
本项目从零实现了KNN和SVM两种算法。1）我们实现了基于L1和L2距离的KNN算法，在自建数据集上测试了算法性能，分析了超参数对于算法表现的影响，绘制了决策边界，
在HIGGS数据集上测试了两种算法的准确率。2）我们实现了基于梯度下降的线性SVM、基于随机梯度下降的线性SVM和基于script.minmize()方法的RBF核SVM。在自制数据
集上分析了算法的性能核超参数的影响，绘制了决策边界，在HIGGS数据集上测试了算法的准确率。此外，我们还利用sklearn机器学习库自动实现了一个RBF核SVM，测试了
大样本下基于核方法的SVM的分类能力。

### 快速开始

Step1: 下载project1文件夹并进入

Step2: 创建环境并安装所需要的库

```
conda create -n ml_hw1 python==3.10
conda activate ml_hw1
pip install requirements.txt
```

Step3: 运行生成数据集代码
```
bash run_generate.sh
```
注意：HIGGS数据集在这里

Step4: 依次运行bash文件。结果将打印在终端或保存在results文件夹中。




