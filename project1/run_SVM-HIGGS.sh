#!/bin/bash
set -e

echo "开始训练 SVM 模型（SGD）..."
python /data1/zhouruiqi/project1/SVM/svm_sgd.py\
    --data_path /data1/zhouruiqi/project1/datasets/HIGGS_datasets/HIGGS.csv \
    --lr 1e-6 \
    --epochs 200 \
    --penalty 10.0 \
    --train_proportion 490000 \
    --val_proportion 5000 \
    --test_proportion 500 \
    --results_dir /data1/zhouruiqi/project1/results/SVM_HIGGS_results \
    --batch_size 256

echo "开始训练 线性SVM 模型（SGD）..."
python /data1/zhouruiqi/project1/SVM/svm_minimize.py\
    --data_path /data1/zhouruiqi/project1/datasets/HIGGS_datasets/HIGGS.csv \
    --train_size 200 \
    --C 1.0 \
    --gamma 0.1 

echo "开始自动化训练 SVM 模型（auto_svm）..."
python /data1/zhouruiqi/project1/SVM/svm_auto.py\
    --data_path /data1/zhouruiqi/project1/datasets/HIGGS_datasets/HIGGS.csv \
    --C 1.0 \
    --gamma 0.1 \
    --train_size 30000