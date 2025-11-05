#echo "数据集生成完成"

#运行模型训练脚本
echo "开始训练 SVM 模型..."
python /data1/zhouruiqi/project1/SVM/svm_model.py\
    --data_dir /data1/zhouruiqi/project1/datasets/4f_datasets \
    --iterations 1000 \
    --lr 2e-5 \
    --penalty 30 \
    --train_proportion 0.8 \
    --val_proportion 0.1 \
    --test_proportion 0.1 \
    --results_dir /data1/zhouruiqi/project1/results/SVM_results

echo "开始绘制决策边界..."
python /data1/zhouruiqi/project1/SVM/draw.py\
    --data_dir /data1/zhouruiqi/project1/datasets/2f_datasets \
    --iterations 1000 \
    --lr 1e-5 \
    --penalty 2 \
    --train_proportion 0.5 \
    --val_proportion 0.1 \
    --test_proportion 0.4 \
    --results_dir /data1/zhouruiqi/project1/results/SVM_decision_boundary