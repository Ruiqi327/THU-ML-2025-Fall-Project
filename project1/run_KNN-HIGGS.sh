echo "Running KNN model on HIGGS dataset"

for TRAIN_SIZE in 100000 200000 300000 490000; do
for DISTANCE_METRIC in L1 L2; do
    python /data1/zhouruiqi/project1/KNN/KNN-HIGGS.py \
        --data_path /data1/zhouruiqi/ml_hw1/SVM-HIGGS/dataset/HIGGS.csv \
        --train_size $TRAIN_SIZE \
        --results_dir /data1/zhouruiqi/project1/results/KNN_HIGGS_results \
        --distance_metric $DISTANCE_METRIC
done    