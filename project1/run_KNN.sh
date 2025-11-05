echo "Running KNN model"

for k_val in 1 3 5
do
    for distance_metric in L1 L2
        do
            echo "Testing with k = $k_val and distance_metric = $distance_metric"
            python /data1/zhouruiqi/project1/KNN/KNN_model.py \
                --data_path /data1/zhouruiqi/project1/datasets/4f_datasets \
                --train_size 0.8 \
                --val_size 0.1 \
                --test_size 0.1 \
                --k $k_val \
                --results_dir /data1/zhouruiqi/project1/results/KNN_results \
                --distance_metric $distance_metric
        done
done

python /data1/zhouruiqi/project1/KNN/draw.py \
    --data_path /data1/zhouruiqi/project1/datasets/2f_datasets \
    --train_size 0.2 \
    --test_size 0.8 \
    --k 5 \
    --results_dir /data1/zhouruiqi/project1/results/KNN_decision_boundary \
    --distance_metric L1