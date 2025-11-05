echo "Generating datasets"

python /data1/zhouruiqi/project1/dataset_generate/generate_4f.py \
    --data_scale 3000 \
    --output_dir /data1/zhouruiqi/project1/datasets/4f_datasets

python /data1/zhouruiqi/project1/dataset_generate/generate_2f.py \
    --data_scale 3000 \
    --output_dir /data1/zhouruiqi/project1/datasets/2f_datasets