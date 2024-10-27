import itertools

models = ["LSTM", "GRU", "LSTM_relu", "GRU_relu", "LSTM_relu_dropALL", "GRU_relu_dropALL", "LSTMWithBatchNorm", "GRUWithBatchNorm"]
feature_keys_list = [
    'all_features', 'pca_only', 'cluster_only', 'medoid_only', 'pca_cluster',
    'pca_medoid', 'cluster_medoid', 'none_pca_cluster_medoid', 
    'cluster_2_only', 'cluster_3_only', 'cluster_4_only', 'cluster_5_only', 'cluster_6_only'
]
hidden_dims = [256, 400, 512, 1024]
regions = ["yg", "gj"]

with open("train_1027_auto.sub", "w") as f:
    f.write("# Auto-generated Condor submit file for all parameter combinations\n")
    f.write("executable = /usr/bin/python3\n")
    f.write("output = logs/$(region)_$(model)_$(feature_keys)_$(hidden_dim).out\n")
    f.write("error = logs/$(region)_$(model)_$(feature_keys)_$(hidden_dim).err\n")
    f.write("log = logs/$(region)_$(model)_$(feature_keys)_$(hidden_dim).log\n")
    f.write("request_gpus = 1\n")
    
    # 공통 arguments
    base_args = (
        "train_DNN.py --model_type $(model) --region $(region) --save_dir test_results "
        "--epochs 1500 --batch_size 24 --lr 0.001 --criterion_type SmoothL1 --weight_decay 0.0001 "
        "--max_grad_norm 1.0 --num_layers 4 --hidden_dim $(hidden_dim) --bidirectional --feature_keys $(feature_keys)"
    )
    f.write(f"arguments = {base_args}\n\n")

    # 조합 생성
    f.write("queue model, feature_keys, hidden_dim, region from (\n")
    for model, feature_keys, hidden_dim, region in itertools.product(models, feature_keys_list, hidden_dims, regions):
        f.write(f"    {model} {feature_keys} {hidden_dim} {region}\n")
    f.write(")\n")
