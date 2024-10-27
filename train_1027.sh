#!/bin/bash

# Slurm options
#SBATCH --job-name=wind_power_train
#SBATCH --output=logs/%x_%j.out  # 로그 파일은 logs 폴더
#SBATCH --error=logs/%x_%j.err   # 에러 로그는 logs 폴더
#SBATCH --gres=gpu:1              # GPU 하나 요청
#SBATCH --partition=LocalQ       # 사용할 파티션 이름

# 공통 설정
SAVE_DIR="test_results"
EPOCHS=1500
BATCH_SIZE=24
LR=0.001
CRITERION_TYPE="SmoothL1"
WEIGHT_DECAY=0.0001
MAX_GRAD_NORM=1.0
NUM_LAYERS=4

# 모델 목록 (SeriesDecompLSTM 제외)
MODELS=("LSTM" "GRU" "LSTM_relu" "GRU_relu" "LSTM_relu_dropALL" "GRU_relu_dropALL" "LSTMWithBatchNorm" "GRUWithBatchNorm")

# Feature keys 목록
FEATURE_KEYS_LIST=(
    'all_features'
    'pca_only'
    'cluster_only'
    'medoid_only'
    'pca_cluster'
    'pca_medoid'
    'cluster_medoid'
    'none_pca_cluster_medoid'
    'cluster_2_only'
    'cluster_3_only'
    'cluster_4_only'
    'cluster_5_only'
    'cluster_6_only'
)

# Hidden dim 목록
HIDDEN_DIMS=(256 400 512 1024)

# 지역 목록
REGIONS=("yg" "gj")

# 작업 예약 루프
for model in "${MODELS[@]}"; do
    for feature_keys in "${FEATURE_KEYS_LIST[@]}"; do
        for hidden_dim in "${HIDDEN_DIMS[@]}"; do
            for region in "${REGIONS[@]}"; do
                srun python train_DNN.py \
                    --model_type "$model" \
                    --region "$region" \
                    --save_dir "$SAVE_DIR" \
                    --epochs "$EPOCHS" \
                    --batch_size "$BATCH_SIZE" \
                    --lr "$LR" \
                    --criterion_type "$CRITERION_TYPE" \
                    --weight_decay "$WEIGHT_DECAY" \
                    --max_grad_norm "$MAX_GRAD_NORM" \
                    --num_layers "$NUM_LAYERS" \
                    --hidden_dim "$hidden_dim" \
                    --bidirectional \
                    --feature_keys "$feature_keys" &
                sleep 1  # 예약 간 딜레이
            done
        done
    done
done

wait  # 모든 작업이 끝날 때까지 대기
