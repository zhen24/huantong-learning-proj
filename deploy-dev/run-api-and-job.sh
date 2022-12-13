#!/usr/bin/env bash
set -euo pipefail
trap "echo 'error: Script failed: see failed command above'" ERR

docker run \
    -d \
    --name huantong-learning-api \
    -e REDIS_HOST='0.0.0.0' \
    -e REDIS_PORT='6399' \
    -e HOST='0.0.0.0' \
    -e PORT='8118' \
    -e TRAINING_DEVICE="cuda:0" \
    --network host \
    huantong_learning_proj/api:0.1.0

nvidia-docker run \
    -d \
    --privileged \
    --name huantong-learning-receiver \
    --network host \
    --ipc=host \
    -v "/home/mnvai/projects/huantong/release/20220801":"/ai_resource" \
    -v "/home/mnvai/projects/huantong/learning":"/resource" \
    -v "/home/mnvai/projects/huantong/learning/logging":"/logging" \
    -e HUANTONG_LEARNING_LOGGING_FOLDER="/logging" \
    -e REDIS_HOST='0.0.0.0' \
    -e REDIS_PORT='6399' \
    -e AI_HOST="192.168.1.74" \
    -e AI_PORT="8500" \
    -e TRAINING_DEVICE="cuda:0" \
    huantong_learning_proj/job:0.1.0 \
    run_job start_receiving \
    --config_json='/resource/release/config_huantong_learning.json' \
    --original_model_path='/ai_resource/huantong/ranker/state_dict_epoch_73.pt' \
    --license_cer='/resource/release/192-168-1-74.cer'

nvidia-docker run \
    -d \
    --privileged \
    --name huantong-learning-trainer \
    --network host \
    --ipc=host \
    -v "/home/mnvai/projects/huantong/release/20220801":"/ai_resource" \
    -v "/home/mnvai/projects/huantong/learning":"/resource" \
    -v "/home/mnvai/projects/huantong/learning/logging":"/logging" \
    -e HUANTONG_LEARNING_LOGGING_FOLDER="/logging" \
    -e SCRIPT_FOLDER="/resource/release/scripts" \
    -e TOKENIZERS_PARALLELISM="false" \
    -e REDIS_HOST='0.0.0.0' \
    -e REDIS_PORT='6399' \
    -e AI_HOST="192.168.1.74" \
    -e AI_PORT="8500" \
    -e TRAINING_DEVICE="cuda:0" \
    huantong_learning_proj/job:0.1.0 \
    run_job start_training \
    --config_json='/resource/release/config_huantong_learning.json' \
    --online_model_fd='/ai_resource/huantong/ranker' \
    --license_cer='/resource/release/192-168-1-74.cer'

docker ps | grep huantong-learning-api
docker ps | grep huantong-learning-receiver
docker ps | grep huantong-learning-trainer
