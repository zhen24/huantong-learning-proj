#!/usr/bin/env bash
set -euo pipefail
trap "echo 'error: Script failed: see failed command above'" ERR

docker run \
    -d \
    --name huantong-learning-api \
    -e REDIS_HOST='0.0.0.0' \
    -e REDIS_PORT='6379' \
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
    -v "/data/oval/release/20220824":"/ai_resource" \
    -v "/data0/learning":"/resource" \
    -v "/data0/learning/logging":"/logging" \
    -e HUANTONG_LEARNING_LOGGING_FOLDER="/logging" \
    -e REDIS_HOST='0.0.0.0' \
    -e REDIS_PORT='6379' \
    -e AI_HOST="10.190.6.12" \
    -e AI_PORT="8500" \
    -e TRAINING_DEVICE="cuda:0" \
    huantong_learning_proj/job:0.1.0 \
    run_job start_receiving \
    --config_json='/resource/release/config_huantong_learning.json' \
    --original_model_path='/ai_resource/huantong/ranker/state_dict_epoch_73.pt' \
    --license_cer='/resource/release/10-190-6-16.cer'

nvidia-docker run \
    -d \
    --privileged \
    --name huantong-learning-trainer \
    --network host \
    --ipc=host \
    -v "/data/oval/release/20220824":"/ai_resource" \
    -v "/data0/learning":"/resource" \
    -v "/data0/learning/logging":"/logging" \
    -e HUANTONG_LEARNING_LOGGING_FOLDER="/logging" \
    -e SCRIPT_FOLDER="/resource/release/scripts" \
    -e TOKENIZERS_PARALLELISM="false" \
    -e REDIS_HOST='0.0.0.0' \
    -e REDIS_PORT='6379' \
    -e AI_HOST="10.190.6.12" \
    -e AI_PORT="8500" \
    -e TRAINING_DEVICE="cuda:0" \
    huantong_learning_proj/job:0.1.0 \
    run_job start_training \
    --config_json='/resource/release/config_huantong_learning.json' \
    --online_model_fd='/ai_resource/huantong/ranker' \
    --license_cer='/resource/release/10-190-6-16.cer'

docker ps | grep huantong-learning-api
docker ps | grep huantong-learning-receiver
docker ps | grep huantong-learning-trainer
