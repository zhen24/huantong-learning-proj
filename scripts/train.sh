#!/usr/bin/env bash
set -eo pipefail
trap "echo 'error: Script failed: see failed command above'" ERR

DATABASE_VERSION="$1"

storage_fd="/resource/train-workspace/${DATABASE_VERSION}/model"
if [ -e "${storage_fd}" ] ; then
    rm -rf "${storage_fd}"
fi

fireball huantong_learning_proj/opt/ranker.py:run_training \
    --bert_pretrained_folder="/resource/release/ranker/bert-base-ernie-vocab-patched" \
    --train_dataset_folder="/resource/dataset/final-train-1" \
    --dev_dataset_folder="/resource/dataset/final-dev-1" \
    --output_folder="${storage_fd}" \
    --device="cuda"
