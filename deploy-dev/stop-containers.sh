#!/usr/bin/env bash
set -euo pipefail
trap "echo 'error: Script failed: see failed command above'" ERR

docker stop huantong-elasticsearch-for-training && docker rm huantong-elasticsearch-for-training
docker stop huantong-redis-for-learning && docker rm huantong-redis-for-learning
docker stop huantong-learning-api && docker rm huantong-learning-api
docker stop huantong-learning-receiver && docker rm huantong-learning-receiver
docker stop huantong-learning-trainer && docker rm huantong-learning-trainer
