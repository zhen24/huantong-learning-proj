#!/usr/bin/env bash
set -euo pipefail
trap "echo 'error: Script failed: see failed command above'" ERR

docker run -d --name huantong-redis -p 6379:6379 redis

docker run -d --name huantong-elasticsearch-for-training \
    -p 8505:9200 \
    -e "discovery.type=single-node" \
    elasticsearch:7.8.1

docker ps | grep huantong-redis
docker ps | grep huantong-elasticsearch-for-training
