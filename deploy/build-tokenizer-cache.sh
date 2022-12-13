#!/usr/bin/env bash
set -euo pipefail
trap "echo 'error: Script failed: see failed command above'" ERR


TENANT_ID=1
echo "TENANT_ID=${TENANT_ID}"
DATABASE_VERSION="$1"
echo "DATABASE_VERSION=${DATABASE_VERSION}"

POSTGRES_CONFIG=$(
cat << 'EOF'
{
    'host': '10.190.6.16',
    'port': '30005',
    'dbname': 'oval',
    'user': 'postgres',
    'password': 'Kt4C4TCHJ3',
    'reuse_connection': True,
}
EOF
)

mkdir -p /tmp/textdog

echo "Loading text."
docker run \
    --rm -it \
    -v "/tmp/textdog":"/data" \
    huantong_learning_proj/job:0.1.0 \
    fireball huantong_learning_proj.opt.retriever:build_ltp_tokenizer_cache_input \
    --postgres_config="$POSTGRES_CONFIG" \
    --tenant_id="$TENANT_ID" \
    --output_txt="/data/ltp-input.txt"

echo "Tokenizing..."
#cp "/data0/learning/train-workspace/ltp-cache.jsl" "/data0/learning/train-workspace/${DATABASE_VERSION}/prep"
nvidia-docker run \
    --rm -it \
    --user "$(id -u):$(id -g)" \
    -v "/data0/learning/release/textdog_data/token/ltp/base":"/model_folder" \
    -v "/tmp/textdog":"/input" \
    -v "/data0/learning/train-workspace/${DATABASE_VERSION}/prep":"/output" \
    huantong_learning_proj/job:0.1.0 \
    fireball huantong_learning_proj.opt.retriever:ltp_tokenize \
    --model_folder="/model_folder" \
    --input_txt="/input/ltp-input.txt" \
    --output_jsl="/output/ltp-cache.jsl" \
    --device="cuda" \
    --batch_size="512"

echo "Cache '/data0/learning/train-workspace/${DATABASE_VERSION}/prep/ltp-cache.jsl' is generated."

