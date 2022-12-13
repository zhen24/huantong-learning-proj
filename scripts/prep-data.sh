#!/usr/bin/env bash
set -eo pipefail
trap "echo 'error: Script failed: see failed command above'" ERR

DATABASE_VERSION="$1"

TENANT_ID='1'
RECEIVER_POSTGRES_CONFIG=$(
cat << 'EOF'
{
    'host': '192.168.1.74',
    'port': '30004',
    'dbname': 'oval',
    'user': 'postgres',
    'password': 'Kt4C4TCHJ3',
    'reuse_connection': True,
}
EOF
)

POSTGRES_CONFIG=$(
cat << 'EOF'
{
    'host': '192.168.1.74',
    'port': '30005',
    'dbname': 'oval',
    'user': 'postgres',
    'password': 'Kt4C4TCHJ3',
    'reuse_connection': True,
}
EOF
)

RETRIEVER_CONFIG=$(
cat << EOF
{
    "ltp_tokenizer_config": {
        "model_folder": "/resource/release/textdog_data/token/ltp/base",
        "device": "cuda",
        "batch_size": 512,
    },
    "ltp_tokenizer_cache_jsls": None,
    "stopwords": [
        "哪里",
        "查看",
        "查找",
        "如何",
    ],
    "enable_trigram_split": True,
    "enable_bigram_split": True,
    "enable_unique_token": True,
    "enable_remove_unigram": True,
    "enable_preserve_non_split_unigram": True,
    "es_bm25_config": {
        "es_host": "192.168.1.74",
        "es_port": "8515"
    },
    "es_bm25_apply_opts_size": 512
}
EOF
)

# Snapshot current receiver tables.
fireball huantong_learning_proj.opt.prep_client_db_dump:pg_backup_tables \
    --postgres_config="${RECEIVER_POSTGRES_CONFIG}" \
    --bak_csv_fd="/resource/train-workspace/${DATABASE_VERSION}/db-csv"

# Organizations.
fireball huantong_learning_proj.opt.prep_client_db_dump:pg_drop_indices \
    --postgres_config="$POSTGRES_CONFIG" \
    --table_name="public.organization"

fireball huantong_learning_proj.opt.prep_client_db_dump:pg_delete_rows \
    --postgres_config="$POSTGRES_CONFIG" \
    --table_name="public.organization" \
    --tenant_id="$TENANT_ID"

fireball huantong_learning_proj.opt.prep_client_db_dump:upload_orgs_to_pg \
    --ovalmaster_csv="/resource/train-workspace/${DATABASE_VERSION}/db-csv/organization.csv" \
    --tenant_id="$TENANT_ID" \
    --postgres_config="$POSTGRES_CONFIG"

fireball huantong_learning_proj.opt.prep_client_db_dump:pg_build_index_orgs \
    --postgres_config="$POSTGRES_CONFIG"

# Aliases.
fireball huantong_learning_proj.opt.prep_client_db_dump:pg_drop_indices \
    --postgres_config="$POSTGRES_CONFIG" \
    --table_name="public.alias"

fireball huantong_learning_proj.opt.prep_client_db_dump:pg_delete_rows \
    --postgres_config="$POSTGRES_CONFIG" \
    --table_name="public.alias" \
    --tenant_id="$TENANT_ID"

fireball huantong_learning_proj.opt.prep_client_db_dump:upload_aliases_to_pg \
    --orgalias_csv="/resource/train-workspace/${DATABASE_VERSION}/db-csv/alias.csv" \
    --ovalmaster_csv="/resource/train-workspace/${DATABASE_VERSION}/db-csv/organization.csv" \
    --tenant_id="$TENANT_ID" \
    --postgres_config="$POSTGRES_CONFIG"

fireball huantong_learning_proj.opt.prep_client_db_dump:pg_build_index_aliases \
    --postgres_config="$POSTGRES_CONFIG"

# Collation.
fireball huantong_learning_proj.opt.prep_client_db_dump:pg_drop_indices \
    --postgres_config="$POSTGRES_CONFIG" \
    --table_name="public.collation"

fireball huantong_learning_proj.opt.prep_client_db_dump:pg_delete_rows \
    --postgres_config="$POSTGRES_CONFIG" \
    --table_name="public.collation" \
    --tenant_id="$TENANT_ID"

fireball huantong_learning_proj.opt.prep_client_db_dump:upload_collations_to_pg \
    --orgcodecollate_csv="/resource/train-workspace/${DATABASE_VERSION}/db-csv/collation.csv" \
    --ovalmaster_csv="/resource/train-workspace/${DATABASE_VERSION}/db-csv/organization.csv" \
    --tenant_id="$TENANT_ID" \
    --postgres_config="$POSTGRES_CONFIG"

fireball huantong_learning_proj.opt.prep_client_db_dump:pg_build_index_collations \
    --postgres_config="$POSTGRES_CONFIG"


## 初步清洗
# Blacklist invalid collations.
fireball huantong_learning_proj/opt/batch_test.py:build_invalid_collation_ids \
    --postgres_config="$POSTGRES_CONFIG" \
    --tenant_id="$TENANT_ID" \
    --output_json="/resource/train-workspace/${DATABASE_VERSION}/prep/conflict-collation-ids.json"

# Hide collations.
fireball huantong_learning_proj.opt.batch_test:drop_invalid_pg_collations \
    --postgres_config="$POSTGRES_CONFIG" \
    --tenant_id="$TENANT_ID" \
    --invalid_collation_ids_json="/resource/train-workspace/${DATABASE_VERSION}/prep/conflict-collation-ids.json"

# Extract dev collations.
fireball huantong_learning_proj.opt.batch_test:extract_dev_collations \
    --postgres_config="$POSTGRES_CONFIG" \
    --tenant_id="$TENANT_ID" \
    --output_pkl="/resource/train-workspace/${DATABASE_VERSION}/prep/dev-clean.pkl" \
    --dev_cnt="10000"

## 再次清洗
fireball huantong_learning_proj/opt/batch_test.py:extract_related_collation_ids_from_batch_test \
    --postgres_config="$POSTGRES_CONFIG" \
    --tenant_id="$TENANT_ID" \
    --batch_test_items_pkl="/resource/train-workspace/${DATABASE_VERSION}/prep/dev-clean.pkl" \
    --output_json="/resource/train-workspace/${DATABASE_VERSION}/prep/dev-collation-ids.json"

# Hide collations.
fireball huantong_learning_proj.opt.batch_test:drop_invalid_pg_collations \
    --postgres_config="$POSTGRES_CONFIG" \
    --tenant_id="$TENANT_ID" \
    --invalid_collation_ids_json="/resource/train-workspace/${DATABASE_VERSION}/prep/dev-collation-ids.json"

# Extract train collation ids.
fireball huantong_learning_proj.opt.batch_test:extract_train_collation_ids \
    --postgres_config="$POSTGRES_CONFIG" \
    --tenant_id="$TENANT_ID" \
    --output_json="/resource/train-workspace/${DATABASE_VERSION}/prep/train-collation-id-and-timestamp.json" \
    --train_cnt="1500000"

## 构建分词索引.
# Generate text for tokenization.
fireball huantong_learning_proj.opt.retriever:build_ltp_tokenizer_cache_input \
    --postgres_config="$POSTGRES_CONFIG" \
    --tenant_id="$TENANT_ID" \
    --output_txt="/tmp/textdog/tenant-${TENANT_ID}-ltp-input.txt"

# Tokenizing...
fireball huantong_learning_proj.opt.retriever:ltp_tokenize \
    --model_folder="/resource/release/textdog_data/token/ltp/base" \
    --input_txt="/tmp/textdog/tenant-${TENANT_ID}-ltp-input.txt" \
    --output_jsl="/resource/train-workspace/ltp-cache.jsl" \
    --device="cuda" \
    --batch_size="512"

# Index.
fireball huantong_learning_proj.opt.retriever:initialize_retriever_cli \
    --retriever_config="$RETRIEVER_CONFIG" \
    --postgres_config="$POSTGRES_CONFIG" \
    --tenant_id="$TENANT_ID" \
    --ltp_cache_jsl="/resource/train-workspace/ltp-cache.jsl"

## 生成格式化的训练数据及验证数据
fireball huantong_learning_proj/opt/train_data.py:generate_train_sample_to_folder \
    --retriever_config="$RETRIEVER_CONFIG" \
    --postgres_config="$POSTGRES_CONFIG" \
    --ac_level_one_json_file="/resource/release/ac/ac_level_one.json" \
    --ac_level_two_json_file="/resource/release/ac/ac_level_two.json" \
    --ac_level_three_json_file="/resource/release/ac/ac_level_three.json" \
    --tenant_id="$TENANT_ID" \
    --collation_id_and_timestamp_json="/resource/train-workspace/${DATABASE_VERSION}/prep/train-collation-id-and-timestamp.json" \
    --output_folder="/resource/dataset/final-train-${TENANT_ID}" \
    --negative_size="20" \
    --chunk_size="5000" \
    --random_seed="13370"

fireball huantong_learning_proj/opt/train_data.py:generate_dev_set_from_batch_test_items \
    --retriever_config="$RETRIEVER_CONFIG" \
    --postgres_config="$POSTGRES_CONFIG" \
    --ac_level_one_json_file="/resource/release/ac/ac_level_one.json" \
    --ac_level_two_json_file="/resource/release/ac/ac_level_two.json" \
    --ac_level_three_json_file="/resource/release/ac/ac_level_three.json" \
    --tenant_id="$TENANT_ID" \
    --batch_test_items_pkl="/resource/train-workspace/${DATABASE_VERSION}/prep/dev-clean.pkl" \
    --output_folder="/resource/dataset/final-dev-${TENANT_ID}" \
    --negative_size="1"
