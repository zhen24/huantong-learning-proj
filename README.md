# huantong_learning_proj

## pg初始化所需参数(按需修改)
```
HUANTONG_PROJ_DATA="/home/mnvai/projects/huantong/HUANTONG_PROJ_DATA"
DATABASE_VERSION='20221101'
POSTGRES_CONFIG=$(
cat << 'EOF'
{
    'host': '192.168.1.74',
    #'port': '30004',
    'port': '30005',
    'dbname': 'oval',
    'user': 'postgres',
    'password': 'Kt4C4TCHJ3',
    'reuse_connection': True,
}
EOF
)

TENANT_ID='1'
BATCH_TEST_VERSION='202105'
```

## 配置开发、训练、测试环境的 postgres
```bash
# 首先登陆 1.74，启动 pg
docker run \
    -d \
    --name huantong-postgres-for-receiver \
    -e POSTGRES_PASSWORD='Kt4C4TCHJ3' \
    -e POSTGRES_USER='postgres' \
    -e POSTGRES_DB='oval' \
    -p "30004":"5432" \
    postgres:11.5

docker run \
    -d \
    --name huantong-postgres-for-trainer \
    -e POSTGRES_PASSWORD='Kt4C4TCHJ3' \
    -e POSTGRES_USER='postgres' \
    -e POSTGRES_DB='oval' \
    -p "30005":"5432" \
    postgres:11.5
```

```bash
# 在本地环境（可能非 1.74）生成当前数据接收表的数据快照,作为训练数据来源(此处是端口30004的pg)
fireball huantong_learning_proj.opt.prep_client_db_dump:pg_backup_tables \
    --postgres_config="$POSTGRES_CONFIG" \
    --bak_csv_fd="${RESOURCE}/train-workspace/${DATABASE_VERSION}/db-csv"

# 登陆 1.74进行开发,指向 1.74 pg
# 执行以下命令,初始化 pg   --此处为用于训练的数据库,端口30005,以下无特殊说明默认是此数据库
# Tables.
fireball huantong_learning_proj.opt.prep_client_db_dump:pg_create_tables \
    --postgres_config="$POSTGRES_CONFIG"

# Organizations.
fireball huantong_learning_proj.opt.prep_client_db_dump:pg_drop_indices \
    --postgres_config="$POSTGRES_CONFIG" \
    --table_name="public.organization"

fireball huantong_learning_proj.opt.prep_client_db_dump:pg_delete_rows \
    --postgres_config="$POSTGRES_CONFIG" \
    --table_name="public.organization" \
    --tenant_id="$TENANT_ID"

fireball huantong_learning_proj.opt.prep_client_db_dump:upload_orgs_to_pg \
    --ovalmaster_csv="${RESOURCE}/train-workspace/${DATABASE_VERSION}/db-csv/organization.csv" \
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
    --orgalias_csv="${RESOURCE}/train-workspace/${DATABASE_VERSION}/db-csv/alias.csv" \
    --ovalmaster_csv="${RESOURCE}/train-workspace/${DATABASE_VERSION}/db-csv/organization.csv" \
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
    --orgcodecollate_csv="${RESOURCE}/train-workspace/${DATABASE_VERSION}/db-csv/collation.csv" \
    --ovalmaster_csv="${RESOURCE}/train-workspace/${DATABASE_VERSION}/db-csv/organization.csv" \
    --tenant_id="$TENANT_ID" \
    --postgres_config="$POSTGRES_CONFIG"

fireball huantong_learning_proj.opt.prep_client_db_dump:pg_build_index_collations \
    --postgres_config="$POSTGRES_CONFIG"
```

## 初步清洗
```bash
# Blacklist invalid collations.
fireball huantong_learning_proj/opt/batch_test.py:build_invalid_collation_ids \
    --postgres_config="$POSTGRES_CONFIG" \
    --tenant_id="$TENANT_ID" \
    --output_json="${RESOURCE}/train-workspace/${DATABASE_VERSION}/prep/conflict-collation-ids.json"

# Hide collations.
# 修改 pg 数据库，删掉无效的对照数据
fireball huantong_learning_proj.opt.batch_test:drop_invalid_pg_collations \
    --postgres_config="$POSTGRES_CONFIG" \
    --tenant_id="$TENANT_ID" \
    --invalid_collation_ids_json="${RESOURCE}/train-workspace/${DATABASE_VERSION}/prep/conflict-collation-ids.json"
```

## 抽取验证集(10000条)
```bash
# 从历史运营数据中抽取并清洗10000条作为验证集(若不足再从人工标注数据中抽取)
fireball huantong_learning_proj.opt.batch_test:extract_dev_collations \
    --postgres_config="$POSTGRES_CONFIG" \
    --tenant_id="$TENANT_ID" \
    --output_pkl="${RESOURCE}/train-workspace/${DATABASE_VERSION}/prep/dev-clean.pkl" \
    --dev_cnt="10000"
```

## 这里面再次清洗，得到明显不应该抽样的对照数据 id
```bash
# Generate data for training.
# 历史运营数据中和验证集中的query_name一致的对照数据
fireball huantong_learning_proj/opt/batch_test.py:extract_related_collation_ids_from_batch_test \
    --postgres_config="$POSTGRES_CONFIG" \
    --tenant_id="$TENANT_ID" \
    --batch_test_items_pkl="${RESOURCE}/train-workspace/${DATABASE_VERSION}/prep/dev-clean.pkl" \
    --output_json="${RESOURCE}/train-workspace/${DATABASE_VERSION}/prep/dev-collation-ids.json"

# Hide collations.
# 修改 pg 数据库，删掉无效的对照数据
fireball huantong_learning_proj.opt.batch_test:drop_invalid_pg_collations \
    --postgres_config="$POSTGRES_CONFIG" \
    --tenant_id="$TENANT_ID" \
    --invalid_collation_ids_json="${RESOURCE}/train-workspace/${DATABASE_VERSION}/prep/dev-collation-ids.json"

# 从人工标注数据中抽取1500000条作为训练集(若不足再从历史运营数据中抽取)
fireball huantong_learning_proj.opt.batch_test:extract_train_collation_ids \
    --postgres_config="$POSTGRES_CONFIG" \
    --tenant_id="$TENANT_ID" \
    --output_json="${RESOURCE}/train-workspace/${DATABASE_VERSION}/prep/train-collation-id-and-timestamp.json" \
    --train_cnt="1500000"
```

## 下一步对 pg 的数据建 elasticsearch 索引
```bash
# 首先生成对应的分词缓存
fireball huantong_learning_proj.opt.retriever:build_ltp_tokenizer_cache_input \
    --postgres_config="$POSTGRES_CONFIG" \
    --tenant_id="$TENANT_ID" \
    --output_txt="/tmp/textdog/tenant-${TENANT_ID}-ltp-input.txt"

nvidia-docker run \
    --rm -it \
    --user "$(id -u):$(id -g)" \
    -v "${RESOURCE}/release/textdog_data/token/ltp/base":"/model_folder" \
    -v "/tmp/textdog":"/input" \
    -v "${RESOURCE}/train-workspace/${DATABASE_VERSION}/prep":"/data" \
    textdog/textdog:0.1.1 \
    ltp_tokenize \
    --model_folder="/model_folder" \
    --input_txt="/data/tenant-${TENANT_ID}-ltp-input.txt" \
    --output_jsl="/data/tenant-${TENANT_ID}-ltp-cache.jsl" \
    --device="cuda" \
    --batch_size="512"

fireball huantong_learning_proj.opt.retriever:ltp_tokenize \
    --model_folder="${RESOURCE}/release/textdog_data/token/ltp/base" \
    --input_txt="/tmp/textdog/tenant-${TENANT_ID}-ltp-input.txt" \
    --output_jsl="${RESOURCE}/train-workspace/ltp-cache.jsl" \
    --device="cuda" \
    --batch_size="512"

# 然后启动 elasticsearch 服务，建立索引
# 你可以换成新一点的 elasticsearch 版本，这个版本有安全问题
docker run -d --name huantong-elasticsearch-for-training \
    -p 8515:9200 \
    -e "discovery.type=single-node" \
    elasticsearch:7.8.1

# TEXTDOG_DATA 对应 textdog-data 文件夹
RETRIEVER_CONFIG=$(
cat << EOF
{
    "ltp_tokenizer_config": {
        "model_folder": "${RESOURCE}/release/textdog_data/token/ltp/base",
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

# Index.
fireball huantong_learning_proj.opt.retriever:initialize_retriever_cli \
    --retriever_config="$RETRIEVER_CONFIG" \
    --postgres_config="$POSTGRES_CONFIG" \
    --tenant_id="$TENANT_ID" \
    --ltp_cache_jsl="${RESOURCE}/train-workspace/ltp-cache.jsl"
```

## 然后生成格式化的训练数据及验证数据
```bash
# (upstream_id,query)对应的retrieved_orgs,其下游ID和对照表中不一致的为负样本
fireball huantong_learning_proj/opt/train_data.py:generate_train_sample_to_folder \
    --retriever_config="$RETRIEVER_CONFIG" \
    --postgres_config="$POSTGRES_CONFIG" \
    --ac_level_one_json_file="${RESOURCE}/release/ac/ac_level_one.json" \
    --ac_level_two_json_file="${RESOURCE}/release/ac/ac_level_two.json" \
    --ac_level_three_json_file="${RESOURCE}/release/ac/ac_level_three.json" \
    --tenant_id="$TENANT_ID" \
    --collation_id_and_timestamp_json="${RESOURCE}/train-workspace/${DATABASE_VERSION}/prep/train-collation-id-and-timestamp.json" \
    --output_folder="${RESOURCE}/dataset/final-train-${TENANT_ID}" \
    --negative_size="20" \
    --chunk_size="5000" \
    --random_seed="13370"

# 格式化的验证数据
fireball huantong_learning_proj/opt/train_data.py:generate_dev_set_from_batch_test_items \
    --retriever_config="$RETRIEVER_CONFIG" \
    --postgres_config="$POSTGRES_CONFIG" \
    --ac_level_one_json_file="${RESOURCE}/release/ac/ac_level_one.json" \
    --ac_level_two_json_file="${RESOURCE}/release/ac/ac_level_two.json" \
    --ac_level_three_json_file="${RESOURCE}/release/ac/ac_level_three.json" \
    --tenant_id="$TENANT_ID" \
    --batch_test_items_pkl="${RESOURCE}/train-workspace/${DATABASE_VERSION}/prep/dev-clean.pkl" \
    --output_folder="${RESOURCE}/dataset/final-dev-${TENANT_ID}" \
    --negative_size="1"
```

## 模型训练
```bash
# 阅读 `rankerdog/README.md`
fireball huantong_learning_proj/opt/ranker.py:run_training \
    --bert_pretrained_folder="${RESOURCE}/release/ranker/bert-base-ernie-vocab-patched" \
    --train_dataset_folder="${RESOURCE}/dataset/final-train-1" \
    --dev_dataset_folder="${RESOURCE}/dataset/final-dev-1" \
    --output_folder="${RESOURCE}/train-workspace/${DATABASE_VERSION}/model" \
    --device="cuda"

# 模型推送见 huantong_learning_proj/job: AutoTrain.launch_state_dict_file  # line 136
```

## 执行端对端测试
在模型训练完之后，基于前面生成的批量测试数据集，执行批量测试
```bash
fireball huantong_learning_proj.opt.batch_test:run_batch_test_retrieval \
    --retriever_config="$RETRIEVER_CONFIG" \
    --postgres_config="$POSTGRES_CONFIG" \
    --ac_level_one_json_file="${RESOURCE}/release/ac/ac_level_one.json" \
    --ac_level_two_json_file="${RESOURCE}/release/ac/ac_level_two.json" \
    --ac_level_three_json_file="${RESOURCE}/release/ac/ac_level_three.json" \
    --tenant_id="$TENANT_ID" \
    --batch_test_items_pkl="${RESOURCE}/train-workspace/${DATABASE_VERSION}/prep/${BATCH_TEST_VERSION}-clean.pkl" \
    --output_pkl="${RESOURCE}/train-workspace/${DATABASE_VERSION}/prep/${BATCH_TEST_VERSION}-retrieved.pkl"

fireball huantong_learning_proj.opt.batch_test:inspect_retrieval_failure \
    --retrieved_items_pkl="${RESOURCE}/train-workspace/${DATABASE_VERSION}/prep/${BATCH_TEST_VERSION}-retrieved.pkl" \
    --output_json="${RESOURCE}/train-workspace/${DATABASE_VERSION}/prep/${BATCH_TEST_VERSION}-retrieved-places.json"

# Copy to share folder.
mkdir -p "/Volumes/share/dev/users/huntzhan/huantong/data/batch-test"
rsync -ah --progress \
    "${RESOURCE}/train-workspace/${DATABASE_VERSION}/prep/${BATCH_TEST_VERSION}-retrieved.pkl" \
    "/Volumes/share/dev/users/huntzhan/huantong/data/batch-test"

# 1.74.
nvidia-docker run \
  --name huantong-proj-batch-test \
  --ipc=host \
  --tmpfs /tmpfs \
  -it \
  --rm \
  --user "$(id -u):$(id -g)" \
  -v "/home/mnvai/projects/huantong":"/MNVAI" \
  -v "/home/mnvai/projects/huantong/HUANTONG_PROJ_DATA":"/data" \
  -e CD_DEFAULT_FOLDER='/MNVAI' \
  -e APT_SET_MIRROR_TENCENT=1 \
  -e PIP_SET_INDEX_TENCENT=1 \
  -v "$SSH_AUTH_SOCK":/run/ssh-auth.sock:shared \
  -e SSH_AUTH_SOCK="/run/ssh-auth.sock" \
  -v "$HOME"/.gitconfig:/etc/gitconfig:ro \
  -v "$HOME"/.bash_history:/run/.bash_history:rw \
  -e HISTFILE=/run/.bash_history \
  wden/wden:devel-cuda11.1.1-cudnn8-ubuntu18.04-python3.8

pip install /MNVAI/torch-1.8.1+cu111-cp38-cp38-linux_x86_64.whl
sudo apt-get -y install libpq-dev

cd huantong_learning_proj/
pip install -e .'[dev,job,job-private,pipeline]'

MODEL_VERSION='20221201'

RANKER_CONFIG=$(
cat << EOF
{
    "bert_pretrained_folder": "/data/release/ranker/bert-base-ernie-vocab-patched-inference",
    "state_dict_file": "/data/train-workspace/20221201/model/state_dict_epoch_73.pt",
    "device": "cuda",
    "classifier_top_k": 10,
    "classifier_thr": 0.5
}
EOF
)

fireball huantong_learning_proj.opt.batch_test:run_batch_test_ranking \
    --ranker_config="$RANKER_CONFIG" \
    --retrieved_items_pkl="/data/batch-test/${BATCH_TEST_VERSION}-retrieved.pkl" \
    --output_pkl="/data/batch-test/${BATCH_TEST_VERSION}-${MODEL_VERSION}-ranked.pkl"

# Copy back to local.
rsync -ah --progress \
    "/Volumes/share/dev/users/huntzhan/huantong/data/batch-test/${BATCH_TEST_VERSION}-${MODEL_VERSION}-ranked.pkl" \
    "${RESOURCE}/train-workspace/${DATABASE_VERSION}/prep/${BATCH_TEST_VERSION}-${MODEL_VERSION}-ranked.pkl"

fireball huantong_learning_proj/opt/batch_test.py:run_batch_test_report \
    --ranked_items_pkl="${RESOURCE}/train-workspace/${DATABASE_VERSION}/prep/${BATCH_TEST_VERSION}-${MODEL_VERSION}-ranked.pkl" \
    --output_folder="${RESOURCE}/train-workspace/${DATABASE_VERSION}/prep/${BATCH_TEST_VERSION}-${MODEL_VERSION}-error"
```
