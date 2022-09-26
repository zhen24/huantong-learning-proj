# huantong_learning_proj

## pg初始化所需参数(按需修改)
```
HUANTONG_PROJ_DATA="/home/zhen24/projects/huantong/HUANTONG_PROJ_DATA"
DATABASE_VERSION='20210709'
POSTGRES_CONFIG=$(
cat << 'EOF'
{
    'host': '192.168.1.74',
    'port': '30000',
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

## 格式化 SQL

```bash
# Convert sql to csv.
mkdir -p "${HUANTONG_PROJ_DATA}/data/db-csv/${DATABASE_VERSION}"

fireball huantong_learning_proj.opt.prep_client_db_dump:convert_sql_to_csv \
    --table_name="ovalmaster" \
    --sql_file="${HUANTONG_PROJ_DATA}/data/db-dump/${DATABASE_VERSION}/OVALMASTER.sql" \
    --output_csv="${HUANTONG_PROJ_DATA}/data/db-csv/${DATABASE_VERSION}/ovalmaster.csv"

fireball huantong_learning_proj.opt.prep_client_db_dump:convert_sql_to_csv \
    --table_name="orgalias" \
    --sql_file="${HUANTONG_PROJ_DATA}/data/db-dump/${DATABASE_VERSION}/ORGALIAS.sql" \
    --output_csv="${HUANTONG_PROJ_DATA}/data/db-csv/${DATABASE_VERSION}/orgalias.csv"

fireball huantong_learning_proj.opt.prep_client_db_dump:convert_sql_to_csv \
    --table_name="orgcodecollate" \
    --sql_file="${HUANTONG_PROJ_DATA}/data/db-dump/${DATABASE_VERSION}/ORGCODECOLLATE.sql" \
    --output_csv="${HUANTONG_PROJ_DATA}/data/db-csv/${DATABASE_VERSION}/orgcodecollate.csv"
```

## 配置开发、训练、测试环境的 postgres

```bash
# 首先登陆 1.74，启动 pg
docker run \
    -d \
    --name huantong-postgres-for-training \
    -e POSTGRES_PASSWORD='Kt4C4TCHJ3' \
    -e POSTGRES_USER='postgres' \
    -e POSTGRES_DB='oval' \
    -p "30000":"5432" \
    postgres:11.5
```

```bash
# 在本地开发环境（非 1.74）
# 指向 1.74 pg
# 执行一下命令初始化 pg
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
    --ovalmaster_csv="${HUANTONG_PROJ_DATA}/data/db-csv/${DATABASE_VERSION}/ovalmaster.csv" \
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
    --orgalias_csv="${HUANTONG_PROJ_DATA}/data/db-csv/${DATABASE_VERSION}/orgalias.csv" \
    --ovalmaster_csv="${HUANTONG_PROJ_DATA}/data/db-csv/${DATABASE_VERSION}/ovalmaster.csv" \
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
    --orgcodecollate_csv="${HUANTONG_PROJ_DATA}/data/db-csv/${DATABASE_VERSION}/orgcodecollate.csv" \
    --ovalmaster_csv="${HUANTONG_PROJ_DATA}/data/db-csv/${DATABASE_VERSION}/ovalmaster.csv" \
    --tenant_id="$TENANT_ID" \
    --postgres_config="$POSTGRES_CONFIG"

fireball huantong_learning_proj.opt.prep_client_db_dump:pg_build_index_collations \
    --postgres_config="$POSTGRES_CONFIG"
```

```bash
# Backup postgres.
docker run \
    --name huantong-postgres-cli \
    --rm -it \
    --network host \
    -v "$(pwd)":"/data" \
    -e POSTGRES_HOST='0.0.0.0' \
    -e POSTGRES_PORT='30000' \
    -e POSTGRES_USER='postgres' \
    -e PGPASSWORD='Kt4C4TCHJ3' \
    --entrypoint /bin/bash \
    postgres

# IN CONTAINER:
TENANT_ID='1'
PG_DUMP_FOLDER="/data/$(date +'%Y%m%d-%H%M%S')"
mkdir "$PG_DUMP_FOLDER"
echo "Dump tenant_id=${TENANT_ID} to ${PG_DUMP_FOLDER}"

pg_dumpall \
    --host="$POSTGRES_HOST" \
    --port="$POSTGRES_PORT" \
    --username="$POSTGRES_USER" \
    --schema-only \
    > "${PG_DUMP_FOLDER}/scheme.sql"

for TABLE in 'organization' 'alias' 'collation' ; do
    echo "Dump ${TABLE}"
    psql \
        --host="$POSTGRES_HOST" \
        --port="$POSTGRES_PORT" \
        --username="$POSTGRES_USER" \
        -c "\copy (SELECT * FROM public.${TABLE} WHERE tenant_id=${TENANT_ID}) To '${PG_DUMP_FOLDER}/${TABLE}.csv' With CSV" \
        oval
done

exit

# Restore postgres.
docker run \
    --name huantong-postgres-cli \
    --rm -it \
    --network host \
    -v "$(pwd)":"/data" \
    -e POSTGRES_HOST='0.0.0.0' \
    -e POSTGRES_PORT='30000' \
    -e POSTGRES_USER='postgres' \
    -e PGPASSWORD='Kt4C4TCHJ3' \
    --entrypoint /bin/bash \
    postgres

# IN CONTAINER:
DUMP_DATE='REQUIRED'

PG_DUMP_FOLDER="/data/${DUMP_DATE}"
echo "Loading from ${PG_DUMP_FOLDER}"

psql \
    --host="$POSTGRES_HOST" \
    --port="$POSTGRES_PORT" \
    --username="$POSTGRES_USER" \
    --file="${PG_DUMP_FOLDER}/scheme.sql" \
    template1

for TABLE in 'organization' 'alias' 'collation' ; do
    echo "Load ${PG_DUMP_FOLDER}/${TABLE}.csv"
    psql \
        --host="$POSTGRES_HOST" \
        --port="$POSTGRES_PORT" \
        --username="$POSTGRES_USER" \
        -c "\copy public.${TABLE} FROM '${PG_DUMP_FOLDER}/${TABLE}.csv' With CSV" \
        oval
done

exit
```

## 端对端批量测试数据集生成

需要从全量数据中抽样、剥离一部分数据，制作端对端批量测试数据集，用于端对端批量测试指标计算

注意，这一步会修改 pg 数据库

之前的批量测试，选取的是 202105 这个月的数据

```bash
# Batch test.
# 简单的格式化
fireball huantong_learning_proj.opt.batch_test:prep_batch_test \
    $HUANTONG_PROJ_DATA/data/batch-test/${BATCH_TEST_VERSION}.csv \
    $HUANTONG_PROJ_DATA/data/batch-test/${BATCH_TEST_VERSION}.pkl

# Remove invalid samples..
# 清洗掉明显不对的标注数据
fireball huantong_learning_proj.opt.batch_test:clean_batch_test_items \
    --postgres_config="$POSTGRES_CONFIG" \
    --tenant_id="$TENANT_ID" \
    --batch_test_items_pkl="$HUANTONG_PROJ_DATA/data/batch-test/${BATCH_TEST_VERSION}.pkl" \
    --output_pkl="$HUANTONG_PROJ_DATA/data/batch-test/${BATCH_TEST_VERSION}-clean.pkl" \
    --removed_output_folder="$HUANTONG_PROJ_DATA/data/batch-test/${BATCH_TEST_VERSION}-removed"

# Hide orgs/aliases/collations.
# 修改 pg 数据库，去掉有可能导致数据泄漏的部分，具体做法是得到批量测试数据的最小操作时间，然后从 pg 中删掉
# 所有操作时间 >= 这个时间的记录
fireball huantong_learning_proj.opt.batch_test:drop_pg_collations_after_opt_time_min \
    --postgres_config="$POSTGRES_CONFIG" \
    --tenant_id="$TENANT_ID" \
    --batch_test_items_pkl="$HUANTONG_PROJ_DATA/data/batch-test/${BATCH_TEST_VERSION}-clean.pkl"
```

下一步对 pg 的数据建 elasticsearch 索引

```bash
# 首先登陆 1.74，启动 elasticsearch 服务，建立索引
# 你可以换成新一点的 elasticsearch 版本，这个版本有安全问题
docker run -d --name huantong-elasticsearch-for-training \
    -p 9200:9200 \
    -e "discovery.type=single-node" \
    elasticsearch:7.8.1

# 然后在本地开发环境执行
# TEXTDOG_DATA 对应 textdog-data 文件夹
MODEL_FOLDER="/home/zhen24/projects/huantong/release/20220801/ltp/base"
RETRIEVER_CONFIG=$(
cat << EOF
{
    "ltp_tokenizer_config": {
#        "model_folder": "$TEXTDOG_DATA/token/ltp/base",
        "model_folder": "$MODEL_FOLDER",
        "device": "cpu",
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
        "es_port": "9200"
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
    --ltp_cache_jsl="${HUANTONG_PROJ_DATA}/data/prep/ltp_tokenizer_cache_jsls/ltp-cache.jsl"
```

## 生成训练用数据集

这里面再次清洗，得到明显不应该抽样的对照数据 id

```bash
# Generate data for training.
fireball huantong_learning_proj/opt/train_data.py:extract_related_collation_ids_from_batch_test \
    --postgres_config="$POSTGRES_CONFIG" \
    --tenant_id="$TENANT_ID" \
    --batch_test_items_pkl="$HUANTONG_PROJ_DATA/data/batch-test/${BATCH_TEST_VERSION}-clean.pkl" \
    --output_json="$HUANTONG_PROJ_DATA/data/train-data/${BATCH_TEST_VERSION}-collation-ids.json"

# Blacklist invalid collations.
fireball huantong_learning_proj/opt/train_data.py:build_invalid_collation_ids \
    --postgres_config="$POSTGRES_CONFIG" \
    --tenant_id="$TENANT_ID" \
    --output_json="$HUANTONG_PROJ_DATA/data/train-data/${DATABASE_VERSION}-invalid-collation-ids.json"

INVALID_COLLATION_IDS_JSONS=$(
cat << EOF
[
    "$HUANTONG_PROJ_DATA/data/train-data/${BATCH_TEST_VERSION}-collation-ids.json",
    "$HUANTONG_PROJ_DATA/data/train-data/${DATABASE_VERSION}-invalid-collation-ids.json",
]
EOF
)

fireball huantong_learning_proj/opt/train_data.py:collect_valid_collations \
    --postgres_config="$POSTGRES_CONFIG" \
    --tenant_id="$TENANT_ID" \
    --invalid_collation_ids_jsons="$INVALID_COLLATION_IDS_JSONS" \
    --output_json="$HUANTONG_PROJ_DATA/data/train-data/${DATABASE_VERSION}-valid-collation-id-and-timestamp.json"
```

然后在全量数据里抽样生成训练数据

```bash
fireball huantong_learning_proj/opt/train_data.py:generate_train_sample_to_folder \
    --retriever_config="$RETRIEVER_CONFIG" \
    --postgres_config="$POSTGRES_CONFIG" \
    --ac_level_one_json_file="$HUANTONG_PROJ_DATA/data/ac/ac_level_one.json" \
    --ac_level_two_json_file="$HUANTONG_PROJ_DATA/data/ac/ac_level_two.json" \
    --ac_level_three_json_file="$HUANTONG_PROJ_DATA/data/ac/ac_level_three.json" \
    --tenant_id="$TENANT_ID" \
    --collation_id_and_timestamp_json="$HUANTONG_PROJ_DATA/data/train-data/${DATABASE_VERSION}-valid-collation-id-and-timestamp.json" \
    --output_folder="$HUANTONG_PROJ_DATA/data/train-data/dataset-${TENANT_ID}" \
    --negative_size="20" \
    --chunk_size="5000" \
    --random_seed="13370"

fireball huantong_learning_proj/opt/train_data.py:generate_dev_set_from_ranked_items \
    --tenant_id="$TENANT_ID" \
    --postgres_config="$POSTGRES_CONFIG" \
    --ranked_items_pkl="$HUANTONG_PROJ_DATA/data/batch-test/${BATCH_TEST_VERSION}-ranked.pkl" \
    --output_folder="${SHARE_FOLDER}/train-data/${BATCH_TEST_VERSION}-dev" \
    --negative_size="1"
```

## 模型训练

阅读 `rankerdog/README.md`

## 执行端对端测试

在模型训练完之后，基于前面生成的批量测试数据集，执行批量测试

```bash
fireball huantong_learning_proj.opt.batch_test:run_batch_test_retrieval \
    --retriever_config="$RETRIEVER_CONFIG" \
    --postgres_config="$POSTGRES_CONFIG" \
    --ac_level_one_json_file="$HUANTONG_PROJ_DATA/data/ac/ac_level_one.json" \
    --ac_level_two_json_file="$HUANTONG_PROJ_DATA/data/ac/ac_level_two.json" \
    --ac_level_three_json_file="$HUANTONG_PROJ_DATA/data/ac/ac_level_three.json" \
    --tenant_id="$TENANT_ID" \
    --batch_test_items_pkl="$HUANTONG_PROJ_DATA/data/batch-test/${BATCH_TEST_VERSION}-clean.pkl" \
    --output_pkl="$HUANTONG_PROJ_DATA/data/batch-test/${BATCH_TEST_VERSION}-retrieved.pkl"

fireball huantong_learning_proj.opt.batch_test:inspect_retrieval_failure \
    --retrieved_items_pkl="$HUANTONG_PROJ_DATA/data/batch-test/${BATCH_TEST_VERSION}-retrieved.pkl" \
    --output_json="$HUANTONG_PROJ_DATA/data/batch-test/${BATCH_TEST_VERSION}-retrieved-places.json"

# Copy to share folder.
mkdir -p "/Volumes/share/dev/users/huntzhan/huantong/data/batch-test"
rsync -ah --progress \
    "$HUANTONG_PROJ_DATA/data/batch-test/${BATCH_TEST_VERSION}-retrieved.pkl" \
    "/Volumes/share/dev/users/huntzhan/huantong/data/batch-test"

# 1.74.
nvidia-docker run \
  --name huantong-proj-batch-test \
  --ipc=host \
  --tmpfs /tmpfs \
  -it \
  --rm \
  --user "$(id -u):$(id -g)" \
  -v "/mnt/md0/home/huntzhan/reco2m":"/MNVAI" \
  -v "/home/zhen24/projects/huantong/HUANTONG_PROJ_DATA":"/share" \
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

cd huantong-proj/
pip install -e .'[dev,job,job-private,pipeline]'

MODEL_VERSION='20210715'

RANKER_CONFIG=$(
cat << EOF
{
    "bert_pretrained_folder": "/share/data/ranker/bert-base-ernie-vocab-patched-inference",
    "state_dict_file": "/share/data/ranker/20210715/state_dict_epoch_73.pt",
    "device": "cuda",
    "classifier_top_k": 10,
    "classifier_thr": 0.5
}
EOF
)

fireball huantong_learning_proj.opt.batch_test:run_batch_test_ranking \
    --ranker_config="$RANKER_CONFIG" \
    --retrieved_items_pkl="/share/data/batch-test/${BATCH_TEST_VERSION}-retrieved.pkl" \
    --output_pkl="/share/data/batch-test/${BATCH_TEST_VERSION}-${MODEL_VERSION}-ranked.pkl"

# Copy back to local.
rsync -ah --progress \
    "/Volumes/share/dev/users/huntzhan/huantong/data/batch-test/${BATCH_TEST_VERSION}-${MODEL_VERSION}-ranked.pkl" \
    "$HUANTONG_PROJ_DATA/data/batch-test/${BATCH_TEST_VERSION}-${MODEL_VERSION}-ranked.pkl"

fireball huantong_learning_proj/opt/batch_test.py:run_batch_test_report \
    --ranked_items_pkl="$HUANTONG_PROJ_DATA/data/batch-test/${BATCH_TEST_VERSION}-${MODEL_VERSION}-ranked.pkl" \
    --output_folder="$HUANTONG_PROJ_DATA/data/batch-test/${BATCH_TEST_VERSION}-${MODEL_VERSION}-error"
```
