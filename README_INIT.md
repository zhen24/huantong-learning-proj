# huantong_learning_proj

## pg初始化所需参数(按需修改)
```
RESOURCE="/home/mnvai/projects/huantong/learning"
DATABASE_VERSION='20210615'
POSTGRES_CONFIG=$(
cat << 'EOF'
{
    'host': '192.168.1.74',
    'port': '30004',
    #'port': '30005',
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
mkdir -p "${RESOURCE}/data/db-csv/${DATABASE_VERSION}"

fireball huantong_learning_proj.opt.prep_client_db_dump:convert_sql_to_csv \
    --table_name="ovalmaster" \
    --sql_file="/share/dev/users/huntzhan/huantong/data/db-dump/20210615/OVALMASTER.sql" \
    --output_csv="${RESOURCE}/data/db-csv/20210615/ovalmaster.csv"

fireball huantong_learning_proj.opt.prep_client_db_dump:convert_sql_to_csv \
    --table_name="orgalias" \
    --sql_file="/share/dev/users/huntzhan/huantong/data/db-dump/20210615/ORGALIAS.sql" \
    --output_csv="${RESOURCE}/data/db-csv/20210615/orgalias.csv"

fireball huantong_learning_proj.opt.prep_client_db_dump:convert_sql_to_csv \
    --table_name="orgcodecollate" \
    --sql_file="/share/dev/users/huntzhan/huantong/data/db-dump/20210615/ORGCODECOLLATE.sql" \
    --output_csv="${RESOURCE}/data/db-csv/20210615/orgcodecollate.csv"
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
# 在本地开发环境（非 1.74）
# 指向 1.74 pg
# 执行一下命令初始化 pg
# Tables.
fireball huantong_learning_proj.opt.prep_client_db_dump:pg_create_tables \
    --postgres_config="$POSTGRES_CONFIG"

fireball huantong_learning_proj.opt.prep_client_db_dump:pg_create_task_table \
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
    --ovalmaster_csv="${RESOURCE}/release/db-csv/${DATABASE_VERSION}/ovalmaster.csv" \
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
    --orgalias_csv="${RESOURCE}/release/db-csv/${DATABASE_VERSION}/orgalias.csv" \
    --ovalmaster_csv="${RESOURCE}/release/db-csv/${DATABASE_VERSION}/ovalmaster.csv" \
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
    --orgcodecollate_csv="${RESOURCE}/release/db-csv/${DATABASE_VERSION}/orgcodecollate.csv" \
    --ovalmaster_csv="${RESOURCE}/release/db-csv/${DATABASE_VERSION}/ovalmaster.csv" \
    --tenant_id="$TENANT_ID" \
    --postgres_config="$POSTGRES_CONFIG"

fireball huantong_learning_proj.opt.prep_client_db_dump:pg_build_index_collations \
    --postgres_config="$POSTGRES_CONFIG"
```

## 端对端批量测试数据集生成需要从全量数据中抽样、剥离一部分数据，制作端对端批量测试数据集，用于端对端批量测试指标计算注意，这一步会修改 pg 数据库之前的批量测试，选取的是 202105 这个月的数据
```bash
# 对 202105 这个月的标注数据(202105.csv)进行简单的格式化
fireball huantong_learning_proj.opt.batch_test:prep_batch_test \
    ${RESOURCE}/train-workspace/${DATABASE_VERSION}/prep/${BATCH_TEST_VERSION}.csv \
    ${RESOURCE}/train-workspace/${DATABASE_VERSION}/prep/${BATCH_TEST_VERSION}.pkl

# 清洗掉明显不对的标注数据(在匹配之后有变更的5月份数据、主数据或别名表中存在精确匹配名称但和标注数据下游id不一致的5月份数据、对照表中存在精确匹配名称但和标注数据上下游id不一致的5月份数据)
fireball huantong_learning_proj.opt.batch_test:clean_batch_test_items \
    --postgres_config="$POSTGRES_CONFIG" \
    --tenant_id="$TENANT_ID" \
    --batch_test_items_pkl="${RESOURCE}/train-workspace/${DATABASE_VERSION}/prep/${BATCH_TEST_VERSION}.pkl" \
    --output_pkl="${RESOURCE}/train-workspace/${DATABASE_VERSION}/prep/${BATCH_TEST_VERSION}-clean.pkl" \
    --removed_output_folder="${RESOURCE}/train-workspace/${DATABASE_VERSION}/prep/${BATCH_TEST_VERSION}-removed"

# Hide orgs/aliases/collations.
# 修改 pg 数据库，去掉有可能导致数据泄漏的部分，具体做法是得到5月份标注数据的最小操作时间(用于测试)，然后从 pg 中删掉
# 所有操作时间 >= 这个时间的记录
fireball huantong_learning_proj.opt.batch_test:drop_pg_collations_after_opt_time_min \
    --postgres_config="$POSTGRES_CONFIG" \
    --tenant_id="$TENANT_ID" \
    --batch_test_items_pkl="${RESOURCE}/train-workspace/${DATABASE_VERSION}/prep/${BATCH_TEST_VERSION}-clean.pkl"
```

## 这里面再次清洗，得到明显不应该抽样的对照数据 id
```bash
# Generate data for training.
# 和5月份标注数据中的query_name一致的对照数据
fireball huantong_learning_proj/opt/batch_test.py:extract_related_collation_ids_from_batch_test \
    --postgres_config="$POSTGRES_CONFIG" \
    --tenant_id="$TENANT_ID" \
    --batch_test_items_pkl="${RESOURCE}/train-workspace/${DATABASE_VERSION}/prep/${BATCH_TEST_VERSION}-clean.pkl" \
    --output_json="${RESOURCE}/train-workspace/${DATABASE_VERSION}/prep/${BATCH_TEST_VERSION}-collation-ids.json"

# Blacklist invalid collations.
# 同样的query_name但下游有冲突的对照数据
fireball huantong_learning_proj/opt/batch_test.py:build_invalid_collation_ids \
    --postgres_config="$POSTGRES_CONFIG" \
    --tenant_id="$TENANT_ID" \
    --output_json="${RESOURCE}/train-workspace/${DATABASE_VERSION}/prep/${DATABASE_VERSION}-invalid-collation-ids.json"

INVALID_COLLATION_IDS_JSONS=$(
cat << EOF
[
    "${RESOURCE}/train-workspace/${DATABASE_VERSION}/prep/${BATCH_TEST_VERSION}-collation-ids.json",
    "${RESOURCE}/train-workspace/${DATABASE_VERSION}/prep/${DATABASE_VERSION}-invalid-collation-ids.json",
]
EOF
)

# 生成有效的对照数据(在测试环境中,其实不妨直接从 pg 中删掉无效的对照数据)
fireball huantong_learning_proj/opt/batch_test.py:collect_valid_collations \
    --postgres_config="$POSTGRES_CONFIG" \
    --tenant_id="$TENANT_ID" \
    --invalid_collation_ids_jsons="$INVALID_COLLATION_IDS_JSONS" \
    --output_json="${RESOURCE}/train-workspace/${DATABASE_VERSION}/prep/${DATABASE_VERSION}-valid-collation-id-and-timestamp.json"

# 修改 pg 数据库，直接从 collation 中删掉无效的对照数据(1.74)
fireball huantong_learning_proj/opt/batch_test.py:drop_invalid_pg_collations \
    --postgres_config="$POSTGRES_CONFIG" \
    --tenant_id="$TENANT_ID" \
    --invalid_collation_ids_json="${RESOURCE}/train-workspace/${DATABASE_VERSION}/prep/${BATCH_TEST_VERSION}-collation-ids.json"

fireball huantong_learning_proj/opt/batch_test.py:drop_invalid_pg_collations \
    --postgres_config="$POSTGRES_CONFIG" \
    --tenant_id="$TENANT_ID" \
    --invalid_collation_ids_json="${RESOURCE}/train-workspace/${DATABASE_VERSION}/prep/${DATABASE_VERSION}-invalid-collation-ids.json"
```

## 生成清洗后的历史运营数据对应快照(下次利用该批数据初始化数据库时,直接导入快照数据即可,不必再次清洗)
```bash
# Backup postgres.
docker run \
    --name huantong-postgres-cli \
    --rm -it \
    --network host \
    -v "$(pwd)":"/data" \
    -e POSTGRES_HOST='0.0.0.0' \
    -e POSTGRES_PORT='30004' \
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
        -c "\copy (SELECT * FROM public.${TABLE} WHERE tenant_id=${TENANT_ID}) To '${PG_DUMP_FOLDER}/${TABLE}.csv' With CSV HEADER" \
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
    -e POSTGRES_PORT='30004' \
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
        -c "\copy public.${TABLE} FROM '${PG_DUMP_FOLDER}/${TABLE}.csv' With CSV HEADER" \
        oval
done

exit
```

## 拷贝到 /home/mnvai/projects/huantong/learning/release/db-csv/20210615 下作为初始化数据库时的数据快照
```bash
cd REQUIRED/
mv organization.csv alias.csv collation.csv /home/mnvai/projects/huantong/learning/release/db-csv/20210615

# 下次初始化数据库时直接使用清洗后的快照数据，可跳过初始化时的清洗
cd /home/mnvai/projects/huantong/learning/release/db-csv/20210615
mv ovalmaster.csv orgalias.csv orgcodecollate.csv bak/
mv organization.csv ovalmaster.csv
mv alias.csv orgalias.csv
mv collation.csv orgcodecollate.csv
```
