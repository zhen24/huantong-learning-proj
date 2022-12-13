#!/usr/bin/env bash
set -euo pipefail
trap "echo 'error: Script failed: see failed command above'" ERR

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
TENANT_ID='1'
DATABASE_VERSION='20221101'

# Organizations.
docker run \
    --rm -it \
    huantong_learning_proj/job:0.1.0 \
    fireball huantong_learning_proj.opt.prep_client_db_dump:pg_drop_indices \
    --postgres_config="$POSTGRES_CONFIG" \
    --table_name="public.organization"

docker run \
    --rm -it \
    huantong_learning_proj/job:0.1.0 \
    fireball huantong_learning_proj.opt.prep_client_db_dump:pg_delete_rows \
    --postgres_config="$POSTGRES_CONFIG" \
    --table_name="public.organization" \
    --tenant_id="$TENANT_ID"

docker run \
    --rm -it \
    -v "/home/mnvai/projects/huantong/learning/train-workspace/${DATABASE_VERSION}/db-csv":"/huantong-db-csv" \
    huantong_learning_proj/job:0.1.0 \
    fireball huantong_learning_proj.opt.prep_client_db_dump:upload_orgs_to_pg \
    --ovalmaster_csv="/huantong-db-csv/organization.csv" \
    --tenant_id="$TENANT_ID" \
    --postgres_config="$POSTGRES_CONFIG"

docker run \
    --rm -it \
    huantong_learning_proj/job:0.1.0 \
    fireball huantong_learning_proj.opt.prep_client_db_dump:pg_build_index_orgs \
    --postgres_config="$POSTGRES_CONFIG"

# Aliases.
docker run \
    --rm -it \
    huantong_learning_proj/job:0.1.0 \
    fireball huantong_learning_proj.opt.prep_client_db_dump:pg_drop_indices \
    --postgres_config="$POSTGRES_CONFIG" \
    --table_name="public.alias"

docker run \
    --rm -it \
    huantong_learning_proj/job:0.1.0 \
    fireball huantong_learning_proj.opt.prep_client_db_dump:pg_delete_rows \
    --postgres_config="$POSTGRES_CONFIG" \
    --table_name="public.alias" \
    --tenant_id="$TENANT_ID"

docker run \
    --rm -it \
    -v "/home/mnvai/projects/huantong/learning/train-workspace/${DATABASE_VERSION}/db-csv":"/huantong-db-csv" \
    huantong_learning_proj/job:0.1.0 \
    fireball huantong_learning_proj.opt.prep_client_db_dump:upload_aliases_to_pg \
    --orgalias_csv="/huantong-db-csv/alias.csv" \
    --ovalmaster_csv="/huantong-db-csv/organization.csv" \
    --tenant_id="$TENANT_ID" \
    --postgres_config="$POSTGRES_CONFIG"

docker run \
    --rm -it \
    huantong_learning_proj/job:0.1.0 \
    fireball huantong_learning_proj.opt.prep_client_db_dump:pg_build_index_aliases \
    --postgres_config="$POSTGRES_CONFIG"

# Collation.
docker run \
    --rm -it \
    huantong_learning_proj/job:0.1.0 \
    fireball huantong_learning_proj.opt.prep_client_db_dump:pg_drop_indices \
    --postgres_config="$POSTGRES_CONFIG" \
    --table_name="public.collation"

docker run \
    --rm -it \
    huantong_learning_proj/job:0.1.0 \
    fireball huantong_learning_proj.opt.prep_client_db_dump:pg_delete_rows \
    --postgres_config="$POSTGRES_CONFIG" \
    --table_name="public.collation" \
    --tenant_id="$TENANT_ID"

docker run \
    --rm -it \
    -v "/home/mnvai/projects/huantong/learning/train-workspace/${DATABASE_VERSION}/db-csv":"/huantong-db-csv" \
    huantong_learning_proj/job:0.1.0 \
    fireball huantong_learning_proj.opt.prep_client_db_dump:upload_collations_to_pg \
    --orgcodecollate_csv="/huantong-db-csv/collation.csv" \
    --ovalmaster_csv="/huantong-db-csv/organization.csv" \
    --tenant_id="$TENANT_ID" \
    --postgres_config="$POSTGRES_CONFIG"

docker run \
    --rm -it \
    huantong_learning_proj/job:0.1.0 \
    fireball huantong_learning_proj.opt.prep_client_db_dump:pg_build_index_collations \
    --postgres_config="$POSTGRES_CONFIG"
