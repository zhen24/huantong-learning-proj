#!/usr/bin/env bash
set -euo pipefail
trap "echo 'error: Script failed: see failed command above'" ERR

POSTGRES_CONFIG=$(
cat << 'EOF'
{
    'host': '10.190.6.16',
    'port': '30000',
    'dbname': 'oval',
    'user': 'postgres',
    'password': 'Kt4C4TCHJ3',
    'reuse_connection': True
}
EOF
)

# Organizations.
docker run \
    --rm -it \
    huantong_learning_proj/job:0.1.0 \
    fireball huantong_learning_proj.opt.prep_client_db_dump:pg_drop_indices \
    --postgres_config="$POSTGRES_CONFIG" \
    --table_name="public.organization"

# Aliases.
docker run \
    --rm -it \
    huantong_learning_proj/job:0.1.0 \
    fireball huantong_learning_proj.opt.prep_client_db_dump:pg_drop_indices \
    --postgres_config="$POSTGRES_CONFIG" \
    --table_name="public.alias"

# Collation.
docker run \
    --rm -it \
    huantong_learning_proj/job:0.1.0 \
    fireball huantong_learning_proj.opt.prep_client_db_dump:pg_drop_indices \
    --postgres_config="$POSTGRES_CONFIG" \
    --table_name="public.collation"
