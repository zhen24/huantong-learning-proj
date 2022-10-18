import contextlib
import logging
import os
import traceback

import attr
import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)


@attr.s
class PostgresConfig:
    host = attr.ib()
    port = attr.ib()
    dbname = attr.ib()
    user = attr.ib()
    password = attr.ib()
    reuse_connection = attr.ib()


def create_pg_conn(config):
    return psycopg2.connect(
        host=config.host,
        port=config.port,
        dbname=config.dbname,
        user=config.user,
        password=config.password,
    )


class PostgresState:
    conn = None
    main_pid = None
    sub_pid = None


def create_or_reuse_pg_conn(config):
    assert config.reuse_connection

    current_pid = os.getpid()
    if PostgresState.conn is None:
        # Init.
        PostgresState.conn = create_pg_conn(config)
        PostgresState.main_pid = current_pid

    if current_pid != PostgresState.main_pid and PostgresState.sub_pid is None:
        PostgresState.conn = create_pg_conn(config)
        PostgresState.sub_pid = current_pid

    conn = PostgresState.conn

    # Check connection.
    try:
        cur = conn.cursor()
        cur.execute('SELECT 1')
    except (psycopg2.OperationalError, psycopg2.InterfaceError):
        logger.warning('Broken connection detected, recreating...')
        PostgresState.conn = create_pg_conn(config)
        conn = PostgresState.conn

    return conn


@contextlib.contextmanager
def create_pg_cursor(config, commit=False, cursor_factory=None, raise_error=True):
    conn = None
    cur = None

    try:
        if config.reuse_connection:
            conn = create_or_reuse_pg_conn(config)
        else:
            conn = create_pg_conn(config)

        cur = conn.cursor(cursor_factory=cursor_factory)
        yield cur

    except Exception:
        logger.error(f'create_pg_cursor, ex={traceback.format_exc()}')
        if raise_error:
            raise

    finally:
        if conn is not None:
            if commit:
                conn.commit()
        if cur is not None:
            cur.close()
        if not config.reuse_connection and conn is not None:
            conn.close()


def get_last_finished_task_end_time(config):
    with create_pg_cursor(config) as cur:
        cur.execute(
            '''
            SELECT end_time
            FROM public.task
            WHERE status = 'finished'
            ORDER BY end_time DESC
            LIMIT 1
            '''
        )
        end_time = None
        result = cur.fetchone()
        if result:
            end_time = result[0]
        return end_time


def get_task_time_range(config, task_id):
    with create_pg_cursor(config) as cur:
        cur.execute(
            '''
            SELECT start_time, end_time
            FROM public.task
            WHERE id = %s
            ''',
            (task_id,),
        )
        start_time, end_time = cur.fetchone()
        return start_time, end_time


def set_task_to_finished(config, task_id):
    with create_pg_cursor(config, commit=True) as cur:
        cur.execute(
            '''
            UPDATE public.task
            SET status = 'finished'
            WHERE id = %s
            ''',
            (task_id,),
        )


def get_task_related_tenant_ids(config, begin_time, end_time):
    with create_pg_cursor(config) as cur:
        tenant_ids = set()

        for table in ['public.organization', 'public.alias']:
            cur.execute(
                f'''
                SELECT DISTINCT tenant_id
                FROM {table}
                WHERE %s <= updated_at AND updated_at <= %s
                ''',
                (begin_time, end_time),
            )
            for row in cur.fetchall():
                tenant_ids.add(row[0])

        return tenant_ids


def yield_pg_objs(cur, obj_cls):
    while True:
        dict_row = cur.fetchone()
        if dict_row is None:
            break
        yield obj_cls(**dict_row)


@attr.s
class PgOrganization:
    tenant_id = attr.ib()
    organization_id = attr.ib()
    name = attr.ib()
    province = attr.ib()
    city = attr.ib()
    county = attr.ib()
    address = attr.ib()
    come_from = attr.ib()
    created_at = attr.ib()
    updated_at = attr.ib()
    deleted_at = attr.ib()

    @property
    def deleted(self):
        return (self.deleted_at is not None)


def yield_pg_orgs(cur):
    yield from yield_pg_objs(cur, PgOrganization)


def get_orgs_in_time_range(config, tenant_id, begin_time, end_time, not_deleted=False):
    with create_pg_cursor(config, cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(
            '''
            SELECT *
            FROM public.organization
            WHERE tenant_id = %s AND updated_at BETWEEN %s AND %s
            ''' + ('AND deleted_at IS NULL' if not_deleted else ''),
            (tenant_id, begin_time, end_time),
        )
        yield from yield_pg_orgs(cur)


def get_orgs(config, tenant_id, organization_ids, not_deleted=False):
    assert isinstance(organization_ids, (tuple, list))

    if not organization_ids:
        return

    with create_pg_cursor(config, cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(
            '''
            SELECT *
            FROM public.organization
            WHERE tenant_id = %s AND organization_id IN %s
            ''' + ('AND deleted_at IS NULL' if not_deleted else ''),
            (
                tenant_id,
                tuple(organization_ids),
            ),
        )
        yield from yield_pg_orgs(cur)


def get_orgs_by_name(config, tenant_id, name, not_deleted=False):
    with create_pg_cursor(config, cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(
            '''
            SELECT *
            FROM public.organization
            WHERE tenant_id = %s AND name = %s
            ''' + ('AND deleted_at IS NULL' if not_deleted else ''),
            (tenant_id, name),
        )
        yield from yield_pg_orgs(cur)


def get_org(config, tenant_id, organization_id, not_deleted=False):
    pg_orgs = list(get_orgs(config, tenant_id, (organization_id,), not_deleted=not_deleted))
    if not pg_orgs:
        return None
    else:
        return pg_orgs[0]


@attr.s
class PgAlias:
    tenant_id = attr.ib()
    id = attr.ib()
    organization_id = attr.ib()
    name = attr.ib()
    come_from = attr.ib()
    created_at = attr.ib()
    updated_at = attr.ib()
    deleted_at = attr.ib()

    @property
    def alias_id(self):
        return self.id

    @property
    def deleted(self):
        return (self.deleted_at is not None)


def yield_pg_aliases(cur):
    yield from yield_pg_objs(cur, PgAlias)


def get_aliases_in_time_range(config, tenant_id, begin_time, end_time, not_deleted=False):
    with create_pg_cursor(config, cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(
            '''
            SELECT *
            FROM public.alias
            WHERE tenant_id = %s AND updated_at BETWEEN %s AND %s
            ''' + ('AND deleted_at IS NULL' if not_deleted else ''),
            (tenant_id, begin_time, end_time),
        )
        yield from yield_pg_aliases(cur)


def get_aliases_by_alias_ids(config, tenant_id, alias_ids, not_deleted=False):
    assert isinstance(alias_ids, (tuple, list))

    if not alias_ids:
        return

    with create_pg_cursor(config, cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(
            '''
            SELECT *
            FROM public.alias
            WHERE tenant_id = %s AND id IN %s
            ''' + ('AND deleted_at IS NULL' if not_deleted else ''),
            (
                tenant_id,
                tuple(alias_ids),
            ),
        )
        yield from yield_pg_aliases(cur)


def get_aliases_by_organization_ids(config, tenant_id, organization_ids, not_deleted=False):
    assert isinstance(organization_ids, (tuple, list))

    if not organization_ids:
        return

    with create_pg_cursor(config, cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(
            '''
            SELECT *
            FROM public.alias
            WHERE tenant_id = %s AND organization_id IN %s
            ''' + ('AND deleted_at IS NULL' if not_deleted else ''),
            (
                tenant_id,
                tuple(organization_ids),
            ),
        )
        yield from yield_pg_aliases(cur)


def get_aliases_by_name(config, tenant_id, name, not_deleted=False):
    with create_pg_cursor(config, cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(
            '''
            SELECT *
            FROM public.alias
            WHERE tenant_id = %s AND name = %s
            ''' + ('AND deleted_at IS NULL' if not_deleted else ''),
            (tenant_id, name),
        )
        yield from yield_pg_aliases(cur)


@attr.s
class PgCollation:
    tenant_id = attr.ib()
    collation_id = attr.ib()
    upstream_id = attr.ib()
    downstream_id = attr.ib()
    query_name = attr.ib()
    query_name_tokens = attr.ib()
    request_at = attr.ib()
    response_at = attr.ib()
    come_from = attr.ib()
    created_at = attr.ib()
    updated_at = attr.ib()
    deleted_at = attr.ib()


def yield_pg_collations(cur):
    yield from yield_pg_objs(cur, PgCollation)


def get_collations_in_time_range(config, tenant_id, begin_time, end_time, not_deleted=False):
    with create_pg_cursor(config, cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(
            '''
            SELECT *
            FROM public.collation
            WHERE tenant_id = %s AND updated_at BETWEEN %s AND %s
            ''' + ('AND deleted_at IS NULL' if not_deleted else ''),
            (tenant_id, begin_time, end_time),
        )
        yield from yield_pg_collations(cur)


def get_collations(config, tenant_id, collation_ids):
    assert isinstance(collation_ids, (tuple, list))

    if not collation_ids:
        return

    with create_pg_cursor(config, cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(
            '''
            SELECT *
            FROM public.collation
            WHERE tenant_id = %s AND collation_id IN %s
            ''',
            (
                tenant_id,
                tuple(collation_ids),
            ),
        )
        yield from yield_pg_collations(cur)


def get_collations_by_query_name(config, tenant_id, query_name, not_deleted=False):
    with create_pg_cursor(config, cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(
            '''
            SELECT *
            FROM public.collation
            WHERE tenant_id = %s AND query_name = %s
            ''' + ('AND deleted_at IS NULL' if not_deleted else ''),
            (tenant_id, query_name),
        )
        yield from yield_pg_collations(cur)


@attr.s
class PgTrain:
    task_id = attr.ib()
    name = attr.ib()
    description = attr.ib()
    status = attr.ib()
    storage_path = attr.ib()
    created_at = attr.ib()
    finished_at = attr.ib()
    deleted_at = attr.ib()


def yield_pg_train(cur):
    yield from yield_pg_objs(cur, PgTrain)


def get_train_tasks_by_status(config, status):
    with create_pg_cursor(config, cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(
            '''
            SELECT *
            FROM public.train
            WHERE status = %s
            ''',
            (status,),
        )
        yield from yield_pg_train(cur)


def get_train_tasks(config):
    with create_pg_cursor(config, cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute('''
            SELECT *
            FROM public.train;
            ''',)
        yield from yield_pg_train(cur)


def set_train_task_status(config, task_id, status):
    with create_pg_cursor(config, commit=True) as cur:
        cur.execute(
            '''
            UPDATE public.train
            SET status = %s
            WHERE task_id = %s
            ''',
            (
                status,
                task_id,
            ),
        )


def set_train_task_storage_path(config, task_id, storage_path):
    with create_pg_cursor(config, commit=True) as cur:
        cur.execute(
            '''
            UPDATE public.train
            SET storage_path = %s
            WHERE task_id = %s
            ''',
            (
                storage_path,
                task_id,
            ),
        )


def get_train_by_task_ids(config, task_ids):
    assert isinstance(task_ids, (tuple, list))

    if not task_ids:
        return

    with create_pg_cursor(config, cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(
            '''
            SELECT *
            FROM public.train
            WHERE task_id in %s
            ''',
            (tuple(task_ids),),
        )
        yield from yield_pg_train(cur)


def delete_train_by_task_ids(config, task_ids):
    assert isinstance(task_ids, (tuple, list))

    if not task_ids:
        return

    with create_pg_cursor(config, commit=True) as cur:
        cur.execute(
            '''
            DELETE FROM public.train
            WHERE task_id in %s
            ''',
            (tuple(task_ids),),
        )


def get_train_by_name(config, name):
    with create_pg_cursor(config, cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(
            '''
            SELECT *
            FROM public.train
            WHERE name = %s
            ''',
            (name,),
        )
        yield from yield_pg_train(cur)


def reset_using_train_task(config):
    with create_pg_cursor(config, commit=True) as cur:
        cur.execute(
            '''
            UPDATE public.train
            SET status = '训练完成'
            WHERE status = '使用中'
            ''',
        )


@attr.s
class PgModel:
    task_id = attr.ib()
    name = attr.ib()
    description = attr.ib()
    model_source = attr.ib()
    status = attr.ib()
    created_at = attr.ib()
    finished_at = attr.ib()
    deleted_at = attr.ib()


def yield_pg_model(cur):
    yield from yield_pg_objs(cur, PgModel)


def get_model_tasks_by_status(config, status):
    with create_pg_cursor(config, cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(
            '''
            SELECT *
            FROM public.model
            WHERE status = %s
            ''',
            (status,),
        )
        yield from yield_pg_model(cur)


def get_model_tasks(config):
    with create_pg_cursor(config, cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute('''
            SELECT *
            FROM public.model
            ''',)
        yield from yield_pg_model(cur)


def set_model_task_status(config, task_id, status):
    with create_pg_cursor(config, commit=True) as cur:
        cur.execute(
            '''
            UPDATE public.model
            SET status = %s
            WHERE task_id = %s
            ''',
            (
                status,
                task_id,
            ),
        )


def get_model_by_task_ids(config, task_ids):
    assert isinstance(task_ids, (tuple, list))

    if not task_ids:
        return

    with create_pg_cursor(config, cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(
            '''
            SELECT *
            FROM public.model
            WHERE task_id in %s
            ''',
            (tuple(task_ids),),
        )
        yield from yield_pg_model(cur)


def delete_model_by_task_ids(config, task_ids):
    assert isinstance(task_ids, (tuple, list))

    if not task_ids:
        return

    with create_pg_cursor(config, commit=True) as cur:
        cur.execute(
            '''
            DELETE FROM public.model
            WHERE task_id in %s
            ''',
            (tuple(task_ids),),
        )


def debug():
    config = PostgresConfig(
        host='192.168.1.172',
        port='30003',
        dbname='oval',
        user='postgres',
        password='Kt4C4TCHJ3',
        reuse_connection=True,
    )
    with create_pg_cursor(config) as cur:
        cur.execute('select distinct tenant_id FROM public.organization')
        print(cur.fetchall())

    print(get_last_finished_task_end_time(config))

    begin_time, end_time = get_task_time_range(config, 261)
    print(begin_time, end_time)
    print(get_task_related_tenant_ids(config, begin_time, end_time))

    print(get_org(config, 0, 106142606))

    idx = 0
    for pg_org in get_orgs_in_time_range(
        config,
        tenant_id=0,
        begin_time=begin_time,
        end_time=end_time,
        not_deleted=True,
    ):
        print(pg_org)
        idx += 1
        if idx >= 10:
            break

    set_task_to_finished(config, 334)


def debug_local():
    config = PostgresConfig(
        host='0.0.0.0',
        port='30000',
        dbname='oval',
        user='postgres',
        password='Kt4C4TCHJ3',
        reuse_connection=True,
    )
    tenant_id = '0'

    from datetime import datetime
    begin_time = datetime.fromisoformat('2021-05-10T00:00:00')
    end_time = datetime.fromisoformat('2021-05-11T00:00:00')

    pg_orgs = list(get_orgs_in_time_range(config, tenant_id, begin_time, end_time))
    print(len(pg_orgs))
    print(pg_orgs[0])
    print()

    org_ids = [pg_org.organization_id for pg_org in pg_orgs[:3]]
    pg_orgs = list(get_orgs(config, tenant_id, org_ids))
    print(org_ids, len(pg_orgs))
    print(pg_orgs)
    print()

    pg_aliases = list(get_aliases_in_time_range(config, tenant_id, begin_time, end_time))
    print(len(pg_aliases))
    print(pg_aliases[0])
    print()

    alias_ids = [pg_alias.alias_id for pg_alias in pg_aliases[:3]]
    pg_aliases = list(get_aliases_by_alias_ids(config, tenant_id, alias_ids))
    print(alias_ids, len(pg_aliases))
    print(pg_aliases)
    print()

    print('get_aliases_by_organization_ids')
    org_ids = [pg_alias.organization_id for pg_alias in pg_aliases]
    pg_aliases = list(get_aliases_by_organization_ids(config, tenant_id, org_ids))
    print(org_ids, len(pg_aliases))
    print(pg_aliases)
    print()

    pg_collations = list(get_collations_in_time_range(config, tenant_id, begin_time, end_time))
    print(len(pg_collations))
    print(pg_collations[0])
    print()

    collation_ids = [pg_collation.collation_id for pg_collation in pg_collations[:3]]
    pg_collations = list(get_collations(config, tenant_id, collation_ids))
    print(collation_ids, len(pg_collations))
    print(pg_collations)
    print()
