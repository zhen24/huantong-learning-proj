import contextlib
import logging
import os
import traceback
from datetime import datetime
from typing import Optional, Any, Union, Sequence

import attr
import psycopg2
from psycopg2.extras import DictCursor

logger = logging.getLogger(__name__)


@attr.s
class PostgresConfig:
    host: str = attr.ib()
    port: Union[int, str] = attr.ib()
    dbname: str = attr.ib()
    user: str = attr.ib()
    password: str = attr.ib()
    reuse_connection: bool = attr.ib()


def create_pg_conn(config: PostgresConfig):
    return psycopg2.connect(
        host=config.host,
        port=config.port,
        dbname=config.dbname,
        user=config.user,
        password=config.password,
    )


class PostgresState:
    conn: Any = None
    main_pid: Optional[int] = None
    sub_pid: Optional[int] = None


def create_or_reuse_pg_conn(config: PostgresConfig):
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
def create_pg_cursor(
    config: PostgresConfig,
    commit: bool = False,
    cursor_factory: Any = None,
    raise_error: bool = True,
):
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


def yield_pg_objs(cur: Any, obj_cls: Any):
    while True:
        dict_row = cur.fetchone()
        if dict_row is None:
            break
        yield obj_cls(**dict_row)


@attr.s
class PgOrganization:
    tenant_id: int = attr.ib()
    organization_id: int = attr.ib()
    name: str = attr.ib()
    province: str = attr.ib()
    city: str = attr.ib()
    county: str = attr.ib()
    address: str = attr.ib()
    come_from: str = attr.ib()
    created_at: datetime = attr.ib()
    updated_at: datetime = attr.ib()
    deleted_at: datetime = attr.ib()

    @property
    def deleted(self):
        return (self.deleted_at is not None)


def yield_pg_orgs(cur: Any):
    yield from yield_pg_objs(cur, PgOrganization)


def get_orgs_in_time_range(
    config: PostgresConfig,
    tenant_id: int,
    begin_time: datetime,
    end_time: datetime,
    not_deleted: bool = False,
):
    with create_pg_cursor(config, cursor_factory=DictCursor) as cur:
        cur.execute(
            '''
            SELECT *
            FROM public.organization
            WHERE tenant_id = %s AND updated_at BETWEEN %s AND %s
            ''' + ('AND deleted_at IS NULL' if not_deleted else ''),
            (tenant_id, begin_time, end_time),
        )
        yield from yield_pg_orgs(cur)


def get_orgs(
    config: PostgresConfig,
    tenant_id: int,
    organization_ids: Sequence[int],
    not_deleted: bool = False,
):
    assert isinstance(organization_ids, (tuple, list))

    if not organization_ids:
        return

    with create_pg_cursor(config, cursor_factory=DictCursor) as cur:
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


def get_orgs_by_name(
    config: PostgresConfig,
    tenant_id: int,
    name: str,
    not_deleted: bool = False,
):
    with create_pg_cursor(config, cursor_factory=DictCursor) as cur:
        cur.execute(
            '''
            SELECT *
            FROM public.organization
            WHERE tenant_id = %s AND name = %s
            ''' + ('AND deleted_at IS NULL' if not_deleted else ''),
            (tenant_id, name),
        )
        yield from yield_pg_orgs(cur)


def get_org(
    config: PostgresConfig,
    tenant_id: int,
    organization_id: int,
    not_deleted: bool = False,
):
    pg_orgs = list(get_orgs(config, tenant_id, (organization_id,), not_deleted=not_deleted))
    if not pg_orgs:
        return None
    else:
        return pg_orgs[0]


@attr.s
class PgAlias:
    tenant_id: int = attr.ib()
    id: int = attr.ib()
    organization_id: int = attr.ib()
    name: str = attr.ib()
    come_from: str = attr.ib()
    created_at: datetime = attr.ib()
    updated_at: datetime = attr.ib()
    deleted_at: datetime = attr.ib()

    @property
    def alias_id(self):
        return self.id

    @property
    def deleted(self):
        return (self.deleted_at is not None)


def yield_pg_aliases(cur: Any):
    yield from yield_pg_objs(cur, PgAlias)


def get_aliases_in_time_range(
    config: PostgresConfig,
    tenant_id: int,
    begin_time: datetime,
    end_time: datetime,
    not_deleted: bool = False,
):
    with create_pg_cursor(config, cursor_factory=DictCursor) as cur:
        cur.execute(
            '''
            SELECT *
            FROM public.alias
            WHERE tenant_id = %s AND updated_at BETWEEN %s AND %s
            ''' + ('AND deleted_at IS NULL' if not_deleted else ''),
            (tenant_id, begin_time, end_time),
        )
        yield from yield_pg_aliases(cur)


def get_aliases_by_alias_ids(
    config: PostgresConfig,
    tenant_id: int,
    alias_ids: Sequence[int],
    not_deleted: bool = False,
):
    assert isinstance(alias_ids, (tuple, list))

    if not alias_ids:
        return

    with create_pg_cursor(config, cursor_factory=DictCursor) as cur:
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


def get_aliases_by_organization_ids(
    config: PostgresConfig,
    tenant_id: int,
    organization_ids: Sequence[int],
    not_deleted: bool = False
):
    assert isinstance(organization_ids, (tuple, list))

    if not organization_ids:
        return

    with create_pg_cursor(config, cursor_factory=DictCursor) as cur:
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


def get_aliases_by_name(
    config: PostgresConfig,
    tenant_id: int,
    name: str,
    not_deleted: bool = False,
):
    with create_pg_cursor(config, cursor_factory=DictCursor) as cur:
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
    tenant_id: int = attr.ib()
    collation_id: int = attr.ib()
    upstream_id: int = attr.ib()
    downstream_id: int = attr.ib()
    query_name: str = attr.ib()
    query_name_tokens: str = attr.ib()
    request_at: datetime = attr.ib()
    response_at: datetime = attr.ib()
    come_from: str = attr.ib()
    created_at: datetime = attr.ib()
    updated_at: datetime = attr.ib()
    deleted_at: datetime = attr.ib()


def yield_pg_collations(cur: Any):
    yield from yield_pg_objs(cur, PgCollation)


def get_collations_in_time_range(
    config: PostgresConfig,
    tenant_id: int,
    begin_time: datetime,
    end_time: datetime,
    not_deleted: bool = False,
):
    with create_pg_cursor(config, cursor_factory=DictCursor) as cur:
        cur.execute(
            '''
            SELECT *
            FROM public.collation
            WHERE tenant_id = %s AND updated_at BETWEEN %s AND %s
            ''' + ('AND deleted_at IS NULL' if not_deleted else ''),
            (tenant_id, begin_time, end_time),
        )
        yield from yield_pg_collations(cur)


def get_collations(
    config: PostgresConfig,
    tenant_id: int,
    collation_ids: Sequence[int],
):
    assert isinstance(collation_ids, (tuple, list))

    if not collation_ids:
        return

    with create_pg_cursor(config, cursor_factory=DictCursor) as cur:
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


def get_collations_by_query_name(
    config: PostgresConfig,
    tenant_id: int,
    query_name: str,
    not_deleted: bool = False,
):
    with create_pg_cursor(config, cursor_factory=DictCursor) as cur:
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
    task_id: int = attr.ib()
    name: str = attr.ib()
    description: str = attr.ib()
    status: str = attr.ib()
    storage_path: str = attr.ib()
    created_at: datetime = attr.ib()
    finished_at: datetime = attr.ib()
    deleted_at: datetime = attr.ib()


def yield_pg_train(cur: Any):
    yield from yield_pg_objs(cur, PgTrain)


def get_train_tasks_by_status(config: PostgresConfig, status: str):
    with create_pg_cursor(config, cursor_factory=DictCursor) as cur:
        cur.execute(
            '''
            SELECT *
            FROM public.train
            WHERE status = %s
            ''',
            (status,),
        )
        yield from yield_pg_train(cur)


def get_train_tasks(config: PostgresConfig):
    with create_pg_cursor(config, cursor_factory=DictCursor) as cur:
        cur.execute('''
            SELECT *
            FROM public.train;
            ''',)
        yield from yield_pg_train(cur)


def set_train_task_status(config: PostgresConfig, task_id: int, status: str):
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


def set_train_task_storage_path(config: PostgresConfig, task_id: int, storage_path: str):
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


def get_train_by_task_ids(config: PostgresConfig, task_ids: Sequence[int]):
    assert isinstance(task_ids, (tuple, list))

    if not task_ids:
        return

    with create_pg_cursor(config, cursor_factory=DictCursor) as cur:
        cur.execute(
            '''
            SELECT *
            FROM public.train
            WHERE task_id in %s
            ''',
            (tuple(task_ids),),
        )
        yield from yield_pg_train(cur)


def delete_train_by_task_ids(config: PostgresConfig, task_ids: Sequence[int]):
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


def get_train_by_name(config: PostgresConfig, name: str):
    with create_pg_cursor(config, cursor_factory=DictCursor) as cur:
        cur.execute(
            '''
            SELECT *
            FROM public.train
            WHERE name = %s
            ''',
            (name,),
        )
        yield from yield_pg_train(cur)


def reset_using_train_task(config: PostgresConfig):
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
    task_id: int = attr.ib()
    name: str = attr.ib()
    description: str = attr.ib()
    model_source: str = attr.ib()
    status: str = attr.ib()
    created_at: datetime = attr.ib()
    finished_at: datetime = attr.ib()
    deleted_at: datetime = attr.ib()


def yield_pg_model(cur: Any):
    yield from yield_pg_objs(cur, PgModel)


def get_model_tasks_by_status(config: PostgresConfig, status: str):
    with create_pg_cursor(config, cursor_factory=DictCursor) as cur:
        cur.execute(
            '''
            SELECT *
            FROM public.model
            WHERE status = %s
            ''',
            (status,),
        )
        yield from yield_pg_model(cur)


def get_model_tasks(config: PostgresConfig):
    with create_pg_cursor(config, cursor_factory=DictCursor) as cur:
        cur.execute('''
            SELECT *
            FROM public.model
            ''',)
        yield from yield_pg_model(cur)


def set_model_task_status(config: PostgresConfig, task_id: int, status: str):
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


def get_model_by_task_ids(config: PostgresConfig, task_ids: Sequence[int]):
    assert isinstance(task_ids, (tuple, list))

    if not task_ids:
        return

    with create_pg_cursor(config, cursor_factory=DictCursor) as cur:
        cur.execute(
            '''
            SELECT *
            FROM public.model
            WHERE task_id in %s
            ''',
            (tuple(task_ids),),
        )
        yield from yield_pg_model(cur)


def delete_model_by_task_ids(config: PostgresConfig, task_ids: Sequence[int]):
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


def debug_local():
    config = PostgresConfig(
        host='0.0.0.0',
        port='30000',
        dbname='oval',
        user='postgres',
        password='Kt4C4TCHJ3',
        reuse_connection=True,
    )
    tenant_id = 1

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
