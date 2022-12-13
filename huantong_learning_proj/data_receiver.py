import logging
from datetime import datetime
from itertools import chain
from typing import Mapping, Sequence, MutableSet, Tuple

import psycopg2.extras

from huantong_learning_proj.db import (
    create_pg_cursor,
    get_train_tasks_by_status,
    get_train_by_task_ids,
    delete_train_by_task_ids,
    get_model_by_task_ids,
    delete_model_by_task_ids,
    get_train_by_name,
    PostgresConfig,
)

logger = logging.getLogger('huantong_learning_proj.job')


def get_org_and_alias_by_result(tenant_id: int, struct: Mapping):
    alias_names = struct['alias_names']
    assert isinstance(alias_names, list)

    org = (
        tenant_id,
        struct['downstream_id'],
        struct['downstream_name'],
        struct['province'],
        struct['city'],
        struct['country'],
        struct['address'],
    )
    aliases = {(
        alias['tenant_id'],
        alias['organization_id'],
        alias['name'],
    ) for alias in alias_names}
    return org, aliases


def delete_orgs_by_organization_ids(
    pg_config: PostgresConfig,
    tenant_id: int,
    organization_ids: Sequence[int],
):
    assert isinstance(organization_ids, (tuple, list))

    if not organization_ids:
        return

    with create_pg_cursor(pg_config, commit=True) as cur:
        cur.execute(
            'DELETE FROM public.organization WHERE tenant_id = %s AND organization_id IN %s;',
            (
                tenant_id,
                tuple(organization_ids),
            ),
        )


def delete_collations_by_collation_ids(pg_config: PostgresConfig, collation_ids: Sequence[int]):
    """ 经过内部探讨, 目前暂不支持多租户问题, 故 tenant_id 固定为 1 """
    assert isinstance(collation_ids, (tuple, list))

    if not collation_ids:
        return

    with create_pg_cursor(pg_config, commit=True) as cur:
        cur.execute(
            'DELETE FROM public.collation WHERE collation_id IN %s;',
            (tuple(collation_ids),),
        )


def insert_labeled_data_to_pg(
    pg_config: PostgresConfig,
    orgs: MutableSet,
    aliases: MutableSet,
    collation: Tuple,
):
    with create_pg_cursor(pg_config, commit=True) as cur:
        psycopg2.extras.execute_values(
            cur,
            '''
            INSERT INTO public.organization
            (
                tenant_id,
                organization_id,
                name,
                province,
                city,
                county,
                address
            )
            VALUES
            %s
            ''',
            orgs,
        )
    with create_pg_cursor(pg_config, commit=True) as cur:
        psycopg2.extras.execute_values(
            cur,
            '''
            INSERT INTO public.alias
            (
                tenant_id,
                organization_id,
                name
            )
            VALUES
            %s
            ''',
            aliases,
        )

    with create_pg_cursor(pg_config, commit=True) as cur:
        cur.execute(
            '''
            INSERT INTO public.collation
            (
                collation_id,
                tenant_id,
                upstream_id,
                downstream_id,
                query_name,
                query_name_tokens,
                request_at,
                response_at
            )
            VALUES
            %s
            ''',
            (tuple(collation),),
        )


def commit_label_data(pg_config: PostgresConfig, task_body: Mapping):
    logger.info('in commit_label_data.')

    label_id = task_body['label_id']
    tenant_id = task_body['tenant_id']
    query_info = task_body['query_info']
    request_at = task_body['request_at']
    response_at = task_body['response_at']
    match_results = task_body['match_results']
    checked_result = task_body['checked_result']
    labeled_results = task_body['labeled_results']
    assert isinstance(query_info, dict), 'query_info should be json!'
    assert query_info['status'] == 41, 'the status of organization should be 41!'
    assert isinstance(match_results, list), 'match_results should be array!'
    assert isinstance(checked_result, dict), 'checked_result should be json!'
    assert isinstance(labeled_results, list), 'labeled_result should be array!'

    request_at = datetime.strptime(request_at[:19], '%Y-%m-%d %H:%M:%S')
    response_at = datetime.strptime(response_at[:19], '%Y-%m-%d %H:%M:%S')
    upstream_id = query_info['upstream_id']
    query_name = query_info['query_name']
    query_name_tokens = query_info['query_name_tokens']
    aliases_from_query = query_info['alias_names']
    logger.info(f'Processing label_id={label_id}, upstream_id={upstream_id}, query={query_name}')

    hit_downstream_ids = {
        struct['downstream_id'] for struct in labeled_results if struct['status'] == 41
    }
    if len(hit_downstream_ids) != 1:
        return {
            'errorMessage':
                "Invalid labeled_results, upstream_id and downstream_id should be one-to-one!!!"
        }

    all_orgs = {(
        tenant_id,
        upstream_id,
        query_info['upstream_name'],
        query_info['province'],
        query_info['city'],
        query_info['country'],
        query_info['address'],
    )}
    all_aliases = {(
        alias['tenant_id'],
        alias['organization_id'],
        alias['name'],
    ) for alias in aliases_from_query}

    for struct in chain(match_results, [checked_result], labeled_results):
        if struct['status'] != 41:
            continue
        org, aliases = get_org_and_alias_by_result(tenant_id, struct)
        all_orgs.add(org)
        all_aliases.update(aliases)

    collation = (
        label_id,
        tenant_id,
        upstream_id,
        next(iter(hit_downstream_ids)),
        query_name,
        query_name_tokens,
        request_at,
        response_at,
    )

    delete_org_ids = [org[1] for org in all_orgs]
    delete_orgs_by_organization_ids(pg_config, tenant_id, delete_org_ids)
    delete_collations_by_collation_ids(pg_config, [collation[0]])

    logger.info(f'collation={collation}')
    insert_labeled_data_to_pg(pg_config, all_orgs, all_aliases, collation)
    logger.info('提交成功!')

    return {'msg': '提交成功!'}


def delete_label_data(pg_config: PostgresConfig, task_body: Mapping[str, Sequence[int]]):
    collation_ids = task_body['label_ids']
    logger.info(f'delete_collation_ids={collation_ids}')
    delete_collations_by_collation_ids(pg_config, collation_ids)
    logger.info('删除成功!')

    return {'msg': '删除成功!'}


def init_train_task(pg_config: PostgresConfig, original_model_path: str):
    initial_train_task = list(get_train_by_task_ids(pg_config, [-1]))
    if not initial_train_task:
        with create_pg_cursor(pg_config, commit=True) as cur:
            cur.execute(
                '''
                INSERT INTO public.train
                (
                    task_id,
                    name,
                    description,
                    status,
                    storage_path
                )
                VALUES
                %s
                ''',
                ((-1, '原初模型', '原始的初期模型(一期)', '训练完成', original_model_path),),
            )


def get_training_task(pg_config: PostgresConfig):
    train_tasks = []
    for train_task in chain(
        get_train_tasks_by_status(pg_config, '准备中'),
        get_train_tasks_by_status(pg_config, '训练中'),
    ):
        train_tasks.append(train_task)

    return train_tasks


def insert_train_task(pg_config: PostgresConfig, task_body: Mapping):
    train_id = task_body['task_id']
    train_task = (train_id, task_body['name'], task_body['description'], '准备中')

    with create_pg_cursor(pg_config, commit=True) as cur:
        cur.execute(
            '''
            INSERT INTO public.train
            (
                task_id,
                name,
                description,
                status
            )
            VALUES
            %s
            ''',
            (train_task,),
        )

    return train_id


def delete_train_task(
    pg_config: PostgresConfig,
    task_ids: Sequence[int],
    force: bool = False,
):
    train_ids = [task_id for task_id in task_ids if task_id != -1]  # 一期原初模型的task_id设为-1
    train_tasks = get_train_by_task_ids(pg_config, train_ids)
    deleted_ids = []
    not_deleted_ids = []
    for train_task in train_tasks:
        if not force and train_task.status in {'训练中', '使用中'}:  # 不可删的
            not_deleted_ids.append(train_task.task_id)
            continue
        deleted_ids.append(train_task.task_id)

    logger.info(f'deleted_ids={deleted_ids}')
    logger.info(f'not_deleted_ids={not_deleted_ids}')

    if deleted_ids:
        delete_train_by_task_ids(pg_config, deleted_ids)
    if not_deleted_ids:
        return {'errorMessage': f'处于训练中或使用中的训练任务不可删除, the task_ids={not_deleted_ids}'}

    return {'msg': '删除成功!'}


def insert_upgrade_task(pg_config: PostgresConfig, task_body: Mapping):
    train_tasks = list(get_train_by_name(pg_config, task_body['model_source']))
    logger.info(f'train_tasks={train_tasks}')
    assert train_tasks and train_tasks[0].status == '训练完成', '仅允许更新已训练完成的模型'

    model_id = task_body['task_id']
    model_task = (
        model_id, task_body['name'], task_body['description'], task_body['model_source'], '升级中'
    )
    logger.info(model_task)

    with create_pg_cursor(pg_config, commit=True) as cur:
        cur.execute(
            '''
            INSERT INTO public.model
            (
                task_id,
                name,
                description,
                model_source,
                status
            )
            VALUES
            %s
            ''',
            (model_task,),
        )

    return model_id, train_tasks[0]


def delete_upgrade_task(
    pg_config: PostgresConfig,
    task_ids: Sequence[int],
    force: bool = False,
):
    model_tasks = get_model_by_task_ids(pg_config, task_ids)
    deleted_ids = []
    not_deleted_ids = []
    for model_task in model_tasks:
        if not force and model_task.status == '升级中':  # 不可删的
            not_deleted_ids.append(model_task.task_id)
            continue
        deleted_ids.append(model_task.task_id)

    logger.info(f'deleted_ids={deleted_ids}')
    logger.info(f'not_deleted_ids={not_deleted_ids}')

    if deleted_ids:
        delete_model_by_task_ids(pg_config, deleted_ids)
    if not_deleted_ids:
        return {'errorMessage': f'处于升级中的升级任务不可删除, the task_ids={not_deleted_ids}'}

    return {'msg': '删除成功!'}
