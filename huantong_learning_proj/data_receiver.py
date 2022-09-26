import logging
from itertools import chain
import psycopg2.extras

from huantong_learning_proj.db import create_pg_cursor

# logger = logging.getLogger('huantong_learning_proj.job')
logger = logging.getLogger('__main__')


def get_org_and_alias_by_result(tenant_id, struct):
    alias_names = struct['alias_names']
    assert isinstance(alias_names, list)

    return (
        tenant_id,
        struct['downstream_id'],
        struct['downstream_name'],
        struct['province'],
        struct['city'],
        struct['country'],
        struct['address'],
    ), {(
        alias['tenant_id'],
        alias['organization_id'],
        alias['name'],
    ) for alias in alias_names}


def delete_orgs_by_organization_ids(pg_config, tenant_id, organization_ids):
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


def delete_collations_by_collation_ids(pg_config, collation_ids):
    assert isinstance(collation_ids, (tuple, list))

    if not collation_ids:
        return

    with create_pg_cursor(pg_config, commit=True) as cur:
        cur.execute(
            'DELETE FROM public.collation WHERE collation_id IN %s;',
            (tuple(collation_ids),),
        )


def insert_labeled_data_to_pg(pg_config, orgs, aliases, collation):
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


def commit_label_data(pg_config, task_body):
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

    query_name = query_info['query_name']
    logger.info(f'Processing label_id={label_id}, tenant_id={tenant_id}, query_name={query_name}')

    hit_downstream_ids = {
        struct['downstream_id'] for struct in labeled_results if struct['status'] == 41
    }
    if len(hit_downstream_ids) != 1:
        return {
            'errorMessage':
                "Invalid labeled_results, upstream_id and downstream_id should be one-to-one!!!"
        }

    upstream_id = query_info['upstream_id']
    aliases_from_query = query_info['alias_names']
    query_name_tokens = query_info['query_name_tokens']
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


def delete_label_data(pg_config, task_body):
    collation_ids = task_body['label_ids']
    logger.info(f'delete_collation_ids={collation_ids}')
    delete_collations_by_collation_ids(pg_config, collation_ids)
    logger.info('删除成功!')

    return {'msg': '删除成功!'}


def get_training_task(pg_config, task_body):
    pass


def add_train_task(pg_config, task_body):
    pass
