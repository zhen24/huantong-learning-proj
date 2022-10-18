import logging
from datetime import datetime

import iolite as io
from textdog.lexicon.normalize import normalize
from tqdm import tqdm
import attr

from huantong_learning_proj.db import (
    PostgresConfig,
)
from retrievaldog.retriever import (
    Retriever,
    retriever_config_from_dict,
)
from huantong_learning_proj.opt.retriever import (
    retrieve_for_rank,
    load_retriever_resource,
)
from huantong_learning_proj.opt.batch_test import BatchTestItem

logger = logging.getLogger(__name__)


def dump_datetime_attr(obj):
    for key in list(obj.keys()):
        val = obj[key]
        if isinstance(val, datetime):
            obj[key] = val.isoformat()
        elif isinstance(val, dict):
            obj[key] = dump_datetime_attr(val)
        elif isinstance(val, (tuple, list)):
            obj[key] = [dump_datetime_attr(v) for v in val]
    return obj


def investigate_retrieval(
    retriever_config,
    postgres_config,
    ac_level_one_json_file,
    ac_level_two_json_file,
    ac_level_three_json_file,
    tenant_id,
    input_jsl,
    output_folder,
):
    logger.info('Creating retriever')
    retriever = Retriever(
        retriever_config_from_dict(retriever_config),
        resource_id=tenant_id,
        reset=False,
    )

    logger.info('Creating retriever_resource')
    retriever_resource = load_retriever_resource(
        postgres_config=PostgresConfig(**postgres_config),
        ac_level_one_json_file=ac_level_one_json_file,
        ac_level_two_json_file=ac_level_two_json_file,
        ac_level_three_json_file=ac_level_three_json_file,
    )

    items = list(io.read_json_lines(input_jsl))
    logger.info(f'{len(items)} items loaded.')

    queries = []
    for item in items:
        queries.append(normalize(item['query']))
    dynamic_ltp_tokenizer_cache = retriever.build_ltp_tokenizer_cache(queries)

    out_fd = io.folder(output_folder, reset=True)
    for idx, item in tqdm(enumerate(items)):
        logger.info(f'Retrieving {item}')
        fast_pass, upstream_pg_org, retrieved_orgs = retrieve_for_rank(
            retriever=retriever,
            retriever_resource=retriever_resource,
            tenant_id=tenant_id,
            upstream_id=item['upstream_id'],
            query=item['query'],
            dynamic_ltp_tokenizer_cache=dynamic_ltp_tokenizer_cache,
        )

        retrieved_orgs = list(map(attr.asdict, retrieved_orgs))
        for place, retrieved_org in enumerate(retrieved_orgs):
            retrieved_org['place'] = place

        output = dump_datetime_attr({
            'input': item,
            'upstream_pg_org': attr.asdict(upstream_pg_org),
            'fast_pass': fast_pass,
            'retrieved_orgs': retrieved_orgs,
        })

        io.write_json(out_fd / f'{idx}.json', output, indent=2, ensure_ascii=False)


def convert_input_jsl_to_batch_test_items(input_jsl, output_pkl):
    batch_test_items = []
    for idx, item in enumerate(io.read_json_lines(input_jsl)):
        batch_test_items.append(
            BatchTestItem(
                id=str(idx),
                upstream_id=item['upstream_id'],
                query=item['query'],
                downstream_id=None,
                opt_time=None,
                recorded_pred_downstream_id=None,
                recorded_update_time=None,
            )
        )
    io.write_joblib(output_pkl, batch_test_items)
