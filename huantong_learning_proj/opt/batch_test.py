import logging
from collections import defaultdict
import json
import random
from datetime import datetime
from typing import Optional, Sequence, Mapping, Dict, Union

import attr
from dateutil.parser import parse as dateutil_parse
import iolite as io
from textdog.lexicon.normalize import normalize
from tqdm import tqdm
import requests

from huantong_learning_proj.db import (
    PostgresConfig,
    get_org,
    get_orgs_by_name,
    get_orgs,
    get_aliases_by_name,
    get_aliases_by_organization_ids,
    get_collations_by_query_name,
    create_pg_cursor,
    get_collations_in_time_range,
    PgOrganization,
)
from retrievaldog.retriever import (
    Retriever,
    retriever_config_from_dict,
)

from huantong_learning_proj.opt.prep_client_db_dump import group_by
from huantong_learning_proj.opt.retriever import (
    retrieve_for_rank,
    load_retriever_resource,
    RetrievedOrg,
)
from huantong_learning_proj.opt.ranker import (
    RankerConfig,
    load_ranker_model,
    ranker_predict_20210715,
)

logger = logging.getLogger(__name__)


@attr.s
class BatchTestItem:
    id: int = attr.ib()
    upstream_id: int = attr.ib()
    query_name: str = attr.ib()
    downstream_id: int = attr.ib()
    opt_time: Optional[datetime] = attr.ib()
    recorded_pred_downstream_id: Optional[int] = attr.ib()
    recorded_update_time: Optional[datetime] = attr.ib()
    query_name_tokens: str = attr.ib(default='')


def prep_batch_test(batch_test_csv: str, output_pkl: str):
    batch_test_items = []

    for struct in io.read_csv_lines(
        batch_test_csv,
        to_dict=True,
        skip_header=True,
    ):
        if not any(struct.values()):
            continue

        recorded_pred_downstream_id = None
        if struct['下游ID'] != '\\N':
            int(struct['下游ID'])
            recorded_pred_downstream_id = int(struct['下游ID'])

        opt_time = dateutil_parse(struct['匹配时间'])
        recorded_update_time = dateutil_parse(struct['主数据最后时间'])
        if struct['别名表时间'] != '\\N':
            recorded_update_time = max(recorded_update_time, dateutil_parse(struct['别名表时间']))

        batch_test_items.append(
            BatchTestItem(
                id=int(struct['ORGNAMECOLLATEID']),
                upstream_id=int(struct['上游ID']),
                query_name=struct['查询名称'],
                downstream_id=int(struct['正确ID']),
                opt_time=opt_time,
                recorded_pred_downstream_id=recorded_pred_downstream_id,
                recorded_update_time=recorded_update_time,
            )
        )

    io.write_joblib(output_pkl, batch_test_items)


def clean_related_to_corrupt_update(
    config: PostgresConfig,
    tenant_id: int,
    batch_test_items: Sequence[BatchTestItem],
):
    opt_times = []
    org_ids = set()
    for batch_test_item in batch_test_items:
        # Consider both upstream & downstream.
        org_ids.add(batch_test_item.upstream_id)
        org_ids.add(batch_test_item.downstream_id)
        opt_times.append(batch_test_item.opt_time)

    opt_time_min = min(opt_times)

    id_to_pg_org = {}
    for pg_org in get_orgs(
        config,
        tenant_id,
        list(org_ids),
        not_deleted=True,
    ):
        id_to_pg_org[pg_org.organization_id] = pg_org

    id_to_pg_aliases = defaultdict(list)
    for pg_alias in get_aliases_by_organization_ids(
        config,
        tenant_id,
        list(org_ids),
        not_deleted=True,
    ):
        id_to_pg_aliases[pg_alias.organization_id].append(pg_alias)

    kept = []
    removed = []

    for batch_test_item in tqdm(batch_test_items):
        updated_ats = []
        updated_ats.append(id_to_pg_org[batch_test_item.upstream_id].updated_at)
        for pg_alias in id_to_pg_aliases.get(batch_test_item.upstream_id, ()):
            updated_ats.append(pg_alias.updated_at)

        updated_ats.append(id_to_pg_org[batch_test_item.downstream_id].updated_at)
        for pg_alias in id_to_pg_aliases.get(batch_test_item.downstream_id, ()):
            updated_ats.append(pg_alias.updated_at)

        updated_at_max = max(updated_ats)
        # Patch timezone.
        updated_at_max = updated_at_max.replace(tzinfo=None)
        if updated_at_max > opt_time_min:
            removed.append((batch_test_item, updated_at_max, opt_time_min))
        else:
            kept.append(batch_test_item)

    logger.info(f'clean_related_to_corrupt_update: kept={len(kept)}, removed={len(removed)}')
    return kept, removed


def clean_exact_match_name_but_incorrect(
    config: PostgresConfig,
    tenant_id: int,
    batch_test_items: Sequence[BatchTestItem],
    strict: bool = False,
):
    kept = []
    removed = []

    for batch_test_item in tqdm(batch_test_items):
        pg_orgs = list(
            get_orgs_by_name(
                config,
                tenant_id,
                batch_test_item.query_name,
                not_deleted=True,
            )
        )
        pg_aliases = list(
            get_aliases_by_name(
                config,
                tenant_id,
                batch_test_item.query_name,
                not_deleted=True,
            )
        )

        if not (pg_orgs or pg_aliases):
            keep = True
        else:
            hit_pg_org_correct = False
            hit_pg_org_incorrect = False
            for pg_org in pg_orgs:
                if pg_org.organization_id == batch_test_item.downstream_id:
                    hit_pg_org_correct = True
                else:
                    hit_pg_org_incorrect = True

            hit_pg_alias_correct = False
            hit_pg_alias_incorrect = False
            for pg_alias in pg_aliases:
                if pg_alias.organization_id == batch_test_item.downstream_id:
                    hit_pg_alias_correct = True
                else:
                    hit_pg_alias_incorrect = True

            if strict:
                keep = ((pg_orgs and hit_pg_org_correct and not hit_pg_org_incorrect)
                        or (pg_aliases and hit_pg_alias_correct and not hit_pg_alias_incorrect))
            else:
                keep = hit_pg_org_correct or hit_pg_alias_correct

        if keep:
            kept.append(batch_test_item)
        else:
            removed.append((batch_test_item, pg_orgs, pg_aliases))

    logger.info(
        f'clean_exact_match_name_but_incorrect(strict={strict}): '
        f'kept={len(kept)}, removed={len(removed)}'
    )
    return kept, removed


def clean_hit_upstream_id_and_query_but_incorrect(
    config: PostgresConfig,
    tenant_id: int,
    batch_test_items: Sequence[BatchTestItem],
    strict: bool = False,
):
    kept = []
    removed = []

    for batch_test_item in tqdm(batch_test_items):
        pg_collations = list(
            get_collations_by_query_name(
                config,
                tenant_id,
                batch_test_item.query_name,
                not_deleted=True,
            )
        )

        if not pg_collations:
            keep = True
        else:
            hit_correct = False
            hit_incorrect = False
            for pg_collation in pg_collations:
                if pg_collation.upstream_id != batch_test_item.upstream_id:
                    continue
                if pg_collation.downstream_id == batch_test_item.downstream_id:
                    hit_correct = True
                else:
                    hit_incorrect = True

            if strict:
                keep = hit_correct and not hit_incorrect
            else:
                keep = hit_correct

        if keep:
            kept.append(batch_test_item)
        else:
            removed.append((batch_test_item, pg_collations))

    logger.info(
        f'clean_hit_upstream_id_and_query_but_incorrect(strict={strict}): '
        f'kept={len(kept)}, removed={len(removed)}'
    )
    return kept, removed


def clean_batch_test_items(
    postgres_config: Mapping,
    tenant_id: int,
    batch_test_items_pkl: str,
    output_pkl: str,
    removed_output_folder: str,
):
    config = PostgresConfig(**postgres_config)
    batch_test_items = io.read_joblib(batch_test_items_pkl)

    removed_out_fd = io.folder(removed_output_folder, touch=True)

    batch_test_items, removed = clean_related_to_corrupt_update(
        config,
        tenant_id,
        batch_test_items,
    )
    io.write_joblib(removed_out_fd / 'clean_related_to_corrupt_update.pkl', removed)

    batch_test_items, removed = clean_exact_match_name_but_incorrect(
        config,
        tenant_id,
        batch_test_items,
        strict=True,
    )
    io.write_joblib(removed_out_fd / 'clean_exact_match_name_but_incorrect.pkl', removed)

    batch_test_items, removed = clean_hit_upstream_id_and_query_but_incorrect(
        config,
        tenant_id,
        batch_test_items,
        strict=True,
    )
    io.write_joblib(removed_out_fd / 'clean_hit_upstream_id_and_query_but_incorrect.pkl', removed)

    io.write_joblib(output_pkl, batch_test_items)


def drop_pg_collations_after_opt_time_min(
    postgres_config: Mapping,
    tenant_id: int,
    batch_test_items_pkl: str,
):
    config = PostgresConfig(**postgres_config)

    batch_test_items = io.read_joblib(batch_test_items_pkl)

    opt_times = []
    org_ids = set()
    for batch_test_item in batch_test_items:
        org_ids.add(batch_test_item.downstream_id)
        opt_times.append(batch_test_item.opt_time)

    opt_time_min = min(opt_times)

    logging.info(f'Dropping orgs after {opt_time_min}')
    with create_pg_cursor(config, commit=True) as cur:
        cur.execute(
            '''
            DELETE FROM public.organization
            WHERE tenant_id = %s AND updated_at >= %s
            ''', (tenant_id, opt_time_min)
        )

    logging.info(f'Dropping aliases after {opt_time_min}')
    with create_pg_cursor(config, commit=True) as cur:
        cur.execute(
            '''
            DELETE FROM public.alias
            WHERE tenant_id = %s AND updated_at >= %s
            ''', (tenant_id, opt_time_min)
        )

    logging.info(f'Dropping collations after {opt_time_min}')
    with create_pg_cursor(config, commit=True) as cur:
        cur.execute(
            '''
            DELETE FROM public.collation
            WHERE tenant_id = %s AND updated_at >= %s
            ''', (tenant_id, opt_time_min)
        )


def extract_related_collation_ids_from_batch_test(
    postgres_config: Mapping,
    tenant_id: int,
    batch_test_items_pkl: str,
    output_json: str,
):
    config = PostgresConfig(**postgres_config)
    batch_test_items = io.read_joblib(batch_test_items_pkl)

    collation_ids = set()
    for batch_test_item in batch_test_items:
        collation_ids.add(batch_test_item.id)
        for pg_collation in get_collations_by_query_name(
            config,
            tenant_id,
            batch_test_item.query_name,
        ):
            if pg_collation.come_from == '历史运营':
                collation_ids.add(pg_collation.collation_id)

    io.write_json(output_json, list(collation_ids), indent=2)


def build_invalid_collation_ids(
    postgres_config: Mapping,
    tenant_id: int,
    output_json: str,
):
    config = PostgresConfig(**postgres_config)

    match_pair_to_pg_collations = defaultdict(list)
    for pg_collation in tqdm(
        get_collations_in_time_range(
            config,
            tenant_id,
            datetime.min,
            datetime.max,
            not_deleted=True,
        )
    ):
        match_pair = (pg_collation.upstream_id, pg_collation.query_name)
        match_pair_to_pg_collations[match_pair].append(pg_collation)

    collation_ids = set()
    for pg_collations in match_pair_to_pg_collations.values():
        invalid = False
        downstream_ids = set()
        for pg_collation in pg_collations:
            if pg_collation.come_from == '人工标注':
                downstream_ids.add(pg_collation.downstream_id)

        if len(downstream_ids) > 1:
            invalid = True
        elif len(downstream_ids) == 0:
            if len({pg_collation.downstream_id for pg_collation in pg_collations}) > 1:
                invalid = True
        else:
            hit_downstream_id = next(iter(downstream_ids))
            for pg_collation in pg_collations:
                if pg_collation.downstream_id != hit_downstream_id:
                    collation_ids.add(pg_collation.collation_id)

        if invalid:
            for pg_collation in pg_collations:
                collation_ids.add(pg_collation.collation_id)

    io.folder(io.file(output_json).parent, touch=True)
    io.write_json(output_json, list(collation_ids), indent=2)


def collect_valid_collations(
    postgres_config: Mapping,
    tenant_id: int,
    invalid_collation_ids_jsons: Sequence[str],
    output_json: str,
):
    invalid_collation_ids = set()
    for json_file in invalid_collation_ids_jsons:
        invalid_collation_ids.update(io.read_json(json_file))

    config = PostgresConfig(**postgres_config)

    collation_id_and_timestamp = []

    for pg_collation in tqdm(
        get_collations_in_time_range(
            config,
            tenant_id,
            datetime.min,
            datetime.max,
            not_deleted=True,
        )
    ):
        if pg_collation.collation_id in invalid_collation_ids:
            continue

        collation_id_and_timestamp.append((
            pg_collation.collation_id,
            pg_collation.created_at.timestamp(),
        ))

    io.write_json(output_json, collation_id_and_timestamp, indent=2)


def stat_collation_id_and_timestamp(collation_id_and_timestamp_json: str):
    collation_id_and_timestamp = io.read_json(collation_id_and_timestamp_json)
    logger.info(f'total={len(collation_id_and_timestamp)}')

    dt = datetime.fromisoformat('2018-01-01T00:00:00.000000')
    filtered = list(
        filter(lambda p: datetime.fromtimestamp(p[1]) >= dt, collation_id_and_timestamp)
    )
    logger.info(f'filtered={len(filtered)}, r={len(filtered) / len(collation_id_and_timestamp)}')


def drop_invalid_pg_collations(
    postgres_config: Mapping,
    tenant_id: int,
    invalid_collation_ids_json: str,
):
    invalid_collation_ids = set()
    for collation_id in io.read_json(invalid_collation_ids_json):
        invalid_collation_ids.add(collation_id)
    logger.info(f'num_invalid_collation_ids={len(invalid_collation_ids)}')

    if invalid_collation_ids:
        config = PostgresConfig(**postgres_config)
        with create_pg_cursor(config, commit=True) as cur:
            cur.execute(
                '''
                DELETE FROM public.collation
                WHERE tenant_id = %s AND collation_id IN %s
                ''', (tenant_id, tuple(invalid_collation_ids))
            )
    logging.info(f'Dropped invalid collations by {invalid_collation_ids_json}')


def extract_dev_collations(
    postgres_config: Mapping,
    tenant_id: int,
    output_pkl: str,
    dev_cnt: int = 10000,
    random_seed: int = 13370,
):
    config = PostgresConfig(**postgres_config)

    history_collations = []
    label_collations = []
    for pg_collation in tqdm(
        get_collations_in_time_range(
            config,
            tenant_id,
            datetime.min,
            datetime.max,
            not_deleted=True,
        )
    ):
        if pg_collation.come_from == '历史运营':
            history_collations.append(pg_collation)
        else:
            label_collations.append(pg_collation)

    dev_collations = []
    random.seed(random_seed)
    random.shuffle(history_collations)
    for history_collations in group_by(history_collations, 1000):
        kept, _ = clean_exact_match_name_but_incorrect(
            config,
            tenant_id,
            history_collations,
            strict=True,
        )
        dev_collations.extend(kept)
        if len(dev_collations) >= dev_cnt:
            dev_collations = dev_collations[:dev_cnt]
            break
    extracted_cnt = len(dev_collations)
    logger.info(f'{extracted_cnt} dev_collations extracted from 历史运营')

    if len(dev_collations) < dev_cnt:
        random.seed(random_seed)
        random.shuffle(label_collations)
        for label_collations in group_by(label_collations, 1000):
            kept, _ = clean_exact_match_name_but_incorrect(
                config,
                tenant_id,
                label_collations,
                strict=True,
            )
            dev_collations.extend(kept)
            if len(dev_collations) >= dev_cnt:
                dev_collations = dev_collations[:dev_cnt]
                break
        logger.info(f'{len(dev_collations) - extracted_cnt} dev_collations extracted from 人工标注')

    batch_test_items = []
    for dev_item in dev_collations:
        batch_test_items.append(
            BatchTestItem(
                id=dev_item.collation_id,
                upstream_id=dev_item.upstream_id,
                query_name=dev_item.query_name,
                downstream_id=dev_item.downstream_id,
                opt_time=None,
                recorded_pred_downstream_id=None,
                recorded_update_time=None,
                query_name_tokens=dev_item.query_name_tokens,
            )
        )
    io.write_joblib(output_pkl, batch_test_items)


def extract_train_collation_ids(
    postgres_config: Mapping,
    tenant_id: int,
    output_json: str,
    train_cnt: int = 1500000,
    random_seed: int = 13370,
):
    config = PostgresConfig(**postgres_config)

    history_collations = []
    label_collations = []
    for pg_collation in tqdm(
        get_collations_in_time_range(
            config,
            tenant_id,
            datetime.min,
            datetime.max,
            not_deleted=True,
        )
    ):
        if pg_collation.come_from == '历史运营':
            history_collations.append(pg_collation)
        else:
            label_collations.append(pg_collation)

    train_collations = list(
        sorted(label_collations, key=lambda collation: collation.updated_at, reverse=True)
    )[:train_cnt]
    extracted_cnt = len(train_collations)
    logger.info(f'{extracted_cnt} train_collations extracted from 人工标注')

    if len(train_collations) < train_cnt:
        random.seed(random_seed)
        random.shuffle(history_collations)
        train_collations.extend(history_collations[:train_cnt - extracted_cnt])
        logger.info(f'{len(train_collations) - extracted_cnt} train_collations extracted from 历史运营')

    collation_id_and_timestamp = [(
        train_collation.collation_id,
        train_collation.created_at.timestamp(),
    ) for train_collation in train_collations]

    io.write_json(output_json, collation_id_and_timestamp, indent=2)


@attr.s
class BatchTestRetrievedItem:
    batch_test_item: BatchTestItem = attr.ib()
    fast_pass: bool = attr.ib()
    upstream_pg_org: PgOrganization = attr.ib()
    retrieved_orgs: Sequence[RetrievedOrg] = attr.ib()


def run_batch_test_retrieval(
    retriever_config: Mapping,
    postgres_config: Mapping,
    ac_level_one_json_file: str,
    ac_level_two_json_file: str,
    ac_level_three_json_file: str,
    tenant_id: int,
    batch_test_items_pkl: str,
    output_pkl: str,
    profiler_limit: Optional[int] = None,
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

    batch_test_items = io.read_joblib(batch_test_items_pkl)
    logger.info(f'{len(batch_test_items)} batch_test_items loaded.')

    if profiler_limit:
        random.shuffle(batch_test_items)
        batch_test_items = batch_test_items[:profiler_limit]

    queries = []
    for batch_test_item in batch_test_items:
        queries.append(normalize(batch_test_item.query_name))
    dynamic_ltp_tokenizer_cache = retriever.build_ltp_tokenizer_cache(queries)

    retrieved_items = []
    for batch_test_item in tqdm(batch_test_items):
        logger.info(f'Retrieving {batch_test_item}')
        fast_pass, upstream_pg_org, retrieved_orgs = retrieve_for_rank(
            retriever=retriever,
            retriever_resource=retriever_resource,
            tenant_id=tenant_id,
            upstream_id=batch_test_item.upstream_id,
            query=batch_test_item.query_name,
            dynamic_ltp_tokenizer_cache=dynamic_ltp_tokenizer_cache,
        )
        retrieved_items.append(
            BatchTestRetrievedItem(
                batch_test_item=batch_test_item,
                fast_pass=fast_pass,
                upstream_pg_org=upstream_pg_org,
                retrieved_orgs=retrieved_orgs,
            )
        )
    io.write_joblib(output_pkl, retrieved_items)


def inspect_retrieval_failure(retrieved_items_pkl: str, output_json: str):
    retrieved_items = io.read_joblib(retrieved_items_pkl)

    num_fast_pass = 0
    num_fast_pass_correct = 0
    num_recalled = 0
    total = 0

    places = []

    for retrieved_item in retrieved_items:
        total += 1

        batch_test_item = retrieved_item.batch_test_item
        retrieved_orgs = retrieved_item.retrieved_orgs

        if retrieved_item.fast_pass:
            num_fast_pass += 1
            if retrieved_orgs[0].organization_id == batch_test_item.downstream_id:
                num_fast_pass_correct += 1

        place = None
        for idx, retrieved_org in enumerate(retrieved_orgs):
            if retrieved_org.organization_id == batch_test_item.downstream_id:
                place = idx
                break
        if place is not None:
            num_recalled += 1
            places.append(place)

    logger.info(f'num_fast_pass={num_fast_pass}, num_fast_pass_correct={num_fast_pass_correct}')
    logger.info(
        f'recalled_cnt={num_recalled}, total={total}, recalled_ratio={num_recalled / total}'
    )

    io.write_json(output_json, places, indent=2)


@attr.s
class BatchTestRankedItem:
    retrieved_item: BatchTestRetrievedItem = attr.ib()
    sorted_retrieved_orgs: Optional[Sequence[RetrievedOrg]] = attr.ib()
    sorted_logits: Optional[Sequence[float]] = attr.ib()


def run_batch_test_ranking(
    ranker_config: Mapping,
    retrieved_items_pkl: str,
    output_pkl: str,
):
    config = RankerConfig(**ranker_config)
    ranker_model, bert_tokenizer = load_ranker_model(config)

    retrieved_items = io.read_joblib(retrieved_items_pkl)

    ranked_items = []
    for retrieved_item in tqdm(retrieved_items):
        batch_test_item = retrieved_item.batch_test_item
        upstream_pg_org = retrieved_item.upstream_pg_org
        retrieved_orgs = retrieved_item.retrieved_orgs

        if not retrieved_item.fast_pass:
            sorted_retrieved_orgs, sorted_logits = ranker_predict_20210715(
                ranker_model,
                bert_tokenizer,
                config.device,
                upstream_pg_org,
                batch_test_item.query_name,
                retrieved_orgs,
                apply_sigmoid=True,
            )
        else:
            sorted_retrieved_orgs = None
            sorted_logits = None

        ranked_items.append(
            BatchTestRankedItem(
                retrieved_item=retrieved_item,
                sorted_retrieved_orgs=sorted_retrieved_orgs,
                sorted_logits=sorted_logits,
            )
        )

    io.write_joblib(output_pkl, ranked_items)


def run_batch_test_report(
    ranked_items_pkl: str,
    output_folder: Optional[str],
    target_thr: Optional[float] = None,
):
    if isinstance(ranked_items_pkl, str):
        ranked_items = io.read_joblib(ranked_items_pkl)
    else:
        ranked_items = ranked_items_pkl

    out_fd = None
    if output_folder:
        out_fd = io.folder(output_folder, reset=True)

    total = 0
    num_affirmative = 0
    num_affirmative_correct = 0

    def dump_datetime_attr(obj: Dict):
        for key in list(obj.keys()):
            val = obj[key]
            if isinstance(val, datetime):
                obj[key] = val.isoformat()
            elif isinstance(val, dict):
                obj[key] = dump_datetime_attr(val)
        return obj

    def dump_batch_test_item(batch_test_item: Union[PgOrganization, BatchTestItem]):
        obj = attr.asdict(batch_test_item)  # type: ignore
        obj = dump_datetime_attr(obj)
        return json.dumps(obj, indent=2, ensure_ascii=False)

    def dump_upstream_pg_org(upstream_pg_org: PgOrganization):
        return dump_batch_test_item(upstream_pg_org)

    def dump_retrieved_org(retrieved_org: RetrievedOrg):
        obj = attr.asdict(retrieved_org)  # type: ignore
        obj['is_alias'] = retrieved_org.is_alias
        obj['name'] = retrieved_org.name
        obj['province'] = retrieved_org.province
        obj['city'] = retrieved_org.city
        obj['county'] = retrieved_org.county
        obj['organization_id'] = retrieved_org.organization_id
        obj = dump_datetime_attr(obj)
        return json.dumps(obj, indent=2, ensure_ascii=False)

    def find_target_thr(
        ranked_items: Sequence[BatchTestRankedItem],
        expected_affirmative_ratio: float = 0.93,
    ):
        import math
        expected_num_affirmative = int(math.ceil(len(ranked_items) * expected_affirmative_ratio))

        num_fast_pass = 0
        non_fast_pass_probs = []
        for ranked_item in ranked_items:
            retrieved_item = ranked_item.retrieved_item
            fast_pass = retrieved_item.fast_pass
            if fast_pass:
                num_fast_pass += 1
            else:
                if ranked_item.sorted_logits:
                    top_prob = ranked_item.sorted_logits[0]
                    non_fast_pass_probs.append(top_prob)

        non_fast_pass_probs = sorted(non_fast_pass_probs, reverse=True)
        thr = non_fast_pass_probs[expected_num_affirmative - num_fast_pass]
        return thr

    if target_thr is None:
        target_thr = find_target_thr(ranked_items)
    logger.info(f'target_thr={target_thr}')

    for idx, ranked_item in enumerate(ranked_items):
        retrieved_item = ranked_item.retrieved_item
        sorted_retrieved_orgs = ranked_item.sorted_retrieved_orgs
        sorted_logits = ranked_item.sorted_logits

        batch_test_item = retrieved_item.batch_test_item
        upstream_pg_org = retrieved_item.upstream_pg_org
        fast_pass = retrieved_item.fast_pass
        retrieved_orgs = retrieved_item.retrieved_orgs

        total += 1

        lines = []

        lines.append('BATCH_TEST_ITEM:')
        lines.append(dump_batch_test_item(batch_test_item))
        lines.append('-' * 30)
        lines.append('')

        lines.append('UPSTREAM_PG_ORG:')
        lines.append(dump_upstream_pg_org(upstream_pg_org))
        lines.append('-' * 30)
        lines.append('')

        lines.append(f'FAST PASS: {fast_pass}')
        lines.append('')

        if fast_pass:
            num_affirmative += 1
            if retrieved_orgs[0].organization_id == batch_test_item.downstream_id:
                num_affirmative_correct += 1

            for pos, retrieved_org in enumerate(retrieved_orgs):
                lines.append(f'POS: {pos}')
                lines.append('RETRIEVED_ORG:')
                lines.append(dump_retrieved_org(retrieved_org))
                lines.append('')

        else:
            if not sorted_retrieved_orgs:
                continue
            top_retrieved_org = sorted_retrieved_orgs[0]
            top_prob = sorted_logits[0]
            if top_prob <= target_thr:
                continue
            num_affirmative += 1
            if top_retrieved_org.organization_id == batch_test_item.downstream_id:
                num_affirmative_correct += 1
            else:
                truth_pos = None
                truth_retrieved_org = None
                truth_prob = None
                for pos, (retrieved_org,
                          prob) in enumerate(zip(sorted_retrieved_orgs, sorted_logits)):
                    if retrieved_org.organization_id == batch_test_item.downstream_id:
                        truth_pos = pos
                        truth_retrieved_org = retrieved_org
                        truth_prob = prob
                        break

                if not truth_retrieved_org:
                    lines.append('TRUTH_RETRIEVED_ORG:')
                    lines.append('NOT RECALLED.')
                else:
                    lines.append(f'TRUTH_POS: {truth_pos}')
                    lines.append(f'TRUTH_PROB: {truth_prob}')
                    lines.append('TRUTH_RETRIEVED_ORG:')
                    lines.append(dump_retrieved_org(truth_retrieved_org))
                lines.append('-' * 30)
                lines.append('')

                lines.append(f'TOP_PROB: {top_prob}')
                lines.append('TOP_RETRIEVED_ORG:')
                lines.append(dump_retrieved_org(top_retrieved_org))
                lines.append('-' * 30)
                lines.append('')

                for pos, (retrieved_org,
                          prob) in enumerate(zip(sorted_retrieved_orgs, sorted_logits)):
                    lines.append(f'POS: {pos}')
                    lines.append(f'PROB: {prob}')
                    lines.append('RETRIEVED_ORG:')
                    lines.append(dump_retrieved_org(retrieved_org))
                    lines.append('')

        if out_fd:
            io.write_text_lines(out_fd / f'{idx}.txt', lines)

    logger.info(
        f'num_affirmative={num_affirmative}, total={total}, '
        f'r={num_affirmative / total}'
    )
    logger.info(
        f'num_affirmative_correct={num_affirmative_correct}, num_affirmative={num_affirmative},'
        f'r={num_affirmative_correct / num_affirmative}'
    )


def run_batch_test_report_target_thrs(ranked_items_pkl: str):
    ranked_items = io.read_joblib(ranked_items_pkl)
    logger.info('Loaded.')

    for num in [
        10,
        20,
        30,
        40,
        50,
        55,
        60,
        65,
        70,
        75,
        80,
        85,
        90,
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        98,
        99,
    ]:
        run_batch_test_report(ranked_items, None, num / 100)
        logger.info('-' * 20)


def run_e2e_batch_test(
    api_url: str,
    callback_url: str,
    tenant_id: int,
    batch_test_items_pkl: str,
    profiler_limit: Optional[int] = None,
):
    batch_test_items = io.read_joblib(batch_test_items_pkl)

    if profiler_limit:
        batch_test_items = batch_test_items[:profiler_limit]

    queries = []
    for batch_test_item in batch_test_items:
        queries.append({
            'upstream_id': batch_test_item.upstream_id,
            'query_name': batch_test_item.query_name,
        })

    body = {
        'tenant_id': tenant_id,
        'queries': queries,
        'callback': callback_url,
    }
    rsp = requests.post(api_url, json=body)
    assert rsp.status_code == 204


def dump_datetime_attr(obj: Dict):
    for key in list(obj.keys()):
        val = obj[key]
        if isinstance(val, datetime):
            obj[key] = val.isoformat()
        elif isinstance(val, dict):
            obj[key] = dump_datetime_attr(val)
    return obj


def run_e2e_batch_test_report(
    postgres_config: Mapping,
    batch_test_items_pkl: str,
    e2e_body_json: str,
    output_json: str,
):
    config = PostgresConfig(**postgres_config)

    batch_test_items = io.read_joblib(batch_test_items_pkl)
    total = len(batch_test_items)

    body = io.read_json(e2e_body_json)
    tenant_id = body['tenant_id']
    matches = body['matches']
    assert len(matches) == total

    num_affirmative = 0
    num_affirmative_correct = 0

    pred_downstreams_max_load_place = 2

    results = []

    for batch_test_item, match in tqdm(zip(batch_test_items, matches)):
        assert match['upstream_id'] == batch_test_item.upstream_id

        upstream = get_org(config, tenant_id, batch_test_item.upstream_id)
        downstream = get_org(config, tenant_id, batch_test_item.downstream_id)

        result = {
            'id': batch_test_item.id,
            'upstream_id': batch_test_item.upstream_id,
            'upstream': dump_datetime_attr(attr.asdict(upstream)) if upstream else None,
            'query': batch_test_item.query_name,
            'downstream_id': batch_test_item.downstream_id,
            'downstream': dump_datetime_attr(attr.asdict(downstream)) if downstream else None,
            'pred_match_type': match['match_type'],
            'pred_correct': False,
        }

        pred_hit_place = -1
        pred_downstreams = []
        for place, pred_downstream_id in enumerate(match['match_downstream_ids']):
            if place <= pred_downstreams_max_load_place:
                pg_org = get_org(config, tenant_id, pred_downstream_id)
                pg_aliases = get_aliases_by_organization_ids(
                    config, tenant_id, [pred_downstream_id]
                )
                dump_org = dump_datetime_attr(attr.asdict(pg_org)) if pg_org else None
                dump_aliases = [
                    dump_datetime_attr(attr.asdict(pg_alias)) for pg_alias in pg_aliases
                ]
            else:
                dump_org = pred_downstream_id
                dump_aliases = None
            pred_downstreams.append({
                'place': place,
                'org': dump_org,
                'aliases': dump_aliases,
            })
            if pred_downstream_id == batch_test_item.downstream_id:
                pred_hit_place = place

        result['pred_hit_place'] = pred_hit_place
        result['pred_downstreams'] = pred_downstreams[:pred_downstreams_max_load_place + 1]

        if match['match_type'] != 'affirmative':
            assert match['match_type'] == 'confused'

        else:
            num_affirmative += 1
            assert match['match_downstream_ids']
            if match['match_downstream_ids'][0] == batch_test_item.downstream_id:
                num_affirmative_correct += 1
                result['pred_correct'] = True

        results.append(result)

    logger.info(
        f'num_affirmative={num_affirmative}, total={total}, '
        f'r={num_affirmative / total}'
    )
    logger.info(
        f'num_affirmative_correct={num_affirmative_correct}, num_affirmative={num_affirmative},'
        f'r={num_affirmative_correct / num_affirmative}'
    )

    io.write_json(output_json, results, indent=2, ensure_ascii=False)


def convert_results_json_to_xlsx(results_json: str, output_folder: str):
    results = io.read_json(results_json)

    import xlsxwriter

    out_fd = io.folder(output_folder, reset=True)

    begin = 0
    step = 200
    while begin < len(results):
        end = min(len(results), begin + step)
        output_xlsx = out_fd / f'{begin}-{end}.xlsx'
        logger.info(f'Generating {output_xlsx}')

        workbook = xlsxwriter.Workbook(output_xlsx)
        worksheet = workbook.add_worksheet()

        fields = [
            'id',
            'upstream_id',
            'query',
            'downstream_id',
            'pred_match_type',
            'pred_correct',
            'pred_hit_place',
            'upstream',
            'downstream',
            'pred_downstreams',
        ]

        # Header.
        for col, field in enumerate(fields):
            worksheet.write_string(0, col, field)

        # Results.
        for row, result in enumerate(results[begin:end], start=1):
            for col, field in enumerate(fields):
                if field != 'pred_downstreams':
                    worksheet.write_string(row, col, str(result[field]))
                else:
                    text = json.dumps(result[field], indent=2, ensure_ascii=False)
                    worksheet.write_string(row, col, text)

        workbook.close()

        begin = end
