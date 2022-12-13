import logging
import random
from datetime import datetime
from typing import Mapping, Optional, Sequence

import Levenshtein
import iolite as io
from attr import attrs, attrib, asdict
from retrievaldog.retriever import (
    Retriever,
    retriever_config_from_dict,
)
from textdog.lexicon.normalize import normalize
from tqdm import tqdm

from huantong_learning_proj.db import (
    PostgresConfig,
    get_collations,
    get_collations_in_time_range,
    get_org,
    get_aliases_by_organization_ids,
    PgCollation,
)
from huantong_learning_proj.opt.retriever import (
    load_retriever_resource,
    retrieve_for_rank,
    RetrieverResource,
)

logger = logging.getLogger(__name__)


@attrs
class TrainOrg:
    province: str = attrib()
    city: str = attrib()
    county: str = attrib()
    name: str = attrib()


@attrs
class TrainSample:
    upstream_org: TrainOrg = attrib()
    query: str = attrib()
    downstream_org: TrainOrg = attrib()
    negative_orgs: Sequence[TrainOrg] = attrib()


def calculate_ned(text0: str, text1: str):
    dis = Levenshtein.distance(text0, text1)
    return dis / max(len(text0), len(text1))


def generate_train_sample(
    tenant_id: int,
    retriever: Retriever,
    retriever_resource: RetrieverResource,
    upstream_id: int,
    query: str,
    downstream_id: int,
    negative_size: int,
    dynamic_ltp_tokenizer_cache: Optional[Mapping[str, Sequence[str]]],
):
    assert not retriever_resource.enable_exact_match_fast_pass
    assert retriever_resource.enable_collation_augmentation
    assert not retriever_resource.enable_collation_fast_pass

    # Load downstream.
    downstream_pg_org = get_org(
        retriever_resource.postgres_config,
        tenant_id,
        downstream_id,
    )
    assert downstream_pg_org

    downstream_pg_aliases = list(
        get_aliases_by_organization_ids(
            retriever_resource.postgres_config,
            tenant_id,
            [downstream_id],
            not_deleted=True,
        )
    )
    if downstream_pg_aliases:
        # Find the best match by NED.
        min_ned = calculate_ned(query, downstream_pg_org.name)
        best_downstream_name = downstream_pg_org.name

        for downstream_pg_alias in downstream_pg_aliases:
            cur_ned = calculate_ned(query, downstream_pg_alias.name)
            if cur_ned < min_ned:
                min_ned = cur_ned
                best_downstream_name = downstream_pg_alias.name

        downstream_org = TrainOrg(
            province=downstream_pg_org.province,
            city=downstream_pg_org.city,
            county=downstream_pg_org.county,
            name=best_downstream_name,
        )

    else:
        downstream_org = TrainOrg(
            province=downstream_pg_org.province,
            city=downstream_pg_org.city,
            county=downstream_pg_org.county,
            name=downstream_pg_org.name,
        )

    # Retrieve for upstream and candidates.
    _, upstream_pg_org, retrieved_orgs = retrieve_for_rank(
        retriever,
        retriever_resource,
        tenant_id,
        upstream_id,
        query,
        dynamic_ltp_tokenizer_cache,
    )
    assert upstream_pg_org

    upstream_org = TrainOrg(
        province=upstream_pg_org.province,
        city=upstream_pg_org.city,
        county=upstream_pg_org.county,
        name=upstream_pg_org.name,
    )

    # Deal with alias.
    negative_orgs = []
    for retrieved_org in retrieved_orgs:
        if len(negative_orgs) >= negative_size:
            break
        if retrieved_org.organization_id == downstream_id:
            continue
        # Add as negative orgs.
        negative_orgs.append(
            TrainOrg(
                province=retrieved_org.province,
                city=retrieved_org.city,
                county=retrieved_org.county,
                name=retrieved_org.name,
            )
        )

    return TrainSample(
        upstream_org=upstream_org,
        query=query,
        downstream_org=downstream_org,
        negative_orgs=negative_orgs,
    )


def generate_train_sample_based_on_pg_collation(
    tenant_id: int,
    retriever: Retriever,
    retriever_resource: RetrieverResource,
    pg_collation: PgCollation,
    negative_size: int,
    dynamic_ltp_tokenizer_cache: Optional[Mapping[str, Sequence[str]]] = None,
):
    return generate_train_sample(
        tenant_id=tenant_id,
        retriever=retriever,
        retriever_resource=retriever_resource,
        upstream_id=pg_collation.upstream_id,
        query=pg_collation.query_name,
        downstream_id=pg_collation.downstream_id,
        negative_size=negative_size,
        dynamic_ltp_tokenizer_cache=dynamic_ltp_tokenizer_cache,
    )


def debug_generate_train_sample(
    retriever_config: Mapping,
    postgres_config: Mapping,
    ac_level_one_json_file: str,
    ac_level_two_json_file: str,
    ac_level_three_json_file: str,
    tenant_id: int,
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
        enable_exact_match_fast_pass=False,
        enable_collation_fast_pass=False,
    )

    pg_collation = list(
        get_collations(
            retriever_resource.postgres_config,
            tenant_id,
            [199865546],
        )
    )[0]

    train_sample = generate_train_sample_based_on_pg_collation(
        tenant_id,
        retriever,
        retriever_resource,
        pg_collation,
        10,
    )
    assert train_sample

    breakpoint()


def generate_train_sample_to_folder(
    retriever_config: Mapping,
    postgres_config: Mapping,
    ac_level_one_json_file: str,
    ac_level_two_json_file: str,
    ac_level_three_json_file: str,
    tenant_id: int,
    collation_id_and_timestamp_json: str,
    output_folder: str,
    negative_size: int = 10,
    chunk_size: int = 5000,
    random_seed: int = 13370,
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
        enable_exact_match_fast_pass=False,
        enable_collation_fast_pass=False,
    )

    logger.info('Loading pg_collations...')
    valid_collation_ids = set()
    for collation_id, _ in io.read_json(collation_id_and_timestamp_json):
        valid_collation_ids.add(collation_id)

    pg_collations = []
    dynamic_ltp_tokenizer_cache = dict()
    for pg_collation in tqdm(
        get_collations_in_time_range(
            retriever_resource.postgres_config,
            tenant_id,
            datetime.min,
            datetime.max,
            not_deleted=True,
        )
    ):
        if pg_collation.collation_id not in valid_collation_ids:
            continue
        pg_collations.append(pg_collation)
        text = normalize(pg_collation.query_name)
        if text and pg_collation.query_name_tokens:
            dynamic_ltp_tokenizer_cache[text] = pg_collation.query_name_tokens.split('|')
    logger.info(f'{len(pg_collations)} loaded.')

    random.seed(random_seed)
    random.shuffle(pg_collations)

    out_fd = io.folder(output_folder, reset=True)
    logger.info(f'Expected num_chunks={len(pg_collations) / chunk_size}')

    chunk_idx = 0
    chunk_fd = io.folder(out_fd / str(chunk_idx), touch=True)
    for pg_collation_idx, pg_collation in enumerate(pg_collations):
        if pg_collation_idx > 0 and pg_collation_idx % chunk_size == 0:
            # Mark the current chunk as ready.
            ready_file = out_fd / f'{chunk_idx}-ready.txt'
            assert not ready_file.exists()
            ready_file.touch()
            logger.info(f'{ready_file} created.')
            # Move to the next chunk.
            chunk_idx += 1
            logger.info(f'Move to chunk_idx={chunk_idx}')
            chunk_fd = io.folder(out_fd / str(chunk_idx), touch=True)

        logger.info(f'Processing pg_collation_idx={pg_collation_idx}')
        train_sample = generate_train_sample_based_on_pg_collation(
            tenant_id,
            retriever,
            retriever_resource,
            pg_collation,
            negative_size,
            dynamic_ltp_tokenizer_cache,
        )
        io.write_json(
            chunk_fd / f'{pg_collation_idx}.json',
            asdict(train_sample),  # type: ignore
            ensure_ascii=False,
            indent=2,
        )


def generate_dev_set_from_ranked_items(
    tenant_id: int,
    postgres_config: Mapping,
    ranked_items_pkl: str,
    output_folder: str,
    negative_size: int = 1,
):
    pg_config = PostgresConfig(**postgres_config)

    out_fd = io.folder(output_folder, reset=True)

    ranked_items = io.read_joblib(ranked_items_pkl)
    for idx, ranked_item in tqdm(enumerate(ranked_items)):
        retrieved_item = ranked_item.retrieved_item
        sorted_retrieved_orgs = ranked_item.sorted_retrieved_orgs

        batch_test_item = retrieved_item.batch_test_item
        upstream_pg_org = retrieved_item.upstream_pg_org
        fast_pass = retrieved_item.fast_pass

        if fast_pass:
            # Doesn't include.
            continue

        # Get downstream.
        downstream_pg_org = get_org(
            pg_config,
            tenant_id,
            batch_test_item.downstream_id,
        )
        assert downstream_pg_org

        downstream_pg_aliases = list(
            get_aliases_by_organization_ids(
                pg_config,
                tenant_id,
                [batch_test_item.downstream_id],
                not_deleted=True,
            )
        )
        if downstream_pg_aliases:
            # Find the best match by NED.
            min_ned = calculate_ned(batch_test_item.query_name, downstream_pg_org.name)
            best_downstream_name = downstream_pg_org.name

            for downstream_pg_alias in downstream_pg_aliases:
                cur_ned = calculate_ned(batch_test_item.query_name, downstream_pg_alias.name)
                if cur_ned < min_ned:
                    min_ned = cur_ned
                    best_downstream_name = downstream_pg_alias.name

            downstream_org = TrainOrg(
                province=downstream_pg_org.province,
                city=downstream_pg_org.city,
                county=downstream_pg_org.county,
                name=best_downstream_name,
            )

        else:
            downstream_org = TrainOrg(
                province=downstream_pg_org.province,
                city=downstream_pg_org.city,
                county=downstream_pg_org.county,
                name=downstream_pg_org.name,
            )

        # Transform upstream.
        upstream_org = TrainOrg(
            province=upstream_pg_org.province,
            city=upstream_pg_org.city,
            county=upstream_pg_org.county,
            name=upstream_pg_org.name,
        )

        # Generate negative samples based on rank.
        negative_orgs = []
        for retrieved_org in sorted_retrieved_orgs:
            if len(negative_orgs) >= negative_size:
                break
            if retrieved_org.organization_id == batch_test_item.downstream_id:
                continue
            # Add as negative orgs.
            negative_orgs.append(
                TrainOrg(
                    province=retrieved_org.province,
                    city=retrieved_org.city,
                    county=retrieved_org.county,
                    name=retrieved_org.name,
                )
            )

        # Save.
        train_sample = TrainSample(
            upstream_org=upstream_org,
            query=batch_test_item.query_name,
            downstream_org=downstream_org,
            negative_orgs=negative_orgs,
        )
        io.write_json(
            out_fd / f'{idx}.json',
            asdict(train_sample),  # type: ignore
            ensure_ascii=False,
            indent=2,
        )


def generate_dev_set_from_batch_test_items(
    retriever_config: Mapping,
    postgres_config: Mapping,
    ac_level_one_json_file: str,
    ac_level_two_json_file: str,
    ac_level_three_json_file: str,
    tenant_id: int,
    batch_test_items_pkl: str,
    output_folder: str,
    negative_size: int = 1,
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

    dynamic_ltp_tokenizer_cache = dict()
    for batch_test_item in batch_test_items:
        text = normalize(batch_test_item.query_name)
        if text and batch_test_item.query_name_tokens:
            dynamic_ltp_tokenizer_cache[text] = batch_test_item.query_name_tokens.split('|')

    pg_config = PostgresConfig(**postgres_config)

    out_fd = io.folder(output_folder, reset=True)

    for idx, batch_test_item in tqdm(enumerate(batch_test_items)):
        logger.info(f'Retrieving {batch_test_item}')
        fast_pass, upstream_pg_org, retrieved_orgs = retrieve_for_rank(
            retriever=retriever,
            retriever_resource=retriever_resource,
            tenant_id=tenant_id,
            upstream_id=batch_test_item.upstream_id,
            query=batch_test_item.query_name,
            dynamic_ltp_tokenizer_cache=dynamic_ltp_tokenizer_cache,
        )
        assert upstream_pg_org

        if fast_pass:
            # Doesn't include.
            continue

        # Get downstream.
        downstream_pg_org = get_org(
            pg_config,
            tenant_id,
            batch_test_item.downstream_id,
        )
        assert downstream_pg_org

        downstream_pg_aliases = list(
            get_aliases_by_organization_ids(
                pg_config,
                tenant_id,
                [batch_test_item.downstream_id],
                not_deleted=True,
            )
        )
        if downstream_pg_aliases:
            # Find the best match by NED.
            min_ned = calculate_ned(batch_test_item.query_name, downstream_pg_org.name)
            best_downstream_name = downstream_pg_org.name

            for downstream_pg_alias in downstream_pg_aliases:
                cur_ned = calculate_ned(batch_test_item.query_name, downstream_pg_alias.name)
                if cur_ned < min_ned:
                    min_ned = cur_ned
                    best_downstream_name = downstream_pg_alias.name

            downstream_org = TrainOrg(
                province=downstream_pg_org.province,
                city=downstream_pg_org.city,
                county=downstream_pg_org.county,
                name=best_downstream_name,
            )

        else:
            downstream_org = TrainOrg(
                province=downstream_pg_org.province,
                city=downstream_pg_org.city,
                county=downstream_pg_org.county,
                name=downstream_pg_org.name,
            )

        # Transform upstream.
        upstream_org = TrainOrg(
            province=upstream_pg_org.province,
            city=upstream_pg_org.city,
            county=upstream_pg_org.county,
            name=upstream_pg_org.name,
        )

        # Generate negative samples based on rank.
        negative_orgs = []
        for retrieved_org in retrieved_orgs:
            if len(negative_orgs) >= negative_size:
                break
            if retrieved_org.organization_id == batch_test_item.downstream_id:
                continue
            # Add as negative orgs.
            negative_orgs.append(
                TrainOrg(
                    province=retrieved_org.province,
                    city=retrieved_org.city,
                    county=retrieved_org.county,
                    name=retrieved_org.name,
                )
            )

        # Save.
        train_sample = TrainSample(
            upstream_org=upstream_org,
            query=batch_test_item.query_name,
            downstream_org=downstream_org,
            negative_orgs=negative_orgs,
        )
        io.write_json(
            out_fd / f'{idx}.json',
            asdict(train_sample),  # type: ignore
            ensure_ascii=False,
            indent=2,
        )
