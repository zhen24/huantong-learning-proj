import logging
from datetime import datetime

import attr
import iolite as io
from tqdm import tqdm

from textdog.lexicon.normalize import normalize
from retrievaldog.retriever import (
    IndexOptType,
    IndexOpt,
    Retriever,
    retriever_config_from_dict,
    build_ac_automaton,
    check_ac_automaton_activation,
)
from huantong_learning_proj.db import (
    PostgresConfig,
    get_aliases_by_alias_ids,
    get_aliases_in_time_range,
    get_aliases_by_organization_ids,
    get_orgs_in_time_range,
    get_orgs_by_name,
    get_orgs,
    get_org,
    get_aliases_by_name,
    get_collations_by_query_name,
)

logger = logging.getLogger(__name__)


class EsBm25State:
    es_bm25 = None


def get_pg_alias_index_id(pg_alias):
    return f'alias:{pg_alias.organization_id}:{pg_alias.alias_id}'


def translate_to_index_opts_in_initialization(pg_orgs, pg_aliases):
    # For indexing alias.
    id_to_pg_org = {}

    for pg_org in pg_orgs:
        assert not pg_org.deleted
        yield IndexOpt(
            id=pg_org.organization_id,
            texts=[
                pg_org.province,
                pg_org.city,
                pg_org.county,
                pg_org.name,
            ],
            type=IndexOptType.CREATE,
        )
        # For indexing alias.
        id_to_pg_org[pg_org.organization_id] = pg_org

    for pg_alias in pg_aliases:
        assert not pg_alias.deleted
        if pg_alias.organization_id not in id_to_pg_org:
            logger.warning(
                f'pg_alias.organization_id={pg_alias.organization_id} not found, ignored.'
            )
            continue

        pg_org = id_to_pg_org[pg_alias.organization_id]
        yield IndexOpt(
            id=get_pg_alias_index_id(pg_alias),
            texts=[
                pg_org.province,
                pg_org.city,
                pg_org.county,
                pg_alias.name,
            ],
            type=IndexOptType.CREATE,
        )


def translate_to_index_opts_in_update(pg_orgs, pg_aliases):
    # For indexing alias.
    id_to_pg_org = {}

    for pg_org in pg_orgs:
        if not pg_org.deleted:
            index_opt_type = IndexOptType.CREATE_OR_UPDATE
        else:
            index_opt_type = IndexOptType.DELETE_IF_EXISTS

        yield IndexOpt(
            id=pg_org.organization_id,
            texts=[
                pg_org.province,
                pg_org.city,
                pg_org.county,
                pg_org.name,
            ],
            type=index_opt_type,
        )
        # For indexing alias.
        id_to_pg_org[pg_org.organization_id] = pg_org

    for pg_alias in pg_aliases:
        index_id = get_pg_alias_index_id(pg_alias)

        if not pg_alias.deleted:
            if pg_alias.organization_id not in id_to_pg_org:
                logger.warning(
                    f'pg_alias.organization_id={pg_alias.organization_id} not found, ignored.'
                )
                continue

            pg_org = id_to_pg_org[pg_alias.organization_id]
            yield IndexOpt(
                id=index_id,
                texts=[
                    pg_org.province,
                    pg_org.city,
                    pg_org.county,
                    pg_alias.name,
                ],
                type=IndexOptType.CREATE_OR_UPDATE,
            )

        else:
            yield IndexOpt(
                id=index_id,
                texts=None,
                type=IndexOptType.DELETE_IF_EXISTS,
            )


def load_pg_orgs_and_pg_aliases_for_initialization(config, tenant_id, end_time):
    # Load orgs.
    pg_orgs = get_orgs_in_time_range(
        config=config,
        tenant_id=tenant_id,
        begin_time=datetime.min,
        end_time=end_time,
        not_deleted=True,
    )

    # Load aliases.
    pg_aliases = get_aliases_in_time_range(
        config=config,
        tenant_id=tenant_id,
        begin_time=datetime.min,
        end_time=end_time,
        not_deleted=True,
    )

    return pg_orgs, pg_aliases


def load_pg_orgs_and_pg_aliases_for_update(config, tenant_id, begin_time, end_time):
    # Load orgs.
    pg_orgs = list(
        get_orgs_in_time_range(
            config=config,
            tenant_id=tenant_id,
            begin_time=begin_time,
            end_time=end_time,
        )
    )
    pg_org_ids = set(pg_org.organization_id for pg_org in pg_orgs)

    # Load aliases.
    pg_aliases = list(
        get_aliases_in_time_range(
            config=config,
            tenant_id=tenant_id,
            begin_time=begin_time,
            end_time=end_time,
        )
    )
    updated_pg_aliases_ref_organization_ids = set(
        pg_alias.organization_id for pg_alias in pg_aliases
    )

    # Load the missing pg_orgs.
    missing_pg_org_ids = []
    for org_id in updated_pg_aliases_ref_organization_ids:
        if org_id not in pg_org_ids:
            missing_pg_org_ids.append(org_id)

    if missing_pg_org_ids:
        new_pg_orgs = get_orgs(
            config=config,
            tenant_id=tenant_id,
            organization_ids=missing_pg_org_ids,
        )
        pg_orgs.extend(new_pg_orgs)

    # If an pg_org is deleted, the correlated pg_aliases should also be deleted.
    deleted_pg_org_ids = set()
    for pg_org in pg_orgs:
        if pg_org.deleted:
            deleted_pg_org_ids.add(pg_org.organization_id)

    # Load the deleted aliases.
    patched_deleted_at = datetime.now()
    for pg_alias in get_aliases_by_organization_ids(
        config=config,
        tenant_id=tenant_id,
        organization_ids=tuple(deleted_pg_org_ids),
    ):
        if not pg_alias.deleted:
            pg_alias = attr.evolve(pg_alias, deleted_at=patched_deleted_at)
        assert pg_alias.deleted
        pg_aliases.append(pg_alias)

    return pg_orgs, pg_aliases


def gen_ltp_cache_input_text(pg_orgs, pg_aliases):
    texts = set()

    logger.info('Loading pg_orgs...')
    for pg_org in tqdm(pg_orgs):
        texts.add(pg_org.province)
        texts.add(pg_org.city)
        texts.add(pg_org.county)
        texts.add(pg_org.name)

    logger.info('Loading pg_aliases...')
    for pg_alias in tqdm(pg_aliases):
        texts.add(pg_alias.name)

    logger.info('Normalizaing...')
    norm_texts = (normalize(text) for text in texts if text)
    return norm_texts


def build_ltp_tokenizer_cache_input(postgres_config, tenant_id, output_txt):
    config = PostgresConfig(**postgres_config)

    pg_orgs, pg_aliases = load_pg_orgs_and_pg_aliases_for_initialization(
        config,
        tenant_id,
        datetime.max,
    )

    norm_texts = gen_ltp_cache_input_text(pg_orgs, pg_aliases)

    io.write_text_lines(
        output_txt,
        norm_texts,
        strip=True,
        skip_empty=True,
    )


def initialize_retriever(
    retriever,
    postgres_config,
    tenant_id,
    end_time,
    texts_for_cache=None,
):
    pg_orgs, pg_aliases = load_pg_orgs_and_pg_aliases_for_initialization(
        postgres_config,
        tenant_id,
        end_time,
    )
    index_opts = translate_to_index_opts_in_initialization(pg_orgs, pg_aliases)
    retriever.index(index_opts, use_high_level_opt=False)

    if texts_for_cache:
        retriever.update_ltp_tokenizer_cache(texts_for_cache)


def initialize_retriever_cli(
    retriever_config,
    postgres_config,
    tenant_id,
    ltp_cache_jsl=None,
):
    if ltp_cache_jsl:
        retriever_config['ltp_tokenizer_cache_jsls'] = [ltp_cache_jsl]

    retriever = Retriever(
        retriever_config_from_dict(retriever_config),
        resource_id=tenant_id,
        reset=True,
    )

    initialize_retriever(
        retriever,
        PostgresConfig(**postgres_config),
        tenant_id,
        datetime.max,
    )


def update_retriever(
    retriever,
    postgres_config,
    tenant_id,
    begin_time,
    end_time,
):
    pg_orgs, pg_aliases = load_pg_orgs_and_pg_aliases_for_update(
        postgres_config,
        tenant_id,
        begin_time,
        end_time,
    )
    index_opts = translate_to_index_opts_in_update(pg_orgs, pg_aliases)
    retriever.index(index_opts)

    texts = gen_ltp_cache_input_text(pg_orgs, pg_aliases)
    new_texts = set()
    for text in texts:
        if text in retriever.ltp_tokenizer_cache.keys():
            continue
        new_texts.add(text)
    new_texts = tuple(texts)
    retriever.update_ltp_tokenizer_cache(new_texts)


@attr.s
class RetrieverResource:
    top_k = attr.ib()
    postgres_config = attr.ib()
    enable_exact_match_fast_pass = attr.ib()
    enable_collation_augmentation = attr.ib()
    enable_collation_fast_pass = attr.ib()
    ac_level_one = attr.ib()
    ac_level_two = attr.ib()
    ac_level_three = attr.ib()


def load_retriever_resource(
    postgres_config,
    ac_level_one_json_file,
    ac_level_two_json_file,
    ac_level_three_json_file,
    top_k=200,
    enable_exact_match_fast_pass=True,
    enable_collation_augmentation=True,
    enable_collation_fast_pass=True,
):
    return RetrieverResource(
        top_k=top_k,
        postgres_config=postgres_config,
        enable_exact_match_fast_pass=enable_exact_match_fast_pass,
        enable_collation_augmentation=enable_collation_augmentation,
        enable_collation_fast_pass=enable_collation_fast_pass,
        ac_level_one=build_ac_automaton(ac_level_one_json_file),
        ac_level_two=build_ac_automaton(ac_level_two_json_file),
        ac_level_three=build_ac_automaton(ac_level_three_json_file),
    )


@attr.s
class RetrievedOrg:
    pg_org = attr.ib()
    pg_alias = attr.ib()
    tag = attr.ib()

    @property
    def is_alias(self):
        return (self.pg_alias is not None)

    @property
    def name(self):
        if self.is_alias:
            return self.pg_alias.name
        else:
            return self.pg_org.name

    @property
    def province(self):
        return self.pg_org.province

    @property
    def city(self):
        return self.pg_org.city

    @property
    def county(self):
        return self.pg_org.county

    @property
    def organization_id(self):
        return self.pg_org.organization_id


def create_retrieved_org_from_pg_org(pg_org, tag):
    return RetrievedOrg(
        pg_org=pg_org,
        pg_alias=None,
        tag=tag,
    )


def create_retrieved_org_from_pg_alias(
    pg_alias,
    id_to_pg_org,
    postgres_config,
    tenant_id,
    tag,
):
    if pg_alias.organization_id in id_to_pg_org:
        pg_org = id_to_pg_org[pg_alias.organization_id]
    else:
        if postgres_config is None or tenant_id is None:
            raise RuntimeError('Check your code.')
        pg_org = get_org(postgres_config, tenant_id, pg_alias.organization_id, not_deleted=True)

    if pg_org is None:
        logger.error(f'valid pg_org not found for pg_alias={pg_alias}')
        return None

    return RetrievedOrg(
        pg_org=pg_org,
        pg_alias=pg_alias,
        tag=tag,
    )


def create_retrieved_orgs(
    pg_orgs,
    pg_aliases,
    postgres_config,
    tenant_id,
    tag,
):
    retrieved_orgs = []

    id_to_pg_org = {}
    for pg_org in pg_orgs:
        retrieved_orgs.append(create_retrieved_org_from_pg_org(pg_org, tag))
        id_to_pg_org[pg_org.organization_id] = pg_org

    for pg_alias in pg_aliases:
        retrieved_org = create_retrieved_org_from_pg_alias(
            pg_alias,
            id_to_pg_org,
            postgres_config,
            tenant_id,
            tag,
        )
        if retrieved_org:
            retrieved_orgs.append(retrieved_org)
    return retrieved_orgs


def retrieve_for_rank(
    retriever,
    retriever_resource,
    tenant_id,
    upstream_id,
    query,
    dynamic_ltp_tokenizer_cache=None,
):
    # Load upstream.
    logger.info('Loading upstream...')
    upstream_pg_org = get_org(retriever_resource.postgres_config, tenant_id, upstream_id)
    if upstream_pg_org is None:
        raise RuntimeError(f'upstream_id={upstream_id} not found.')
    logger.info(f'upstream={upstream_pg_org}')

    # Fast pass for exact match.
    if retriever_resource.enable_exact_match_fast_pass:
        exact_match_retrieved_orgs = create_retrieved_orgs(
            get_orgs_by_name(
                retriever_resource.postgres_config,
                tenant_id,
                query,
                not_deleted=True,
            ),
            get_aliases_by_name(
                retriever_resource.postgres_config,
                tenant_id,
                query,
                not_deleted=True,
            ),
            retriever_resource.postgres_config,
            tenant_id,
            'exact_match_fast_pass',
        )
        if exact_match_retrieved_orgs:
            logger.info('Hit enable_exact_match_fast_pass')
            return True, upstream_pg_org, exact_match_retrieved_orgs

    retrieved_orgs = []

    # Load from prev collations.
    if retriever_resource.enable_collation_augmentation:
        pg_collations = list(
            get_collations_by_query_name(
                retriever_resource.postgres_config,
                tenant_id,
                query,
                not_deleted=True,
            )
        )

        if retriever_resource.enable_collation_fast_pass:
            # Make sure the latest one got hit first.
            pg_collations = sorted(pg_collations, key=lambda col: col.updated_at, reverse=True)

            hit_downstream_id = None
            for pg_collation in pg_collations:
                if pg_collation.upstream_id == upstream_pg_org.organization_id:
                    hit_downstream_id = pg_collation.downstream_id
                    break

            if hit_downstream_id:
                logger.info('Hit enable_collation_fast_pass')
                hit_pg_org = get_org(
                    retriever_resource.postgres_config,
                    tenant_id,
                    hit_downstream_id,
                    not_deleted=True,
                )
                if hit_pg_org:
                    hit_pg_aliases = get_aliases_by_organization_ids(
                        retriever_resource.postgres_config,
                        tenant_id,
                        [hit_downstream_id],
                        not_deleted=True,
                    )

                    retrieved_orgs = create_retrieved_orgs(
                        [hit_pg_org],
                        hit_pg_aliases,
                        # Force not loading.
                        None,
                        None,
                        'collation_fast_pass',
                    )
                    assert retrieved_orgs
                    return True, upstream_pg_org, retrieved_orgs
                else:
                    logger.warning(f'Failed to load hit_downstream_id={hit_downstream_id}')

        # Load prev downstreams.
        org_ids = set()
        for pg_collation in pg_collations:
            org_ids.add(pg_collation.downstream_id)
        org_ids = list(org_ids)

        pg_orgs = get_orgs(
            retriever_resource.postgres_config,
            tenant_id,
            org_ids,
            not_deleted=True,
        )
        pg_aliases = get_aliases_by_organization_ids(
            retriever_resource.postgres_config,
            tenant_id,
            org_ids,
            not_deleted=True,
        )
        retrieved_orgs.extend(
            create_retrieved_orgs(
                pg_orgs,
                pg_aliases,
                # Force not loading.
                None,
                None,
                'collation_augmentation',
            )
        )
        logger.info(f'Load {len(retrieved_orgs)} retrieved_orgs from prev collations.')

    # Invoke retriever now.
    texts = [query]
    if not any([
        check_ac_automaton_activation(retriever_resource.ac_level_one, query),
        check_ac_automaton_activation(retriever_resource.ac_level_two, query),
        check_ac_automaton_activation(retriever_resource.ac_level_three, query),
    ]):
        texts.append(normalize(upstream_pg_org.province))
    logger.info(f'Query texts={texts}')

    raw_retrieved_orgs = retriever.retrieve(
        texts,
        top_k=retriever_resource.top_k,
        dynamic_ltp_tokenizer_cache=dynamic_ltp_tokenizer_cache,
    )

    org_ids = []
    alias_ids = []
    for raw_retrieved_org in raw_retrieved_orgs:
        if ':' in raw_retrieved_org.id:
            tag, org_id, alias_id = raw_retrieved_org.id.split(':')
            assert tag == 'alias'
            alias_ids.append(int(alias_id))
        else:
            org_id = raw_retrieved_org.id
            org_ids.append(int(org_id))

    pg_orgs = get_orgs(
        retriever_resource.postgres_config,
        tenant_id,
        org_ids,
        not_deleted=True,
    )
    pg_aliases = get_aliases_by_alias_ids(
        retriever_resource.postgres_config,
        tenant_id,
        alias_ids,
        not_deleted=True,
    )
    retrieved_orgs.extend(
        create_retrieved_orgs(
            pg_orgs,
            pg_aliases,
            retriever_resource.postgres_config,
            tenant_id,
            'default',
        )
    )
    logger.info(f'Retrieve {len(retrieved_orgs)} candidates.')

    return False, upstream_pg_org, retrieved_orgs
