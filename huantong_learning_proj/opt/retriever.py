import logging
from datetime import datetime
from typing import Optional, Mapping, Sequence, Iterable, Dict

import attr
import iolite as io
from retrievaldog.retriever import (
    IndexOptType,
    IndexOpt,
    Retriever,
    retriever_config_from_dict,
    build_ac_automaton,
    check_ac_automaton_activation,
)
from textdog.lexicon.normalize import normalize
from textdog.token.ltp import (
    LtpTokenizer,
    LtpTokenizerConfig,
)
from tqdm import tqdm

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
    PgOrganization,
    PgAlias,
)
from huantong_learning_proj.opt.token_query import merge_words

logger = logging.getLogger(__name__)


def get_pg_alias_index_id(pg_alias: PgAlias):
    return f'alias:{pg_alias.organization_id}:{pg_alias.alias_id}'


def translate_to_index_opts_in_initialization(
    pg_orgs: Iterable[PgOrganization], pg_aliases: Iterable[PgAlias]
):
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


def load_pg_orgs_and_pg_aliases_for_initialization(
    config: PostgresConfig, tenant_id: int, end_time: datetime
):
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


def gen_ltp_cache_input_text(pg_orgs: Iterable[PgOrganization], pg_aliases: Iterable[PgAlias]):
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


def build_ltp_tokenizer_cache_input(postgres_config: Mapping, tenant_id: int, output_txt: str):
    config = PostgresConfig(**postgres_config)

    pg_orgs, pg_aliases = load_pg_orgs_and_pg_aliases_for_initialization(
        config,
        tenant_id,
        datetime.max,
    )

    norm_texts = gen_ltp_cache_input_text(pg_orgs, pg_aliases)

    io.folder(io.file(output_txt).parent, touch=True)
    io.write_text_lines(
        output_txt,
        norm_texts,
        strip=True,
        skip_empty=True,
    )


def ltp_tokenize(
    model_folder: str,
    input_txt: str,
    output_jsl: str,
    device: str = 'cpu',
    batch_size: Optional[int] = None,
):
    ltp_tokenizer = LtpTokenizer(
        LtpTokenizerConfig(
            model_folder=model_folder,
            device=device,
            batch_size=batch_size,
        )
    )

    base_ltp_cache = {}
    if io.file(output_jsl).exists:
        logger.info(f'loading {output_jsl}')
        for item in io.read_json_lines(output_jsl, tqdm=True):
            base_ltp_cache[item['text']] = item['tokens']

    def ltp_tokenize_input_txt():
        logger.info(f'loading {input_txt}')
        texts = []
        for text in io.read_text_lines(input_txt, strip=True, skip_empty=True):
            if text in base_ltp_cache:
                yield {'text': text, 'tokens': base_ltp_cache[text]}
            else:
                texts.append(text)

        logger.info('Tokenizing ...')
        for idx, tokens in enumerate(tqdm(ltp_tokenizer.batch_tokenize(texts))):
            text = texts[idx]
            tokens = merge_words(tokens)
            yield {'text': text, 'tokens': tokens}

    io.write_json_lines(
        output_jsl,
        ltp_tokenize_input_txt(),
        ensure_ascii=False,
    )


def initialize_retriever(
    retriever: Retriever,
    postgres_config: PostgresConfig,
    tenant_id: int,
    end_time: datetime,
    texts_for_cache: Optional[Sequence[str]] = None,
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
    retriever_config: Dict,
    postgres_config: Mapping,
    tenant_id: int,
    ltp_cache_jsl: Optional[str] = None,
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


@attr.s
class RetrieverResource:
    top_k: int = attr.ib()
    postgres_config: PostgresConfig = attr.ib()
    enable_exact_match_fast_pass: bool = attr.ib()
    enable_collation_augmentation: bool = attr.ib()
    enable_collation_fast_pass: bool = attr.ib()
    ac_level_one: str = attr.ib()
    ac_level_two: str = attr.ib()
    ac_level_three: str = attr.ib()


def load_retriever_resource(
    postgres_config: PostgresConfig,
    ac_level_one_json_file: str,
    ac_level_two_json_file: str,
    ac_level_three_json_file: str,
    top_k: int = 200,
    enable_exact_match_fast_pass: bool = True,
    enable_collation_augmentation: bool = True,
    enable_collation_fast_pass: bool = True,
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
    pg_org: PgOrganization = attr.ib()
    pg_alias: Optional[PgAlias] = attr.ib()
    tag: str = attr.ib()

    @property
    def is_alias(self):
        return (self.pg_alias is not None)

    @property
    def name(self):
        if self.pg_alias:
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


def create_retrieved_org_from_pg_org(pg_org: PgOrganization, tag: str):
    return RetrievedOrg(
        pg_org=pg_org,
        pg_alias=None,
        tag=tag,
    )


def create_retrieved_org_from_pg_alias(
    pg_alias: PgAlias,
    id_to_pg_org: Mapping[int, PgOrganization],
    postgres_config: Optional[PostgresConfig],
    tenant_id: Optional[int],
    tag: str,
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
    pg_orgs: Iterable[PgOrganization],
    pg_aliases: Iterable[PgAlias],
    postgres_config: Optional[PostgresConfig],
    tenant_id: Optional[int],
    tag: str,
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
    retriever: Retriever,
    retriever_resource: RetrieverResource,
    tenant_id: int,
    upstream_id: int,
    query: str,
    dynamic_ltp_tokenizer_cache: Optional[Mapping[str, Sequence[str]]] = None,
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
