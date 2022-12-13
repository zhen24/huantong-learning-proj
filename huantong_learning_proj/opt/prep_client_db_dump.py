import logging
from collections import defaultdict
from datetime import datetime
from typing import Sequence, Optional, Mapping, Iterable

import iolite as io
import psycopg2.extras
from textdog.lexicon.type import get_lexicon_type, LexiconType

from huantong_learning_proj.db import PostgresConfig, create_pg_cursor
from huantong_learning_proj.opt import extract_sql_fields

logger = logging.getLogger(__name__)


def get_table_keys(table_name: str):
    assert table_name in {'ovalmaster', 'orgalias', 'orgcodecollate'}

    if table_name == 'ovalmaster':
        return extract_sql_fields.org_fields

    if table_name == 'orgalias':
        return extract_sql_fields.alias_fields

    return extract_sql_fields.collection_fields


def convert_insert_sql_to_structs(table_name: str, table_keys: Sequence[str], sql_file: str):
    prefix = f'INSERT INTO `{table_name}` VALUES ('
    suffix = ');'
    for line in io.read_text_lines(sql_file, strip=True, tqdm=True):
        assert line.startswith(prefix)
        assert line.endswith(suffix)

        fields_text = line[len(prefix):-len(suffix)]
        fields_text = fields_text.replace(' null,', ' None,')
        fields_text = fields_text.replace(' NULL,', ' None,')
        fields_text = fields_text.replace('\\0', '')

        fields = eval(f'[{fields_text}]')
        assert len(fields) == len(table_keys)
        yield dict(filter(lambda x: x[0] != '-', zip(table_keys, fields)))


def convert_sql_to_csv(table_name: str, sql_file: str, output_csv: str):
    logger.info(f'Converting {sql_file}')

    table_keys = get_table_keys(table_name)
    io.write_csv_lines(
        output_csv,
        convert_insert_sql_to_structs(table_name.upper(), table_keys, sql_file),
        from_dict=True,
    )


def load_ovalmaster_csv(ovalmaster_csv: str, tqdm: bool = True):
    for struct in io.read_csv_lines(
        ovalmaster_csv,
        to_dict=True,
        skip_header=True,
        tqdm=tqdm,
    ):
        if struct.get('status', '41') != '41':
            continue

        organization_id = struct['organization_id']
        name = struct['name']
        province = struct['province']
        city = struct['city']
        county = struct['county']
        come_from = struct.get('come_from', '历史运营')
        created_at = struct['created_at'][:19]
        updated_at = struct['updated_at'][:19]

        yield (
            organization_id,
            name,
            (province, city, county),
            come_from,
            created_at,
            updated_at,
        )


def load_orgalias_csv_default(orgalias_csv: str, tqdm: bool = True):
    for struct in io.read_csv_lines(
        orgalias_csv,
        to_dict=True,
        skip_header=True,
        tqdm=tqdm,
    ):
        alias_id = struct['id']
        organization_id = struct['organization_id']
        org_name = struct.get('org_name')
        alias_name = struct['name']
        come_from = struct.get('come_from', '历史运营')
        created_at = struct['created_at'][:19]
        updated_at = struct['updated_at'][:19]

        if not organization_id:
            continue
        if not alias_name and not org_name:
            continue

        yield (
            alias_id,
            organization_id,
            org_name,
            alias_name,
            come_from,
            created_at,
            updated_at,
        )


def load_orgalias_csv_and_patch(orgalias_csv: str, ovalmaster_csv: str, tqdm: bool = True):
    org_id_to_name = {}
    for (
        organization_id,
        name,
        _,
        _,
        _,
        _,
    ) in load_ovalmaster_csv(ovalmaster_csv):
        org_id_to_name[organization_id] = name

    org_id_to_raw_aliases = defaultdict(list)
    for (
        alias_id,
        organization_id,
        org_name,
        alias_name,
        come_from,
        created_at,
        updated_at,
    ) in load_orgalias_csv_default(orgalias_csv, tqdm=tqdm):
        if organization_id not in org_id_to_name:
            continue
        org_id_to_raw_aliases[organization_id].append((
            alias_id,
            organization_id,
            org_name,
            alias_name,
            come_from,
            created_at,
            updated_at,
        ))

    for org_id, aliases in org_id_to_raw_aliases.items():
        std_org_name = org_id_to_name[org_id]

        first_pass_names = set()
        for (
            alias_id,
            organization_id,
            org_name,
            alias_name,
            come_from,
            created_at,
            updated_at,
        ) in aliases:
            if not alias_name:
                continue
            if alias_name == std_org_name:
                # logger.info(f'id={id}, alias_name == std_org_name')
                continue
            first_pass_names.add(alias_name)
            yield (
                alias_id,
                organization_id,
                alias_name,
                come_from,
                created_at,
                updated_at,
            )

        # Use not visited org_name as alias.
        for (
            alias_id,
            organization_id,
            org_name,
            alias_name,
            come_from,
            created_at,
            updated_at,
        ) in aliases:
            if not org_name:
                continue
            if org_name == std_org_name:
                continue
            if org_name in first_pass_names:
                continue
            # logger.info(f'id={id}, use org_name as aliase')
            yield (
                # Use negative value to avoid conflict.
                -int(alias_id),
                organization_id,
                org_name,
                come_from,
                created_at,
                updated_at,
            )


def load_orgalias_csv(orgalias_csv: str, ovalmaster_csv: Optional[str] = None, tqdm: bool = True):
    if not ovalmaster_csv:
        logger.info('Use load_orgalias_csv_default')
        yield from load_orgalias_csv_default(orgalias_csv, tqdm=tqdm)
    else:
        logger.info('Use load_orgalias_csv_and_patch')
        yield from load_orgalias_csv_and_patch(orgalias_csv, ovalmaster_csv, tqdm=tqdm)


def chinese_in_text(text: str):
    for char in text:
        if get_lexicon_type(char) == LexiconType.CHINESE:
            return True
    return False


def load_orgcodecollate_csv_default(orgcodecollate_csv: str, tqdm: bool = True):
    for struct in io.read_csv_lines(
        orgcodecollate_csv,
        to_dict=True,
        skip_header=True,
        tqdm=tqdm,
    ):
        collation_id = struct['collation_id']
        upstream_id = struct['upstream_id']
        downstream_id = struct['downstream_id']
        query = struct['query_name']
        query_tokens = struct['query_name_tokens']
        request_at = struct['request_at'][:19]
        response_at = struct['response_at'][:19]
        come_from = struct.get('come_from', '历史运营')
        created_at = struct['created_at'][:19]
        updated_at = struct['updated_at'][:19]

        yield (
            collation_id,
            upstream_id,
            downstream_id,
            query,
            query_tokens,
            request_at,
            response_at,
            come_from,
            created_at,
            updated_at,
        )


def load_orgcodecollate_csv_and_patch(
    orgcodecollate_csv: str,
    ovalmaster_csv: str,
    tqdm: bool = True,
):
    org_ids = set()
    for item in load_ovalmaster_csv(ovalmaster_csv):
        organization_id = item[0]
        org_ids.add(organization_id)

    total = 0
    num_invalid_upstream_ids = 0
    num_invalid_downstream_ids = 0
    num_invalid_query = 0
    num_skip = 0

    for (
        collation_id,
        upstream_id,
        downstream_id,
        query,
        query_tokens,
        request_at,
        response_at,
        come_from,
        created_at,
        updated_at,
    ) in load_orgcodecollate_csv_default(orgcodecollate_csv, tqdm=tqdm):
        total += 1

        skip = False

        if upstream_id not in org_ids:
            num_invalid_upstream_ids += 1
            skip = True
        if downstream_id not in org_ids:
            num_invalid_downstream_ids += 1
            skip = True
        if not chinese_in_text(query):
            num_invalid_query += 1
            skip = True

        if skip:
            num_skip += 1
            continue

        yield (
            collation_id,
            upstream_id,
            downstream_id,
            query,
            query_tokens,
            request_at,
            response_at,
            come_from,
            created_at,
            updated_at,
        )

    logger.info(
        f'total={total}, '
        f'num_skip={num_skip}, '
        f'num_invalid_upstream_ids={num_invalid_upstream_ids}, '
        f'num_invalid_downstream_ids={num_invalid_downstream_ids}, '
        f'num_invalid_query={num_invalid_query}'
    )


def load_orgcodecollate_csv(
    orgcodecollate_csv: str,
    ovalmaster_csv: Optional[str] = None,
    tqdm: bool = True,
):
    if not ovalmaster_csv:
        logger.info('Use load_orgcodecollate_csv_default')
        yield from load_orgcodecollate_csv_default(orgcodecollate_csv, tqdm=tqdm)
    else:
        logger.info('Use load_orgcodecollate_csv_and_patch')
        yield from load_orgcodecollate_csv_and_patch(orgcodecollate_csv, ovalmaster_csv, tqdm=tqdm)


def pg_create_tables(postgres_config: Mapping):
    config = PostgresConfig(**postgres_config)

    with create_pg_cursor(config, commit=True) as cur:
        cur.execute(
            '''
--
-- PostgreSQL database dump
--

-- Dumped from database version 11.5
-- Dumped by pg_dump version 13.3

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

--
-- Name: organization; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.organization (
    tenant_id bigint NOT NULL,
    organization_id bigint NOT NULL,
    name character varying(255) NOT NULL,
    province character varying(255),
    city character varying(255),
    county character varying(255),
    address character varying(255),
    come_from character varying(255) DEFAULT '人工标注',
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    deleted_at timestamp with time zone
);


ALTER TABLE public.organization OWNER TO postgres;

--
-- Name: organization organization_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.organization
    ADD CONSTRAINT organization_pkey PRIMARY KEY (tenant_id, organization_id);


--
-- Name: index_organization_organization_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX index_organization_organization_id ON public.organization USING hash (organization_id);


--
-- PostgreSQL database dump complete
--
            '''
        )

    with create_pg_cursor(config, commit=True) as cur:
        cur.execute(
            '''
--
-- PostgreSQL database dump
--

-- Dumped from database version 11.5
-- Dumped by pg_dump version 13.3

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

--
-- Name: alias; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.alias (
    tenant_id bigint NOT NULL,
    id bigint NOT NULL,
    organization_id bigint NOT NULL,
    name character varying(255) NOT NULL,
    come_from character varying(255) DEFAULT '人工标注',
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    deleted_at timestamp with time zone
);


ALTER TABLE public.alias OWNER TO postgres;

--
-- Name: alias_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.alias_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.alias_id_seq OWNER TO postgres;

--
-- Name: alias_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.alias_id_seq OWNED BY public.alias.id;


--
-- Name: alias id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.alias ALTER COLUMN id SET DEFAULT nextval('public.alias_id_seq'::regclass);


--
-- Name: alias alias_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.alias
    ADD CONSTRAINT alias_pkey PRIMARY KEY (id, tenant_id);


--
-- Name: alias alias_fkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.alias
    ADD CONSTRAINT alias_fkey FOREIGN KEY (tenant_id, organization_id)
        REFERENCES public.organization (tenant_id, organization_id) MATCH SIMPLE
        ON UPDATE CASCADE
        ON DELETE CASCADE;


--
-- PostgreSQL database dump complete
--
            '''
        )

    with create_pg_cursor(config, commit=True) as cur:
        cur.execute(
            '''
--
-- PostgreSQL database dump
--

-- Dumped from database version 11.5
-- Dumped by pg_dump version 13.3

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

--
-- Name: collation; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public."collation" (
    tenant_id bigint NOT NULL,
    collation_id bigint NOT NULL,
    upstream_id bigint NOT NULL,
    downstream_id bigint NOT NULL,
    query_name character varying(255) NOT NULL,
    query_name_tokens character varying(255),
    request_at timestamp with time zone,
    response_at timestamp with time zone,
    come_from character varying(255) DEFAULT '人工标注',
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    deleted_at timestamp with time zone
);


ALTER TABLE public."collation" OWNER TO postgres;

--
-- Name: collation collation_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public."collation"
    ADD CONSTRAINT collation_pkey PRIMARY KEY (tenant_id, collation_id);


--
-- PostgreSQL database dump complete
--
            '''
        )


def pg_create_task_table(postgres_config: Mapping):
    config = PostgresConfig(**postgres_config)

    with create_pg_cursor(config, commit=True) as cur:
        cur.execute(
            '''
--
-- PostgreSQL database dump
--

-- Dumped from database version 11.5
-- Dumped by pg_dump version 13.3

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

--
-- Name: train; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE IF NOT EXISTS public.train
(
    task_id bigint NOT NULL,
    name character varying(255) UNIQUE NOT NULL,
    description character varying(255),
    status character varying(255) NOT NULL,
    storage_path character varying(255),
    created_at timestamp with time zone DEFAULT now(),
    finished_at timestamp with time zone,
    deleted_at timestamp with time zone
);

ALTER TABLE public.train OWNER to postgres;

--
-- Name: train train_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.train
    ADD CONSTRAINT train_pkey PRIMARY KEY (task_id);


--
-- PostgreSQL database dump complete
--
            '''
        )

    with create_pg_cursor(config, commit=True) as cur:
        cur.execute(
            '''
--
-- PostgreSQL database dump
--

-- Dumped from database version 11.5
-- Dumped by pg_dump version 13.3

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

--
-- Name: model; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE IF NOT EXISTS public.model
(
    task_id bigint NOT NULL,
    name character varying(255) NOT NULL,
    description character varying(255),
    model_source character varying(255),
    status character varying(255) NOT NULL,
    created_at timestamp with time zone DEFAULT now(),
    finished_at timestamp with time zone,
    deleted_at timestamp with time zone
);

ALTER TABLE public.model OWNER to postgres;

--
-- Name: model model_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.model
    ADD CONSTRAINT model_pkey PRIMARY KEY (task_id);


--
-- Name: model model_fkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.model
    ADD CONSTRAINT model_fkey FOREIGN KEY (model_source)
        REFERENCES public.train (name) MATCH SIMPLE
        ON UPDATE CASCADE
        ON DELETE CASCADE;


--
-- PostgreSQL database dump complete
--
            '''
        )


def pg_drop_indices(postgres_config: Mapping, table_name: str):
    config = PostgresConfig(**postgres_config)

    with create_pg_cursor(config) as cur:
        cur.execute(
            f'''
            SELECT indexrelid::regclass::text
            FROM   pg_index  i
            LEFT   JOIN pg_depend d ON d.objid = i.indexrelid AND d.deptype = 'i'
            WHERE  i.indrelid = '{table_name}'::regclass
            AND    d.objid IS NULL
            '''
        )
        indices = cur.fetchall()

    for index, in indices:
        logger.info(f'Dropping index={index}')
        with create_pg_cursor(config, commit=True) as cur:
            cur.execute(f'DROP INDEX {index};')


def pg_delete_rows(postgres_config: Mapping, table_name: str, tenant_id: int):
    config = PostgresConfig(**postgres_config)

    logger.info(f'Deleting all rows from {table_name} with tenant_id = {tenant_id}')
    with create_pg_cursor(config, commit=True) as cur:
        cur.execute(f'DELETE FROM {table_name} WHERE tenant_id = %s;', (tenant_id,))


def group_by(items: Iterable, size: int):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def upload_orgs_to_pg(ovalmaster_csv: str, tenant_id: int, postgres_config: Mapping):
    config = PostgresConfig(**postgres_config)

    num_rows = 0
    for orgs in group_by(load_ovalmaster_csv(ovalmaster_csv), 10000):
        rows = []
        for (
            organization_id,
            name,
            (province, city, county),
            come_from,
            created_at,
            updated_at,
        ) in orgs:
            updated_at = datetime.strptime(updated_at, '%Y-%m-%d %H:%M:%S')
            if created_at:
                created_at = datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S')
            else:
                created_at = updated_at
            rows.append((
                tenant_id,
                organization_id,
                name,
                province,
                city,
                county,
                come_from,
                created_at,
                updated_at,
            ))
        num_rows += len(rows)

        with create_pg_cursor(config, commit=True) as cur:
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
                    come_from,
                    created_at,
                    updated_at
                )
                VALUES
                %s
                ''',
                rows,
            )
    logger.info(f'num_rows = {num_rows}')


def pg_build_index_orgs(postgres_config: Mapping):
    config = PostgresConfig(**postgres_config)

    logger.info('Creating index...')
    with create_pg_cursor(config, commit=True) as cur:
        cur.execute(
            '''
            CREATE INDEX index_organization_organization_id
            ON public.organization
            USING hash
            (organization_id);
            '''
        )

    with create_pg_cursor(config, commit=True) as cur:
        cur.execute(
            '''
            CREATE INDEX index_organization_name
            ON public.organization
            USING hash
            (name);
            '''
        )


def upload_aliases_to_pg(
    orgalias_csv: str,
    ovalmaster_csv: str,
    tenant_id: int,
    postgres_config: Mapping,
):
    config = PostgresConfig(**postgres_config)

    num_rows = 0
    for aliases in group_by(load_orgalias_csv(orgalias_csv, ovalmaster_csv), 10000):
        rows = []
        for (
            alias_id,
            organization_id,
            name,
            come_from,
            created_at,
            updated_at,
        ) in aliases:
            created_at = datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S')
            updated_at = datetime.strptime(updated_at, '%Y-%m-%d %H:%M:%S')
            rows.append((
                tenant_id,
                alias_id,
                organization_id,
                name,
                come_from,
                created_at,
                updated_at,
            ))
        num_rows += len(rows)

        with create_pg_cursor(config, commit=True) as cur:
            psycopg2.extras.execute_values(
                cur,
                '''
                INSERT INTO public.alias
                (
                    tenant_id,
                    id,
                    organization_id,
                    name,
                    come_from,
                    created_at,
                    updated_at
                )
                VALUES
                %s
                ''',
                rows,
            )
    logger.info(f'num_rows = {num_rows}')


def pg_build_index_aliases(postgres_config: Mapping):
    config = PostgresConfig(**postgres_config)

    logger.info('Creating index...')
    with create_pg_cursor(config, commit=True) as cur:
        cur.execute(
            '''
            CREATE INDEX index_aliases_organization_id
            ON public.alias
            USING hash
            (organization_id);
            '''
        )

    with create_pg_cursor(config, commit=True) as cur:
        cur.execute(
            '''
            CREATE INDEX index_aliases_name
            ON public.alias
            USING hash
            (name);
            '''
        )


def upload_collations_to_pg(
    orgcodecollate_csv: str,
    ovalmaster_csv: str,
    tenant_id: int,
    postgres_config: Mapping,
):
    config = PostgresConfig(**postgres_config)

    num_rows = 0
    for collations in group_by(load_orgcodecollate_csv(orgcodecollate_csv, ovalmaster_csv), 10000):
        rows = []
        for (
            collation_id,
            upstream_id,
            downstream_id,
            query,
            query_tokens,
            request_at,
            response_at,
            come_from,
            created_at,
            updated_at,
        ) in collations:
            request_at = datetime.strptime(request_at, '%Y-%m-%d %H:%M:%S')
            response_at = datetime.strptime(response_at, '%Y-%m-%d %H:%M:%S')
            updated_at = datetime.strptime(updated_at, '%Y-%m-%d %H:%M:%S')
            if created_at:
                created_at = datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S')
            else:
                created_at = updated_at
            rows.append((
                tenant_id,
                collation_id,
                upstream_id,
                downstream_id,
                query,
                query_tokens,
                request_at,
                response_at,
                come_from,
                created_at,
                updated_at,
            ))
        num_rows += len(rows)

        with create_pg_cursor(config, commit=True) as cur:
            psycopg2.extras.execute_values(
                cur,
                '''
                INSERT INTO public.collation
                (
                    tenant_id,
                    collation_id,
                    upstream_id,
                    downstream_id,
                    query_name,
                    query_name_tokens,
                    request_at,
                    response_at,
                    come_from,
                    created_at,
                    updated_at
                )
                VALUES
                %s
                ''',
                rows,
            )
    logger.info(f'num_rows = {num_rows}')


def pg_build_index_collations(postgres_config: Mapping):
    config = PostgresConfig(**postgres_config)

    logger.info('Creating index...')
    with create_pg_cursor(config, commit=True) as cur:
        cur.execute(
            '''
            CREATE INDEX index_collation_query_name
            ON public.collation
            USING hash
            (query_name);
            '''
        )


def pg_backup_tables(postgres_config: Mapping, bak_csv_fd: str):
    config = PostgresConfig(**postgres_config)

    out_fd = io.folder(bak_csv_fd, touch=True)
    for t_name in ('organization', 'alias', 'collation'):
        bak_csv = out_fd / f'{t_name}.csv'
        logger.info(f'Backup table={t_name}')
        with create_pg_cursor(config) as cur:
            with bak_csv.open('w') as fp:
                cur.copy_expert(f"COPY public.{t_name} TO STDOUT WITH CSV HEADER", fp)
    logger.info(f'Saved to {out_fd}')
