import logging
import importlib.resources
from collections import defaultdict
from datetime import datetime

import iolite as io
import psycopg2.extras

from textdog.lexicon.type import get_lexicon_type, LexiconType
from huantong_learning_proj.db import PostgresConfig, create_pg_cursor

logger = logging.getLogger(__name__)


def get_scheme_sql(table_name):
    with importlib.resources.path(
        'huantong_learning_proj.opt.client_db_scheme',
        f'{table_name}.sql',
    ) as path:
        return path


def extract_keys_from_scheme_sql_text(scheme_sql_text):
    table_keys = []
    for line in scheme_sql_text.split('\n'):
        line = line.strip()
        if not line.startswith('`'):
            continue
        key = line.split()[0]
        assert key.startswith('`') and key.endswith('`')
        key = key[1:-1]
        table_keys.append(key)
    return table_keys


def convert_insert_sql_to_structs(table_name, table_keys, sql_file):
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
        yield dict(zip(table_keys, fields))


def convert_sql_to_csv(table_name, sql_file, output_csv):
    logger.info(f'Converting {sql_file}')

    table_name = table_name.lower()
    scheme_sql = get_scheme_sql(table_name)
    assert scheme_sql.is_file()

    table_keys = extract_keys_from_scheme_sql_text(scheme_sql.read_text())
    io.write_csv_lines(
        output_csv,
        convert_insert_sql_to_structs(table_name.upper(), table_keys, sql_file),
        from_dict=True,
    )


def load_ovalmaster_csv(ovalmaster_csv, tqdm=True):
    for struct in io.read_csv_lines(
        ovalmaster_csv,
        to_dict=True,
        skip_header=True,
        tqdm=tqdm,
    ):
        if struct['STATUS'] != '41':
            continue

        organization_id = struct['OVALMASTERID']
        province = struct['REGIONNAME']
        city = struct['CITYNAME']
        county = struct['COUNTYNAME']
        name = struct['ORGNAME']
        created_at = struct['CREATEDATE']
        updated_at = struct['DATELASTUPDATED']

        yield (
            organization_id,
            (province, city, county),
            name,
            created_at,
            updated_at,
        )


def load_orgalias_csv_default(orgalias_csv, tqdm=True):
    for struct in io.read_csv_lines(
        orgalias_csv,
        to_dict=True,
        skip_header=True,
        tqdm=tqdm,
    ):
        id = struct['ORGALIASID']
        organization_id = struct['ORGID']
        org_name = struct['ORGNAME']
        alias_name = struct['ALIASNAME']
        created_at = struct['CREATEDATE']
        updated_at = struct['UPDATETIME']

        if not organization_id:
            continue
        if not alias_name or not org_name:
            continue

        yield (
            id,
            organization_id,
            org_name,
            alias_name,
            created_at,
            updated_at,
        )


def load_orgalias_csv_and_patch(orgalias_csv, ovalmaster_csv, tqdm=True):
    org_id_to_name = {}
    for (
        organization_id,
        _,
        name,
        _,
        _,
    ) in load_ovalmaster_csv(ovalmaster_csv):
        org_id_to_name[organization_id] = name

    org_id_to_raw_aliases = defaultdict(list)
    for (
        id,
        organization_id,
        org_name,
        alias_name,
        created_at,
        updated_at,
    ) in load_orgalias_csv_default(orgalias_csv, tqdm=tqdm):
        if organization_id not in org_id_to_name:
            continue
        org_id_to_raw_aliases[organization_id].append((
            id,
            organization_id,
            org_name,
            alias_name,
            created_at,
            updated_at,
        ))

    for org_id, aliases in org_id_to_raw_aliases.items():
        std_org_name = org_id_to_name[org_id]

        first_pass_names = set()
        for (
            id,
            organization_id,
            org_name,
            alias_name,
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
                id,
                organization_id,
                alias_name,
                created_at,
                updated_at,
            )

        # Use not visited org_name as alias.
        for (
            id,
            organization_id,
            org_name,
            alias_name,
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
                -int(id),
                organization_id,
                org_name,
                created_at,
                updated_at,
            )


def load_orgalias_csv(orgalias_csv, ovalmaster_csv=None, tqdm=True):
    if not ovalmaster_csv:
        logger.info('Use load_orgalias_csv_default')
        yield from load_orgalias_csv_default(orgalias_csv, tqdm=tqdm)
    else:
        logger.info('Use load_orgalias_csv_and_patch')
        yield from load_orgalias_csv_and_patch(orgalias_csv, ovalmaster_csv, tqdm=tqdm)


def chinese_in_text(text):
    for char in text:
        if get_lexicon_type(char) == LexiconType.CHINESE:
            return True
    return False


def load_orgcodecollate_csv_default(orgcodecollate_csv, tqdm=True):
    for struct in io.read_csv_lines(
        orgcodecollate_csv,
        to_dict=True,
        skip_header=True,
        tqdm=tqdm,
    ):
        id = struct['ORGCODECOLLATEID']
        upstream_id = struct['DTID']
        downstream_id = struct['ORGID']
        query = struct['ORGNAME']
        query_tokens = struct['ORGNAMETOKENS']
        request_at = struct['REQUESTDATE']
        response_at = struct['RESPONSEDATE']
        created_at = struct['CREATEDATE']
        updated_at = struct['UPDATETIME']

        yield (
            id,
            upstream_id,
            downstream_id,
            query,
            query_tokens,
            request_at,
            response_at,
            created_at,
            updated_at,
        )


def load_orgcodecollate_csv_and_patch(orgcodecollate_csv, ovalmaster_csv, tqdm=True):
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
        id,
        upstream_id,
        downstream_id,
        query,
        query_tokens,
        request_at,
        response_at,
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
            id,
            upstream_id,
            downstream_id,
            query,
            query_tokens,
            request_at,
            response_at,
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


def load_orgcodecollate_csv(orgcodecollate_csv, ovalmaster_csv=None, tqdm=True):
    if not ovalmaster_csv:
        logger.info('Use load_orgcodecollate_csv_default')
        yield from load_orgcodecollate_csv_default(orgcodecollate_csv, tqdm=tqdm)
    else:
        logger.info('Use load_orgcodecollate_csv_and_patch')
        yield from load_orgcodecollate_csv_and_patch(orgcodecollate_csv, ovalmaster_csv, tqdm=tqdm)


def pg_create_tables(postgres_config):
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
    id bigint NOT NULL,
    tenant_id bigint NOT NULL,
    organization_id bigint NOT NULL,
    name character varying(255) NOT NULL,
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


def pg_create_task_table(postgres_config):
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
-- Name: task; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.task (
    id bigint NOT NULL,
    start_time timestamp with time zone NOT NULL,
    end_time timestamp with time zone NOT NULL,
    status character varying(255) NOT NULL,
    scheduled_at timestamp with time zone,
    ended_at timestamp with time zone
);


ALTER TABLE public.task OWNER TO postgres;

--
-- Name: task_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.task_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.task_id_seq OWNER TO postgres;

--
-- Name: task_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.task_id_seq OWNED BY public.task.id;


--
-- Name: task id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.task ALTER COLUMN id SET DEFAULT nextval('public.task_id_seq'::regclass);


--
-- Name: task task_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.task
    ADD CONSTRAINT task_pkey PRIMARY KEY (id);


--
-- Name: task_id_index; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX task_id_index ON public.task USING btree (id) INCLUDE (id);


--
-- PostgreSQL database dump complete
--
            '''
        )


def pg_drop_indices(postgres_config, table_name):
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


def pg_delete_rows(postgres_config, table_name, tenant_id):
    config = PostgresConfig(**postgres_config)

    logger.info(f'Deleting all rows from {table_name} with tenant_id = {tenant_id}')
    with create_pg_cursor(config, commit=True) as cur:
        cur.execute(f'DELETE FROM {table_name} WHERE tenant_id = %s;', (tenant_id,))


def group_by(items, size):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def upload_orgs_to_pg(ovalmaster_csv, tenant_id, postgres_config):
    config = PostgresConfig(**postgres_config)

    num_rows = 0
    for orgs in group_by(load_ovalmaster_csv(ovalmaster_csv), 10000):
        rows = []
        for (
            organization_id,
            (province, city, county),
            name,
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
                    created_at,
                    updated_at
                )
                VALUES
                %s
                ''',
                rows,
            )
    logger.info(f'num_rows = {num_rows}')


def pg_build_index_orgs(postgres_config):
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


def upload_aliases_to_pg(orgalias_csv, ovalmaster_csv, tenant_id, postgres_config):
    config = PostgresConfig(**postgres_config)

    num_rows = 0
    for aliases in group_by(load_orgalias_csv(orgalias_csv, ovalmaster_csv), 10000):
        rows = []
        for (
            id,
            organization_id,
            name,
            created_at,
            updated_at,
        ) in aliases:
            created_at = datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S')
            updated_at = datetime.strptime(updated_at, '%Y-%m-%d %H:%M:%S')
            rows.append((
                id,
                tenant_id,
                organization_id,
                name,
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
                    id,
                    tenant_id,
                    organization_id,
                    name,
                    created_at,
                    updated_at
                )
                VALUES
                %s
                ''',
                rows,
            )
    logger.info(f'num_rows = {num_rows}')


def pg_build_index_aliases(postgres_config):
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


def upload_collations_to_pg(orgcodecollate_csv, ovalmaster_csv, tenant_id, postgres_config):
    config = PostgresConfig(**postgres_config)

    num_rows = 0
    for collations in group_by(load_orgcodecollate_csv(orgcodecollate_csv, ovalmaster_csv), 10000):
        rows = []
        for (
            id,
            upstream_id,
            downstream_id,
            query,
            query_tokens,
            request_at,
            response_at,
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
                id,
                tenant_id,
                upstream_id,
                downstream_id,
                query,
                query_tokens,
                request_at,
                response_at,
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
                    collation_id,
                    tenant_id,
                    upstream_id,
                    downstream_id,
                    query_name,
                    query_name_tokens,
                    request_at,
                    response_at,
                    created_at,
                    updated_at
                )
                VALUES
                %s
                ''',
                rows,
            )
    logger.info(f'num_rows = {num_rows}')


def pg_build_index_collations(postgres_config):
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
