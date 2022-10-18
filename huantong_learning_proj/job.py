import json
import logging
import shutil
import traceback
from datetime import datetime, date, timedelta

import attr
import fire
import iolite as io
import redis
import requests
from miser.protect import load_license, validate_license

from huantong_learning_proj.data_receiver import (
    commit_label_data,
    delete_label_data,
    insert_train_task,
    delete_train_task,
    insert_upgrade_task,
    delete_upgrade_task,
    get_training_task,
    init_train_task,
)
from huantong_learning_proj.db import (
    PostgresConfig,
    set_train_task_status,
    get_train_tasks,
    get_train_tasks_by_status,
    get_train_by_task_ids,
    get_model_tasks_by_status,
    get_model_by_task_ids,
    set_model_task_status,
    get_model_tasks,
    get_train_by_name,
    reset_using_train_task,
    set_train_task_storage_path,
)
from huantong_learning_proj.settings import (
    HUANTONG_LEARNING_LOGGING_FOLDER,
    REDIS_HOST,
    REDIS_PORT,
    CURRENT_HOST,
    PORT,
    TRAINING_DEVICE,
    AI_HOST,
    AI_PORT,
    BIND_HOST,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s: [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def dump_datetime_attr(obj):
    from datetime import datetime
    for key in list(obj.keys()):
        val = obj[key]
        if isinstance(val, datetime):
            obj[key] = val.isoformat()
        elif isinstance(val, dict):
            obj[key] = dump_datetime_attr(val)
    return obj


def execute_training_process():
    pass


def start_receiving(config_json, original_model_path, license_cer):
    try:
        license = load_license(license_cer)
        validate_license(license)
    except Exception:
        logger.error(f'Invalid license, ex={traceback.format_exc()}')
        return 1

    config = io.read_json(config_json)
    postgres_config = PostgresConfig(**config.pop('postgres_config'))

    try:
        init_train_task(postgres_config, original_model_path)
    except Exception:
        logger.exception('pg数据库操作异常')
        return

    logger.info(f'Connecting to redis ({REDIS_HOST}, {REDIS_PORT})')
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

    huantong_logging_root_fd = None
    if HUANTONG_LEARNING_LOGGING_FOLDER:
        huantong_logging_root_fd = io.folder(
            HUANTONG_LEARNING_LOGGING_FOLDER + 'receiver',
            touch=True,
        )

    while True:
        item = redis_client.blpop('receiving_tasks', timeout=300)

        date_today = date.today()
        if item is None:
            if huantong_logging_root_fd:
                logger.info('Cleanup logs...')
                max_timedelta = timedelta(days=30)
                for date_fd in huantong_logging_root_fd.glob('*'):
                    if not date_fd.is_dir():
                        continue

                    try:
                        date_prev = date.fromisoformat(date_fd.name)
                    except ValueError:
                        logger.warning(f'invalid date_fd={date_fd}, skip')
                        continue

                    date_timedelta = date_today - date_prev
                    if date_timedelta <= max_timedelta:
                        continue

                    logger.info(f'Removing log folder {date_fd}')
                    shutil.rmtree(date_fd)

            continue

        # Setup logging file handler.
        logging_file_handler = None
        if huantong_logging_root_fd:
            # Logging path.
            huantong_logging_fd = io.folder(
                huantong_logging_root_fd / date_today.isoformat(),
                touch=True,
            )
            logging_file_handler = logging.FileHandler(
                huantong_logging_fd / datetime.now().strftime('%Y%m%d-%H%M%S-%f.txt')
            )
            # Logging format.
            formatter = logging.Formatter('%(asctime)s: [%(levelname)s] %(message)s')
            logging_file_handler.setFormatter(formatter)
            # Attach.
            logger.addHandler(logging_file_handler)

        try:
            assert item[0] == b'receiving_tasks'
            task_str = item[1]
            task = json.loads(task_str)
            task_id = task['id']
            task_key = task['key']
            task_body = task['body']
        except Exception:
            logger.error(f'Failed to decode item={item}')

            # Detach.
            if logging_file_handler:
                logging_file_handler.close()
                logger.removeHandler(logging_file_handler)

            continue

        logger.info(f'Processing task_key={task_key}')

        if task_key == 'label_commit':
            try:
                # label_id = task_body['label_id']
                # query_name = task_body['query_info']['query_name']
                #
                # if huantong_logging_root_fd:
                #     huantong_label_fd = io.folder(
                #         huantong_logging_root_fd.parent / 'labeled_data' / date_today.isoformat(),
                #         touch=True,
                #     )
                #
                #     filename = query_name[:12].replace('/', '')
                #     io.write_json(
                #         huantong_label_fd / f'{label_id}-{filename}.txt',
                #         task_body,
                #         encoding='utf-8',
                #         ensure_ascii=False,
                #         indent=2
                #     )

                result = commit_label_data(postgres_config, task_body)
                redis_client.rpush(f'res-{task_id}', json.dumps(result))
            except Exception:
                error = traceback.format_exc()
                logger.error(f'Failed to label_commit, ex={error}')
                redis_client.rpush(f'res-{task_id}', json.dumps({'errorMessage': error}))

        elif task_key == 'label_delete':
            try:
                result = delete_label_data(postgres_config, task_body)
                redis_client.rpush(f'res-{task_id}', json.dumps(result))
            except Exception:
                error = traceback.format_exc()
                logger.error(f'Failed to label_delete, ex={error}')
                redis_client.rpush(f'res-{task_id}', json.dumps({'errorMessage': error}))

        elif task_key == 'train_commit':
            try:
                training_tasks = get_training_task(postgres_config)
                if training_tasks:
                    logger.info('已存在正在执行或正在准备的训练任务!!!')
                    logger.info(f'training_tasks={training_tasks}')
                    result = {'errorMessage': '已存在正在执行或正在准备的训练任务!!!'}
                    redis_client.rpush(f'res-{task_id}', json.dumps(result))
                    continue

                train_id = insert_train_task(postgres_config, task_body)
                train_task = list(get_train_by_task_ids(postgres_config, [train_id]))
                assert train_task
                logger.info(train_task)
                result = dump_datetime_attr(attr.asdict(train_task[0]))
                redis_client.rpush(f'res-{task_id}', json.dumps(result))

                try:
                    url = f'http://{AI_HOST}:{AI_PORT}/remove'  # noqa
                    logger.info(f'POST to {url}')
                    data = {
                        'loc': f'{CURRENT_HOST}:{TRAINING_DEVICE}',
                        'callback': f'http://{CURRENT_HOST}:{PORT}/start_traing'  # noqa
                    }
                    rsp = requests.post(url, json=data, timeout=5)
                    logger.info(f'status code = {rsp.status_code}')
                except Exception:
                    error = traceback.format_exc()
                    logger.error(f'Failed to remove_the_ranker_actor, ex={error}')
                    set_train_task_status(postgres_config, train_id, '训练失败')

            except Exception:
                error = traceback.format_exc()
                logger.error(f'Failed to train_commit, ex={error}')
                redis_client.rpush(f'res-{task_id}', json.dumps({'errorMessage': error}))

        elif task_key == 'train_getInfo':
            try:
                train_tasks = []
                for train_task in get_train_tasks(postgres_config):
                    train_tasks.append(dump_datetime_attr(attr.asdict(train_task)))
                logger.info(
                    f'train_tasks=\n{json.dumps(train_tasks, ensure_ascii=False, indent=2)}'
                )
                redis_client.rpush(f'res-{task_id}', json.dumps(train_tasks))
            except Exception:
                error = traceback.format_exc()
                logger.error(f'Failed to train_getInfo, ex={error}')
                redis_client.rpush(f'res-{task_id}', json.dumps({'errorMessage': error}))

        elif task_key == 'train_delete':
            try:
                train_ids = task_body['task_ids']
                result = delete_train_task(postgres_config, train_ids)
                redis_client.rpush(f'res-{task_id}', json.dumps(result))
            except Exception:
                error = traceback.format_exc()
                logger.error(f'Failed to train_delete, ex={error}')
                redis_client.rpush(f'res-{task_id}', json.dumps({'errorMessage': error}))

        elif task_key == 'model_fetch':
            try:
                train_tasks = []
                for train_task in get_train_tasks_by_status(postgres_config, '训练完成'):
                    train_tasks.append(dump_datetime_attr(attr.asdict(train_task)))
                logger.info(
                    f'train_tasks=\n{json.dumps(train_tasks, ensure_ascii=False, indent=2)}'
                )
                redis_client.rpush(f'res-{task_id}', json.dumps(train_tasks))
            except Exception:
                error = traceback.format_exc()
                logger.error(f'Failed to model_fetch, ex={error}')
                redis_client.rpush(f'res-{task_id}', json.dumps({'errorMessage': error}))

        elif task_key == 'model_commit':
            try:
                upgrading_tasks = list(get_model_tasks_by_status(postgres_config, '升级中'))
                if upgrading_tasks:
                    result = {'errorMessage': '已存在正在执行的升级任务!!!'}
                    redis_client.rpush(f'res-{task_id}', json.dumps(result))
                    continue

                model_id = insert_upgrade_task(postgres_config, task_body)
                model_task = list(get_model_by_task_ids(postgres_config, [model_id]))
                assert model_task
                logger.info(model_task)
                result = dump_datetime_attr(attr.asdict(model_task[0]))
                redis_client.rpush(f'res-{task_id}', json.dumps(result))

                train_tasks = list(get_train_by_name(postgres_config, task_body['model_source']))
                assert train_tasks
                state_dict_filename = io.file(train_tasks[0].storage_path).name
                try:
                    url = f'http://{AI_HOST}:{AI_PORT}/upgrade'  # noqa
                    logger.info(f'POST to {url}')
                    data = {
                        'state_dict_filename': state_dict_filename,
                        'callback': f'http://{CURRENT_HOST}:{PORT}/complete'  # noqa
                    }
                    rsp = requests.post(url, json=data)
                    logger.info(f'status code = {rsp.status_code}')

                    set_model_task_status(postgres_config, model_id, '升级完成')
                    reset_using_train_task(postgres_config)
                    set_train_task_status(postgres_config, train_tasks[0].task_id, '使用中')
                except Exception:
                    error = traceback.format_exc()
                    logger.error(f'Failed to upgrade_ai, ex={error}')
                    set_model_task_status(postgres_config, model_id, '升级失败')

            except Exception:
                error = traceback.format_exc()
                logger.error(f'Failed to model_commit, ex={error}')
                redis_client.rpush(f'res-{task_id}', json.dumps({'errorMessage': error}))

        elif task_key == 'model_getInfo':
            try:
                model_tasks = []
                for model_task in get_model_tasks(postgres_config):
                    model_tasks.append(dump_datetime_attr(attr.asdict(model_task)))
                logger.info(
                    f'train_tasks=\n{json.dumps(model_tasks, ensure_ascii=False, indent=2)}'
                )
                redis_client.rpush(f'res-{task_id}', json.dumps(model_tasks))
            except Exception:
                error = traceback.format_exc()
                logger.error(f'Failed to model_getInfo, ex={error}')
                redis_client.rpush(f'res-{task_id}', json.dumps({'errorMessage': error}))

        elif task_key == 'model_delete':
            try:
                model_ids = task_body['task_ids']
                result = delete_upgrade_task(postgres_config, model_ids)
                redis_client.rpush(f'res-{task_id}', json.dumps(result))
            except Exception:
                error = traceback.format_exc()
                logger.error(f'Failed to model_delete, ex={error}')
                redis_client.rpush(f'res-{task_id}', json.dumps({'errorMessage': error}))

        else:
            logger.error(f'Invalid task_key={task_key}')

        logger.info('Done')

        # Detach.
        if logging_file_handler:
            logging_file_handler.close()
            logger.removeHandler(logging_file_handler)


def start_training(config_json, license_cer):
    try:
        license = load_license(license_cer)
        validate_license(license)
    except Exception:
        logger.error(f'Invalid license, ex={traceback.format_exc()}')
        return 1

    config = io.read_json(config_json)
    postgres_config = PostgresConfig(**config.pop('postgres_config'))

    logger.info(f'Connecting to redis ({REDIS_HOST}, {REDIS_PORT})')
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

    huantong_logging_root_fd = None
    if HUANTONG_LEARNING_LOGGING_FOLDER:
        huantong_logging_root_fd = io.folder(
            HUANTONG_LEARNING_LOGGING_FOLDER + 'trainer',
            touch=True,
        )

    while True:
        item = redis_client.blpop('training_tasks', timeout=600)

        date_today = date.today()
        if item is None:
            if huantong_logging_root_fd:
                logger.info('Cleanup logs...')
                max_timedelta = timedelta(days=30)
                for date_fd in huantong_logging_root_fd.glob('*'):
                    if not date_fd.is_dir():
                        continue

                    try:
                        date_prev = date.fromisoformat(date_fd.name)
                    except ValueError:
                        logger.warning(f'invalid date_fd={date_fd}, skip')
                        continue

                    date_timedelta = date_today - date_prev
                    if date_timedelta <= max_timedelta:
                        continue

                    logger.info(f'Removing log folder {date_fd}')
                    shutil.rmtree(date_fd)

            continue

        # Setup logging file handler.
        logging_file_handler = None
        if huantong_logging_root_fd:
            # Logging path.
            huantong_logging_fd = io.folder(
                huantong_logging_root_fd / date_today.isoformat(),
                touch=True,
            )
            logging_file_handler = logging.FileHandler(
                huantong_logging_fd / datetime.now().strftime('%Y%m%d-%H%M%S-%f.txt')
            )
            # Logging format.
            formatter = logging.Formatter('%(asctime)s: [%(levelname)s] %(message)s')
            logging_file_handler.setFormatter(formatter)
            # Attach.
            logger.addHandler(logging_file_handler)

        try:
            assert item[0] == b'training_tasks'
            task_str = item[1]
            task = json.loads(task_str)
            task_key = task['key']
            # task_body = task['body']
            train_tasks = get_training_task(postgres_config)
            assert len(train_tasks) == 1
            train_id = train_tasks[0].task_id

        except Exception:
            logger.error(f'Failed to decode item={item}')

            # Detach.
            if logging_file_handler:
                logging_file_handler.close()
                logger.removeHandler(logging_file_handler)

            continue

        logger.info(f'Processing task_key={task_key}')

        if task_key == 'start_training':
            try:
                set_train_task_status(postgres_config, train_id, '训练中')

                # todo, 训练中
                execute_training_process()

                set_train_task_storage_path(postgres_config, train_id, '')
                set_train_task_status(postgres_config, train_id, '训练完成')

                try:
                    url = f'http://{AI_HOST}:{AI_PORT}/restore'  # noqa
                    logger.info(f'POST to {url}')
                    data = {
                        'loc': f'{CURRENT_HOST}:{TRAINING_DEVICE}',
                        'callback': f'http://{CURRENT_HOST}:{PORT}/restore_ai',  # noqa
                        'bind_host': BIND_HOST,
                    }
                    rsp = requests.post(url, json=data)
                    logger.info(f'status code = {rsp.status_code}')
                except Exception:
                    error = traceback.format_exc()
                    logger.error(f'Failed to restore_ai, ex={error}')

            except Exception:
                error = traceback.format_exc()
                logger.error(f'Failed to start_training, ex={error}')
                set_train_task_status(postgres_config, train_id, '训练失败')

        else:
            logger.error(f'Invalid task_key={task_key}')

        logger.info('Done')

        # Detach.
        if logging_file_handler:
            logging_file_handler.close()
            logger.removeHandler(logging_file_handler)


def fire_run_job():
    fire.Fire({
        'start_receiving': start_receiving,
        'start_training': start_training,
    })


def debug():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = io.file(tmp_dir) / 'tmp_file.json'
        with tmp_file.open('w') as fp:
            json.dump(
                {
                    'postgres_config': {
                        'host': '192.168.0.33',
                        'port': '5432',
                        'dbname': 'oval',
                        'user': 'postgres',
                        'password': '123456',
                        'reuse_connection': True,
                    }
                },
                fp,
            )

        start_receiving(
            config_json=tmp_file,
            original_model_path='/ai_resource/huantong/ranker/state_dict_epoch_73.pt',
            license_cer=None
        )
