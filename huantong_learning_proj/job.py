import atexit
import json
import logging
import os
import shutil
import subprocess
import traceback
from datetime import datetime, date, timedelta
from typing import Dict, Any

import attr
import fire
import iolite as io
import psutil
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


def dump_datetime_attr(obj: Dict):
    for key in list(obj.keys()):
        val = obj[key]
        if isinstance(val, datetime):
            obj[key] = val.isoformat()
        elif isinstance(val, dict):
            obj[key] = dump_datetime_attr(val)
    return obj


def stop_all_children_processes():
    procs = psutil.Process().children()
    for proc in procs:
        proc.terminate()

    _, alive = psutil.wait_procs(procs, timeout=10)
    for proc in alive:
        proc.kill()


class AutoTrain:

    def __init__(self, online_model_fd: str):
        self.env = os.environ.copy()
        self.online_model_fd = io.folder(online_model_fd, exists=True)

        # All processes in the current process group will be terminated
        # with the lead process.
        os.setpgrp()
        self.pgid = os.getpgrp()

        atexit.register(stop_all_children_processes)

    def exec_script(self, script_file_name: str, *args: Any):
        logger.info(f'Executing {script_file_name}')
        params = " ".join(json.dumps(str(arg), ensure_ascii=False) for arg in args)
        logger.info(f'shell params={params}\n\n')

        proc = subprocess.Popen(
            [f'$SCRIPT_FOLDER/{script_file_name} {params}'],
            # Attach to the current process group.
            preexec_fn=lambda: os.setpgid(0, self.pgid),
            # Shell & env.
            shell=True,
            env=self.env,
            text=True,
            # line buffered.
            bufsize=1,
            # Redirect all output to stdout.
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        while proc.poll() is None:
            line = proc.stdout.readline().rstrip()  # type: ignore
            logger.info(line)

        return proc.returncode == 0

    def prep_data(self, database_version: str):
        return self.exec_script('prep-data.sh', database_version)

    def train(self, database_version: str):
        return self.exec_script('train.sh', database_version)

    def run_with_break(self, database_version: str):
        succeeded = self.prep_data(database_version)
        if not succeeded:
            err = 'prep_data failed.'
            logger.error(err)
            raise RuntimeError(err)

        succeeded = self.train(database_version)
        if not succeeded:
            err = 'train failed.'
            logger.error(err)
            raise RuntimeError(err)

    def launch_state_dict_file(self, database_version: str):
        storage_fd = f'/resource/train-workspace/{database_version}/model'
        fd = io.folder(storage_fd, exists=True)

        pt_files = []
        st = ''
        log_file = io.file(fd / 'log.txt', exists=True)
        for line in io.read_text_lines(log_file):
            if line.startswith('Validating'):
                st = line
            if line.startswith(('avg_dev_loss', '!!! ')):
                st += line
                if line.startswith('!!! Saving'):
                    logger.info(st)
                    pt_files.append(line.split('/')[-1].split()[0])

        selected_file = io.file(fd / pt_files[-2], exists=True)
        target_file = self.online_model_fd / f'{selected_file.stem}-{database_version}.pt'
        shutil.copy(selected_file, self.online_model_fd)
        logger.info(f'{selected_file} Copied to {target_file}')

        removed_pt = pt_files[:-2]
        for pt_file in fd.glob('*.pt'):
            if pt_file.name in removed_pt:
                pt_file.unlink()

        return target_file


def execute_training_process():
    raise RuntimeError('警告: 需和AI系统交互,且一经开始不可中断,暂不可使用(此行仅用于测试接口是否可访问)!!!')


def start_receiving(
    config_json: str,
    original_model_path: str,
    license_cer: str,
):
    try:
        license = load_license(license_cer)
        validate_license(license)
    except Exception:
        logger.error(f'Invalid license, ex={traceback.format_exc()}')
        return 1

    config = io.read_json(config_json)
    receiver_pg = PostgresConfig(**config.pop('receiver_postgres'))

    try:
        init_train_task(receiver_pg, original_model_path)
    except Exception:
        logger.exception('pg数据库操作异常')
        return

    logger.info(f'Connecting to redis ({REDIS_HOST}, {REDIS_PORT})')
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

    huantong_logging_root_fd = None
    if HUANTONG_LEARNING_LOGGING_FOLDER:
        huantong_logging_root_fd = io.folder(
            HUANTONG_LEARNING_LOGGING_FOLDER + '/receiver',
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

                result = commit_label_data(receiver_pg, task_body)
                redis_client.rpush(f'res-{task_id}', json.dumps(result))
            except Exception:
                error = traceback.format_exc()
                logger.error(f'Failed to label_commit, ex={error}')
                redis_client.rpush(f'res-{task_id}', json.dumps({'errorMessage': error}))

        elif task_key == 'label_delete':
            try:
                result = delete_label_data(receiver_pg, task_body)
                redis_client.rpush(f'res-{task_id}', json.dumps(result))
            except Exception:
                error = traceback.format_exc()
                logger.error(f'Failed to label_delete, ex={error}')
                redis_client.rpush(f'res-{task_id}', json.dumps({'errorMessage': error}))

        elif task_key == 'train_commit':
            try:
                training_tasks = get_training_task(receiver_pg)
                if training_tasks:
                    logger.info('已存在正在执行或正在准备的训练任务!!!')
                    logger.info(f'training_tasks={training_tasks}')
                    result = {'errorMessage': '已存在正在执行或正在准备的训练任务!!!'}
                    redis_client.rpush(f'res-{task_id}', json.dumps(result))
                    continue

                train_id = insert_train_task(receiver_pg, task_body)
                train_tasks = list(get_train_by_task_ids(receiver_pg, [train_id]))
                assert train_tasks
                logger.info(train_tasks)
                result = dump_datetime_attr(attr.asdict(train_tasks[0]))
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
                    set_train_task_status(receiver_pg, train_id, '训练失败')

            except Exception:
                error = traceback.format_exc()
                logger.error(f'Failed to train_commit, ex={error}')
                redis_client.rpush(f'res-{task_id}', json.dumps({'errorMessage': error}))

        elif task_key == 'train_getInfo':
            try:
                train_tasks = []
                for train_task in get_train_tasks(receiver_pg):
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
                force = task_body.get('force', False)
                result = delete_train_task(receiver_pg, train_ids, force)
                redis_client.rpush(f'res-{task_id}', json.dumps(result))
            except Exception:
                error = traceback.format_exc()
                logger.error(f'Failed to train_delete, ex={error}')
                redis_client.rpush(f'res-{task_id}', json.dumps({'errorMessage': error}))

        elif task_key == 'model_fetch':
            try:
                task_names = []
                train_tasks = []
                for train_task in get_train_tasks_by_status(receiver_pg, '训练完成'):
                    task_names.append(train_task.name)
                    train_tasks.append(dump_datetime_attr(attr.asdict(train_task)))
                logger.info(
                    f'train_tasks=\n{json.dumps(train_tasks, ensure_ascii=False, indent=2)}'
                )
                redis_client.rpush(f'res-{task_id}', json.dumps(task_names))
            except Exception:
                error = traceback.format_exc()
                logger.error(f'Failed to model_fetch, ex={error}')
                redis_client.rpush(f'res-{task_id}', json.dumps({'errorMessage': error}))

        elif task_key == 'model_commit':
            try:
                upgrading_tasks = list(get_model_tasks_by_status(receiver_pg, '升级中'))
                if upgrading_tasks:
                    logger.info('已存在正在执行的升级任务!!!')
                    logger.info(f'upgrading_tasks={upgrading_tasks}')
                    result = {'errorMessage': '已存在正在执行的升级任务!!!'}
                    redis_client.rpush(f'res-{task_id}', json.dumps(result))
                    continue

                model_id, train_task = insert_upgrade_task(receiver_pg, task_body)
                model_tasks = list(get_model_by_task_ids(receiver_pg, [model_id]))
                assert model_tasks
                logger.info(model_tasks)
                result = dump_datetime_attr(attr.asdict(model_tasks[0]))
                redis_client.rpush(f'res-{task_id}', json.dumps(result))

                try:
                    url = f'http://{AI_HOST}:{AI_PORT}/upgrade'  # noqa
                    logger.info(f'POST to {url}')
                    state_dict_filename = io.file(train_task.storage_path, exists=True).name
                    data = {
                        'state_dict_filename': state_dict_filename,
                        'callback': f'http://{CURRENT_HOST}:{PORT}/complete'  # noqa
                    }
                    rsp = requests.post(url, json=data)
                    logger.info(f'status code = {rsp.status_code}')

                    set_model_task_status(receiver_pg, model_id, '升级完成')
                    reset_using_train_task(receiver_pg)
                    set_train_task_status(receiver_pg, train_task.task_id, '使用中')
                except Exception:
                    error = traceback.format_exc()
                    logger.error(f'Failed to upgrade_ai, ex={error}')
                    set_model_task_status(receiver_pg, model_id, '升级失败')

            except Exception:
                error = traceback.format_exc()
                logger.error(f'Failed to model_commit, ex={error}')
                redis_client.rpush(f'res-{task_id}', json.dumps({'errorMessage': error}))

        elif task_key == 'model_getInfo':
            try:
                model_tasks = []
                for model_task in get_model_tasks(receiver_pg):
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
                force = task_body.get('force', False)
                result = delete_upgrade_task(receiver_pg, model_ids, force)
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


def start_training(
    config_json: str,
    online_model_fd: str,
    license_cer: str,
):
    try:
        license = load_license(license_cer)
        validate_license(license)
    except Exception:
        logger.error(f'Invalid license, ex={traceback.format_exc()}')
        return 1

    config = io.read_json(config_json)
    receiver_pg = PostgresConfig(**config.pop('receiver_postgres'))

    auto_train = AutoTrain(online_model_fd)

    logger.info(f'Connecting to redis ({REDIS_HOST}, {REDIS_PORT})')
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

    huantong_logging_root_fd = None
    if HUANTONG_LEARNING_LOGGING_FOLDER:
        huantong_logging_root_fd = io.folder(
            HUANTONG_LEARNING_LOGGING_FOLDER + '/trainer',
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
            train_tasks = get_training_task(receiver_pg)
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
                set_train_task_status(receiver_pg, train_id, '训练中')

                # todo, 训练中
                database_version = datetime.now().strftime('%Y%m%d')
                logger.info(f'database_version={database_version}')
                # execute_training_process()
                auto_train.run_with_break(database_version)
                set_train_task_status(receiver_pg, train_id, '训练完成')

                storage_path = auto_train.launch_state_dict_file(database_version)
                set_train_task_storage_path(receiver_pg, train_id, storage_path)

            except Exception:
                error = traceback.format_exc()
                logger.error(f'Failed to start_training, ex={error}')
                set_train_task_status(receiver_pg, train_id, '训练失败')

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
            license_cer=''
        )


def debug_auto_train():
    auto_train = AutoTrain('/ai_resource/huantong/ranker')
    auto_train.run_with_break('20221101')
