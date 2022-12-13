import json
import logging
import time
import traceback
from typing import Mapping, Sequence, Any, Dict

import redis
import shortuuid
from flask import Flask, request, jsonify

from huantong_learning_proj.settings import (
    REDIS_HOST,
    REDIS_PORT,
    CURRENT_HOST,
    TRAINING_DEVICE,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s: [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
app = Flask(__name__)


def get_task_result(task_id: str):
    # logger.info('redis waiting...')
    item = redis_client.blpop(f'res-{task_id}', 100)
    if item is None:
        return {'errorMessage': 'Response timeout...'}
    result: Dict[str, Any] = json.loads(item[1])
    return result


def add_task(
    task_key: str,
    task_body: Mapping[str, Any],
    inspect_keys: Sequence[str],
):
    inspect_body = {inspect_key: task_body.get(inspect_key) for inspect_key in inspect_keys}
    task_id = str(shortuuid.uuid())
    task = {
        'id': task_id,
        'key': task_key,
        'body': task_body,
    }
    task_str = json.dumps(task)
    logger.info(
        f'Add task key={task_key}, inspect_body={inspect_body}, len(task_str)={len(task_str)}'
    )
    redis_client.rpush('receiving_tasks', task_str)
    return task_id


def api_add_task(task_key: str, inspect_keys: Sequence[str]):
    try:
        if inspect_keys:
            method = 'POST'
            task_body = request.get_json(force=True)
        else:
            method = 'GET'
            task_body = dict()
        logger.info(f'{method} /{task_key}')

        task_id = add_task(task_key, task_body, inspect_keys)
        api_result = get_task_result(task_id)
        # logger.info(f'api_result={api_result}')

        if not isinstance(api_result, dict) or not api_result.get('errorMessage'):
            return jsonify(api_result), 200
        return jsonify(api_result), 400
    except Exception:
        error = traceback.format_exc()
        logger.error(error)
        return jsonify({'errorMessage': error}), 400


@app.route('/label/commit', methods=['POST', 'OPTIONS'])
def label_commit():
    return api_add_task(
        'label_commit',
        [
            'label_id',
            'tenant_id',
        ],
    )


@app.route('/label/delete', methods=['POST', 'OPTIONS'])
def label_delete():
    return api_add_task(
        'label_delete',
        [
            'label_ids',
        ],
    )


@app.route('/train/commit', methods=['POST', 'OPTIONS'])
def train_commit():
    return api_add_task(
        'train_commit',
        [
            'task_id',
            'name',
            'description',
        ],
    )


@app.route('/start_training')
def start_training():
    logger.info('GET /start_training')
    logger.info(f"The ranker_actor on {CURRENT_HOST}:{TRAINING_DEVICE} has been removed.")

    task = {
        'execution_time': time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    task_str = json.dumps(task)
    logger.info(f'Add task key=start_training, inspect_body={task}, len(task_str)={len(task_str)}')
    redis_client.rpush('training_tasks', task_str)
    logger.info('已启动训练任务.')

    return jsonify({'msg': '已启动训练任务.'}), 200


@app.route('/train/getInfo')
def train_get_info():
    return api_add_task(
        'train_getInfo',
        [],
    )


@app.route('/restore_ai')
def restore_ai():
    logger.info('GET /restore_ai')
    logger.info(f"The ranker_actor on {CURRENT_HOST}:{TRAINING_DEVICE} has been restored.")
    logger.info('训练完成.')

    return jsonify({'msg': '训练完成.'}), 200


@app.route('/train/delete', methods=['POST', 'OPTIONS'])
def train_delete():
    return api_add_task(
        'train_delete',
        [
            'task_ids',
        ],
    )


@app.route('/model/fetch')
def model_fetch():
    return api_add_task(
        'model_fetch',
        [],
    )


@app.route('/model/commit', methods=['POST', 'OPTIONS'])
def model_commit():
    return api_add_task(
        'model_commit',
        [
            'task_id',
            'name',
            'description',
            'model_source',
        ],
    )


@app.route('/model/getInfo')
def model_get_info():
    return api_add_task(
        'model_getInfo',
        [],
    )


@app.route('/complete')
def complete():
    logger.info('GET /complete')
    logger.info('The model is updated successfully')
    logger.info('模型更新完成.')

    return jsonify({'msg': '模型更新完成.'}), 200


@app.route('/model/delete', methods=['POST', 'OPTIONS'])
def model_delete():
    return api_add_task(
        'model_delete',
        [
            'task_ids',
        ],
    )


if __name__ == "__main__":
    HOST = '0.0.0.0'
    PORT = 8118
    app.run(host=HOST, port=PORT)
