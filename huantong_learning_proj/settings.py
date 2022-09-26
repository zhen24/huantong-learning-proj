import os
import socket

# redis
REDIS_HOST = os.getenv('REDIS_HOST', '0.0.0.0')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))

assert REDIS_HOST and REDIS_PORT


def get_current_host():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(('114.114.114.114', 80))
        host = s.getsockname()[0]
        return host


# current host and port.
CURRENT_HOST = get_current_host()
PORT = int(os.getenv('PORT', 8118))

# device of training.
TRAINING_DEVICE = os.getenv('TRAINING_DEVICE', 'cuda:0')

# HUANTONG_LOGGING_FOLDER
HUANTONG_LOGGING_FOLDER = os.getenv('HUANTONG_LOGGING_FOLDER')

# ai_host and ai_port
AI_HOST = os.getenv('AI_HOST', '10.190.6.12')
AI_PORT = int(os.getenv('AI_PORT', 8500))
BIND_HOST = True
