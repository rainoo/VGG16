import logging
import sys


def init_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(module)s : %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")

    if not logger.handlers:
        logger.addHandler(init_stream_handler(formatter))
        logger.addHandler(init_file_handler(formatter))

    return logger


def init_stream_handler(formatter):
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    return stream_handler


def init_file_handler(formatter, log_filename='log.log', mode='w'):
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    return file_handler
