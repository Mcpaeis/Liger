import logging
import sys


def config_root_logger_stout():
    """
    Configures the root logger to use the standard out instead of sterr
    :return:
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch)

def config_root_logger_file_handler(file_path, append=True):
    root = logging.getLogger()
    if append:
        mode='a'
    else:
        mode='w'
    handler = logging.FileHandler(file_path,mode=mode)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)

