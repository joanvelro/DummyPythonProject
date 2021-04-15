#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
    This script define the different function to be used in the project

    (C) Tessella by Capgemini Engineering - 2021
    joseangel.velascorodriguez@altran.com
"""


def set_up_logger(path):
    """ Set up logger to capture logs
    :param path: path where to store logs example: 'logs\\log_file_name'
    :return logger: logger
    """
    import logging
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('{}.log'.format(path))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # logging.getLogger().addHandler(logging.StreamHandler())  # to display in console message
    return logger


def load_data(path):
    """ Load input data
    :param path: path of the csv file to upload
    :return df: dataframe with the data
    """
    import pandas
    try:
        df = pandas.read_csv(filepath_or_buffer=path, sep=';')
        return df
    except Exception as exception_msg:
        df = []
        print('(!) Error in load_data: ' + str(exception_msg))
        return df
