#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
.. module::Data Processing
   :synopsis: This script ...

.. moduleauthor:: Jose Angel Velasco - (C) Tessella Spain by Capgemini Engineering - 2021


"""


def load_data(path, logger):
    """
        **Load input data**

        Load the crimes input data

        :param path: path of the csv file to upload
        :type path: str
        :param logger: logger to record exception
        :return: dataframe with the data


    """
    import pandas
    try:
        df = pandas.read_csv(filepath_or_buffer=path, sep=';')
        return df
    except Exception as exception_msg:
        df = []
        logger.error('(!) Error in load_data: ' + str(exception_msg))
        return df


def check_nan(dataframe, logger):
    """
        **Check NaNs**

        Check if the input dataframe contains NaN values and fill with 0

        :param dataframe: Data Frame if input data
        :param logger: logger to record exception
        :return: output dataframe with 0 values in NaN values
    """
    df0 = dataframe.copy()
    try:
        if df0.isna().sum().sum() > 0:
            for col in df0.columns:
                if df0[col].isna().sum() > 0:
                    df0[col].fillna(value=0, inplace=True)
        return df0
    except Exception as exception_msg:
        logger.error('(!) Error in check_nan: ' + str(exception_msg))
        return df0


def load_config_info(path):
    """
        *Load Configuration Info*

        Load configuration info

        :param path: path of the config.ini file
        :return: dictionary with config data
    """
    import configparser
    try:
        config_object = configparser.ConfigParser()
        config_object.read(path)

        return config_object._sections
    except Exception as exception_msg:
        print('(!) Error in load_config_info:{}'.format(str(exception_msg)))
