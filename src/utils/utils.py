#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
.. module:: utils
   :synopsis: This script define the different functions to be used in the project

.. moduleauthor:: Jose Angel Velasco (C) Tessella Spain by Capgemini Engineering - 2021


"""

# GLOBAL VARIABLES
FONTSIZE = 12


def set_up_logger(path):
    """
        **Set up logger**

        Configure the logger to record all the events in the execution of the code

        :param path: path where to store reports example: 'reports\\log_file_name'
        :type path: str
        :param logger: logger to record exception
        :return: logger logger

    """
    import logging
    try:
        logger = logging.getLogger(path)
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler('{}.log'.format(path))
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logging.getLogger().addHandler(logging.StreamHandler())  # to display in console message
        return logger
    except Exception as exception_msg:
        logger = []
        print('(!) Error in set_up_logger: ' + str(exception_msg))
        return logger


def get_frequencies(df, column, logger):
    """
        **Get frequencies**

        Obtain some distribution of frequency to known how is distributed the incidences

        :param df: Data frame
        :param column: Column to obtain the distribution of frequency of occurrence
        :param logger: logger to record exception
        :return: dataframe with the crimes frequencies
    """

    try:
        # Check that exist a column named index
        if not 'index' in df:
            df['index'] = df.index
        # Get distribution of frequency of OFFENSE_CODE
        df_aux = df.groupby([column])['index'].count().reset_index() \
            .sort_values(by=['index'], ascending=False) \
            .reset_index(drop=True).rename(columns={'index': 'COUNT_CRIMES'})
        # Get frequency in percentage
        df_aux['FREQ_CRIMES_PERCENTAGE'] = 100 * df_aux['COUNT_CRIMES'] / df.shape[0]

        return df_aux

    except Exception as exception_msg:
        logger.error('(!) Error in get_frequencies: ' + str(exception_msg))
        df_aux = []
        return df_aux



def enconde_input_features(x, logger):
    """
        **Encode Input Features**
        Encode the input features using label encoder

        :param x: INput features dataframe
        :param logger: logger:
        :return x: Data frame with encoded input features (_ENCODED)
    """
    import sklearn.preprocessing

    try:
        # Initialize encoder for input features
        logger.info('Encode input features (label encoder)')
        le = sklearn.preprocessing.LabelEncoder()

        # encode input features
        for column in x.columns:
            le.fit(x[column])
            x[column + '_ENCODED'] = le.transform(x[column])

        return x
    except Exception as exception_msg:
        x = []
        logger.error('(!) Error in enconde_input_features:{}'.format(str(exception_msg)))
        return x
