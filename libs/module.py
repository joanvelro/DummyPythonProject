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
    try:
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
    except Exception as exception_msg:
        logger = []
        print('(!) Error in set_up_logger: ' + str(exception_msg))
        return logger


def load_data(path, logger):
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
        logger.error('(!) Error in load_data: ' + str(exception_msg))
        return df


def check_nan(dataframe, logger):
    """ Check if the input dataframe contains NaN values and fill with 0
    :param dataframe: Data Frame if input data
    :return df0: output dataframe with 0 values in NaN values
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


def get_frequencies(df, column, logger):
    """ Obtain some distribution of frequency to known how is distributed the incidences
    :param df: Data frame
    :param column: Column to obtain the distribution of frequency of occurrence
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


def plot_barplot(df, var_x, var_y, path, logger):
    """ Plot a barplot to compare the incidence of crimes according to characteristics
    :param df: Dataframe to plot
    :param var_x: x-axis variable
    :param var_y: y-axis variable
    :param path: path where it is saved plot in png format
    """
    import matplotlib.pyplot

    try:
        matplotlib.pyplot.bar(x=df[var_x].values,
                              height=df[var_y].values)
        matplotlib.pyplot.grid()
        matplotlib.pyplot.show()
        matplotlib.pyplot.savefig(path)
        matplotlib.pyplot.close()
    except Exception as exception_msg:
        logger.error('(!) Error in plot_barplot:{} when: {} '.format(str(exception_msg), var_x))

