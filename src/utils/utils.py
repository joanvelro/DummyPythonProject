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

        Configure the logger to record all the envents in the execution of the code

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


def plot_barplot(df, var_x, var_y, path, logger):
    """

        **Plot Barplot**

        Plot a barplot to compare the incidence of crimes according to characteristics

        :param df: Dataframe to plot
        :param var_x: x-axis variable
        :param var_y: y-axis variable
        :param path: path where it is saved plot in png format
        :param logger: logger to record exception
    """
    import matplotlib.pyplot

    try:
        matplotlib.pyplot.figure(figsize=(14, 14))
        matplotlib.pyplot.barh(y=df[var_x].values,
                               width=df[var_y].values,
                               align='center',
                               )
        matplotlib.pyplot.grid(True)
        matplotlib.pyplot.xlabel(var_x, fontsize=FONTSIZE)
        matplotlib.pyplot.ylabel(var_y, fontsize=FONTSIZE)
        matplotlib.pyplot.xticks(rotation=45)
        matplotlib.pyplot.tick_params(axis='both', labelsize=FONTSIZE)
        matplotlib.pyplot.savefig(path)
        matplotlib.pyplot.show()
        matplotlib.pyplot.close()
    except Exception as exception_msg:
        logger.error('(!) Error in plot_barplot:{}'.format(str(exception_msg)))


def plot_scatterplot(df, var_x, var_y, scale, path, logger):
    """

        **Plot scatter**

        plot with the coordinates of the crimes

        :param df: Dataframe to plot
        :param var_x: x-axis variable
        :param var_y: y-axis variable
        :param scale: scale of the point
        :param path: path where it is saved plot in png format
    """
    import matplotlib.pyplot

    try:
        matplotlib.pyplot.figure(figsize=(14, 10))
        matplotlib.pyplot.scatter(x=df[var_x],
                                  y=df[var_y],
                                  s=200 * df[scale],
                                  alpha=0.9,
                                  edgecolors='blue')
        matplotlib.pyplot.grid(True)
        matplotlib.pyplot.xlabel(var_x, fontsize=FONTSIZE)
        matplotlib.pyplot.ylabel(var_y, fontsize=FONTSIZE)
        matplotlib.pyplot.savefig(path)
        matplotlib.pyplot.show()
        matplotlib.pyplot.close()
    except Exception as exception_msg:
        logger.error('(!) Error in plot_scatterplot:{}'.format(str(exception_msg)))







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
