#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
.. module:: Visualize
   :synopsis: Visualize results

.. moduleauthor:: Jose Angel Velasco - (C) Tessella Spain by Capgemini Engineering - 2021
"""
# Import dependencies
import configparser
from sys import path
path.append("../../")

# Import custom libraries
import src.utils.utils

# Load configuration info
try:
    config_object = configparser.ConfigParser()
    config_object.read('..\\config.ini')
    data_path = config_object._sections['paths']['data_path']
    logs_path = config_object._sections['paths']['logs_path']
    figures_path = config_object._sections['paths']['figures_path']
    processed_file = config_object._sections['dataset']['processed']
except Exception as exception_msg:
    print('(!) Error in visualize.load_configuration:{}'.format(str(exception_msg)))
    raise

# Initialize reports
try:
    log_file_name = 'visualize'
    logger = src.utils.utils.set_up_logger(path='..\\..\\' + logs_path + log_file_name)
    logger.info('::: Start visualize :::')
except Exception as exception_msg:
    print('(!) Error in visualize.initialize_logging:{}'.format(str(exception_msg)))
    raise

# Load data
try:
    logger.info('Loading processed data')
    df = src.utils.utils.load_data(path='..\\..\\' + data_path + 'processed\\' + processed_file, logger=logger)
    df.reset_index(inplace=True)
except Exception as exception_msg:
    logger.error('(!) Error in visualize.loading_data:{}'.format(str(exception_msg)))
    raise

# Plot scatterplot of crimes
try:
    logger.info('scatterplots')

    # Determine frequency of crimes per location
    df_freq_loc_crimes = src.utils.utils.get_frequencies(df=df,
                                                         column='Location',
                                                         logger=logger)

    # Extract X and Y coordinates
    df_freq_loc_crimes['Location_X'] = df_freq_loc_crimes['Location'].apply(
        lambda x: float(x.strip('()').split(',')[0]))
    df_freq_loc_crimes['Location_Y'] = df_freq_loc_crimes['Location'].apply(
        lambda x: float(x.strip('()').split(',')[1]))

    # Plot
    error = src.utils.utils.plot_scatterplot(df=df_freq_loc_crimes,
                                             var_x='Location_X',
                                             var_y='Location_Y',
                                             scale='FREQ_CRIMES_PERCENTAGE',
                                             path='..\\..\\' + figures_path + 'crimes_map.png',
                                             logger=logger)
except Exception as exception_msg:
    logger.error('(!) Error in visualize.plot_scatter:{}'.format(str(exception_msg)))
    raise

# Bar plot distributions
try:
    logger.info('Bar plot distributions')
    # Obtain some distribution of frequency of occurrence to known how is distributed the crimes
    # Add results to the logging file
    logger.info('Obtain distribution of frequency of crimes occurence')

    # Define relevant columns
    columns = ['OFFENSE_CODE_GROUP', 'REPORTING_AREA', 'DAY_OF_WEEK', 'HOUR', 'MONTH']

    # Iter per columns ang ger the distribution of crimes ocurrence
    logger.info('Iter per columns ang ger the distribution of crimes ocurrence')
    logger.info('save figures in {}'.format(figures_path))

    for column in columns:
        logger.info(column)

        # Get crimes frequencies
        df_freq = src.utils.utils.get_frequencies(df=df,
                                                  column=column,
                                                  logger=logger)

        # Plot crimes frequencies
        src.utils.utils.plot_barplot(df=df_freq[0:20],
                                     var_x=column,
                                     var_y='FREQ_CRIMES_PERCENTAGE',
                                     path='..\\..\\' + figures_path + 'crimes_distribution_per_{}.png'.format(column),
                                     logger=logger)

        # report the first 5 values in the log
        for obs in range(0, 5):
            logger.info('column value:{} frequency of crimes:{}'.format(df_freq.loc[obs][column],
                                                                        df_freq.loc[obs][
                                                                            'FREQ_CRIMES_PERCENTAGE']))
except Exception as exception_msg:
    logger.error('(!) Error in visualize.bar_plot_distributions:{}'.format(str(exception_msg)))
    raise

logger.info('::: Finish :::')
