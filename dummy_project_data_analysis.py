#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
.. module:: Exploratory Data Analysis
   :synopsis: Exploratory Data Analysis of the data set that contains the crimes in the city
    Data path and other configuration settings are indicated in config.ini

.. moduleauthor:: Jose Angel Velasco - (C) Tessella Spain by Capgemini Engineering - 2021


"""
# Import dependencies
import configparser
# TO DO: Import only sklearn
import sklearn
import sklearn.preprocessing
import sklearn.metrics
import sklearn.model_selection
import sklearn.ensemble
import matplotlib.pyplot
import pandas

# Import custom libraries
import dummy_project_utils


def main():
    """
        **Main program of Crimes Exploratory Data Analysis**

        This function execute a descriptive analysis of the crimes data set and provides insight about
        and it can support de decision process related to surveillance schedule
    """
    # Load configuration info
    config_object = configparser.ConfigParser()
    config_object.read('config.ini')
    data_path = config_object._sections['paths']['data_path']
    logs_path = config_object._sections['paths']['logs_path']
    figures_path = config_object._sections['paths']['figures_path']

    # Initialize logs
    log_file_name = 'exploratory_data_analysis'
    logger = dummy_project_utils.set_up_logger(path=logs_path + log_file_name)
    logger.info('Initialize logger')

    # Load data
    logger.info('Load data')
    df = dummy_project_utils.load_data(path=data_path, logger=logger)
    df.reset_index(inplace=True)

    # Exist registers in REPORTING AREA column without an acceptable value, change for "unknown"
    df.loc[df['REPORTING_AREA'] == ' ', ['REPORTING_AREA']] = 'unknown'

    # Check that the dataframe do not contains NaN values
    logger.info('Check NaN values')
    df = dummy_project_utils.check_nan(dataframe=df, logger=logger)

    # Extract X and Y coordinates
    df['Location_X'] = df['Location'].apply(lambda x: float(x.strip('()').split(',')[0]))
    df['Location_Y'] = df['Location'].apply(lambda x: float(x.strip('()').split(',')[1]))

    # Filter by the crimes that have a valid location
    df = df[(df['Location_X'] != 0) & (df['Location_Y'] != 0)]
    df = df[(df['Location_X'] != -1) | (df['Location_Y'] != -1)]

    # Determine frequency of crimes per location
    df_freq_loc_crimes = dummy_project_utils.get_frequencies(df=df,
                                                             column='Location',
                                                             logger=logger)
    # Extract X and Y coordinates
    df_freq_loc_crimes['Location_X'] = df_freq_loc_crimes['Location'].apply(
        lambda x: float(x.strip('()').split(',')[0]))
    df_freq_loc_crimes['Location_Y'] = df_freq_loc_crimes['Location'].apply(
        lambda x: float(x.strip('()').split(',')[1]))

    # Plot scatterplot of crimes
    error = dummy_project_utils.plot_scatterplot(df=df_freq_loc_crimes,
                                                 var_x='Location_X',
                                                 var_y='Location_Y',
                                                 scale='FREQ_CRIMES_PERCENTAGE',
                                                 path=figures_path + 'crimes_map.png',
                                                 logger=logger)

    # Obtain some distribution of frequency of occurrence to known how is distributed the crimes
    # Add results to the logging file
    logger.info('Obtain distribution of frequency of crimes occurence')

    # Define relevant columns
    columns = ['OFFENSE_CODE_GROUP', 'REPORTING_AREA', 'DAY_OF_WEEK', 'HOUR', 'MONTH']

    # Iter per columns ang ger the distribution of crimes ocurrence
    for column in columns:
        logger.info(column)

        # Get crimes frequencies
        df_freq = dummy_project_utils.get_frequencies(df=df,
                                                      column=column,
                                                      logger=logger)

        # Plot crimes frequencies
        dummy_project_utils.plot_barplot(df=df_freq[0:20],
                                         var_x=column,
                                         var_y='FREQ_CRIMES_PERCENTAGE',
                                         path=figures_path + 'crimes_distribution_per_{}.png'.format(column),
                                         logger=logger)

        # report the first 5 values in the log
        for obs in range(0, 5):
            logger.info('column value:{} frequency of crimes:{}'.format(df_freq.loc[obs][column],
                                                                        df_freq.loc[obs]['FREQ_CRIMES_PERCENTAGE']))


    logger.info('Finish')


if __name__ == "__main__":
    main()
