#!/usr/bin/python
# -*- coding: utf-8 -*-
""" Exploratory Data Analysis of the data set that contains the crimes in the city
    Data path and other configuration settings are indicated in config.ini

    (C) Tessella by Capgemini Engineering - 2021
    joseangel.velascorodriguez@altran.com
"""
# Import dependencies
import pandas
import numpy
import configparser
import matplotlib.pyplot

# Import custom libraries
from libs import module


def main():
    """ Main program of Crime Data Analysis
    """
    # Load configuration info
    config_object = configparser.ConfigParser()
    config_object.read('config.ini')
    data_path = config_object._sections['paths']['data_path']
    logs_path = config_object._sections['paths']['logs_path']
    figures_path = config_object._sections['paths']['figures_path']

    # Initialize logs
    logger = module.set_up_logger(path=logs_path)
    logger.info('Initialize logger')

    # Load data
    logger.info('Load data')
    df = module.load_data(path=data_path, logger=logger)
    df.reset_index(inplace=True)

    # Exist registers in REPORTING AREA column without an acceptable value, change for "unknown"
    df.loc[df['REPORTING_AREA'] == ' ', ['REPORTING_AREA']] = 'unknown'

    # Exist registers in REPORTING AREA column without an acceptable value, change for "unknown"
    df.loc[df['REPORTING_AREA'] == ' ', ['REPORTING_AREA']] = 'unknown'

    # Check that the dataframe do not contains NaN values
    logger.info('Check NaN values')
    df = module.check_nan(dataframe=df, logger=logger)

    # Obtain some distribution of frequency of occurrence to known how is distributed the crimes
    # Add results to the logging file
    logger.info('Obtain distribution of frequency of crimes occurence')

    # Define relevant columns
    columns = ['OFFENSE_CODE', 'OFFENSE_CODE_GROUP', 'REPORTING_AREA', 'DAY_OF_WEEK', 'MONTH', 'UCR_PART']

    # Iter per columns ang ger the distribution of crimes ocurrence
    for column in columns:
        logger.info(column)

        # Get crimes frequencies
        df_freq = module.get_frequencies(df=df,
                                         column=column,
                                         logger=logger)

        # Plot crimes frequencies
        module.plot_barplot(df=df_freq,
                            var_x=column,
                            var_y='FREQ_CRIMES_PERCENTAGE',
                            path=figures_path + '\\plot_{}.png'.format(column),
                            logger=logger)

        # report the first 5 values in the log
        for obs in range(0, 5):
            logger.info('column value:{} frequency of crimes:{}'.format(df_freq.loc[obs][column],
                                                                        df_freq.loc[obs]['FREQ_CRIMES_PERCENTAGE']))


if __name__ == "__main__":
    main()
