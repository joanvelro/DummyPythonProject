#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
    Exploratory Data Analysis of the data set that contains the crimes in the city
    Data path and other configuration settings are indicated in config.ini

    (C) Tessella by Capgemini Engineering - 2021
    joseangel.velascorodriguez@altran.com
"""
# Import dependencies
import pandas
import numpy
import configparser

# Import custom libraries
import module


def main_data_analysis():
    """ Main program of Data Analysis

    """
    # Load configuration info
    config_object = configparser.ConfigParser()
    config_object.read('config.ini')
    data_path = config_object._sections['paths']['data_path']
    logs_path = config_object._sections['paths']['logs_path']

    # Initialize logs
    logger = module.set_up_logger(path=logs_path)
    logger.info('Initialize logger')

    # Load data
    logger.info('Load data')
    df = module.load_data(path=data_path)
    df.reset_index(inplace=True)

    # Exist registers in REPORTING AREA column without an acceptable value, change for "unknown"
    df.loc[df['REPORTING_AREA'] == ' ', ['REPORTING_AREA']] = 'unknown'

    # Check that the dataframe do not contains NaN values
    logger.info('Check NaN values')
    df = module.check_nan(dataframe=df)

    # Obtain some distribution of frequency of occurrence to known how is distributed the crimes
    # Add results to the logging file
    logger.info('Obtain distribution of frequency of crimes occurence')
    df_freq_var1 = module.get_frequencies(df, 'OFFENSE_CODE')
    logger.info('OFFENSE_CODE')
    for obs in range(0, 3):
        logger.info(df_freq_var1.loc[obs]['FREQ_CRIMES_PERCENTAGE'])
    df_freq_var2 = module.get_frequencies(df, 'OFFENSE_CODE_GROUP')
    df_freq_var3 = module.get_frequencies(df, 'REPORTING_AREA')
    df_freq_var4 = module.get_frequencies(df, 'DAY_OF_WEEK')
    df_freq_var5 = module.get_frequencies(df, 'MONTH')
    df_freq_var6 = module.get_frequencies(df, 'UCR_PART')


#if __name__ == "__main__":
#    main_data_analysis()
