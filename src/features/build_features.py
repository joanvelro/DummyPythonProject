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
from sys import path

path.append("../../")

# Import custom libraries
import src.utils.utils


def main():
    """
        **Main program of Crimes Exploratory Data Analysis**

        This function execute a descriptive analysis of the crimes data set and provides insight about
        and it can support de decision process related to surveillance schedule

    """
    # Load configuration info
    try:
        config_object = configparser.ConfigParser()
        config_object.read('..//config.ini')
        data_path = config_object._sections['paths']['data_path']
        logs_path = config_object._sections['paths']['logs_path']
        figures_path = config_object._sections['paths']['figures_path']
        raw_file = config_object._sections['dataset']['raw']
        processed_file = config_object._sections['dataset']['processed']
    except Exception as exception_msg:
        print('(!) Error in build_features.load_configuration:{}'.format(str(exception_msg)))
        raise

    # Initialize reports
    try:
        log_file_name = 'build_features'
        logger = src.utils.utils.set_up_logger(path='..//..//' + logs_path + log_file_name)
        logger.info('::: Start Exploratory data analysis :::')
    except Exception as exception_msg:
        print('(!) Error in build_features.initialize_logging:{}'.format(str(exception_msg)))
        raise

    # Load data
    try:
        logger.info('Loading raw data')
        df = src.utils.utils.load_data(path='..\\..\\' + data_path + 'raw\\' + raw_file, logger=logger)
        df.reset_index(inplace=True)
    except Exception as exception_msg:
        logger.error('(!) Error in build_features.loading_data:{}'.format(str(exception_msg)))
        raise

    # preprocessing
    try:
        logger.info('Data Pre-processing')
        # Exist registers in REPORTING AREA column without an acceptable value, change for "unknown"
        df.loc[df['REPORTING_AREA'] == ' ', ['REPORTING_AREA']] = 'unknown'

        # Check that the dataframe do not contains NaN values
        logger.info('Check NaN values')
        df = src.utils.utils.check_nan(dataframe=df, logger=logger)

        # Extract X and Y coordinates
        df['Location_X'] = df['Location'].apply(lambda x: float(x.strip('()').split(',')[0]))
        df['Location_Y'] = df['Location'].apply(lambda x: float(x.strip('()').split(',')[1]))

        # Filter by the crimes that have a valid location
        df = df[(df['Location_X'] != 0) & (df['Location_Y'] != 0)]
        df = df[(df['Location_X'] != -1) | (df['Location_Y'] != -1)]

    except Exception as exception_msg:
        logger.error('(!) Error in build_features.pre_processing:{}'.format(str(exception_msg)))
        raise

    # Save processed data set
    try:
        logger.info('Save processed data set')
        df.drop(columns=['index'], inplace=True)
        df.to_csv(path_or_buf='..\\..\\' + data_path + 'processed\\' + processed_file, index=False, sep=';')
    except Exception as exception_msg:
        logger.error('(!) Error in build_features.save_processed_data:{}'.format(str(exception_msg)))
        raise

    logger.info('::: Finish :::')


if __name__ == "__main__":
    main()
