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

# Obtain some distribution of frequency to known how is distributed the incidences
# Get distribution of frequency of OFFENSE_CODE
df_aux1 = df.groupby(['OFFENSE_CODE'])['index'].count().reset_index() \
    .sort_values(by=['index'], ascending=False) \
    .reset_index(drop=True).rename(columns={'index': 'COUNT_CRIMES'})

# Get distribution of frequency of OFFENSE_CODE
df_aux2 = df.groupby(['OFFENSE_CODE_GROUP'])['index'].count().reset_index() \
    .sort_values(by=['index'], ascending=False) \
    .reset_index(drop=True).rename(columns={'index': 'COUNT_CRIMES'})

# Exist registers in REPORTING AREA column without an acceptable value, change for "unknown"
df.loc[df['REPORTING_AREA'] == ' ', ['REPORTING_AREA']] = 'unknown'

# Get distribution of frequency of OFFENSE_CODE
df_aux3 = df.groupby(['REPORTING_AREA'])['index'].count().reset_index() \
    .sort_values(by=['index'], ascending=False) \
    .reset_index(drop=True).rename(columns={'index': 'COUNT_CRIMES'})

# Get distribution of frequency in TIME
df_aux4 = df.groupby(['YEAR', 'MONTH', 'DAY_OF_WEEK', 'HOUR'])['index'].count() \
    .reset_index().rename(columns={'index': 'COUNT_CRIMES'}).sort_values(by=['COUNT_CRIMES'], ascending=False)

# Get distribution of frequency in DAY_OF_WEEK
df_aux5 = df.groupby(['DAY_OF_WEEK'])['index'].count() \
    .reset_index().rename(columns={'index': 'COUNT_CRIMES'}) \
    .sort_values(by=['COUNT_CRIMES'], ascending=False)

# Get distribution of frequency in MONTH
df_aux6 = df.groupby(['MONTH'])['index'].count() \
    .reset_index().rename(columns={'index': 'COUNT_CRIMES'}) \
    .sort_values(by=['COUNT_CRIMES'], ascending=False)
# Get frequency in percentage
df_aux6['FREQ_CRIMES_PERC'] = 100 * df_aux6['COUNT_CRIMES'] / df.shape[0]

# Get distribution of frequency in UCR_PART
df_aux7 = df.groupby(['UCR_PART'])['index'].count() \
    .reset_index().rename(columns={'index': 'COUNT_CRIMES'}) \
    .sort_values(by=['COUNT_CRIMES'], ascending=False)
