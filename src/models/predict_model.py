#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
.. module::Inference
   :synopsis: This function execute a inference procedure to decide whether a crime is violent or not based on the machine learning
        model trained for classification

.. moduleauthor:: Jose Angel Velasco - (C) Tessella Spain by Capgemini Engineering - 2021


"""
# Import dependencies
import configparser
import pickle
# TO DO: Import only sklearn
import sklearn
import sklearn.preprocessing
import sklearn.metrics
import sklearn.model_selection
import sklearn.ensemble
import matplotlib.pyplot
import pandas
import glob

from sys import path

path.append("../../")

# Import custom libraries
try:
    import src.utils.utils
    import src.data.data_processing
    import src.models.train_model
except Exception as exception_msg:
    print('(!) Error in importing custom functions: ' + str(exception_msg))

# Load configuration info
configuration = src.data.data_processing.load_config_info(path='..\\config.ini')

data_path = configuration['paths']['data_path']
logs_path = configuration['paths']['logs_path']
models_path = configuration['paths']['models_path']
file_inference = configuration['dataset']['inference']

# Initialize reports
log_file_name = 'predict_model'
logger = src.utils.utils.set_up_logger(path=logs_path + log_file_name)
logger.info('Initialize logger')
logger.info('::: Start Inference :::')

# Load data
logger.info('Load input data from: {}'.format(data_path))
df = src.data.data_processing.load_data(path=data_path + 'processed\\' + file_inference, logger=logger)

# Define features used
# TO DO: load features used in predictive modelling
features = ['REPORTING_AREA_ENCODED', 'OFFENSE_CODE_GROUP_ENCODED', 'DAY_OF_WEEK_ENCODED', 'COUNT_CRIMES_ENCODED']

# Split input features
logger.info('Split input features')
X = df[features]
y_true = df['VIOLENT_CRIME_FLAG'].values

# Load models
model_names = glob.glob(models_path + '*.sav')
for model in model_names:
    model = pickle.load(open(model, 'rb'))
    y_hat = model.predict(X=X)

    # Calculate classification metrics
    logger.info('Calculate classification metrics')
    metrics = src.models.train_model.calculate_classification_metrics(y_hat=y_hat,
                                                                      y_test=y_true,
                                                                      logger=logger)

    # Plot test confusion matrix
    logger.info('Plot test confusion matrix')
    sklearn.metrics.plot_confusion_matrix(model, X, y_hat)
    matplotlib.pyplot.show()

logger.info('::: Finish ::: ')
