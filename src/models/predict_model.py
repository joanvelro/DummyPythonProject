#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
.. module::Inference
   :synopsis: This script loads the predictive model fitted and provides new predictions for a new data set with the
   the same structure that the data set of the predictive modelling phase

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

# Import custom libraries
import dummy_project_utils


def main():
    """
        **Main program of Crimes Inference**

        This function execute a inference procedure to decide whether a crime is violent or not based on the machine learning
        model trained for classification
    """

    try:
        # Load configuration info
        config_object = configparser.ConfigParser()
        config_object.read('config.ini')
        data_path = config_object._sections['paths']['data_path']
        logs_path = config_object._sections['paths']['logs_path']
        models_path = config_object._sections['paths']['models_path']
        file_inference = config_object._sections['dataset']['inference']

        # Initialize reports
        log_file_name = 'inference'
        logger = dummy_project_utils.set_up_logger(path=logs_path + log_file_name)
        logger.info('Initialize logger')

        logger.info('::: Start Inference :::')

        # Load data
        logger.info('Load input data from: {}'.format(data_path))
        df = dummy_project_utils.load_data(path=data_path + file_inference, logger=logger)

        # Define features used
        # TO DO: load features used in predictive modelling
        features = ['REPORTING_AREA_ENCODED', 'OFFENSE_CODE_GROUP_ENCODED', 'DAY_OF_WEEK_ENCODED', 'COUNT_CRIMES_ENCODED']

        # Split input features
        logger.info('Split input features')
        X = df[features]
        y_true = df['VIOLENT_CRIME_FLAG'].values

        # Load model
        # to do_ load models names from congif ini
        model_name = 'clf_model_GradientBoostingClassifier.sav'
        model = pickle.load(open(models_path + model_name, 'rb'))
        y_hat = model.predict(X=X)

        # Calculate classification metrics
        logger.info(' Calculate classification metrics')
        dummy_project_utils.calculate_classification_metrics(y_hat=y_hat,
                                                             y_test=y_true,
                                                             logger=logger)

        # Plot test confusion matrix
        logger.info('Plot test confusion matrix')
        sklearn.metrics.plot_confusion_matrix(model, X, y_hat)
        matplotlib.pyplot.show()

        logger.info('::: Finish ::: ')

    except Exception as exception_msg:
        logger.error('(!) Error in dummy_project_inference.main:{}'.format(str(exception_msg)))


if __name__ == "__main__":
    main()
