#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
.. module::Predictive Modelling
   :synopsis: This script fits a machine learning model to classify whether or not a crime register is a violent or not.
   First approach is based on a gradient boosted decission trees model

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
        **Main program of Crimes Predictive Modelling**

        This function execute a predictive modelling process in which the crimes data set is grouped in time-space
        basis, and a flag indicating violent crime or not is added. Then a machine learning model is trained to classify
        the crimes registers into violent /non-violent crimes

    """

    try:
        # Load configuration info
        config_object = configparser.ConfigParser()
        config_object.read('config.ini')
        data_path = config_object._sections['paths']['data_path']
        logs_path = config_object._sections['paths']['logs_path']
        models_path = config_object._sections['paths']['models_path']
        processed_file = config_object._sections['dataset']['processed']
        inference_file = config_object._sections['dataset']['inference']

        # Initialize logs
        log_file_name = 'predictive_modelling'
        logger = dummy_project_utils.set_up_logger(path=logs_path + log_file_name)
        logger.info('Initialize logger')

        logger.info('::: Start Predictive modelling :::')

        # Load data
        logger.info('Load data from: {}'.format(data_path))
        df = dummy_project_utils.load_data(path=data_path + processed_file, logger=logger)
        df.reset_index(inplace=True)

        # Check that the dataframe do not contains NaN values
        logger.info('Check NaN values')
        df = dummy_project_utils.check_nan(dataframe=df, logger=logger)

        # Aggregate historical crime data for each spatial unit (reporting areas) and for each day
        df_aux = df.groupby(['REPORTING_AREA', 'OFFENSE_CODE_GROUP', 'DAY_OF_WEEK'])['index'].count().reset_index() \
            .rename(columns={'index': 'COUNT_CRIMES'}).sort_values(by=['COUNT_CRIMES'], ascending=False)

        # Group crime types into violent and nonviolent crimes.
        # 0: no violent crime, 1: violent crime
        crimes_dict = {
            'Fraud': 0,
            'Investigate Property': 0,
            'Property Lost': 0,
            'Other': 0,
            'Confidence Games': 0,
            'Larceny': 0,
            'Auto Theft': 0,
            'Residential Burglary': 0,
            'Violations': 1,
            'Harassment': 1,
            'Counterfeiting': 0,
            'Larceny From Motor Vehicle': 1,
            'Police Service Incidents': 1,
            'Investigate Person': 1,
            'Recovered Stolen Property': 1,
            'Embezzlement': 0,
            'Motor Vehicle Accident Response': 1,
            'Simple Assault': 1,
            'Warrant Arrests': 1,
            'Bomb Hoax': 1,
            'Vandalism': 1,
            'Missing Person Reported': 1,
            'License Plate Related Incidents': 0,
            'Aggravated Assault': 1,
            'Property Found': 0,
            'Robbery': 1,
            'Restraining Order Violations': 1,
            'Property Related Damage': 1,
            'Missing Person Located': 1,
            'Landlord/Tenant Disputes': 1,
            'Disorderly Conduct': 1,
            'Auto Theft Recovery': 0,
            'Medical Assistance': 0,
            'License Violation': 0,
            'Towed': 0,
            'Service': 0,
            'Verbal Disputes': 1,
            'Liquor Violation': 0,
            'Fire Related Reports': 0,
            'Ballistics': 1,
            'Evading Fare': 0,
            'Operating Under the Influence': 1,
            'Offenses Against Child / Family': 1,
            'Drug Violation': 1,
            'Other Burglary': 0,
            'Firearm Violations': 0,
            'Commercial Burglary': 0,
            'Search Warrants': 0,
            'Biological Threat': 0,
            'Harbor Related Incidents': 0,
            'Firearm Discovery': 1,
            'Prisoner Related Incidents': 1,
            'Homicide': 1,
            'Assembly or Gathering Violations': 1,
            'Manslaughter': 1,
            'Arson': 1,
            'Criminal Harassment': 1,
            'Prostitution': 1,
            'HOME INVASION': 1,
            'Phone Call Complaints': 0,
            'Aircraft': 0,
            'Gambling': 0,
            'INVESTIGATE PERSON': 0,
            'Explosives': 1,
            'HUMAN TRAFFICKING': 1,
            'HUMAN TRAFFICKING - INVOLUNTARY SERVITUDE': 1,
            'Burglary - No Property Taken': 0
        }

        # Creates new column with violent crimes flag classification
        df_aux['VIOLENT_CRIME_FLAG'] = df_aux['OFFENSE_CODE_GROUP'].map(crimes_dict)

        # Split input features
        logger.info('Split input features')
        X = df_aux[['REPORTING_AREA', 'OFFENSE_CODE_GROUP', 'DAY_OF_WEEK', 'COUNT_CRIMES']]
        y = df_aux['VIOLENT_CRIME_FLAG'].values

        # Determine the ratio of inbalance between violent crimes and non-violent crimes
        df_aux2 = df_aux.groupby(['VIOLENT_CRIME_FLAG'])['VIOLENT_CRIME_FLAG'].count()

        ratio_imbalance = df_aux2.loc[0] / df_aux2.loc[1]
        logger.info('Ratio imbalance (no. samples label 1/ no. sables plabel 0): {}'.format(ratio_imbalance))

        # Encode input features
        X = dummy_project_utils.enconde_input_features(x=X, logger=logger)

        # Split train and test data sets
        logger.info('Split train-test data sets')
        # TO DO: replace unpacking like this
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X.iloc[:, -4:],
                                                                                    y,
                                                                                    test_size=0.33,
                                                                                    random_state=42)

        # Initalize classification algorithm (gradient boosted decision trees) with hyperparameters manually introduced
        # To do : train with more classification models (logistic, random forest, adaboost, neural network,..)
        # To do : apply cross-validation and hyperparameter optimization
        model_name = 'GradientBoostingClassifier'
        logger.info('Initialize classification model: {}'.format(model_name))
        model = sklearn.ensemble.GradientBoostingClassifier(loss='deviance',
                                                            n_estimators=100,
                                                            learning_rate=1.0,
                                                            max_depth=1,
                                                            min_samples_split=2,
                                                            min_samples_leaf=1,
                                                            min_weight_fraction_leaf=0,
                                                            max_features=None,
                                                            verbose=0,
                                                            max_leaf_nodes=None,
                                                            criterion='mse',
                                                            random_state=0)
        # Fit model to training data
        logger.info('fit model')
        model.fit(X=X_train,
                  y=y_train)

        # Get mean accuracy on the given test data and labels.
        logger.info('calculate accuracy training')
        accuracy_train = model.score(X=X_train,
                                     y=y_train)
        logger.info('Train Accuracy: {}'.format(accuracy_train))

        # Predict in test
        logger.info('Predict on test dataset')
        y_hat = model.predict(X=X_test)

        # Calculate classification metrics
        dummy_project_utils.calculate_classification_metrics(y_hat=y_hat,
                                                             y_test=y_test,
                                                             logger=logger)

        # Plot test confusion matrix
        sklearn.metrics.plot_confusion_matrix(model, X_test, y_test)
        matplotlib.pyplot.show()

        # Save model
        # to do: Names of the model have to be stored in config ini
        dummy_project_utils.save_model(model=model,
                                       model_name=model_name,
                                       logger=logger,
                                       models_path=models_path)

        ## Save test data set
        logger.info('Save test data set')
        df_test = pandas.DataFrame({'VIOLENT_CRIME_FLAG': y_test})
        X_test.reset_index(inplace=True, drop=True)
        df_test = pandas.concat([df_test, X_test], axis=1)
        df_test.to_csv(path_or_buf=data_path + inference_file, index=False, sep=';')

        logger.info('::: Finish ::: ')

    except Exception as exception_msg:
        logger.error('(!) Error in dummy_project_predictive_modeling.main:{}'.format(str(exception_msg)))


if __name__ == "__main__":
    main()
