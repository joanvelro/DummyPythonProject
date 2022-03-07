#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
.. module::Predictive Modelling
   :synopsis: This function execute a predictive modelling process in which the crimes data set is grouped in time-space
        basis, and a flag indicating violent crime or not is added. Then a machine learning model is trained to classify
        the crimes registers into violent /non-violent crimes

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
import sklearn.neighbors
import sklearn.linear_model
import sklearn.neural_network

import pandas
from sys import path

path.append("../../")

# Import custom libraries
try:
    import src.utils.utils
    import src.data.data_processing
except Exception as exception_msg:
    print('(!) Error in importing custom functions: ' + str(exception_msg))


def calculate_classification_metrics(y_hat, y_test, logger):
    """
        **Calculate Classification Metrics**

         Calculate deveral classifcation metrics: confusion matrix, precision, recall and AUC ROC

         :param y_hat: actual labels obteined
         :param y_test: target labels
         :param logger: logger to record errors
         :return: list with metrics: confusion matrix, precision, recall, roc

    """
    import sklearn.metrics

    try:
        # Calculate confusion matrix
        CM = sklearn.metrics.confusion_matrix(y_true=y_test,
                                              y_pred=y_hat)

        # Get  true positives  / (true positives  + false positives)
        precision = sklearn.metrics.precision_score(y_true=y_test,
                                                    y_pred=y_hat)

        # Get  ratio true positives / (true positives + false negatives)
        recall = sklearn.metrics.recall_score(y_true=y_test,
                                              y_pred=y_hat)

        # get Area Under the Receiver Operating Characteristic Curve (ROC AUC)
        roc = sklearn.metrics.roc_auc_score(y_true=y_test,
                                            y_score=y_hat,
                                            average='macro')

        return [CM, precision, recall, roc]
    except Exception as exception_msg:
        return []
        logger.error('(!) Error in calculate_classification_metrics:{}'.format(str(exception_msg)))


def save_data_set(path, target, target_label, input_features):
    """
        *Save data set test*

        Save the dataset for testing

        :param path:
        :param target:
        :param input_features:
    """
    import pandas

    try:
        df_test = pandas.DataFrame({target_label: target})
        input_features.reset_index(inplace=True, drop=True)
        df_test = pandas.concat([df_test, input_features], axis=1)
        df_test.to_csv(path_or_buf=path, index=False, sep=';')
    except Exception as exception_msg:
        print('(!) Error in save_data_set:{}'.format(str(exception_msg)))


def save_model(model, model_name, logger, models_path):
    """
        ** Save Model**

        Save the predictive mdel in .sav file in models path indicated in config.ini file

        :param model: Model
        :param model_name: model name (str)
        :param logger: logger
        :param models_path: path to the models folder (str)
    """

    import pickle
    try:
        model_file_name = 'clf_model_{}.sav'.format(model_name)
        pickle.dump(model, open(models_path + model_file_name, 'wb'))
    except Exception as exception_msg:
        logger.error('(!) Error in save_model:{}'.format(str(exception_msg)))


def main():
    # Load configuration info
    configuration = src.data.data_processing.load_config_info(path='..\\config.ini')
    data_path = configuration['paths']['data_path']
    logs_path = configuration['paths']['logs_path']
    models_path = configuration['paths']['models_path']
    processed_file = configuration['dataset']['processed']
    inference_file = configuration['dataset']['inference']

    # Initialize reports
    log_file_name = 'train_model'
    logger = src.utils.utils.set_up_logger(path=logs_path + log_file_name)
    logger.info('Initialize logger')
    logger.info('::: Start Predictive modelling :::')

    # Load data
    logger.info('Load data from')
    df = src.data.data_processing.load_data(path=data_path + 'processed\\' + processed_file, logger=logger)
    df.reset_index(inplace=True)

    # Check that the dataframe do not contains NaN values
    logger.info('Check NaN values')
    df = src.data.data_processing.check_nan(dataframe=df, logger=logger)

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
    features = ['REPORTING_AREA', 'OFFENSE_CODE_GROUP', 'DAY_OF_WEEK', 'COUNT_CRIMES']
    X = df_aux[features]
    y = df_aux['VIOLENT_CRIME_FLAG'].values

    # Determine the ratio of inbalance between violent crimes and non-violent crimes
    df_aux2 = df_aux.groupby(['VIOLENT_CRIME_FLAG'])['VIOLENT_CRIME_FLAG'].count()

    ratio_imbalance = df_aux2.loc[0] / df_aux2.loc[1]
    logger.info('Ratio imbalance (no. samples label 1/ no. sables plabel 0): {}'.format(ratio_imbalance))

    # Encode input features
    X = src.utils.utils.enconde_input_features(x=X, logger=logger)

    # Split train and test data sets
    logger.info('Split train-test data sets')
    # TO DO: replace unpacking like this
    split = sklearn.model_selection.train_test_split(X.iloc[:, -4:],
                                                     y,
                                                     test_size=0.33,
                                                     random_state=42)
    X_train = split[0]
    X_test = split[1]
    y_train = split[2]
    y_test = split[3]

    logger.info('Save test data set')
    save_data_set(path=data_path + inference_file,
                  target=y_test,
                  target_label='VIOLENT_CRIME_FLAG',
                  input_features=X_test)

    # Initalize classification algorithm (gradient boosted decision trees) with hyperparameters manually introduced
    # To do : train with more classification models (logistic, random forest, adaboost, neural network,..)
    # To do : apply cross-validation and hyperparameter optimization

    models = ['GradientBoostingClassifier', 'SGDClassifier', 'LinearSVC', 'MLPClassifier', 'RandomForestClassifier']
    precision_ = []
    recall_ = []
    roc_ = []
    for modeln in models:
        logger.info('Initialize classification model: {}'.format(modeln))
        if modeln == 'GradientBoostingClassifier':
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
        if modeln == 'KNeighborsClassifier':
            model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5,
                                                           weights='uniform',
                                                           algorithm='auto',
                                                           leaf_size='30',
                                                           metric='minkowski')
        if modeln == 'SGDClassifier':
            model = sklearn.linear_model.SGDClassifier()
        if modeln == 'LinearSVC':
            model = sklearn.svm.LinearSVC(max_iter=1000)
        if modeln == 'MLPClassifier':
            model = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100, 100, 100),
                                                         solver='sgd',
                                                         activation='relu')
        if modeln == 'RandomForestClassifier':
            model = sklearn.ensemble.RandomForestClassifier()

        # Fit model to training data
        logger.info('fit model')
        try:
            model.fit(X=X_train,
                      y=y_train)
            error = 0
        except Exception as exception_msg:
            logger.error('(!) Error in fit: ' + str(exception_msg))
            error = 1

        if error == 0:
            # Get mean accuracy on the given test data and labels.
            logger.info('calculate accuracy training')
            accuracy_train = model.score(X=X_train,
                                         y=y_train)
            logger.info('Train Accuracy: {}'.format(accuracy_train))

            # Predict in test
            logger.info('Predict on test dataset')
            y_hat = model.predict(X=X_test)

            # Calculate classification metrics
            metrics = calculate_classification_metrics(y_hat=y_hat,
                                                       y_test=y_test,
                                                       logger=logger)
            CM = metrics[0]  # confusion matrix
            FP = CM[0, 1]  # False Positive
            TN = CM[1, 0]  # True Negatives
            precision = metrics[1]  # Precision
            recall = metrics[2]  # Recall
            roc = metrics[3]  # Roc

            precision_.append(precision)
            recall_.append(recall)
            roc_.append(roc)


            logger.info('Test Confusion Matrix: {}'.format(CM))
            logger.info('Test Precision: {}'.format(precision))
            logger.info('Test Recall: {}'.format(recall))
            logger.info('Test AUC ROC: {}'.format(roc))

            # Plot test confusion matrix
            sklearn.metrics.plot_confusion_matrix(model, X_test, y_test)
            matplotlib.pyplot.show()



            # Save model
            logger.info('Save model')
            # to do: Names of the model have to be stored in config ini
            save_model(model=model,
                       model_name=modeln,
                       logger=logger,
                       models_path=models_path)

    logger.info('::: Finish ::: ')


if __name__ == "__main__":
    main()
