"""
This script provides a diagnostic overview ML model.
Diagnostic features consists of :
    1. execution time for training
    2. provides model prediction
    3. Dataset value analysis
    4. Environment correctness

author: Ondrej Ploteny <ondrej.ploteny@thermofisher.com>
Nov 2023
"""

import pickle
import subprocess
import sys

import pandas as pd
import timeit
import os
import json
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path'])


def model_predictions(data_to_predict: pd.DataFrame):
    """
    Function to get model predictions
    read the deployed model and a test dataset, calculate predictions
    return value should be a list containing all predictions

    :param data_to_predict: pd.DataFrame
    :return:
    """

    model_path = os.path.join(prod_deployment_path, 'trainedmodel.pkl')

    model = pickle.load(open(model_path, 'rb'))
    predictions = model.predict(data_to_predict)

    assert len(data_to_predict) == len(predictions)

    return predictions.tolist()


def dataframe_summary():
    """
    Function to get summary statistics
    calculate summary statistics here
    return value should be a list containing all summary statistics
    :return:
    """
    dataset_path = os.path.join(dataset_csv_path, 'finaldata.csv')

    data_df = pd.read_csv(dataset_path)
    data_df = data_df.drop(['exited'], axis=1)
    data_df = data_df.select_dtypes(include='number')

    numeric_values = data_df.select_dtypes(include='number')
    result_dict = numeric_values.agg(['mean', 'median', 'std'])
    result_list = [
        {'column': col,
         'mean': result_dict[col][0],
         'median': result_dict[col][1],
         'std_dev': result_dict[col][2]}
        for col in result_dict.columns
    ]
    return result_list


def _measure_subprocess_execution_time(command):
    """
    Measure the execution time of a subprocess.

    :param command:
    :return:
    """
    start_time = timeit.default_timer()
    _ = subprocess.run(command, capture_output=True)
    duration = timeit.default_timer() - start_time
    return duration


def execution_time():
    """
    Function to get timings
    calculate timing of training.py and ingestion.py
    :return: a list of 2 timing values in seconds
    """
    command_list = [
        ['python', 'training.py'],
        ['python', 'ingestion.py']
    ]
    timing_list = list(map(_measure_subprocess_execution_time, command_list))
    return timing_list


def missing_data():
    """
    Calculate percentage of missing values per each column of dataset
    :return dictionary, keys are column names, values are percentages
    """
    dataset_path = os.path.join(dataset_csv_path, 'finaldata.csv')
    data = pd.read_csv(dataset_path)
    missing_percentage = (data.isna().mean() * 100).round(2).to_dict()
    return missing_percentage


def outdated_packages_list():
    """
    Function to check dependencies
    :return: list of dictionaries - column name, current version, latest version
    """
    pip_outdated = [sys.executable, '-m', 'pip', 'list', '--outdated', '--format=json']
    outdated_packages = subprocess.check_output(pip_outdated).decode('utf-8')
    return outdated_packages


if __name__ == '__main__':
    logging.info("STEP: diagnostics, begin")

    test_dataset_path = os.path.join(test_data_path, 'testdata.csv')
    test_df = pd.read_csv(test_dataset_path)
    X_df = test_df.drop(['corporation', 'exited'], axis=1)
    preds = model_predictions(data_to_predict=X_df)
    logging.info(f"STEP: diagnostics, predictions: {str(preds)}")

    statistics = dataframe_summary()
    logging.info(f"STEP: diagnostics, dataframe statistics: {str(statistics)}")

    t = execution_time()
    logging.info(f"STEP: diagnostics, execution time: {str(t)}")

    missing = missing_data()
    logging.info(f"STEP: diagnostics, missing percentage: {str(missing)}")

    outdated_list = outdated_packages_list()
    logging.info(f"STEP: diagnostics, outdated packages: {str(outdated_list)}")
    logging.info("STEP: diagnostics, done")





    
