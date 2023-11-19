"""
This script runs fully automated mlops process.

author: Ondrej Ploteny <ondrej.ploteny@thermofisher.com>
Nov 2023
"""

import os
import json
import sys

import pandas as pd

import ingestion
import training
import scoring
import deployment
import diagnostics
import reporting
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

with open('config.json', 'r') as f:
    config = json.load(f)

ingested_file_path = os.path.join(config['prod_deployment_path'], "ingestedfiles.txt")
dataset_file_path = os.path.join(config['output_folder_path'], 'finaldata.csv')
score_file_path = os.path.join(config['prod_deployment_path'], "latestscore.txt")
model_path = os.path.join(config['prod_deployment_path'], 'trainedmodel.pkl')
data_path = os.path.join(config['output_model_path'], 'finaldata.csv')


def load_previous_score(path: str):
    """
    Load the latest score from file
    :param path: path to file containing the latest score
    :return: float with the latest score
    """
    with open(path, 'r') as file:
        latest_score = float(file.read())
    return latest_score


def load_ingested_files(path: str):
    """
    Load list of previously ingested files
    :param path: str path to file
    :return: list: list of ingested datasets
    """
    with open(path) as file:
        file_content_list = [line.strip().split(' ')[1] for line in file.readlines()]
    return set(file_content_list)


def get_files_per_directory(dir_path: str):
    """
    Get list of all files in directory
    :param dir_path: str path to folder
    :return: set of filepaths in folder
    """
    directory_file_list = [os.path.join(dir_path, filename) for filename in os.listdir(dir_path)]
    return set(directory_file_list)


def check_for_new_dataset():
    """
    Compare previously ingested files with currently available files in directory
    :return: bool, True if new dataset is available, False otherwise
    """
    ingested_files = load_ingested_files(ingested_file_path)
    actual_files_in_dir = get_files_per_directory(config['input_folder_path'])
    unprocessed_files = set(actual_files_in_dir) - set(ingested_files)
    return unprocessed_files


def full_process():
    """
    Full mlops process:
        check if new dataset is available
        check if data drift occurs and re-train new model if it is needed
        deploy new model if score is sufficient
        log all metrics
    :return:
    """

    # if you found new data, you should proceed. otherwise, do end the process here
    if not check_for_new_dataset():
        logging.info("STEP: ingestion, no new dataset occurs")
        return

    # process new ingested data
    logging.info("STEP: ingestion, new dataset occurs - check drift")
    ingestion.merge_multiple_dataframe()

    # check for data drift
    deployed_score = load_previous_score(score_file_path)
    actual_f1_score = scoring.score_model(is_dump=False)

    if actual_f1_score >= deployed_score:
        logging.info("STEP: ingestion, No model drift occurred")
        return None

    # drift occurs, make re-training
    logging.info("STEP: re-training model")
    training.train_model()

    logging.info("STEP: re-scoring model")
    retrain_f1 = reporting.score_model()
    logging.info(f"STEP: re-scoring model, new F1 score: {retrain_f1}")

    logging.info(f"STEP: model deploying")
    deployment.store_model_into_pickle()

    logging.info("STEP: diagnostics, begin")
    test_dataset_path = os.path.join(config['test_data_path'], 'testdata.csv')
    test_df = pd.read_csv(test_dataset_path)
    X_df = test_df.drop(['corporation', 'exited'], axis=1)
    preds = diagnostics.model_predictions(data_to_predict=X_df)
    logging.info(f"STEP: diagnostics, predictions: {str(preds)}")

    statistics = diagnostics.dataframe_summary()
    logging.info(f"STEP: diagnostics, dataframe statistics: {str(statistics)}")

    t = diagnostics.execution_time()
    logging.info(f"STEP: diagnostics, execution time: {str(t)}")

    missing = diagnostics.missing_data()
    logging.info(f"STEP: diagnostics, missing percentage: {str(missing)}")

    outdated_list = diagnostics.outdated_packages_list()
    logging.info(f"STEP: diagnostics, outdated packages: {str(outdated_list)}")
    logging.info("STEP: diagnostics, done")


if __name__ == "__main__":
    logging.info("STEP: Full MLops process, begin")
    full_process()
    logging.info("STEP: Full MLops process, done")
