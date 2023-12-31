"""
This script provides a raw data ingestion.

author: Ondrej Ploteny <ondrej.ploteny@thermofisher.com>
Nov 2023
"""


import pandas as pd
import os
import sys
import json
from datetime import datetime
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


def locate_datasets(directory_path: str, extension: str = '.csv'):
    """
    This function returns file path existing in directory.
    Files can be filtered by extension.
    :param directory_path:
    :param extension:
    :return: list of csv files
    """
    all_files = os.listdir(directory_path)
    for file in all_files:
        if file.endswith(extension):
            yield os.path.join(directory_path, file)


def merge_multiple_dataframe():
    """
    Function for data ingestion, check for datasets, compile them together, and write to an output file

    :return:
    """
    final_dataset_log = list()
    final_dataset = pd.DataFrame()
    final_dataset_path = os.path.join(output_folder_path, 'finaldata.csv')
    final_log_path = os.path.join(output_folder_path, 'ingestedfiles.txt')

    for dataset_path in locate_datasets(input_folder_path):
        curr_dataset = pd.read_csv(dataset_path)
        final_dataset = pd.concat([final_dataset, curr_dataset], ignore_index=True)
        curr_timestamp = datetime.now().strftime('%d/%m/%Y-%H:%M:%S')
        final_dataset_log.append(f"{curr_timestamp} {dataset_path}")
        logging.info(f"STEP: ingestion, partial dataset loaded {dataset_path}")

    final_dataset.drop_duplicates(inplace=True)

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    final_dataset.to_csv(final_dataset_path, index=False)
    logging.info(f"STEP: ingestion, dataset dumped to {final_dataset_path}")

    with open(final_log_path, "w") as file:
        file.write("\n".join(final_dataset_log))
        logging.info(f"STEP: ingestion, log dumped to {final_log_path}")


if __name__ == '__main__':
    logging.info("STEP: ingestion, begin")
    merge_multiple_dataframe()
    logging.info("STEP: ingestion, done")
