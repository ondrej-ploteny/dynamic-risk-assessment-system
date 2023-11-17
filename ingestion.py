import pandas as pd
import numpy as np
import os
import json
from datetime import datetime


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

    for dataset_path in locate_datasets(input_folder_path):
        curr_dataset = pd.read_csv(dataset_path)
        final_dataset = pd.concat([final_dataset, curr_dataset], ignore_index=True)
        curr_timestamp = datetime.now().strftime('%d/%m/%Y-%H:%M:%S')
        final_dataset_log.append(f"{curr_timestamp} {dataset_path}")

    final_dataset.drop_duplicates(inplace=True)

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    final_dataset.to_csv(os.path.join(output_folder_path, 'finaldata.csv'), index=False)

    with open(os.path.join(output_folder_path, 'ingestedfiles.txt'), "w") as file:
        file.write("\n".join(final_dataset_log))


if __name__ == '__main__':
    merge_multiple_dataframe()
