"""
This script makes a ML model performance report

author: Ondrej Ploteny <ondrej.ploteny@thermofisher.com>
Nov 2023
"""

import os
import sys
import json
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from diagnostics import model_predictions

import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')
confusion_matrix_path = os.path.join(config['output_model_path'], 'confusionmatrix.png')


def score_model():
    """
    Function for reporting
    calculate a confusion matrix using the test data and the deployed model
    write the confusion matrix to the workspace
    """

    target_column_name = 'exited'
    predictor_column_name = ['lastmonth_activity',
                             'lastyear_activity',
                             'number_of_employees']

    x_test = pd.read_csv(test_data_path, usecols=predictor_column_name)
    y_true = pd.read_csv(test_data_path, usecols=[target_column_name]).values
    y_pred = model_predictions(x_test)

    matrix = metrics.confusion_matrix(y_true, y_pred)
    logging.info(f"STEP: training, confusion matrix: {str(matrix.tolist())}")

    plt.figure(figsize=(4, 4))
    ax = sns.heatmap(matrix, annot=True, fmt='d', cmap='Reds', cbar=False,
                     xticklabels=['Predicted 0', 'Predicted 1'],
                     yticklabels=['Actual 0', 'Actual 1'])
    ax.set_title('Confusion matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.savefig(confusion_matrix_path)
    logging.info(f"STEP: training, report saved as {confusion_matrix_path}")


if __name__ == '__main__':
    logging.info("STEP: reporting, begin")
    score_model()
    logging.info("STEP: reporting, done")
