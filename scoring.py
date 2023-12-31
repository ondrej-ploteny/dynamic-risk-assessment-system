"""
This script provides a ML model scoring functions

author: Ondrej Ploteny <ondrej.ploteny@thermofisher.com>
Nov 2023
"""


import pandas as pd
import pickle
import os
import sys
from sklearn import metrics
import json
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f) 

output_model_path = os.path.join(config['output_model_path'])
test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')


def score_model(is_dump: bool = True):
    """
    Function for model scoring,
    this function should take a trained model, load test data,
    calculate an F1 score for the model relative to the test data
    it should write the result to the latestscore.txt file
    :return:
    """
    model_path = os.path.join(output_model_path, 'trainedmodel.pkl')
    score_path = os.path.join(output_model_path, 'latestscore.txt')

    logging.info(f"STEP: scoring, loading model from {model_path}")
    with open(model_path, 'rb') as m:
        model = pickle.load(m)

    test_df = pd.read_csv(test_data_path)
    logging.info(f"STEP: scoring, loading testdata from {test_data_path}, size: {len(test_df)}")

    X = test_df[['lastmonth_activity',
                 'lastyear_activity',
                 'number_of_employees']]
    y_test = test_df['exited']

    logging.info("STEP: scoring, making prediction")
    y_pred = model.predict(X)
    f1_score = metrics.f1_score(y_test, y_pred)
    logging.info(f"STEP: scoring, f1 score: {f1_score}")

    if is_dump:
        with open(score_path, "w") as file:
            file.write(str(f1_score))
            logging.info(f"STEP: scoring, score dumped to {score_path}")

    return f1_score


if __name__ == '__main__':
    logging.info("STEP: scoring, begin")
    _ = score_model()
    logging.info("STEP: scoring, done")
