import pandas as pd
import pickle
import os
import sys
from sklearn.linear_model import LogisticRegression
import json
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 


# Function for training the model
def train_model():
    training_dataset_path = os.path.join(dataset_csv_path, 'finaldata.csv')
    final_model_path = os.path.join(model_path, 'trainedmodel.pkl')
    
    # use this logistic regression for training
    lr = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                            intercept_scaling=1, l1_ratio=None, max_iter=100,
                            multi_class='auto', n_jobs=None, penalty='l2',
                            random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                            warm_start=False)
    
    # fit the logistic regression to your data
    df = pd.read_csv(training_dataset_path)
    logging.info(f"STEP: training, dataset path {training_dataset_path}, size: {len(df)}")

    X = df[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']]
    y = df['exited']

    lr.fit(X, y)
    
    # write the trained model to your workspace in a file called trainedmodel.pkl
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    with open(final_model_path, 'wb') as file:
        pickle.dump(lr, file)
        logging.info(f"STEP: training, model dumped to {final_model_path}")


if __name__ == '__main__':
    logging.info("STEP: training, begin")
    train_model()
    logging.info("STEP: training, done")
