import os
import json
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


# Load config.json and correct path variable
with open('config.json', 'r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
output_model_path = os.path.join(config['output_model_path'])


def store_model_into_pickle():
    """
    function for deployment
    copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    """
    command = 'cp'

    if not os.path.exists(prod_deployment_path):
        os.makedirs(prod_deployment_path)

    model_filename = 'trainedmodel.pkl'
    model_path_src = os.path.join(output_model_path, model_filename)
    model_path_dst = os.path.join(prod_deployment_path, model_filename)
    os.system(f'{command} {model_path_src} {model_path_dst}')
    logging.info(f"STEP: deploying, {model_filename} deployed")

    score_filename = 'latestscore.txt'
    score_path_src = os.path.join(output_model_path, score_filename)
    score_path_dst = os.path.join(prod_deployment_path, score_filename)
    os.system(f'{command} {score_path_src} {score_path_dst}')
    logging.info(f"STEP: deploying, {score_filename} deployed")

    ingested_filename = 'ingestedfiles.txt'
    ingested_path_src = os.path.join(dataset_csv_path, ingested_filename)
    ingested_path_dst = os.path.join(prod_deployment_path, ingested_filename)
    os.system(f'{command} {ingested_path_src} {ingested_path_dst}')
    logging.info(f"STEP: deploying, {ingested_filename} deployed")


if __name__ == '__main__':
    logging.info("STEP: deploying, begin")
    store_model_into_pickle()
    logging.info("STEP: deploying, done")
