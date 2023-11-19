import json
import os
import pandas as pd
import requests

# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

# Get test data:
# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

# path to data used for analysis
test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')
report_path = os.path.join(config['output_model_path'], 'apireturns.txt')


def api_call():
    header = {"Content-Type": "application/json"}
    body = {'filepath': test_data_path}
    response1 = requests.post(
        url=URL + '/prediction',
        headers=header,
        json=body
    ).json()

    # scoring
    response2 = requests.get(URL + '/scoring').json()

    # statistics
    response3 = requests.get(URL + '/summarystats').json()

    # diagnostics
    response4 = requests.get(URL + '/diagnostics').json()

    data = [
        {'key': 'prediction', 'value': response1},
        {'key': 'scoring', 'value': response2},
        {'key': 'statistics', 'value': response3},
        {'key': 'missing pct', 'value': response4['missing']},
        {'key': 'execution time', 'value': response4['time_check']},
        {'key': 'outdated packages', 'value': response4['outdated']}
    ]

    api_report = pd.DataFrame(data, columns=['key', 'value'])
    api_report.to_csv(report_path, index=False, header=False, sep='\t')


if __name__ == "__main__":
    api_call()
