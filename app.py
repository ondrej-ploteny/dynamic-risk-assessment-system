from flask import Flask, session, jsonify, request
import pandas as pd
import json
import os
from diagnostics import model_predictions, dataframe_summary, missing_data, execution_time, outdated_packages_list
from scoring import score_model

# Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    """
    call the prediction function you created in Step 3
    add return value for prediction outputs
    :return:
    """
    dataset_path = request.get_json()['filepath']

    predictor_column_name = ['lastmonth_activity',
                             'lastyear_activity',
                             'number_of_employees']

    df = pd.read_csv(dataset_path, usecols=predictor_column_name)
    y_pred = model_predictions(df)
    return jsonify(y_pred)


@app.route("/scoring", methods=['GET', 'OPTIONS'])
def stats():
    """
    Scoring Endpoint
    check the score of the deployed model
    :return: a single F1 score number
    """

    f1_score = score_model(is_dump=False)
    return jsonify([f1_score])


@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def summary():
    """
    Summary Statistics Endpoint
    :return:
    """
    col_stats = dataframe_summary()
    return jsonify(col_stats)


@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diagnostics():
    """
    Diagnostics Endpoint
    :return:
    """
    diagnostics_dict = {
        'missing': missing_data(),
        'time_check': execution_time(),
        'outdated': outdated_packages_list()
    }

    return jsonify(diagnostics_dict)


if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
