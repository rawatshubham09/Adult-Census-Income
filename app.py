from flask import Flask, request
from censusIncome.util.util import read_yaml_file, write_yaml_file
from censusIncome.logger import logging
from censusIncome.exception import CensusException
import os
import numpy as np
import sys
import json
from censusIncome.config.configuration import Configuartion
from censusIncome.constant import CONFIG_DIR, get_current_time_stamp
from censusIncome.pipeline.pipeline import Pipeline
from censusIncome.entity.census_prediction import CensusPredictor, CensusData
from flask import send_file, abort, render_template
from censusIncome.logger import get_log_dataframe

ROOT_DIR = os.getcwd()
LOG_FOLDER_NAME = "logs"
PIPELINE_FOLDER_NAME = "censusIncome"
SAVED_MODELS_DIR_NAME = "saved_models"
MODEL_CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, "model.yaml")
LOG_DIR = os.path.join(ROOT_DIR, LOG_FOLDER_NAME)
PIPELINE_DIR = os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME)
MODEL_DIR = os.path.join(ROOT_DIR, SAVED_MODELS_DIR_NAME)


CENSUS_DATA_KEY = "census_data"
MEDIAN_CENSUS_VALUE_KEY = "median_census_value"

app = Flask(__name__)


@app.route('/artifact', defaults={'req_path': 'censusIncome'})
@app.route('/artifact/<path:req_path>')
def render_artifact_dir(req_path):
    os.makedirs("censusIncome", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        if ".html" in abs_path:
            with open(abs_path, "r", encoding="utf-8") as file:
                content = ''
                for line in file.readlines():
                    content = f"{content}{line}"
                return content
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file_name): file_name for file_name in os.listdir(abs_path) if
             "artifact" in os.path.join(abs_path, file_name)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('files.html', result=result)


@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return str(e)


@app.route('/view_experiment_hist', methods=['GET', 'POST'])
def view_experiment_history():
    experiment_df = Pipeline.get_experiments_status()
    context = {
        "experiment": experiment_df.to_html(classes='table table-striped col-12')
    }
    return render_template('experiment_history.html', context=context)


@app.route('/train', methods=['GET', 'POST'])
def train():
    message = ""
    pipeline = Pipeline(config=Configuartion(current_time_stamp=get_current_time_stamp()))
    if not Pipeline.experiment.running_status:
        message = "Training started."
        pipeline.start()
    else:
        message = "Training is already in progress."
    context = {
        "experiment": pipeline.get_experiments_status().to_html(classes='table table-striped col-12'),
        "message": message
    }
    return render_template('train.html', context=context)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    context = {
        CENSUS_DATA_KEY: None,
        MEDIAN_CENSUS_VALUE_KEY: None
    }

    if request.method == 'POST':
        age = np.int64(request.form['age'])
        work_class = str(request.form['workclass'])
        fnlwgt = np.int64(request.form['fnlwgt'])
        education = str(request.form['education'])
        education_num = np.int64(request.form['education-num'])
        marital_status = str(request.form['marital-status'])
        occupation = str(request.form['occupation'])
        relationship = str(request.form['relationship'])
        race = str(request.form['race'])
        sex = str(request.form['sex'])
        capital_gain = np.int64(request.form['capital-gain'])
        capital_loss = np.int64(request.form['capital-loss'])
        hours_per_week = np.int64(request.form['hours-per-week'])
        country = str(request.form['country'])
        salary = str(request.form['salary'])

        census_data = CensusData(age=age,
                                 work_class=work_class,
                                 fnlwgt=fnlwgt,
                                 education=education,
                                 education_num=education_num,
                                 marital_status=marital_status,
                                 occupation=occupation,
                                 relationship=relationship,
                                 race=race,
                                 sex=sex,
                                 capital_gain=capital_gain,
                                 capital_loss=capital_loss,
                                 hours_per_week=hours_per_week,
                                 country=country,
                                 salary=salary
                                 )
        census_df = census_data.get_census_input_data_frame()
        census_predictor = CensusPredictor(model_dir=MODEL_DIR)
        median_census_value = census_predictor.predict(X=census_df)
        context = {
            CENSUS_DATA_KEY: census_data.get_census_data_as_dict(),
            MEDIAN_CENSUS_VALUE_KEY: median_census_value,
        }
        return render_template('predict.html', context=context)
    return render_template("predict.html", context=context)


@app.route('/saved_models', defaults={'req_path': 'saved_models'})
@app.route('/saved_models/<path:req_path>')
def saved_models_dir(req_path):
    os.makedirs("saved_models", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('saved_models_files.html', result=result)


@app.route("/update_model_config", methods=['GET', 'POST'])
def update_model_config():
    try:
        if request.method == 'POST':
            model_config = request.form['new_model_config']
            model_config = model_config.replace("'", '"')
            print(model_config)
            model_config = json.loads(model_config)

            write_yaml_file(file_path=MODEL_CONFIG_FILE_PATH, data=model_config)

        model_config = read_yaml_file(file_path=MODEL_CONFIG_FILE_PATH)
        return render_template('update_model.html', result={"model_config": model_config})

    except Exception as e:
        logging.exception(e)
        return str(e)


@app.route(f'/logs', defaults={'req_path': f'{LOG_FOLDER_NAME}'})
@app.route(f'/{LOG_FOLDER_NAME}/<path:req_path>')
def render_log_dir(req_path):
    os.makedirs(LOG_FOLDER_NAME, exist_ok=True)
    # Joining the base and the requested path
    logging.info(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        log_df = get_log_dataframe(abs_path)
        context = {"log": log_df.to_html(classes="table-striped", index=False)}
        return render_template('log.html', context=context)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('log_files.html', result=result)


if __name__ == "__main__":
    app.run()
