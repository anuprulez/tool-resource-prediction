import numpy
from onnxconverter_common import FloatTensorType
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
import xgboost as xgb
import onnxruntime as onnx_rt
import onnxmltools
import onnx

# Models
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import sys
import time
from datetime import datetime
import json
import numpy as np
import pandas as pd
import multiprocessing
import yaml
from yaml import SafeLoader
from typing import Union


def get_train_and_test_set(do_scaling: bool, seed: int, is_mixed_data: bool = False, run_config=None,
                           doStandardScale=False, remove_outliers=False):
    if is_mixed_data:
        column_names = ["Tool_id", "Filesize", "Number_of_files", "Slots", "Memory_bytes", "Create_time", "Validity"]
    else:
        column_names = ["Tool_id", "Filesize", "Number_of_files", "Slots", "Memory_bytes", "Create_time"]

    dataset_path = run_config["dataset_path"]
    print(f"Get train and test set from path {dataset_path} ...")
    data = pd.read_csv(dataset_path, sep=',', names=column_names)

    # Extract tool name
    tool_name = data["Tool_id"][0]
    start_idx = 0
    idx = tool_name.rfind('/')
    if idx != -1:
        start_idx = tool_name[0:idx].rfind('/') + 1
    tool_name = tool_name[start_idx:]

    if is_mixed_data:
        relevant_columns_x = ["Filesize", "Number_of_files", "Slots", "Create_time", "Validity"]
    else:
        relevant_columns_x = ["Filesize", "Number_of_files", "Slots", "Create_time"]
    X = data[relevant_columns_x].values
    if is_mixed_data:
        X[:, 0:-2] = X[:, 0:-2].astype('float64')
    else:
        X[:, 0:-1] = X[:, 0:-1].astype('float64')

    y = data["Memory_bytes"].values
    y = y.astype('float64')

    if do_scaling:
        # scale with GB
        scaling = 1000000000
        # Scale bytes of filesize
        X[:, 0] /= scaling
        # Scale bytes of memory bytes
        y /= scaling

    X_train_orig, X_test_orig, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    y_train = y_train.astype('float64')
    y_test = y_test.astype('float64')

    if remove_outliers:
        print("Outliers get removed.")
        # Find entries that are outside of mean +- 2*std
        upper_threshold = np.mean(y) + 2 * np.std(y)
        lower_threshold = np.mean(y) - 2 * np.std(y)
        remove_indices_train = np.where(np.logical_or(y_train > upper_threshold, y_train < lower_threshold))
        # Remove those entries (only in train set)
        X_train_orig = np.delete(X_train_orig, remove_indices_train, axis=0)
        y_train = np.delete(y_train, remove_indices_train)

        nr_samples_outside = len(remove_indices_train[0])
        percentage_within = 1 - (nr_samples_outside / len(y_train))
        print(f"Percentage data within mean +- 2 * std: {percentage_within}")
    else:
        print("Outliers are not removed.")

    if doStandardScale:
        sc = StandardScaler()
        if is_mixed_data:
            # Scale all columns beside 'Validity' & 'Create_time
            X_train = sc.fit_transform(X_train_orig[:, 0:-2])
            X_test = sc.transform(X_test_orig[:, 0:-2])
        else:
            # Scale all columns beside 'Create_time
            X_train = sc.fit_transform(X_train_orig[:, 0:-1])
            X_test = sc.transform(X_test_orig[:, 0:-1])

        X_test_unscaled = sc.inverse_transform(X_test)
    else:
        X_train = X_train_orig[:, 0:-1]
        X_test = X_test_orig[:, 0:-1]
        X_test_unscaled = X_test
    return X_train, X_test, X_test_orig, X_test_unscaled, y_train, y_test, tool_name


def save_training_results(y_pred, training_stats, X_train, X_test, X_test_orig, X_test_unscaled, y_train,
                          y_test, tool_name, seed: int,
                          is_mixed_data: bool = False, run_config=None):
    time_for_training_mins = training_stats["time_for_training_mins"]
    feature_importances = training_stats["feature_importances"]
    mean_abs_error = training_stats["mean_abs_error"]
    mean_squared_error = training_stats["mean_squared_error"]
    root_mean_squared_error = training_stats["root_mean_squared_error"]
    r2_score = training_stats["r2_score"]
    model_type = run_config["model_type"]
    model_params = training_stats["model_params"]
    if model_type == "rf":
        mean_abs_error_with_uncertainty = training_stats["mean_abs_error_with_uncertainty"]
        mean_squared_error_with_uncertainty = training_stats["mean_squared_error_with_uncertainty"]
        root_mean_squared_error_with_uncertainty = training_stats["root_mean_squared_error_with_uncertainty"]
        r2_score_with_uncertainty = training_stats["r2_score_with_uncertainty"]

    filename_str = f"saved_data/training_{model_type}_{tool_name.replace('/', '-')}_{str(datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p'))}.txt"
    print(f"Save training results to file {filename_str}")
    with open(filename_str, 'a+') as f:
        f.write(f"Tool name: {tool_name}\n")
        f.write(f"Dataset_path: {run_config['dataset_path']}\n")
        f.write(f"Is_mixed_data: {is_mixed_data}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Model_params: {json.dumps(model_params)}\n")
        f.write(f"Time for training in mins: {time_for_training_mins}\n")
        f.write(f"Feature importance: {feature_importances}\n")
        f.write(f"Mean absolute error: {mean_abs_error}\n")
        f.write(f"Mean squared error: {mean_squared_error}\n")
        f.write(f"Root mean squared error: {root_mean_squared_error}\n")
        f.write(f"R2 Score: {r2_score}\n")
        # With uncertainty
        # Only supported for RF
        if model_type == "rf":
            f.write(f"Mean absolute error with uncertainty: {mean_abs_error_with_uncertainty}\n")
            f.write(f"Mean squared error with uncertainty: {mean_squared_error_with_uncertainty}\n")
            f.write(f"Root mean squared error with uncertainty: {root_mean_squared_error_with_uncertainty}\n")
            f.write(f"R2 Score with uncertainty: {r2_score_with_uncertainty}\n")
        f.write("############################\n")
        if is_mixed_data:
            f.write("Filesize, Prediction, Target, Create_time, Validity\n")
            f.write("############################\n")
            for idx, entry in enumerate(X_test_orig):
                filesize = X_test_unscaled[idx][0]
                validity = entry[-1]
                prediction = y_pred[idx]
                target = y_test[idx]
                create_time = entry[-1]
                f.write(f"{filesize},{prediction},{target},{create_time},{validity}")
                f.write("\n")
        else:
            f.write("Filesize, Prediction, Target, Create_time\n")
            f.write("############################\n")
            for idx, entry in enumerate(X_test_orig):
                filesize = X_test_unscaled[idx][0]
                prediction = y_pred[idx]
                target = y_test[idx]
                create_time = entry[-1]
                f.write(f"{filesize},{prediction},{target},{create_time}")
                f.write("\n")


def save_evaluation_results(y_pred, evaluation_stats, X_test, X_test_orig, y_test, tool_name, is_mixed_data: bool = False, run_config=None, isBaseline=False):
    mean_abs_error = evaluation_stats["mean_abs_error"]
    mean_squared_error = evaluation_stats["mean_squared_error"]
    root_mean_squared_error = evaluation_stats["root_mean_squared_error"]
    r2_score = evaluation_stats["r2_score"]
    model_type = run_config["model_type"]
    # TODO: find out if this is possible
    # if model_type == "rf":
    #     mean_abs_error_with_uncertainty = training_stats["mean_abs_error_with_uncertainty"]
    #     mean_squared_error_with_uncertainty = training_stats["mean_squared_error_with_uncertainty"]
    #     root_mean_squared_error_with_uncertainty = training_stats["root_mean_squared_error_with_uncertainty"]
    #     r2_score_with_uncertainty = training_stats["r2_score_with_uncertainty"]

    str_eval_or_baseline = "evaluation"
    if isBaseline:
        str_eval_or_baseline = "baseline"
    filename_str = f"saved_data/{str_eval_or_baseline}_{model_type}_{tool_name.replace('/', '-')}_{str(datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p'))}.txt"
    print(f"Save {str_eval_or_baseline} results to file {filename_str}")
    with open(filename_str, 'a+') as f:
        f.write(f"Tool name: {tool_name}\n")
        f.write(f"Dataset_path: {run_config['dataset_path']}\n")
        f.write(f"Is_mixed_data: {is_mixed_data}\n")
        f.write(f"Mean absolute error: {mean_abs_error}\n")
        f.write(f"Mean squared error: {mean_squared_error}\n")
        f.write(f"Root mean squared error: {root_mean_squared_error}\n")
        f.write(f"R2 Score: {r2_score}\n")
        # TODO: find out if this is possible
        # # With uncertainty
        # # Only supported for RF
        # if model_type == "rf":
        #     f.write(f"Mean absolute error with uncertainty: {mean_abs_error_with_uncertainty}\n")
        #     f.write(f"Mean squared error with uncertainty: {mean_squared_error_with_uncertainty}\n")
        #     f.write(f"Root mean squared error with uncertainty: {root_mean_squared_error_with_uncertainty}\n")
        #     f.write(f"R2 Score with uncertainty: {r2_score_with_uncertainty}\n")
        f.write("############################\n")
        if is_mixed_data:
            f.write("Filesize, Prediction, Target, Create_time, Validity\n")
            f.write("############################\n")
            for idx, entry in enumerate(X_test_orig):
                filesize = X_test[idx][0]
                validity = entry[-1]
                prediction = y_pred[idx]
                target = y_test[idx]
                create_time = entry[-1]
                f.write(f"{filesize},{prediction},{target},{create_time},{validity}")
                f.write("\n")
        else:
            f.write("Filesize, Prediction, Target, Create_time\n")
            f.write("############################\n")
            for idx, entry in enumerate(X_test_orig):
                filesize = X_test[idx][0]
                prediction = y_pred[idx]
                target = y_test[idx]
                create_time = entry[-1]
                f.write(f"{filesize},{prediction},{target},{create_time}")
                f.write("\n")


def fit_model(X_train, y_train, hyper_param_opt, run_config):
    start_time = time.time()

    if run_config["model_type"] == "rf":
        print("Fit Random Forest...")
        # criterion='absolute_error', bootstrap=False, warm_start=True
        regressor = RandomForestRegressor(**run_config["model_params"], verbose=2)
        if hyper_param_opt:
            clf = GridSearchCV(regressor, {'max_depth': np.arange(1, 9), 'n_estimators': [50, 100, 200, 400],
                                           'criterion': ['absolute_error', 'poisson', 'squared_error']}, verbose=2,
                               n_jobs=2)
            clf.fit(X_train, y_train)
            regressor = clf.best_estimator_
        else:
            regressor.fit(X_train, y_train)

    if run_config["model_type"] == "xgb":
        print("Fit XGB model...")
        xgb.set_config(verbosity=2)

        # TODO: maybe try XGB RF Regressor
        # xgb.XGBRFRegressor
        regressor = xgb.XGBRegressor(**run_config["model_params"], n_jobs=multiprocessing.cpu_count() // 2)
        if hyper_param_opt:
            # From 0.03 to 0.3
            lr_space = np.logspace(-2, -1, num=10) * 3
            clf = GridSearchCV(regressor, {'max_depth': np.arange(1, 9), 'n_estimators': [50, 100, 200, 400],
                                           'learning_rate': lr_space},
                               verbose=2, n_jobs=2)
            clf.fit(X_train, y_train)
            regressor = clf.best_estimator_
        else:
            regressor.fit(X_train, y_train)

    end_time = time.time()
    time_for_training_mins = (end_time - start_time) / 60
    return regressor, time_for_training_mins


def train_and_predict(X_train, X_test, X_test_orig, X_test_unscaled, y_train, y_test, tool_name, seed: int,
                      is_mixed_data: bool = False, run_config=None) \
        -> tuple[np.ndarray, np.ndarray, dict, Union[RandomForestRegressor, XGBRegressor]]:
    """
    Returns:
    y_pred, y_true, training_stats, regressor (the fitted model)
    """
    np.set_printoptions(threshold=sys.maxsize)

    regressor, time_for_training_mins = fit_model(X_train=X_train, y_train=y_train, hyper_param_opt=False,
                                                  run_config=run_config)

    regressor_params = regressor.get_params()
    relevant_params = {"n_estimators", "random_state", "criterion", "max_depth", "learning_rate", "bootstrap",
                       "min_samples_leaf"}
    # Extract the relevant params from the model
    model_params = [f"{key}: {regressor_params.get(key)}" for key in (regressor_params.keys() & relevant_params)]

    print("Time taken in minutes for training:", time_for_training_mins)

    feature_importances = regressor.feature_importances_
    print("Feature importances:", feature_importances)

    print("Predict...")
    y_pred = regressor.predict(X_test)

    mean_abs_error = metrics.mean_absolute_error(y_test, y_pred)
    mean_squared_error = metrics.mean_squared_error(y_test, y_pred)
    root_mean_squared_error = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    r2_score = metrics.r2_score(y_test, y_pred)
    print('Mean Absolute Error:', mean_abs_error)
    print('Mean Squared Error:', mean_squared_error)
    print('Root Mean Squared Error:', root_mean_squared_error)
    print('R2 Score:', r2_score)

    training_stats = {
        "mean_abs_error": mean_abs_error,
        "mean_squared_error": mean_squared_error,
        "root_mean_squared_error": root_mean_squared_error,
        "r2_score": r2_score,
        "time_for_training_mins": time_for_training_mins,
        "feature_importances": feature_importances,
        "model_params": model_params
    }

    # With uncertainty
    # only supported for RF
    if run_config["model_type"] == "rf":
        print("Predict with uncertainty...")
        y_pred_with_uncertainty = pred_with_uncertainty(regressor, X_test, 90)

        mean_abs_error_with_uncertainty = metrics.mean_absolute_error(y_test, y_pred_with_uncertainty)
        mean_squared_error_with_uncertainty = metrics.mean_squared_error(y_test, y_pred_with_uncertainty)
        root_mean_squared_error_with_uncertainty = np.sqrt(metrics.mean_squared_error(y_test, y_pred_with_uncertainty))
        r2_score_with_uncertainty = metrics.r2_score(y_test, y_pred_with_uncertainty)
        print('Mean Absolute Error with uncertainty:', mean_abs_error_with_uncertainty)
        print('Mean Squared Error with uncertainty:', mean_squared_error_with_uncertainty)
        print('Root Mean Squared Error with uncertainty:', root_mean_squared_error_with_uncertainty)
        print('R2 Score with uncertainty:', r2_score_with_uncertainty)

        training_stats_uncertainty = {
            "mean_abs_error_with_uncertainty": mean_abs_error_with_uncertainty,
            "mean_squared_error_with_uncertainty": mean_squared_error_with_uncertainty,
            "root_mean_squared_error_with_uncertainty": root_mean_squared_error_with_uncertainty,
            "r2_score_with_uncertainty": r2_score_with_uncertainty
        }
        training_stats.update(training_stats_uncertainty)

    return y_pred, y_test, training_stats, regressor


def save_model_as_onnx(regressor, train_and_test_data, run_config):
    X_train = train_and_test_data['X_train']
    tool_name = train_and_test_data['tool_name']
    model_type = run_config['model_type']
    initial_types = [
        ('X', FloatTensorType([None, X_train.shape[1]])),
    ]
    if model_type == "xgb":
        onnx_model = onnxmltools.convert_xgboost(regressor, initial_types=initial_types,
                                                 name='XGB Regressor')
    if model_type == "rf":
        onnx_model = onnxmltools.convert_sklearn(regressor, initial_types=initial_types,
                                                 name='Random Forest')

    filename_str = f"saved_models/model_{model_type}_{tool_name.replace('/', '-')}_{str(datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p'))}.onnx"
    print(f"Save model to file {filename_str}")
    onnxmltools.save_model(onnx_model, filename_str)


def load_model_and_predict(model_path, X_test, X_test_orig, y_test, tool_name):
    print(f"Load model from path {model_path}")
    print("Predict...")
    sess = onnx_rt.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    y_pred = sess.run([label_name], {input_name: X_test.astype(numpy.float32)})[0]

    mean_abs_error = metrics.mean_absolute_error(y_test, y_pred)
    mean_squared_error = metrics.mean_squared_error(y_test, y_pred)
    root_mean_squared_error = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    r2_score = metrics.r2_score(y_test, y_pred)
    print('Mean Absolute Error:', mean_abs_error)
    print('Mean Squared Error:', mean_squared_error)
    print('Root Mean Squared Error:', root_mean_squared_error)
    print('R2 Score:', r2_score)

    evaluation_stats = {
        "mean_abs_error": mean_abs_error,
        "mean_squared_error": mean_squared_error,
        "root_mean_squared_error": root_mean_squared_error,
        "r2_score": r2_score
    }

    # With uncertainty
    # only supported for RF
    # TODO: find out how to do this for ONNX model
    # if sess.get_modelmeta().graph_name == "Random Forest":
    #     y_pred_with_uncertainty = pred_with_uncertainty(regressor, X_test, 90)
    #
    #     mean_abs_error_with_uncertainty = metrics.mean_absolute_error(y_test, y_pred_with_uncertainty)
    #     mean_squared_error_with_uncertainty = metrics.mean_squared_error(y_test, y_pred_with_uncertainty)
    #     root_mean_squared_error_with_uncertainty = np.sqrt(metrics.mean_squared_error(y_test, y_pred_with_uncertainty))
    #     r2_score_with_uncertainty = metrics.r2_score(y_test, y_pred_with_uncertainty)
    #     print('Mean Absolute Error with uncertainty:', mean_abs_error_with_uncertainty)
    #     print('Mean Squared Error with uncertainty:', mean_squared_error_with_uncertainty)
    #     print('Root Mean Squared Error with uncertainty:', root_mean_squared_error_with_uncertainty)
    #     print('R2 Score with uncertainty:', r2_score_with_uncertainty)
    #
    #     training_stats_uncertainty = {
    #         "mean_abs_error_with_uncertainty": mean_abs_error_with_uncertainty,
    #         "mean_squared_error_with_uncertainty": mean_squared_error_with_uncertainty,
    #         "root_mean_squared_error_with_uncertainty": root_mean_squared_error_with_uncertainty,
    #         "r2_score_with_uncertainty": r2_score_with_uncertainty
    #     }
    #     training_stats.update(training_stats_uncertainty)
    #
    return y_pred, y_test, evaluation_stats


def training_pipeline(run_configuration, save: bool, remove_outliers: bool):
    method_params = {
        "do_scaling": True,
        "seed": run_configuration["seed"],
        "is_mixed_data": run_configuration["is_mixed_data"],
        "run_config": run_configuration,
        "remove_outliers": remove_outliers
    }
    # Data loading
    X_train, X_test, X_test_orig, X_test_unscaled, y_train, y_test, tool_name = get_train_and_test_set(**method_params)
    train_and_test_data = {
        "X_train": X_train,
        "X_test": X_test,
        "X_test_orig": X_test_orig,
        "X_test_unscaled": X_test_unscaled,
        "y_train": y_train,
        "y_test": y_test,
        "tool_name": tool_name
    }
    # Remove unnecessary params for the next steps
    method_params.pop("do_scaling")
    method_params.pop("remove_outliers")
    # Model training and predicting
    y_pred, y_test, training_stats, regressor = train_and_predict(**train_and_test_data, **method_params)
    if save:
        save_training_results(y_pred, training_stats, **train_and_test_data, **method_params)
        save_model_as_onnx(regressor, train_and_test_data, run_configuration)


def baseline_pipeline(run_configuration, remove_outliers: bool, save: bool):
    method_params = {
        "do_scaling": True,
        "seed": run_configuration["seed"],
        "is_mixed_data": run_configuration["is_mixed_data"],
        "run_config": run_configuration,
        "remove_outliers": remove_outliers,
    }
    # Data loading
    X_train, X_test, X_test_orig, X_test_unscaled, y_train, y_test, tool_name = get_train_and_test_set(**method_params)
    train_and_test_data = {
        "X_train": X_train,
        "X_test": X_test,
        "X_test_orig": X_test_orig,
        "X_test_unscaled": X_test_unscaled,
        "y_train": y_train,
        "y_test": y_test,
        "tool_name": tool_name
    }
    method_params.pop("do_scaling")
    method_params.pop("remove_outliers")
    method_params.pop("seed")
    y_pred, y_test, baseline_stats = calc_metrics_for_baseline(**train_and_test_data, **method_params)
    if save:
        save_evaluation_results(y_pred, baseline_stats, X_test, X_test_orig, y_test, tool_name, **method_params, isBaseline=True)


def evaluate_model_pipeline(run_configuration, model_path, save):
    method_params = {
        "do_scaling": True,
        "is_mixed_data": run_configuration["is_mixed_data"],
        "run_config": run_configuration
    }
    # Data loading
    X_test_orig, X_test, y_test, tool_name = load_data(**method_params)
    data = {
        "X_test": X_test,
        "X_test_orig": X_test_orig,
        "y_test": y_test,
        "tool_name": tool_name
    }
    # Remove unnecessary params for the next steps
    method_params.pop("do_scaling")

    y_pred, y_test, evaluation_stats = load_model_and_predict(model_path, **data)
    if save:
        save_evaluation_results(y_pred, evaluation_stats, **method_params, **data)


def load_data(do_scaling: bool, is_mixed_data: bool = False, run_config=None):
    if is_mixed_data:
        column_names = ["Tool_id", "Filesize", "Number_of_files", "Slots", "Memory_bytes", "Create_time", "Validity"]
    else:
        column_names = ["Tool_id", "Filesize", "Number_of_files", "Slots", "Memory_bytes", "Create_time"]

    dataset_path = run_config["dataset_path"]
    print(f"Load data from path {dataset_path} ...")

    data = pd.read_csv(dataset_path, sep=',', names=column_names)

    # Extract tool name
    tool_name = data["Tool_id"][0]
    start_idx = 0
    idx = tool_name.rfind('/')
    if idx != -1:
        start_idx = tool_name[0:idx].rfind('/') + 1
    tool_name = tool_name[start_idx:]

    if is_mixed_data:
        relevant_columns_x = ["Filesize", "Number_of_files", "Slots", "Create_time", "Validity"]
    else:
        relevant_columns_x = ["Filesize", "Number_of_files", "Slots", "Create_time"]
    X_orig = data[relevant_columns_x].values
    if is_mixed_data:
        X_orig[:, 0:-2] = X_orig[:, 0:-2].astype('float64')
    else:
        X_orig[:, 0:-1] = X_orig[:, 0:-1].astype('float64')

    X_test = X_orig[:, 0:-1]
    y_test = data["Memory_bytes"].values
    y_test = y_test.astype('float64')

    if do_scaling:
        # scale with GB
        scaling = 1000000000
        # Scale bytes of filesize
        X_orig[:, 0] /= scaling
        # Scale bytes of memory bytes
        y_test /= scaling

    # X_test is data without Create_time
    # X_orig is data with Create_time
    # y is Memory_bytes
    return X_orig, X_test, y_test, tool_name


def calc_metrics_for_baseline(X_train, X_test, X_test_orig, X_test_unscaled, y_train, y_test, tool_name,
                              is_mixed_data: bool = False, run_config=None, doStandardScale=True):
    print("Calculate metrics for the baseline...")
    with open("../processed_data/tool_destinations.yaml") as f:
        tool_configs = yaml.load(f, Loader=SafeLoader)

    y_pred_constant = 0

    # Find latest occurrence of '/' to remove the version of the tool
    idx = tool_name.rfind('/')
    use_tool_name = tool_name
    if idx != -1:
        use_tool_name = tool_name[0:idx]
    if use_tool_name in tool_configs:
        if "mem" in tool_configs[use_tool_name].keys():
            y_pred_constant = tool_configs[use_tool_name]["mem"]
        else:
            # Tool does not have memory entry in config file. This means it is assigned default value 1 GB
            y_pred_constant = 1
    else:
        # Tool is not in config file. This means it is assigned default value 1 GB
        y_pred_constant = 1
    y_pred = np.ones_like(y_test) * y_pred_constant
    mean_abs_error = metrics.mean_absolute_error(y_test, y_pred)
    mean_squared_error = metrics.mean_squared_error(y_test, y_pred)
    root_mean_squared_error = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    r2_score = metrics.r2_score(y_test, y_pred)
    print("Tool name", tool_name)
    print("y_pred assigned by Galaxy:", y_pred_constant)
    print('Mean Absolute Error:', mean_abs_error)
    print('Mean Squared Error:', mean_squared_error)
    print('Root Mean Squared Error:', root_mean_squared_error)
    print('R2 Score:', r2_score)

    baseline_stats = {
        "mean_abs_error": mean_abs_error,
        "mean_squared_error": mean_squared_error,
        "root_mean_squared_error": root_mean_squared_error,
        "r2_score": r2_score
    }

    return y_pred, y_test, baseline_stats


# Code taken from http://blog.datadive.net/prediction-intervals-for-random-forests/
def pred_with_uncertainty(model, X, percentile=95):
    preds = []
    for decision_tree in model.estimators_:
        prediction = decision_tree.predict(X)
        preds.append(prediction)
    preds = np.vstack(preds).T
    err_up = np.percentile(preds, 100 - (100 - percentile) / 2., axis=1, keepdims=True)
    err_up = err_up.reshape(-1, )
    return err_up
