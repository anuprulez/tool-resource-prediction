from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Models
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

import sys
import time
from datetime import datetime
import json
import numpy as np
import pandas as pd


def train_and_predict_random_forest(do_scaling: bool, seed: int, is_mixed_data: bool = False, run_config=None) \
        -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
    y_pred, y_true
    """
    np.set_printoptions(threshold=sys.maxsize)

    if is_mixed_data:
        column_names = ["Tool_id", "Filesize", "Number_of_files", "Slots", "Memory_bytes", "Create_time", "Validity"]
    else:
        column_names = ["Tool_id", "Filesize", "Number_of_files", "Slots", "Memory_bytes", "Create_time"]

    # dataset_path = "../processed_data/mixed_data/4000_faulty_20000_valid_tool_0.txt"
    # dataset_path = "../processed_data/mixed_data/500_faulty_19500_valid_tool_0.txt"
    # dataset_path = "../processed_data/labelled_valid/first_20000_samples_of_tool_number_0_seed_100.txt"
    dataset_path = "../processed_data/first_20000_samples_of_tool_number_0_seed_100.txt"

    # data = pd.read_csv("../processed_data/50_samples_of_top_1_tools_seed_100.txt", sep=',', names=column_names)
    # data = pd.read_csv("../processed_data/5000_samples_of_top_1_tools_seed_100.txt", sep=',',
    #                    names=column_names)
    # data = pd.read_csv("../processed_data/5000_samples_of_tool_number_18_seed_100.txt", sep=',',
    #                    names=column_names)
    # data = pd.read_csv("../processed_data/200000_samples_of_tool_number_0_seed_100.txt", sep=',',
    #                    names=column_names)
    # data = pd.read_csv("../processed_data/first_20000_samples_of_tool_number_0_seed_100.txt", sep=',',
    #                    names=column_names)
    if run_config is not None:
        dataset_path = run_config["dataset_path"]
    data = pd.read_csv(dataset_path, sep=',', names=column_names)

    # scale with GB
    scaling = 1000000000
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
        # Scale bytes of filesize
        X[:, 0] /= scaling
        # Scale bytes of memory bytes
        y /= scaling

    X_train_orig, X_test_orig, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    y_train = y_train.astype('float64')
    y_test = y_test.astype('float64')

    sc = StandardScaler()
    if is_mixed_data:
        # Scale all columns beside 'Validity' & 'Create_time
        X_train = sc.fit_transform(X_train_orig[:, 0:-2])
        X_test = sc.transform(X_test_orig[:, 0:-2])
    else:
        # Scale all columns beside 'Create_time
        X_train = sc.fit_transform(X_train_orig[:, 0:-1])
        X_test = sc.transform(X_test_orig[:, 0:-1])

    # TODO: try out some of these params
    # criterion='absolute_error', bootstrap=False, warm_start=True
    # regressor = RandomForestRegressor(n_estimators=200, random_state=seed)
    # regressor = RandomForestRegressor(n_estimators=200, random_state=0, criterion='absolute_error')
    # regressor = RandomForestRegressor(n_estimators=50, random_state=0, criterion='absolute_error')
    if run_config is not None:
        regressor = RandomForestRegressor(**run_config["model_params"])

    start_time = time.time()
    regressor.fit(X_train, y_train)
    end_time = time.time()
    time_for_training_mins = (end_time - start_time) / 60
    print("Time taken in minutes for training:", time_for_training_mins)

    feature_importance = regressor.feature_importances_
    print("Feature importance:", feature_importance)

    y_pred = regressor.predict(X_test)
    mean_abs_error = metrics.mean_absolute_error(y_test, y_pred)
    mean_squared_error = metrics.mean_squared_error(y_test, y_pred)
    root_mean_squared_error = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    print('Mean Absolute Error:', mean_abs_error)
    print('Mean Squared Error:', mean_squared_error)
    print('Root Mean Squared Error:', root_mean_squared_error)

    X_test_unscaled = sc.inverse_transform(X_test)

    # Extract tool name
    tool_name = data["Tool_id"][0]
    start_idx = 0
    idx = tool_name.rfind('/')
    if idx != -1:
        start_idx = tool_name[0:idx].rfind('/') + 1
    tool_name = tool_name[start_idx:]

    filename_str = "saved_data/training_results_" + str(datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")) + ".txt"
    with open(filename_str, 'a+') as f:
        f.write(f"Tool name: {tool_name}\n")
        f.write(f"Dataset_path: {dataset_path}\n")
        f.write(f"Is_mixed_data: {is_mixed_data}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Model_params: {json.dumps(run_config['model_params'])}\n")
        f.write(f"Time for training in mins: {time_for_training_mins}\n")
        f.write(f"Feature importance: {feature_importance}\n")
        f.write(f"Mean absolute error: {mean_abs_error}\n")
        f.write(f"Mean squared error: {mean_squared_error}\n")
        f.write(f"Root mean squared error: {root_mean_squared_error}\n")
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

    return y_pred, y_test
