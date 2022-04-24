from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Models
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

import sys
import numpy as np
import pandas as pd


def train_and_predict_random_forest(do_scaling: bool) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
    y_pred, y_true
    """
    np.set_printoptions(threshold=sys.maxsize)

    # data = pd.read_csv("../processed_data/50_samples_of_top_1_tools_seed_100.txt", sep=',', names=["Tool_id", "Filesize", "Number_of_files", "Memory_bytes"])
    # data = pd.read_csv("../processed_data/5000_samples_of_top_1_tools_seed_100.txt", sep=',',
    #                    names=["Tool_id", "Filesize", "Number_of_files", "Runtime_seconds", "Slots", "Memory_bytes"])
    # data = pd.read_csv("../processed_data/5000_samples_of_tool_number_18_seed_100.txt", sep=',',
    #                    names=["Tool_id", "Filesize", "Number_of_files", "Runtime_seconds", "Slots", "Memory_bytes"])
    data = pd.read_csv("../processed_data/20000_samples_of_tool_number_0_seed_100.txt", sep=',',
                       names=["Tool_id", "Filesize", "Number_of_files", "Runtime_seconds", "Slots", "Memory_bytes"])

    # TODO: maybe scale with GB
    scaling = 1000000000
    relevant_columns_x = ["Filesize", "Number_of_files", "Runtime_seconds", "Slots"]
    X = data[relevant_columns_x].values
    X = X.astype('float64')

    y = data["Memory_bytes"].values
    y = y.astype('float64')

    if do_scaling:
        # Scale bytes of filesize
        X[:, 0] /= scaling
        # Scale bytes of memory bytes
        y /= scaling

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    y_train = y_train.astype('float64')
    y_test = y_test.astype('float64')

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    print("X_train: ", X_train[0:5])
    print("X_test: ", X_test[0:5])
    print("y_train: ", y_train[0:5])
    print("y_test ", y_test[0:5])

    # TODO: try out some of these params
    # criterion='absolute_error', bootstrap=False, warm_start=True
    regressor = RandomForestRegressor(n_estimators=200, random_state=0, criterion='absolute_error')
    regressor.fit(X_train, y_train)
    print("Feature importance: ", regressor.feature_importances_)
    y_pred = regressor.predict(X_test)

    print("y_pred: ", y_pred[0:5])
    print("y_test: ", y_test[0:5])

    # print("Absolute error: \n", np.abs(y_pred - y_test))

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    return y_pred, y_test

# def train_and_predict