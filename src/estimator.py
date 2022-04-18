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


def train_and_predict(do_scaling: bool):
    np.set_printoptions(threshold=sys.maxsize)

    # data = pd.read_csv("../processed_data/50_samples_of_top_1_tools_seed_100.txt", sep=',', names=["Tool_id", "Filesize", "Number_of_files", "Memory_bytes"])
    # data = pd.read_csv("../processed_data/5000_samples_of_top_1_tools_seed_100.txt", sep=',',
    #                    names=["Tool_id", "Filesize", "Number_of_files", "Runtime_seconds", "Slots", "Memory_bytes"])
    data = pd.read_csv("../processed_data/5000_samples_of_tool_number_18_seed_100.txt", sep=',',
                       names=["Tool_id", "Filesize", "Number_of_files", "Runtime_seconds", "Slots", "Memory_bytes"])

    scaling = 1000000
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

    # data = pd.read_csv("../processed_data/petrol_consumption.csv")
    # X = data.iloc[:, 0:4].values
    # y = data.iloc[:, 4].values

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

    # criterion='absolute_error', bootstrap=False, warm_start=True
    regressor = RandomForestRegressor(n_estimators=200, random_state=0)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    print("y_pred: ", y_pred[0:5])
    print("y_test: ", y_test[0:5])

    # print("Absolute error: \n", np.abs(y_pred - y_test))

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    ######################
    # # TODO: shuffle samples with seed
    # y = data[:, 3]
    # X = pd.DataFrame(data[:, 0])
    #
    # # Calc correlation
    # df = pd.DataFrame(data[:, (1, 2, 3)])
    # headers = ["Filesize", "Number_of_files", "Memory_bytes"]
    # df.columns = headers
    # df = df.astype('int64')
    # print(df.corr())
    #
    # label_encoder = preprocessing.LabelEncoder()
    # X_2 = X.apply(label_encoder.fit_transform)
    #
    # enc = preprocessing.OneHotEncoder()
    # enc.fit(X_2)
    # one_hot_labels = enc.transform(X_2)
    #
    # regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
    # regr.fit(X, y)
