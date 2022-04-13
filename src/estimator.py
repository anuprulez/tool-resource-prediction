from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn import preprocessing

import numpy as np
import pandas as pd

data = np.loadtxt("../processed_data/150_samples_of_top_100_tools_seed_100.txt", delimiter=',', dtype=str)

# TODO: shuffle samples with seed
y = data[:, 3]
X = pd.DataFrame(data[:, 0])

# Calc correlation
df = pd.DataFrame(data[:, (1, 2, 3)])
headers = ["Filesize", "Number_of_files", "Memory_bytes"]
df.columns = headers
df = df.astype('int64')
print(df.corr())

label_encoder = preprocessing.LabelEncoder()
X_2 = X.apply(label_encoder.fit_transform)

enc = preprocessing.OneHotEncoder()
enc.fit(X_2)
one_hot_labels = enc.transform(X_2)

regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
regr.fit(X, y)
