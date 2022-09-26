import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

X = [0.01, 0.07, 0.16, 0.31, 0.45]

# data
df_rf = pd.DataFrame({
  'x_axis': X,
  'y_axis': [-0.41, 0.01, -0.18, 0.09, 0.61]
})
df_xgb = pd.DataFrame({
  'x_axis': X,
  'y_axis': [-0.63, -0.43, -0.22, -0.12, 0.65]
})
df_lr = pd.DataFrame({
  'x_axis': X,
  'y_axis': [0.01, 0.44, 0.01, 0.14, 0.21]
})
df_svr = pd.DataFrame({
  'x_axis': X,
  'y_axis': [-0.02, 0.09, 0.01, 0.07, 0.63]
})

sns.set_style("darkgrid")
sns.set_palette(sns.color_palette("Set2"))

# plot
plt.plot('x_axis', 'y_axis', data=df_rf, linestyle='--', marker='o', label="Random Forest")
plt.plot('x_axis', 'y_axis', data=df_xgb, linestyle='--', marker='o', label="XGB")
plt.plot('x_axis', 'y_axis', data=df_lr, linestyle='--', marker='o', label="Linear Regression")
plt.plot('x_axis', 'y_axis', data=df_svr, linestyle='--', marker='o', label="SVR")
plt.xlabel("Pearson correlation")
plt.ylabel("R2 Score")
plt.title("Relationship between correlation and model performances")
plt.legend()
plt.show()
