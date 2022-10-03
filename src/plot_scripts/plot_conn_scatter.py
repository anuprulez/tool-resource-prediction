import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# High correlation
# X = [0.35, 0.55, 0.66, 0.74, 0.95]
# Low correlation
X = [0.01, 0.07, 0.16, 0.31, 0.45]

# High correlation
# df_rf = pd.DataFrame({
#   'x_axis': X,
#   'y_axis': [0.13, 0.01, 0.89, 0.76, 0.96]
# })
# df_xgb = pd.DataFrame({
#   'x_axis': X,
#   'y_axis': [0.01, 0.03, 0.89, 0.72, 0.99]
# })
# df_lr = pd.DataFrame({
#   'x_axis': X,
#   'y_axis': [0.13, 0.13, 0.60, 0.59, 0.99]
# })
# df_svr = pd.DataFrame({
#   'x_axis': X,
#   'y_axis': [0.04, 0.20, 0.91, 0.76, 0.82]
# })

#Low correlation
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
plt.scatter('x_axis', 'y_axis', data=df_rf, label="Random Forest")
plt.scatter('x_axis', 'y_axis', data=df_xgb, label="XGB")
plt.scatter('x_axis', 'y_axis', data=df_lr, label="Linear Regression")
plt.scatter('x_axis', 'y_axis', data=df_svr, label="SVR")
corr_line = np.arange(-0.5, 1.1, 0.1)
plt.plot(corr_line, corr_line, 'silver', linestyle="dashed")
plt.xlabel("Pearson correlation")
plt.ylabel("R2 Score")
plt.title("Relationship between correlation and model performances")
plt.legend()
plt.show()
