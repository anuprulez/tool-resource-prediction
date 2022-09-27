import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# df = pd.DataFrame(columns=["Dataset", "Model", "R2 Score"])
# df_entry = pd.DataFrame({""}, index=[0])
# # df_num_entries = pd.concat([df_num_entries, df_entry], ignore_index=True)
results_rf = [0.96, 0.76]

X = ["bowtie2", "hisat2"]
rf_scores = [0.90, 0.75]
xgb_scores = [0.88, 0.72]
lr_scores = [0.38, 0.42]
svr_scores = [0.87, 0.55]

X_axis = np.arange(len(X), 0, -1) * 3

sns.set_style("darkgrid")
sns.set_palette(sns.color_palette("Set2"))

# Horizontal
plt.barh(X_axis + 1.2, rf_scores, 0.4, label='Random Forest')
plt.barh(X_axis + 0.7, xgb_scores, 0.4, label='XGB')
plt.barh(X_axis + 0.2, lr_scores, 0.4, label='Linear Regression')
plt.barh(X_axis - 0.3, svr_scores, 0.4, label='SVR')
plt.yticks(X_axis, X)
plt.ylabel("Tools")
plt.xlabel("R2 Score")

# Vertical
# plt.bar(X_axis - 0.3, rf_scores, 0.4, label='Random Forest')
# plt.bar(X_axis + 0.2, xgb_scores, 0.4, label='XGB')
# plt.bar(X_axis + 0.7, lr_scores, 0.4, label='Linear Regression')
# plt.bar(X_axis + 1.2, svr_scores, 0.4, label='SVR')
# plt.xticks(X_axis, X)
# plt.xlabel("Tools")
# plt.ylabel("R2 Score")

plt.title("Results for Kamali's data")
plt.legend()
plt.show()
