import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# df = pd.DataFrame(columns=["Dataset", "Model", "R2 Score"])
# df_entry = pd.DataFrame({""}, index=[0])
# # df_num_entries = pd.concat([df_num_entries, df_entry], ignore_index=True)
results_rf = [0.96, 0.76]

X = ["lofreq_indelqual/2.1.4", "bamleftalign/1.3.1", "fastqc/0.73", "umi_tools_extract/0.5.5.1", "bedtools_intersectbed/2.30.0"]
rf_scores = [0.96, 0.76, 0.89, 0.01, 0.13]
xgb_scores = [0.99, 0.72, 0.89, 0.03, 0.01]
lr_scores = [0.99, 0.59, 0.60, 0.13, 0.13]
svr_scores = [0.82, 0.76, 0.91, 0.20, 0.04]

X_axis = np.arange(len(X)) * 3

sns.set_style("darkgrid")
sns.set_palette(sns.color_palette("Set2"))
plt.bar(X_axis - 0.3, rf_scores, 0.4, label='Random Forest')
plt.bar(X_axis + 0.2, xgb_scores, 0.4, label='XGB')
plt.bar(X_axis + 0.7, lr_scores, 0.4, label='Linear Regression')
plt.bar(X_axis + 1.2, svr_scores, 0.4, label='SVR')
# plt.bar(X_axis - 0.3, rf_scores, 0.4, label='Random Forest', color="forestgreen")
# plt.bar(X_axis + 0.2, xgb_scores, 0.4, label='XGB', color="black")
# plt.bar(X_axis + 0.7, lr_scores, 0.4, label='Linear Regression', color="gold")
# plt.bar(X_axis + 1.2, svr_scores, 0.4, label='SVR')

plt.xticks(X_axis, X)
plt.xlabel("Tools")
plt.ylabel("R2 Score")
plt.title("Results for data with moderate to high correlation")
plt.legend()
plt.show()
