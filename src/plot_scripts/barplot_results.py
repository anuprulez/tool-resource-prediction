import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# df = pd.DataFrame(columns=["Dataset", "Model", "R2 Score"])
# df_entry = pd.DataFrame({""}, index=[0])
# # df_num_entries = pd.concat([df_num_entries, df_entry], ignore_index=True)
results_rf = [0.96, 0.76]

# X = ["lofreq_indelqual/2.1.4", "bamleftalign/1.3.1", "vcf2tsv/1.0.0_rc3", "rna_star/2.7.2b", "rna_star/2.7.5b",
#      "kraken2/2.1.1", "fasta2tab/1.1.1", "gmx_sim/2020.4"]
X = ["lofreq_indelqual/2.1.4", "bamleftalign/1.3.1", "vcf2tsv/1.0.0_rc3", "rna_star/2.7.2b"]

# rf_scores = [0.96, 0.76, 0.09, -0.18, -0.33, -0.09, 0.69, -0.01]
# rf_scores_hpo = [0.98, 0.79, 0.15, 0.04, 0.09, 0.18, 0.80, 0.10]
# xgb_scores = [0.99, 0.72, -0.12, -0.22, -0.35, -0.02, 0.72, 0.01]
# xgb_scores_hpo = [0.96, 0.78, 0.07, 0.04, 0.09, 0.17, 0.80, 0.07]
# lr_scores = [0.99, 0.59, 0.14, 0.01, 0.05, 0.08, 0.25, 0.07]
# lr_scores_hpo = [0.99, 0.59, 0.14, 0.01, 0.05, 0.08, 0.25, 0.07]
# svr_scores = [0.74, 0.77, -0.01, -0.01, 0.07, -0.01, 0.81, 0.05]
# svr_scores_hpo = [0.86, 0.77, -0.04, 0.01, 0.07, -0.04, 0.78, 0.06]

rf_scores = [0.96, 0.76, 0.09, -0.18]
rf_scores_hpo = [0.98, 0.79, 0.15, 0.04]
xgb_scores = [0.99, 0.72, -0.12, -0.22]
xgb_scores_hpo = [0.96, 0.78, 0.07, 0.04]
lr_scores = [0.99, 0.59, 0.14, 0.01]
lr_scores_hpo = [0.99, 0.59, 0.14, 0.01]
svr_scores = [0.74, 0.77, -0.01, -0.01]
svr_scores_hpo = [0.86, 0.77, -0.04, 0.01]


X_axis = np.arange(len(X), 0, -1) * 3

sns.set_style("darkgrid")
# sns.set_palette(sns.color_palette("Set2"))

# hpo_scores = np.concatenate([rf_scores_hpo, xgb_scores_hpo, svr_scores_hpo])
# axis_hpo = np.concatenate([X_axis + 1.2, X_axis + 0.7, X_axis - 0.3])
hpo_scores = np.concatenate([rf_scores_hpo])
axis_hpo = np.concatenate([X_axis + 1.2])

# Horizontal
plt.barh(axis_hpo, hpo_scores, 0.4, label='with HPO', color="green")
plt.barh(X_axis + 1.2, rf_scores, 0.4, label='Random Forest')
# plt.barh(X_axis + 0.7, xgb_scores, 0.4, label="XGB")
# plt.barh(X_axis + 0.2, lr_scores, 0.4, label='Linear Regression')
# plt.barh(X_axis - 0.3, svr_scores, 0.4, label='SVR')
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

plt.title("R2 Score with and without HPO")
plt.legend()
plt.show()
