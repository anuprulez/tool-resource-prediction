import pandas as pd
import numpy as np

df = pd.read_csv('saved_data/pearson_corr_scores_not_sorted.csv', sep=",", names=["tool_id", "pearson_correlation", "nr_samples"], skiprows=1)

nan_entries = df.loc[df['pearson_correlation'] == " nan"]
df = df[df["pearson_correlation"] != " nan"]
df["pearson_correlation"] = df["pearson_correlation"].astype(np.float32)
df = df.sort_values(by=["pearson_correlation", "nr_samples"], ascending=False)
combined_data = pd.concat([df, nan_entries])
combined_data.to_csv("saved_data/pearson_corr_scores_fixed.csv", index=False)