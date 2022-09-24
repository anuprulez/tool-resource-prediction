import pandas as pd
import numpy as np
from scipy import stats
import re
from tqdm import tqdm

df = pd.read_csv('../processed_data/pearson_corr_scores.csv', sep=",", skiprows= 1,
                 names=["tool_id", "pearson_correlation", "nr_samples"])

df = df[df.nr_samples >= 10000]
filename = "saved_data/pearson_corr_scores_min_10000_samples.csv"
print(f"Save correlation scores to file {filename}...")
df.to_csv(filename, index=False)