import pandas as pd
import numpy as np
from scipy import stats
import re
from tqdm import tqdm

df = pd.read_csv('../processed_data/avg_memory_for_all_tools.csv', sep=",", skiprows= 1,
                 names=["tool_id", "pearson_correlation", "nr_samples"])

df = df[df.nr_samples >= 10000]
filename = "saved_data/avg_memory_min_10000_samples.csv"
print(f"Save file to {filename}...")
df.to_csv(filename, index=False)