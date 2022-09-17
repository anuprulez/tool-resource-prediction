import pandas as pd
import numpy as np
from scipy import stats
import re
from tqdm import tqdm

df = pd.read_csv('../processed_data/dataset_labeled/valid_data.txt', sep=",", usecols=["tool_id", "filesize", "memory_bytes"],
                 names=["tool_id", "filesize", "num_files", "slots", "memory_bytes", "create_time"])

corr_scores = dict()
df_corr_scores = pd.DataFrame(columns=["tool_id", "pearson_correlation", "nr_samples"])

for idx, entry in tqdm(df.iterrows()):
    filesize = entry["filesize"]
    memory_bytes = entry["memory_bytes"]

    # Extract tool name
    tool_name = entry["tool_id"]
    start_idx = 0
    idx = tool_name.rfind('/')
    if idx != -1:
        start_idx = tool_name[0:idx].rfind('/') + 1
    tool_name = tool_name[start_idx:]
    # Remove +...galaxy0 or +...galaxy 1 from name
    tool_name = re.sub(r"\+.*galaxy\d", "", tool_name)
    if tool_name in corr_scores:
        corr_scores[tool_name]["filesize"].append(filesize)
        corr_scores[tool_name]["memory_bytes"].append(memory_bytes)
    else:
        corr_scores[tool_name] = dict()
        corr_scores[tool_name]["filesize"] = [filesize]
        corr_scores[tool_name]["memory_bytes"] = [memory_bytes]


for tool in tqdm(corr_scores):
    filesizes = corr_scores[tool]["filesize"]
    memory_bytes = corr_scores[tool]["memory_bytes"]
    nr_samples = len(filesizes)
    # Need at least two samples to calc pearson correlation
    if nr_samples >= 2:
        pearson_corr = stats.pearsonr(filesizes, memory_bytes)[0]
    # Otherwise nan is assigned
    else:
        pearson_corr = np.nan
    corr_scores[tool] = (pearson_corr, nr_samples)

for key, value in corr_scores.items():
    df_entry = pd.DataFrame({"tool_id": key,
                             "pearson_correlation": value[0],
                             "nr_samples": value[1]
                             }, index=[0])
    df_corr_scores = pd.concat([df_corr_scores, df_entry], ignore_index=True)

df_corr_scores = df_corr_scores.sort_values(by=["pearson_correlation", "nr_samples"], ascending=False)
filename = "saved_data/pearson_corr_scores.csv"
print(f"Save correlation scores to file {filename}...")
df_corr_scores.to_csv(filename, index=False, na_rep="nan")