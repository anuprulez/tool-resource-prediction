import pandas as pd
from scipy import stats
import re
from tqdm import tqdm

df = pd.read_csv('../processed_data/dataset_labeled/valid_data.txt', sep=",", usecols=["tool_id", "filesize", "memory_bytes"],
                 names=["tool_id", "filesize", "num_files", "slots", "memory_bytes", "create_time"])

corr_scores = dict()

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
    # Remove galaxy0 or galaxy 1 from name
    tool_name = re.sub(r"\+galaxy\d", "", tool_name)
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
    # Otherwise, 0 is assigned
    else:
        pearson_corr = 0
    corr_scores[tool] = (pearson_corr, nr_samples)

# Sort list such that the tool are ordered according to their correlation
corr_scores = corr_scores.items()
sorted_corr_scores = sorted(corr_scores, key=lambda tup: -tup[1][0])

filename = "saved_data/pearson_corr_scores_not_sorted.csv"
print(f"Save correlation scores to file {filename}...")
with open(filename, 'a+') as f:
    f.write("tool_id, pearson_correlation, nr_samples \n")
    for entry in sorted_corr_scores:
        tool_name = entry[0]
        corr_score = entry[1][0]
        nr_samples = entry[1][1]
        f.write(f"{tool_name}, {corr_score}, {nr_samples} \n")