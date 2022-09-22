import pandas as pd
import numpy as np
from tqdm import tqdm
import re

data_path = '../processed_data/dataset_labeled/faulty_data.txt'
column_names = ["tool_id", "filesize", "num_files", "slots", "memory_bytes", "create_time"]

df = pd.read_csv(data_path, sep=",", usecols=["tool_id"], names=column_names, skiprows=1)
df_num_entries = pd.DataFrame(columns=["tool_id", "nr_samples"])
dict_tool_entries = {}

for idx, entry in tqdm(df.iterrows()):
    # Extract tool name
    tool_name = entry["tool_id"]
    start_idx = 0
    idx = tool_name.rfind('/')
    if idx != -1:
        start_idx = tool_name[0:idx].rfind('/') + 1
    tool_name = tool_name[start_idx:]
    # Remove +...galaxy0 or +...galaxy 1 from name
    tool_name = re.sub(r"\+.*galaxy\d", "", tool_name)
    if tool_name not in dict_tool_entries:
        dict_tool_entries[tool_name] = 1
    else:
        dict_tool_entries[tool_name] = dict_tool_entries[tool_name] + 1

for key, value in dict_tool_entries.items():
    df_entry = pd.DataFrame({"tool_id": key, "nr_samples": value}, index=[0])
    df_num_entries = pd.concat([df_num_entries, df_entry], ignore_index=True)

df_num_entries = df_num_entries.sort_values(by=["nr_samples", "tool_id"], ascending=False)
filename = "saved_data/num_entries_per_tool_faulty_data.csv"
print(f"Save number entries per tool to file {filename}...")
df_num_entries.to_csv(filename, index=False, na_rep="nan")