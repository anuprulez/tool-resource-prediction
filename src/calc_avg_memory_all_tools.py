import pandas as pd
import re
from tqdm import tqdm

df = pd.read_csv('../processed_data/dataset_labeled/valid_data.txt', sep=",", usecols=["tool_id", "memory_bytes"],
                 names=["tool_id", "filesize", "num_files", "slots", "memory_bytes", "create_time"])

df_to_save = pd.DataFrame(columns=["tool_id", "avg_memory_in_gb", "nr_samples"])
dict_tool_memory = {}

for idx, entry in tqdm(df.iterrows()):
    memory_in_gb = entry["memory_bytes"] / 1000000000
    # Process tool name
    tool_name = entry["tool_id"]
    start_idx = 0
    idx = tool_name.rfind('/')
    if idx != -1:
        start_idx = tool_name[0:idx].rfind('/') + 1
    tool_name = tool_name[start_idx:]
    # Remove +...galaxy0 or +...galaxy 1 from name
    tool_name = re.sub(r"\+.*galaxy\d", "", tool_name)
    if tool_name not in dict_tool_memory:
        dict_tool_memory[tool_name] = (memory_in_gb, 1)
    else:
        old_avg_memory = dict_tool_memory[tool_name][0]
        old_nr_samples = dict_tool_memory[tool_name][1]
        nr_samples = old_nr_samples + 1
        avg_memory = (old_avg_memory * old_nr_samples + memory_in_gb) / nr_samples
        dict_tool_memory[tool_name] = (avg_memory, nr_samples)

for key, value in dict_tool_memory.items():
    df_entry = pd.DataFrame({"tool_id": key,
                             "avg_memory_in_gb": value[0],
                             "nr_samples": value[1]
                             }, index=[0])
    df_to_save = pd.concat([df_to_save, df_entry], ignore_index=True)

df_to_save = df_to_save.sort_values(by=['avg_memory_in_gb'], ascending=False)
filename = "saved_data/avg_memory_for_all_tools.csv"
print(f"Save avg memory of all tools to file {filename}...")
df_to_save.to_csv(filename, index=False)
