import pandas as pd
import numpy as np
import re
from matplotlib import pyplot as plt
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler

memory_bytes = "Memory_bytes"
filesize = "Filesize"
# data_path = "../../processed_data/most_memory_tools/tools_with_trinity/trinity/2.9.1.txt"
# data_path = "../../processed_data/most_memory_tools/rna_star/2.7.5b.txt"
# data_path = "../../processed_data/most_memory_tools/rna_star/2.7.2b.txt"
# data_path = "../../processed_data/sampled_data/bowtie2-2.3.4.3_5000_samples_seed_0.txt"
# data_path = "../../processed_data/other_tools/bwa-0.7.17.4.txt"
# data_path = "../../experiments/Experiment 2 - Transformations/data/bowtie2-2.3.4.3/bowtie2-2.3.4.3-ready.txt"
data_path = "../../processed_data/other_tools/chembl-0.10.1.txt"
column_names = ["Tool_id", "Filesize", "Number_of_files", "Slots", "Memory_bytes", "Create_time"]
data = pd.read_csv(data_path, sep=",", names=column_names, usecols=["Tool_id", "Filesize", "Memory_bytes"])

# Extract tool name
tool_name = data["Tool_id"][0]
start_idx = 0
idx = tool_name.rfind('/')
if idx != -1:
    start_idx = tool_name[0:idx].rfind('/') + 1
tool_name = tool_name[start_idx:]
# Remove +...galaxy0 or +...galaxy 1 from name
tool_name = re.sub(r"\+.*galaxy\d", "", tool_name)

# Scale bytes to GB
data[memory_bytes] = (data[memory_bytes].values / 1000000000).astype('float64')
data[filesize] = (data[filesize].values / 1000000000).astype('float64')

var_to_use = filesize

f, (ax0, ax1) = plt.subplots(1, 2)

tmpScaler = MinMaxScaler()
data_to_use = tmpScaler.fit_transform(data[var_to_use].values.reshape(-1, 1))
ax0.hist(data_to_use, bins=500)
ax0.set_ylabel("Number of instances")
ax0.set_xlabel(f"{var_to_use}")
ax0.set_title(f"{var_to_use} histogram")

# transformed = np.log1p(data[var_to_use])
pt = PowerTransformer()
transformed = pt.fit_transform(data[var_to_use].values.reshape(-1, 1))

minMaxScaler = MinMaxScaler()
transformed = minMaxScaler.fit_transform(transformed)

ax1.hist(transformed, bins=500)
ax1.set_ylabel("Number of instances")
ax1.set_xlabel(f"log({var_to_use} + 1)")
ax1.set_title(f"Transformed {var_to_use} histogram")
f.suptitle(tool_name)

plt.show()