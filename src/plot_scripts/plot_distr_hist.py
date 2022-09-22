import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import PowerTransformer

memory_bytes = "Memory_bytes"
filesize = "Filesize"
# data_path = "../../processed_data/most_memory_tools/tools_with_trinity/trinity/2.9.1.txt"
# data_path = "../../processed_data/most_memory_tools/rna_star/2.7.5b.txt"
data_path = "../../processed_data/most_memory_tools/rna_star/2.7.2b.txt"
column_names = ["Tool_id", "Filesize", "Number_of_files", "Slots", "Memory_bytes", "Create_time"]
data = pd.read_csv(data_path, sep=",", names=column_names, usecols=["Filesize", "Memory_bytes"])
# Scale bytes to GB
data[memory_bytes] = (data[memory_bytes].values / 1000000000).astype('float64')
data[filesize] = (data[filesize].values / 1000000000).astype('float64')

var_to_use = filesize

f, (ax0, ax1) = plt.subplots(1, 2)

ax0.hist(data[var_to_use], bins=200, density=True)
ax0.set_ylabel("Probability")
ax0.set_xlabel(f"{var_to_use}")
ax0.set_title(f"{var_to_use} distribution")

transformed = np.log1p(data[var_to_use])
# pt = PowerTransformer()
# transformed = pt.fit_transform(data[var_to_use].values.reshape(-1, 1))
ax1.hist(transformed, bins=200, density=True)
ax1.set_ylabel("Probability")
ax1.set_xlabel(f"{var_to_use} transformed")
ax1.set_title(f"Transformed {var_to_use} distribution")

plt.show()