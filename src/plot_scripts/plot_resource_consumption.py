import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re

##Kraken2
# data_path = "../../experiments/Experiment 6 - HPO & Comparison to Galaxy/results/Second run/rf/training_results_rf_kraken2-2.1.1+galaxy1_2022_09_29-01_19_56_AM.txt"
# 0.99 uncertainty
# data_path = "../../experiments/Experiment 7 - Accuracy - Failure trade-off/results/evaluation_results_rf_kraken2-2.1.1+galaxy1_2022_09_29-07_58_16_PM.txt"
# assigned_memory_galaxy = 64

##rna_star/2.7.5b
# data_path = "../../experiments/Experiment 6 - HPO & Comparison to Galaxy/results/Second run/rf/training_results_rf_rna_star-2.7.5b_2022_09_29-01_15_33_AM.txt"
# 0.99 uncertainty
data_path = "../../experiments/Experiment 7 - Accuracy - Failure trade-off/results/evaluation_results_rf_rna_star-2.7.5b_2022_09_29-08_12_23_PM.txt"
assigned_memory_galaxy = 90

file = open(data_path, 'r')
Lines = file.readlines()
skiprows = 0
tool_name = ""
for idx, line in enumerate(Lines):
    line = line.strip()
    if line == "Filesize (GB), Prediction (GB), Target (GB), Create_time":
        skiprows = idx + 2
    search_str = "Tool name: "
    if line.startswith(search_str):
        tool_name = line[(line.find(search_str) + len(search_str)):]
df = pd.read_csv(filepath_or_buffer=data_path, sep=",", skiprows=skiprows,
                 names=["Filesize", "Prediction", "Target", "Create_time"])

target_values = df.Target.values
prediction_values = df.Prediction.values

# Extract tool name
start_idx = 0
idx = tool_name.rfind('/')
if idx != -1:
    start_idx = tool_name[0:idx].rfind('/') + 1
tool_name = tool_name[start_idx:]
# Remove +...galaxy0 or +...galaxy 1 from name
tool_name = re.sub(r"\+.*galaxy\d", "", tool_name)
print(f"Tool: {tool_name}")

##Galaxy
# Find values where there were too many resources assigned
resource_difference_galaxy = assigned_memory_galaxy - target_values
resource_abundance_per_entry_galaxy = resource_difference_galaxy[resource_difference_galaxy >= 0]
# Find jobs where too few resources were assigned resulting in failed jobs by Galaxy
nr_failed_jobs_galaxy = np.sum(resource_difference_galaxy < 0)
percentage_failed_jobs_galaxy = nr_failed_jobs_galaxy / len(df)
print(f"Total overallocation in GB (Galaxy): {np.sum(resource_abundance_per_entry_galaxy)}")
print(f"Average overallocation in GB (Galaxy): {np.mean(resource_abundance_per_entry_galaxy)}")
print(f"Percentage of failed jobs due to underallocation (Galaxy): {percentage_failed_jobs_galaxy * 100}%")

##Model
# Find values where there were too many resources assigned
resource_difference_model = prediction_values - target_values
resource_abundance_per_entry_model = resource_difference_model[resource_difference_model >= 0]
# Find jobs where too few resources were assigned resulting in failed jobs by Galaxy
nr_failed_jobs_model = np.sum(resource_difference_model < 0)
percentage_failed_jobs_model = nr_failed_jobs_model / len(df)
print(f"Total overallocation in GB (Model): {np.sum(resource_abundance_per_entry_model)}")
print(f"Average overallocation in GB (Model): {np.mean(resource_abundance_per_entry_model)}")
print(f"Percentage of failed jobs due to underallocation (Model): {percentage_failed_jobs_model * 100}%")

sns.set_style("darkgrid")
sns.set_theme()
# plt.axhline(y=assigned_memory_galaxy, color="red", label=f"{assigned_memory_galaxy} GB assigned by Galaxy")
galaxy_error = assigned_memory_galaxy - target_values
model_error = df.Prediction.values - target_values
# plt.plot(np.arange(len(df.head(200))), 'Target', data=df.head(200), linestyle='-', marker='o', label="True values")
# plt.plot(np.arange(len(df.head(200))), 'Prediction', data=df.head(200), linestyle='-', marker='o', label="Model prediction")
plt.plot(np.arange(100), galaxy_error[:100], linestyle='-', marker='o', label="Galaxy error")
plt.plot(np.arange(100), model_error[:100], linestyle='-', marker='o', label="Random Forest error")
plt.legend()
plt.xlabel("Data points")
plt.ylabel("Difference to actual value [GB]")
plt.title(f"Tool: {tool_name}")
plt.show()
