import pandas as pd
import yaml
from scipy import stats
from yaml import SafeLoader


def calc_avg_and_std(run_config):
    filepath = run_config["dataset_path"]
    column_names = ["Tool_id", "Filesize", "Num_files", "Slots", "Memory_bytes", "Create_time"]
    data = pd.read_csv(filepath, ",", names=column_names)
    memory_bytes = data["Memory_bytes"] / 1000000000
    avg = memory_bytes.mean()
    std = memory_bytes.std()
    print("Tool: " + data["Tool_id"][0])
    print("Mean: " + str(avg))
    print("Std: " + str(std))
    print("################")

def calc_pearson_corr(run_config):
    filepath = run_config["dataset_path"]
    column_names = ["Tool_id", "Filesize", "Num_files", "Slots", "Memory_bytes", "Create_time"]
    data = pd.read_csv(filepath, ",", names=column_names)
    memory_bytes = data["Memory_bytes"]
    filesize = data["Filesize"]
    # memory_bytes = data["Memory_bytes"] / 1000000000
    # filesize = data["Filesize"] / 1000000000
    pearson_corr = stats.pearsonr(filesize, memory_bytes)
    print("Pearson correlation:", pearson_corr)

with open("../run_configurations/specific.yaml") as f:
    run_configs = yaml.load(f, Loader=SafeLoader)
for key in run_configs.keys():
    run_configuration = run_configs[key]
    calc_avg_and_std(run_config=run_configuration)
    calc_pearson_corr(run_config=run_configuration)
