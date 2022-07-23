import pandas as pd
import yaml
from yaml import SafeLoader


def calc_avg_and_std(run_config):
    filepath = "../" + run_config["dataset_path"]
    column_names = ["Tool_id", "Filesize", "Num_files", "Slots", "Memory_bytes", "Create_time"]
    data = pd.read_csv(filepath, ",", names=column_names)
    memory_bytes = data["Memory_bytes"] / 1000000000
    avg = memory_bytes.mean()
    std = memory_bytes.std()
    print("Tool: " + data["Tool_id"][0])
    print("Mean: " + str(avg))
    print("Std: " + str(std))
    print("################")


with open("../../run_configurations/remove_outliers1.yaml") as f:
    run_configs = yaml.load(f, Loader=SafeLoader)
for key in run_configs.keys():
    run_configuration = run_configs[key]
    calc_avg_and_std(run_config=run_configuration)
