import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re

def plot_memory_bytes_over_time():
    create_time = "Create_time"
    memory_bytes = "Memory_bytes"
    data = pd.read_csv("../../processed_data/example_data/bowtie2_transformed.txt", ",",
                       names=["Tool_id", "Filesize", "Number_of_files", "Slots", "Memory_bytes", "Create_time"])
    # Scale memory bytes by GB
    data[memory_bytes] = (data[memory_bytes].values / 1000000000).astype('float64')
    data_2020 = data[data.Create_time.str.startswith('2020')]
    data_2021 = data[data.Create_time.str.startswith('2021')]
    data_2022 = data[data.Create_time.str.startswith('2022')]

    # print(data["Memory_bytes"].mean())

    sorted_data = data.sort_values(create_time)
    sorted_data['7point_avg'] = sorted_data[memory_bytes].rolling(7).mean().shift(-3)
    # Extract dates from data with format YYYY-MM-DD
    dates = sorted_data[create_time].str.slice(stop=10)
    unique_dates = dates.unique()

    x_ticks = [unique_dates[0],
               unique_dates[int(len(unique_dates) / 8)],
               unique_dates[int(len(unique_dates) * 2 / 8)],
               unique_dates[int(len(unique_dates) * 3 / 8)],
               unique_dates[int(len(unique_dates) / 2)],
               unique_dates[int(len(unique_dates) * 5 / 8)],
               unique_dates[int(len(unique_dates) * 6 / 8)],
               unique_dates[int(len(unique_dates) * 7 / 8)],
               unique_dates[-1]]

    line_plt = sns.lineplot(x=dates, y=memory_bytes, label="Single point", data=sorted_data, ci=None)
    line_plt.set_ylabel(memory_bytes + " in GB")
    line_plt.set(title=data["Tool_id"].iloc[0])

    # avg_plot = sns.lineplot(x=dates, y="7point_avg", label="7-point Avg", data=sorted_data, ci=None)
    # graph = sns.scatterplot(data=data_2021, x="Filesize", y="Memory_bytes")
    # graph.set_xscale('log')
    # graph.set_yscale('log')
    plt.xticks(x_ticks)
    plt.show()


def plot_file_size_memory_bytes():
    doLogTrafo = False
    memory_bytes = "Memory_bytes"
    filesize = "Filesize"
    column_names = ["Tool_id", "Filesize", "Number_of_files", "Slots", "Memory_bytes", "Create_time"]
    # data_path = "../../processed_data/other_tools/smf_utils_estimate-energy-smf-v1.7-0_utils-v2.1.1-2.txt"
    # data_path = "../../experiments/Experiment 1 - Removing faulty data/data/mimodd_reheader-0.1.8_1/Ready/mimodd_reheader-0.1.8_1-only-valid.txt"
    # data_path = "../../experiments/Experiment 1 - Removing faulty data/data/fastqc-0.72/Ready/fastqc-0.72-only-valid.txt"
    # data_path = "../../experiments/Experiment 1 - Removing faulty data/data/ivar_trim-1.2.2/Ready/ivar_trim-1.2.2-only-valid.txt"
    # data_path = "../../experiments/Experiment 1 - Removing faulty data/data/ivar_removereads-1.2.2/Ready/ivar_removereads-1.2.2-only-valid.txt"
    # data_path = "../../experiments/Experiment 1 - Removing faulty data/data/cutadapt-1.16.5/Ready/cutadapt-1.16.5-only-valid.txt"
    # data_path = "../../processed_data/sampled_data/bowtie2-2.3.4.3_5000_samples_seed_0.txt"
    # data_path = "../../processed_data/kamali's data/bowtie2_transformed.txt"
    data_path = "../../experiments/Experiment 6 - HPO & Comparison to Galaxy/data/low memory/fasta2tab-1.1.1.txt"
    data = pd.read_csv(data_path, sep=",", names=column_names)
    # Scale memory bytes by GB
    data[filesize] = (data[filesize].values / 1000000000).astype('float64')
    data[memory_bytes] = (data[memory_bytes].values / 1000000000).astype('float64')
    if doLogTrafo:
        data[filesize] = np.log1p(data[filesize].values).astype('float64')
        # data[memory_bytes] = np.log1p(data[memory_bytes].values / 1000000000).astype('float64')

    scatter_plt = sns.scatterplot(data=data, x=filesize, y=memory_bytes)
    # scatter_plt = sns.histplot(data=data, x=filesize)
    scatter_plt.set_xlabel(filesize + " in GB")
    scatter_plt.set_ylabel("Memory bytes in GB")

    # Extract tool name
    tool_name = data["Tool_id"].iloc[0]
    start_idx = 0
    idx = tool_name.rfind('/')
    if idx != -1:
        start_idx = tool_name[0:idx].rfind('/') + 1
    tool_name = tool_name[start_idx:]
    # Remove +...galaxy0 or +...galaxy 1 from name
    tool_name = re.sub(r"\+.*galaxy\d", "", tool_name)
    scatter_plt.set(title=tool_name)
    plt.show()


if __name__ == "__main__":
    sns.set_style("darkgrid")
    sns.set_palette(sns.color_palette("Set2"))
    # plot_memory_bytes_over_time()
    plot_file_size_memory_bytes()