import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
    memory_bytes = "Memory_bytes"
    filesize = "Filesize"
    data = pd.read_csv("../../processed_data/top_10_tools/Add_a_column1-1.6.txt", ",",
                       names=["Tool_id", "Filesize", "Number_of_files", "Slots", "Memory_bytes", "Create_time"])
    # Scale memory bytes by GB
    data[memory_bytes] = (data[memory_bytes].values / 1000000000).astype('float64')
    data[filesize] = (data[filesize].values / 1000000000).astype('float64')
    # data[memory_bytes] = np.log1p(data[memory_bytes].values / 1000000000).astype('float64')
    # data[filesize] = np.log1p(data[filesize].values / 1000000000).astype('float64')

    scatter_plt = sns.scatterplot(data=data, x=filesize, y=memory_bytes)
    # scatter_plt = sns.histplot(data=data, x=filesize)
    scatter_plt.set_xlabel(filesize + " in GB")
    scatter_plt.set_ylabel(memory_bytes + " in GB")
    scatter_plt.set(title=data["Tool_id"].iloc[0])
    plt.show()


if __name__ == "__main__":
    # plot_memory_bytes_over_time()
    plot_file_size_memory_bytes()