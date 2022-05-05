import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = pd.read_csv("../processed_data/200000_samples_of_tool_number_1_seed_100.txt", ",",
                       names=["Tool_id", "Filesize", "Number_of_files", "Slots", "Memory_bytes", "Create_time"])

    filesize = data["Filesize"].values / 1000000000
    memory_bytes = data["Memory_bytes"].values / 1000000000
    print(memory_bytes.mean())
    # graph = sb.scatterplot(data=data, x="Filesize", y="Memory_bytes")
    # graph.set_xscale('log')
    # graph.set_yscale('log')
    # graph = sb.histplot(data=data, x="Filesize")
    # sns.jointplot(x=x[1:10000], y=y[1:10000], kind="hex", color="#4CB391")
    sns.set_theme(style="ticks")
    df = pd.DataFrame({"x": filesize, "y": memory_bytes})

    graph = sns.jointplot(data=df, x="x", y="y", kind="scatter", color="#4CB391")
    plt.show()
