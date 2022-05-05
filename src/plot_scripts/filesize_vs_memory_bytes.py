import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # filepath = "../predictions_vs_targets/500_faulty_19500_valid_tool_0.txt"
    filepath = "../saved_data/training_results_2022_05_05-01_56_18_AM.txt"
    # column_names = ["Validity", "Filesize", "Prediction", "Target"]
    column_names = ["Filesize", "Prediction", "Target"]
    data = pd.read_csv(filepath, ",", names=column_names, skiprows=12)
    absolute_error = (data["Prediction"] - data["Target"]).abs()
    scatter_plt = sns.scatterplot(data=data, x="Filesize", y="Target")
    scatter_plt.set_xlabel("Filesize in GB")
    scatter_plt.set_ylabel("Memory bytes in GB")
    scatter_plt.set(title="lofreq_call/lofreq_call/2.1.5+galaxy0")
    plt.show()