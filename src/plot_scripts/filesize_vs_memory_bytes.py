import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # filepath = "../predictions_vs_targets/500_faulty_19500_valid_tool_0.txt"
    filepath = "../saved_data/training_results_2022_05_10-04_17_13_PM.txt"
    # column_names = ["Validity", "Filesize", "Prediction", "Target"]
    column_names = ["Filesize", "Prediction", "Target", "Create_time"]
    data = pd.read_csv(filepath, ",", names=column_names, skiprows=13)
    with open(filepath, "r") as file:
        tool_name = file.readline().replace("\n", "")

    def determine_quarter(create_time: str):
        quarter_str = ""
        year = create_time[0:4]
        month = int(create_time[5:7])
        if month <= 3:
            quarter_str += "1st "
        if 4 <= month <= 6:
            quarter_str += "2nd "
        if 7 <= month <= 9:
            quarter_str += "3rd "
        if 10 <= month <= 12:
            quarter_str += "4th "
        quarter_str += "quarter of " + year
        return quarter_str

    data["Quarter"] = [determine_quarter(row["Create_time"]) for index, row in data.iterrows()]
    absolute_error = (data["Prediction"].astype("float64") - data["Target"].astype("float64")).abs()
    scatter_plt = sns.scatterplot(data=data, x="Filesize", y="Target", hue="Quarter")
    scatter_plt.set_xlabel("Filesize in GB")
    scatter_plt.set_ylabel("Memory bytes in GB")
    scatter_plt.set(title=tool_name)
    plt.show()