import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # rna_star/2.5.2b-1
    filepath = "../saved_data/training_results_2022_05_11-11_28_18_PM.txt"

    # trinity/2.9.1
    # filepath = "../saved_data/training_results_2022_05_12-12_11_46_AM.txt"

    # trinity_align_and_estimate_abundance/2.9.1
    # filepath = "../saved_data/training_results_2022_05_12-12_49_37_PM.txt"

    # trinity_abundance_estimates_to_matrix/2.9.1
    # filepath = "../saved_data/training_results_2022_05_12-12_10_41_PM.txt"

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
    data["abs_error"] = (data["Prediction"].astype("float64") - data["Target"].astype("float64")).abs()
    data["Filesize"] = data["Filesize"] * 1000000000
    data["Target"] = data["Target"] * 1000000000

    quarters_2019 = np.sort(data["Quarter"].loc[data["Quarter"].str.contains('2019')].unique())
    quarters_2020 = np.sort(data["Quarter"].loc[data["Quarter"].str.contains('2020')].unique())
    quarters_2021 = np.sort(data["Quarter"].loc[data["Quarter"].str.contains('2021')].unique())
    quarters_2022 = np.sort(data["Quarter"].loc[data["Quarter"].str.contains('2022')].unique())
    hue_order = np.append(np.append(np.append(quarters_2019, quarters_2020), quarters_2021), quarters_2022)

    plt.figure()
    scatter_plt = sns.scatterplot(data=data, x="Filesize", y="Target", hue="Quarter", hue_order=hue_order)
    scatter_plt.set_xlabel("Filesize")
    scatter_plt.set_ylabel("Memory bytes")
    scatter_plt.set(title=tool_name)
    plt.show(block=False)

    plt.figure()
    scatter_plt2 = sns.scatterplot(data=data, x="Filesize", y=data["abs_error"], hue="Quarter", hue_order=hue_order)
    scatter_plt2.set_xlabel("Filesize")
    scatter_plt2.set_ylabel("Absolute error")
    scatter_plt2.set(title=tool_name)
    plt.show()