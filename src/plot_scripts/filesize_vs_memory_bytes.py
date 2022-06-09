import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Before removing
    filepath = "../saved_data/training_rf_snpSift_filter-4.3+t.galaxy1_2022_06_09-12_26_29_PM.txt"
    filepath = "../saved_data/training_rf_rna_star-2.5.2b-1_2022_06_09-12_26_30_PM.txt"
    # After removing
    filepath = "../saved_data/training_rf_snpSift_filter-4.3+t.galaxy1_2022_06_09-12_29_45_PM.txt"
    filepath = "../saved_data/training_rf_rna_star-2.5.2b-1_2022_06_09-12_29_45_PM.txt"

    # column_names = ["Validity", "Filesize", "Prediction", "Target"]
    column_names = ["Filesize", "Prediction", "Target", "Create_time"]
    data = pd.read_csv(filepath, ",", names=column_names, skiprows=14)
    with open(filepath, "r") as file:
        tool_name = file.readline().replace("\n", "")
        while True:
            line = file.readline()
            if line.startswith("Mean absolute error"):
                mean_abs_error = line.rstrip('\n')[0:27]
            if line.startswith("R2 Score"):
                r2_score = line.rstrip('\n')[0:16]
                break

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
    data["Filesize"] = data["Filesize"]
    data["Target"] = data["Target"]

    quarters_2019 = np.sort(data["Quarter"].loc[data["Quarter"].str.contains('2019')].unique())
    quarters_2020 = np.sort(data["Quarter"].loc[data["Quarter"].str.contains('2020')].unique())
    quarters_2021 = np.sort(data["Quarter"].loc[data["Quarter"].str.contains('2021')].unique())
    quarters_2022 = np.sort(data["Quarter"].loc[data["Quarter"].str.contains('2022')].unique())
    hue_order = np.append(np.append(np.append(quarters_2019, quarters_2020), quarters_2021), quarters_2022)

    plt.figure()
    scatter_plt = sns.scatterplot(data=data, x="Filesize", y="Target", hue="Quarter", hue_order=hue_order)
    scatter_plt.set_xlabel("Filesize in GB")
    scatter_plt.set_ylabel("Memory in GB")
    scatter_plt.set(title=f"{tool_name} ({r2_score}, {mean_abs_error})")
    plt.show(block=False)

    plt.figure()
    scatter_plt2 = sns.scatterplot(data=data, x="Filesize", y=data["abs_error"], hue="Quarter", hue_order=hue_order)
    scatter_plt2.set_xlabel("Filesize in GB")
    scatter_plt2.set_ylabel("Absolute error")
    scatter_plt2.set(title=f"{tool_name} ({r2_score}, {mean_abs_error})")
    plt.show()