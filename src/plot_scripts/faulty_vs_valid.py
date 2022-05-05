import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # filepath = "../predictions_vs_targets/500_faulty_19500_valid_tool_0.txt"
    filepath = "../predictions_vs_targets/20000_valid_tool_0.txt"
    column_names = ["Validity", "Filesize", "Prediction", "Target"]
    data = pd.read_csv(filepath, ",", names=column_names, skiprows=2)
    data["Validity"] = data["Validity"].replace(to_replace=-1, value="Faulty").replace(to_replace=1, value="Valid")
    absolute_error = (data["Prediction"] - data["Target"]).abs()
    scatter_plt = sns.scatterplot(data=data, x="Filesize", y=absolute_error, hue="Validity")
    scatter_plt.set_xlabel("Filesize in GB")
    scatter_plt.set_ylabel("Absolute error")
    scatter_plt.set(title="snpsift/snpSift_filter/4.3+t.galaxy1")
    plt.show()