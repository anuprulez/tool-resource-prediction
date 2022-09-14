# from dask import dataframe as dd
import pandas as pd

df = pd.read_csv('../processed_data/dataset_labeled/valid_data.txt', sep=",", usecols=["filesize", "memory_bytes"],
                 names=["tool_id", "filesize", "num_files", "slots", "memory_bytes", "create_time"])

print(df.head())