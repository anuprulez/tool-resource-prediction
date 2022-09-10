import pandas as pd

data = pd.read_csv('../processed_data/example_data/bowtie2.csv')

with open('saved_data/bowtie2_transformed.txt', 'a+') as f:
    for idx, entry in data.iterrows():
        f.write("bowtie2,")
        own_file_size = int(entry[1])
        input_1_size = int(entry[2])
        input_2_size = int(entry[3])
        sum_filesize = own_file_size + input_1_size + input_2_size
        f.write(f"{sum_filesize},")
        num_files = (1 if own_file_size > 0 else 0) + (1 if input_1_size > 0 else 0) + (1 if input_2_size > 0 else 0)
        f.write(f"{num_files},")
        # Cores is assigned 8 by Galaxy
        f.write("8,")
        f.write(f"{entry[0]},")
        # No create time
        f.write("0\n")