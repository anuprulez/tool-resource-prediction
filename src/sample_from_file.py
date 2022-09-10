import numpy as np
from tqdm import tqdm

dir_path = "../processed_data/other_tools/"
# filepaths = ["Add_a_column1-1.6.txt", "bwa_mem-0.7.17.1.txt", "Cut1.txt", "fastp-0.20.1+galaxy0.txt",
#              "lofreq_call-2.1.5+galaxy0.txt", "lofreq_filter-2.1.5+galaxy0.txt",
#              "snpSift_extractFields-4.3+t.galaxy0.txt", "snpSift_filter-4.3+t.galaxy1.txt",
#              "tp_find_and_replace-1.1.3.txt", "vcfvcfintersect-1.0.0_rc3+galaxy0.txt"
#              ]
filepaths = ["bowtie2-2.3.4.3.txt"]
for filepath in tqdm(filepaths):
    original_data = np.loadtxt(dir_path + filepath, delimiter=',', dtype=str)

    num_samples = 5000
    # num_samples = 20000
    seed_rnd = 0
    np.random.seed(seed_rnd)
    sampled_data = original_data[np.random.choice(original_data.shape[0], size=num_samples, replace=False), :]

    # Extract tool name
    tool_name = original_data[0, 0]
    start_idx = 0
    idx = tool_name.rfind('/')
    if idx != -1:
        start_idx = tool_name[0:idx].rfind('/') + 1
    tool_name = tool_name[start_idx:].replace("/", "-")
    filename = f"{tool_name}_{num_samples}_samples_seed_{seed_rnd}.txt"
    with open(f'saved_data/samples/{filename}', 'a+') as f:
        for row_nr, entry in enumerate(sampled_data):
            for idx, data_feature in enumerate(entry):
                # No comma if last element
                if idx != (len(entry) - 1):
                    f.write("%s," % data_feature)
                else:
                    f.write("%s" % data_feature)
            # No new line if last line
            if row_nr != sampled_data.shape[0] - 1:
                f.write("\n")
