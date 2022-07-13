import argparse
import numpy as np
import random
import sys
import yaml
from yaml.loader import SafeLoader

np.set_printoptions(threshold=sys.maxsize)


# This method is used to process the original raw dataset
# It filters all jobs without file size or number of files and removes the columns job_id & state
# The processed data (new_data) has form tool_id, file size, num_files, runtime_seconds, slots, memory_bytes, create_time
def process_dataset(filename: str, number_rows=1000000, use_dict_tools=False):
    # 39 because we have between 38 and 39 million entries
    for i in range(39):
        # Filter out columns job_id, state
        original_data = np.loadtxt(filename, delimiter='|', skiprows=2 + i * number_rows, dtype=str,
                                   usecols=(1, 3, 4, 5, 6, 7, 8), max_rows=number_rows)
        new_data = []
        # Dict: tool_id --> tool_id, file size, num_files, runtime_seconds, slots, memory_bytes, create_time
        dict_tools = {}
        for idx, row in enumerate(original_data):
            stripped = [column.strip() for column in row]
            # We skip rows with no file size, number of files or memory_bytes
            if stripped[1] != "" and stripped[2] != "" and stripped[5] != "":
                file_size = int(stripped[1])
                num_files = int(stripped[2])
                # Skip invalid data
                if not ((file_size == 0 and num_files > 0) or (num_files == 0 and file_size > 0)):
                    tool_name = stripped[0]
                    # Find latest occurrence of '/' to remove the version of the tool
                    idx = tool_name.rfind('/')
                    if idx == -1:
                        idx = len(tool_name)
                    tool_name_without_version = tool_name[0:idx]
                    if use_dict_tools:
                        if tool_name_without_version in dict_tools:
                            dict_tools[tool_name_without_version].append(stripped)
                        else:
                            dict_tools[tool_name_without_version] = [stripped]
                    stripped[3] = stripped[3].replace(".0000000", "")
                    stripped[4] = stripped[4].replace(".0000000", "")
                    stripped[5] = stripped[5].replace(".0000000", "")
                    new_data.append(stripped)
        # Write all entries to a file
        with open('saved_data/dataset_stripped.txt', 'a+') as f:
            for entry in new_data:
                for idx, data_feature in enumerate(entry):
                    # No comma if last element
                    if idx != (len(entry) - 1):
                        f.write("%s," % data_feature)
                    else:
                        f.write("%s" % data_feature)
                f.write("\n")


def find_most_used_tools(filename: str, rows_per_chunk: int, save: bool, distinction_between_tools=False):
    # Dictionary: toolname_without_version --> number of entries in dataset
    dict_tools = {}

    # range should be such that it is rounded up to number of rows of file
    for i in range(26):
        # data has form tool_id, filesize, num_files, runtime_seconds, slots, memory_bytes, create_time
        data = np.loadtxt(filename, delimiter=',', skiprows=i * rows_per_chunk, dtype=str, max_rows=rows_per_chunk,
                          usecols=0)
        for tool_name in data:
            # Find latest occurrence of '/' to remove the version of the tool
            idx = tool_name.rfind('/')
            if idx == -1:
                idx = len(tool_name)
            tool_name_without_version = tool_name[0:idx]
            use_tool_name = tool_name if distinction_between_tools else tool_name_without_version
            if use_tool_name in dict_tools:
                dict_tools[use_tool_name] += 1
            else:
                dict_tools[use_tool_name] = 1

    list_tools_number_entries = []
    for key in dict_tools:
        list_tools_number_entries.append((key, dict_tools[key]))

    # Sort list such that most used tool is at the beginning
    list_tools_number_entries = sorted(list_tools_number_entries, key=lambda tup: -tup[1])

    if save:
        # Write list of most used tools into a file
        with open('saved_data/most_used_tools.txt', 'a+') as f:
            for entry in list_tools_number_entries:
                f.write("%s, %s\n" % (entry[0], entry[1]))


# This method extracts only the rows from the dataset that are given in the list of tools
def extract_entries_from_data(filename_dataset: str, filename_list_of_tools, number_tools: int, rows_per_chunk: int,
                              rndm_seed=0, sample_data=False, number_samples_per_tool=100,
                              distinction_between_versions: bool = False, specific_tool_number=None):
    tool_data = np.loadtxt(filename_list_of_tools, delimiter=',', dtype=str, max_rows=number_tools).reshape((-1, 2))
    if specific_tool_number is None:
        set_of_tools = set(tool_data[:, 0])
    else:
        set_of_tools = {tool_data[specific_tool_number, 0]}

    # Dictionary: tool_id --> tool_id, file size, num_files and memory_bytes
    dict_tools = {}

    for i in range(5):
        # we extract tool_id, filesize, num_files, slots, memory_bytes & create_time
        data = np.loadtxt(filename_dataset, delimiter=',', skiprows=i * rows_per_chunk, dtype=str,
                          max_rows=rows_per_chunk)
        for entry in data:
            tool_name = entry[0]
            # Find latest occurrence of '/' to remove the version of the tool
            idx = tool_name.rfind('/')
            if idx == -1:
                idx = len(tool_name)
            tool_name_without_version = tool_name[0:idx]
            use_tool = tool_name if distinction_between_versions else tool_name_without_version
            entry[0] = use_tool
            # We only consider entries which are in the set of tools
            if use_tool in set_of_tools:
                if use_tool in dict_tools:
                    dict_tools[use_tool].append(entry)
                else:
                    dict_tools[use_tool] = [entry]

    if sample_data:
        sampled_data_dict = {}
        if specific_tool_number is None:
            filename_str = 'saved_data/' + str(number_samples_per_tool) + '_samples_of_top_' + str(number_tools) + \
                           '_tools_seed_' + str(rndm_seed) + '.txt'
        else:
            filename_str = 'saved_data/' + str(number_samples_per_tool) + '_samples_of_tool_number_' + \
                           str(specific_tool_number) + '_seed_' + str(rndm_seed) + '.txt'
        for tool in dict_tools:
            random.seed(rndm_seed)
            sampled_data_dict[tool] = random.sample(dict_tools[tool], number_samples_per_tool)
            # Write sampled data to a file
            # Data format: tool_id, filesize, num_files, slots, memory bytes & create_time
            with open(filename_str, 'a+') as f:
                for entry in sampled_data_dict[tool]:
                    for idx, data_feature in enumerate(entry):
                        if idx != (len(entry) - 1):
                            f.write("%s," % data_feature)
                        else:
                            f.write("%s" % data_feature)
                    f.write("\n")


def remove_faulty_entries_from_data(filename_dataset: str, filename_tool_config: str, rows_per_chunk: int,
                                    save_data: bool):
    with open(filename_tool_config) as f:
        tool_configs = yaml.load(f, Loader=SafeLoader)

    # range should be such that it is rounded up to number of rows of file
    for i in range(26):
        valid_data = []
        faulty_data = []

        # data has form tool_id, filesize, num_files, runtime_seconds, slots, memory_bytes, create_time
        data = np.loadtxt(filename_dataset, delimiter=',', skiprows=i * rows_per_chunk, dtype=str,
                          max_rows=rows_per_chunk, usecols=(0, 1, 2, 4, 5, 6))
        for entry in data:
            tool_name = entry[0]
            # Scale memory bytes to GB
            memory_bytes = int(entry[4]) / 1000000000
            # Find latest occurrence of '/' to remove the version of the tool
            idx = tool_name.rfind('/')
            use_tool_name = tool_name
            if idx != -1:
                tool_name_without_version = tool_name[0:idx]
                idx = tool_name_without_version.rfind('/')
                if idx != -1:
                    use_tool_name = tool_name_without_version[idx + 1:]
            # If memory_bytes is greater than 1 TB it is faulty data
            if memory_bytes > 1000:
                faulty_data.append(entry)
                continue
            if use_tool_name in tool_configs:
                if "mem" in tool_configs[use_tool_name].keys():
                    memory_assigned = tool_configs[use_tool_name]["mem"]
                    # If used memory is over assigned memory times 8 it is faulty
                    if memory_bytes > (memory_assigned * 8):
                        faulty_data.append(entry)
                    else:
                        valid_data.append(entry)
                else:
                    # Tool does not have memory entry in config file. This means it is assigned default value 8 GB
                    if memory_bytes > 8:
                        faulty_data.append(entry)
                    else:
                        valid_data.append(entry)
            else:
                # Tool is not in config file. This means it is assigned default value 8 GB
                if memory_bytes > 8:
                    faulty_data.append(entry)
                else:
                    valid_data.append(entry)

        if save_data:
            with open('saved_data/valid_data.txt', 'a+') as f:
                for entry in valid_data:
                    for idx, data_feature in enumerate(entry):
                        if idx != (len(entry) - 1):
                            f.write("%s," % data_feature)
                        else:
                            f.write("%s" % data_feature)
                    f.write("\n")
            with open('saved_data/faulty_data.txt', 'a+') as f:
                for entry in faulty_data:
                    for idx, data_feature in enumerate(entry):
                        if idx != (len(entry) - 1):
                            f.write("%s," % data_feature)
                        else:
                            f.write("%s" % data_feature)
                    f.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This script allows you to process the dataset in various forms")
    parser.add_argument('task', metavar='task',
                        choices=['process_data', 'most_used', 'extract_entries', 'remove_faulty'],
                        help="Choose one of the following tasks you want to execute: "
                             "'process_data': process the original raw dataset "
                             "'most_used': find the most used tools in the dataset "
                             " 'extract_entries': given a file with a list of tools, extract only the "
                             "rows from the dataset that are given in the list of tools "
                             "'remove_faulty: remove entries in the data with faulty memory bytes")
    parser.add_argument('dataset_path', metavar='dataset_path',
                        help="The path to the dataset you want to apply the task on")
    args = parser.parse_args()

    if args.task == 'process_data':
        if args.dataset_path is None:
            parser.error("process_data requires --dataset_path")
        # process_dataset('../Galaxy1-[jobs_runs_resources_23_05_22.csv].txt', 1000000, args.save)
        process_dataset(args.dataset_path)
