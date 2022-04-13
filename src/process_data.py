import numpy as np
import random

# This method is used to process the original raw dataset
# It filters all jobs without file size or number of files and removes the columns job_id, state & create_time
# The processed data (new_data) has form tool_id, filesize, num_files, runtime_seconds, slots, memory_bytes
def process_dataset(filename: str, number_rows: int, save_data: bool):
    # 34 because we have between 33 and 34 million entries
    for i in range(34):
        # Filter out columns job_id, state, create_time
        original_data = np.loadtxt(filename, delimiter='|', skiprows=2 + i * number_rows, dtype=str,
                                   usecols=(1, 3, 4, 5, 6, 7), max_rows=number_rows)
        new_data = []
        # Dictionary: toolname_without_version --> tool_id, filesize, num_files, runtime_seconds, slots, memory_bytes
        dict_tools = {}
        for row in original_data:
            stripped = [column.strip() for column in row]
            # We skip rows with no file size or number of files
            if stripped[1] != "" and stripped[2] != "":
                tool_name = stripped[0]
                # TODO: maybe add option to make distinction between different versions of tools
                # Find latest occurrence of '/' to remove the version of the tool
                idx = tool_name.rfind('/')
                if idx == -1:
                    idx = len(tool_name)
                tool_name_without_version = tool_name[0:idx]
                if tool_name_without_version in dict_tools:
                    dict_tools[tool_name_without_version].append(stripped)
                else:
                    dict_tools[tool_name_without_version] = [stripped]
                new_data.append(stripped)

        # print("Number of different tools: ", len(dict_tools))

        # Find out which tool has most entries

        # number_of_entries_per_tool = {}
        # for key in dict_tools:
        #     entries = dict_tools[key]
        #     number_of_entries_per_tool[key] = len(entries)
        # highest_value = max(number_of_entries_per_tool, key=number_of_entries_per_tool.get)
        # print("Tool", highest_value, "has most entries. In total", number_of_entries_per_tool[highest_value])
        # print("####################################")
        #
        # list_tools_number_entries = []
        # for key in dict_tools:
        #     entries = dict_tools[key]
        #     list_tools_number_entries.append((key, len(entries)))
        #
        # # Sort list such that most used tool is at the beginning
        # list_tools_number_entries = sorted(list_tools_number_entries, key=lambda tup: -tup[1])

        if save_data:
            # # Write data of tool with most entries into a file
            # with open('saved_data/tool_with_most_entries.txt', 'a') as f:
            #     for item in dict_tools[highest_value]:
            #         f.write("%s\n" % item)
            #
            # # Write list of most used tools into a file
            # with open('saved_data/most_used_tools.txt', 'a') as f:
            #     for entry in list_tools_number_entries:
            #         f.write("%s, %s\n" % (entry[0], entry[1]))

            # Write all entries to a file
            with open('saved_data/all_entries.txt', 'a+') as f:
                for entry in new_data:
                    for idx, data_feature in enumerate(entry):
                        if idx != (len(entry) - 1):
                            f.write("%s," % data_feature)
                        else:
                            f.write("%s" % data_feature)
                    f.write("\n")


def find_most_used_tools(filename: str, rows_per_chunk: int, save: bool):
    # Dictionary: toolname_without_version --> number of entries in dataset
    dict_tools = {}

    # 25 because we have between 24 and 25 million entries
    for i in range(25):
        # data has form tool_id, filesize, num_files, runtime_seconds, slots, memory_bytes
        data = np.loadtxt(filename, delimiter=',', skiprows=i * rows_per_chunk, dtype=str, max_rows=rows_per_chunk,
                          usecols=0)
        for tool_name in data:
            # Find latest occurrence of '/' to remove the version of the tool
            idx = tool_name.rfind('/')
            if idx == -1:
                idx = len(tool_name)
            tool_name_without_version = tool_name[0:idx]
            if tool_name_without_version in dict_tools:
                dict_tools[tool_name_without_version] += 1
            else:
                dict_tools[tool_name_without_version] = 1

    list_tools_number_entries = []
    for key in dict_tools:
        list_tools_number_entries.append((key, dict_tools[key]))

    # Sort list such that most used tool is at the beginning
    list_tools_number_entries = sorted(list_tools_number_entries, key=lambda tup: -tup[1])

    if save:
        # Write list of most used tools into a file
        with open('saved_data/most_used_tools.txt', 'a') as f:
            for entry in list_tools_number_entries:
                f.write("%s, %s\n" % (entry[0], entry[1]))


# This method extracts only the rows from the dataset that are given in the list of tools
def extract_entries_from_data(filename_dataset: str, filename_list_of_tools, number_tools: int, rows_per_chunk: int,
                              rndm_seed=0, sample_data=False, number_samples_per_tool=100):
    set_of_tools = set(np.loadtxt(filename_list_of_tools, delimiter=',', dtype=str, usecols=0, max_rows=number_tools))

    # Dictionary: toolname_without_version --> toolname_without_version, filesize, num_files and memory_bytes
    dict_tools = {}

    for i in range(10):
        # we extract only tool_id, filesize, num_files and memory_bytes
        data = np.loadtxt(filename_dataset, delimiter=',', skiprows=i * rows_per_chunk, dtype=str,
                          max_rows=rows_per_chunk, usecols=(0, 1, 2, 5))
        for entry in data:
            tool_name = entry[0]
            # Find latest occurrence of '/' to remove the version of the tool
            idx = tool_name.rfind('/')
            if idx == -1:
                idx = len(tool_name)
            tool_name_without_version = tool_name[0:idx]
            entry[0] = tool_name_without_version
            # We only consider entries which are in the set of tools
            if tool_name_without_version in set_of_tools:
                if tool_name_without_version in dict_tools:
                    dict_tools[tool_name_without_version].append(entry)
                else:
                    dict_tools[tool_name_without_version] = [entry]

    if sample_data:
        sampled_data_dict = {}
        filename_str = 'saved_data/sampled_data_top_' + str(number_tools) + '_tools.txt'
        for tool in dict_tools:
            random.seed(rndm_seed)
            sampled_data_dict[tool] = random.sample(dict_tools[tool], number_samples_per_tool)
            # Write sampled data to a file
            with open(filename_str, 'a+') as f:
                for entry in sampled_data_dict[tool]:
                    for idx, data_feature in enumerate(entry):
                        if idx != (len(entry) - 1):
                            f.write("%s," % data_feature)
                        else:
                            f.write("%s" % data_feature)
                    f.write("\n")
