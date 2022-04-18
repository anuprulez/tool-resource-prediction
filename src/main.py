import estimator
from process_data import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument("--save",
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help="set to save data to files (default: false)")

    args = parser.parse_args()

    # process_dataset('../Galaxy1-[galaxy_tool_resources_2y.csv].txt', 1000000, args.save)
    # find_most_used_tools("../processed_data/dataset_stripped.txt", 1000000, args.save)

    # extract_entries_from_data("../processed_data/dataset_stripped.txt",
    #                           "../processed_data/most_used_tools.txt",
    #                           number_tools=100,
    #                           rows_per_chunk=1000000,
    #                           rndm_seed=100,
    #                           sample_data=True,
    #                           number_samples_per_tool=150,
    #                           distinction_between_versions=False)

    # extract_entries_from_data("../processed_data/dataset_stripped.txt",
    #                           "../processed_data/most_used_tools.txt",
    #                           number_tools=100,
    #                           rows_per_chunk=1000000,
    #                           rndm_seed=100,
    #                           sample_data=True,
    #                           number_samples_per_tool=5000,
    #                           distinction_between_versions=False,
    #                           specific_tool_number=18)

    estimator.train_and_predict(do_scaling=True)
