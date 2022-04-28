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

    # remove_faulty_entries_from_data(filename_dataset="../processed_data/dataset_stripped.txt",
    #                                 filename_tool_config="../processed_data/tool_destinations.yaml",
    #                                 rows_per_chunk=1000000,
    #                                 save_data=args.save)

    # find_most_used_tools("../processed_data/dataset_labeled/valid_data.txt", 1000000, args.save,
    #                      distinction_between_tools=True)

    # extract_entries_from_data("../processed_data/dataset_stripped.txt",
    #                           "../processed_data/most_used_tools.txt",
    #                           number_tools=100,
    #                           rows_per_chunk=1000000,
    #                           rndm_seed=100,
    #                           sample_data=True,
    #                           number_samples_per_tool=150,
    #                           distinction_between_versions=False)

    # extract_entries_from_data("../processed_data/dataset_labeled/valid_data.txt",
    #                           "../processed_data/most_used_tools.txt",
    #                           number_tools=100,
    #                           rows_per_chunk=1000000,
    #                           rndm_seed=100,
    #                           sample_data=True,
    #                           number_samples_per_tool=200000,
    #                           distinction_between_versions=True,
    #                           specific_tool_number=0)

    y_pred, y_true = estimator.train_and_predict_random_forest(do_scaling=True, seed=50)
    y_pred, y_true = estimator.train_and_predict_random_forest(do_scaling=True, seed=150)
    y_pred, y_true = estimator.train_and_predict_random_forest(do_scaling=True, seed=1000)
