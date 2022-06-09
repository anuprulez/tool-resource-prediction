import estimator
from process_data import *
import argparse
import yaml
from yaml.loader import SafeLoader
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument("--save",
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help="set to save data to files (default: false)")

    args = parser.parse_args()

    # process_dataset('../Galaxy1-[jobs_runs_resources_23_05_22.csv].txt', 1000000, args.save)

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

    # tools_to_extract = [1858, 2352, 4300, 4303, 4574]
    # for tool_num in tools_to_extract:
    #     extract_entries_from_data("../processed_data/dataset_labeled/valid_data.txt",
    #                               "../processed_data/most_used_tools.txt",
    #                               number_tools=5053,
    #                               rows_per_chunk=1000000,
    #                               rndm_seed=100,
    #                               sample_data=True,
    #                               number_samples_per_tool=20000,
    #                               distinction_between_versions=True,
    #                               specific_tool_number=tool_num)

    # specific = True
    # specific = False
    # if specific:
    #     specific_run_config = {
    #         "model_type": "rf",
    #         "dataset_path": "../processed_data/most_memory_tools/tools_with_trinity/trinity/2.9.1.txt",
    #         "is_mixed_data": False,
    #         "seed": 0,
    #         "model_params": {
    #             "n_estimators": 200,
    #             "random_state": 0,
    #             "criterion": "absolute_error"
    #         }}
    #     estimator.training_pipeline(run_configuration=specific_run_config, save=args.save)
    # else:
    #     with open("../run_configurations/remove_outliers1.yaml") as f:
    #         run_configs = yaml.load(f, Loader=SafeLoader)
    #     for key in tqdm(run_configs.keys()):
    #         run_configuration = run_configs[key]
    #         estimator.training_pipeline(run_configuration=run_configuration, save=args.save)

    # Baseline pipeline
    with open("../run_configurations/remove_outliers1.yaml") as f:
        run_configs = yaml.load(f, Loader=SafeLoader)
    for key in tqdm(run_configs.keys()):
        run_configuration = run_configs[key]
        estimator.baseline_pipeline(run_configuration=run_configuration, save=args.save)
