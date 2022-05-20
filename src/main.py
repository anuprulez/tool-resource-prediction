import estimator
from process_data import *
import argparse
import yaml
from yaml.loader import SafeLoader

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

    specific = True
    # specific = False
    if specific:
        specific_run_config = {
            "dataset_path": "../processed_data/most_memory_tools/tools_with_trinity/trinity/2.9.1.txt",
            "is_mixed_data": False,
            "seed": 0,
            "model_params": {
                "n_estimators": 200,
                "random_state": 0,
                "criterion": "absolute_error"
            }}
        estimator.train_and_predict_random_forest(do_scaling=True, seed=specific_run_config["seed"],
                                                  is_mixed_data=specific_run_config["is_mixed_data"],
                                                  run_config=specific_run_config)
    else:
        with open("../run_configurations/high_memory_tools_1.yaml") as f:
            run_configs = yaml.load(f, Loader=SafeLoader)
        for key in run_configs.keys():
            run_configuration = run_configs[key]
            estimator.train_and_predict_random_forest(do_scaling=True, seed=run_configuration["seed"],
                                                      is_mixed_data=run_configuration["is_mixed_data"],
                                                      run_config=run_configuration)

    # y_pred, y_true = estimator.train_and_predict_random_forest(do_scaling=True, seed=0, is_mixed_data=False)
    # y_pred, y_true = estimator.train_and_predict_random_forest(do_scaling=True, seed=50)
    # y_pred, y_true = estimator.train_and_predict_random_forest(do_scaling=True, seed=150)
    # y_pred, y_true = estimator.train_and_predict_random_forest(do_scaling=True, seed=1000)
