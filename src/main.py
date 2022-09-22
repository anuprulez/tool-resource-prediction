import estimator
import time
from process_data import *
import argparse
import yaml
from yaml.loader import SafeLoader
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument('run_config',
                        help="The path to the run configuration yaml file, where the trainings that should be done are configured. See example in README")
    parser.add_argument("--save",
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Set to save training/test results and the fitted model (default: false). Not possible for baseline.")
    parser.add_argument("--baseline",
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Set to evaluate the baseline on the run configuration (default: false). The baseline is given by the tool destination file from galaxy.")
    parser.add_argument("--remove_outliers",
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Set to remove outliers from the data (default: false). This means data outside of mean +- 2 * standard deviation gets removed")
    parser.add_argument('--model',
                        help="The path to the model (ONNX-file) you want to load and predict on")
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

    with open(args.run_config) as f:
        run_configs = yaml.load(f, Loader=SafeLoader)
    for key in tqdm(run_configs.keys()):
        # Wait 2 seconds between each run to avoid that files are overwritten
        time.sleep(2)
        run_configuration = run_configs[key]
        print("Following run config is used: ")
        print(run_configuration)
        if args.baseline:
            print("Run baseline...")
            estimator.baseline_pipeline(run_configuration=run_configuration, save=args.save)
        elif args.model:
            print("Evaluate given model...")
            estimator.evaluate_model_pipeline(run_configuration=run_configuration, model_path=args.model, save=args.save)
        else:
            print("Train model...")
            estimator.training_pipeline(run_configuration=run_configuration, remove_outliers=args.remove_outliers, save=args.save)
