# Tool Resource Prediction for Genomic Datasets

The original dataset taken from Galaxy can be found here: https://usegalaxy.eu/u/kumara/h/tool-resources

## Requirements
Python 3.9 was used.
Install the necessary packages by using the [requirements file](requirements.txt).
```
pip install -r requirements.txt
```

## How to use

- All scripts have to be run with working directory set to "src" directory
- All the data files have to be .txt files except the files for running trainings and the tool configuration file from galaxy. These have to be .yaml files

## Example scripts

### Run baseline on trinity/2.9.1 dataset
```
python main.py ../run_configurations/example_run_config.yaml --baseline --save
```

### Train model on trinity/2.9.1 dataset
```
python main.py ../run_configurations/example_run_config.yaml --save
```

### Evaluate an existing model on trinity/2.9.1 dataset
```
python main.py ../run_configurations/example_run_config.yaml --model=saved_models/model_rf_trinity-2.9.1_2022_08_30-02_45_03_PM.onnx --save
```

## Format of the dataset
The data used for the training, baseline or evaluation has to be in the following format (seperated by commas):
```
Tool_id, Filesize (in bytes), Number_of_files, Slots, Memory_bytes (in bytes), Create_time
```

## Run configuration file
- The run configuration file has to be a yaml-file inside the "run_configurations" folder
- In this file you can define multiple trainings that will be run sequentially in the given order
- Following model types are supported: 
  - "rf" for Random Forest
  - "xgb" for XGBoost
  - "lr" for Linear Regression
  - "svr" for Support Vector Regression
- The file must have the following format:
```yaml
<arbitrary unique name>:
  model_type: <one of the model types defined above>
  dataset_path: <path to the dataset that you want to use for training/evaluation/baseline>
  seed: <seed for splitting data into training and test set in case of training>
  probability_uncertainty (optional): probability used for uncertainty prediction in range [0,1]
  doStandardScale (optional): set True to scale the inputs using StandardScaler (default --> False)
  doHPO (optional): set True to do Hyperparameter Optimization (default --> false)
  do_cross_validation (optional): set True to do Cross Validation (default --> false)
  remove_outliers (optional): set True to remove outliers from the data before training (default --> false)Outliers are data points outside of mean +- 2 * standard deviation
  model_params:
      <parameters for the model you want to use>
```

### Example run configuration (yaml-file)

```yaml
train1:
  model_type: "xgb"
  dataset_path: "../processed_data/most_memory_tools/tools_with_trinity/trinity/2.9.1.txt"
  seed: 0
  doStandardScale: False
  doHPO: True 
  do_cross_validation: False
  model_params:
    n_estimators: 200
    random_state: 0
train2:
  model_type: "rf"
  dataset_path: "../processed_data/most_memory_tools/tools_with_trinity/trinity/2.9.1.txt"
  seed: 0
  doStandardScale: True
  model_params:
    n_estimators: 200
    random_state: 0
    bootstrap: False
```

## How to train a model:
- run "main.py" in the "src" folder with following parameters:
```
  - first parameter: path to the run configuration file
  - (optional) --save: save the training results and the trained model to a file (default: false)
```
The model predicts the memory bytes in GB

### Saved data
By setting "--save" when running the "main.py" the training results, train set, test set and the trained model will be saved in distinct files.
The trained model is hereby saved as an ONNX-file & as a .joblib-file to the "saved_models" directory inside the "src" folder.
The training results, train set & test set are saved to the "saved_data" directory inside the "src" folder. 

The training results hereby have the following format:
```yaml
Tool name: ...
Run configuratiom: <the keys and values of the used run configuration>
Model_params: ...
Time for training in mins: ...
Percentage above mean + 2 * std that got removed: <only if remove_outliers was True in run configuration>
Feature importance: <feature importances for Filesize, Number_of_files, Slots>
Mean absolute error: ...
Mean squared error: ...
Root mean squared error: ...
R2 Score: ...
Mean cross-validation test score: <only if cross-validation was done>
<if model type was Random Forest and a probability was given as parameter the following metrics are also given>
Mean absolute error with uncertainty: ...
Mean squared error with uncertainty: ...
Root mean squared error with uncertainty: ...
R2 Score with uncertainty: ...
############################
Filesize (GB), Prediction (GB), Target (GB), Create_time
############################
<here all the data points from the test set are listed with Filesize (GB), Prediction (GB), Target (GB), Create_time> 
```

## How to load a model and evaluate on given data

- run "main.py" in the "src" folder with following parameters:
```
  - first parameter: path to the run configuration file. The run configuration file has to be in the same format as mentioned above
  - (optional) --save: save the evaluation results to a file (default: false)
  - (optional) --model: The path to the model (ONNX-file or .joblib-file) you want to load and predict with
```

Prediction with uncertainty is only available if the model is loaded as a .joblib-file

Don't forget to set the "dataset_path" in the run configuration file to the actual data you want to use for evaluation.

The model predicts the memory bytes in GB

## How to run the Galaxy baseline on given data

- run "main.py" in the "src" folder with following parameters:
```
  - first parameter: path to the run configuration file. The run configuration file has to be in the same format as mentioned above
  - (optional) --save: save the baseline results to a file (default: false)
  - (optional) --baseline: flag that the baseline should run on the data
```

Don't forget to set the "dataset_path" in the run configuration file to the actual data you want to use for evaluation.

## Information about the dataset

There are several files that give us more information about the dataset and the specific tools.

- [pearson correlation for all tools](processed_data/pearson_corr_scores.csv):
  This file lists all the tools with their respective pearson correlation between filesize & memory_bytes
- [average memory for all tools](processed_data/avg_memory_for_all_tools.csv):
  This file lists all the tools with their respective avg. memory in GBs
- [faulty data](processed_data/dataset_labeled/faulty_data.txt):
  This file lists all the entries in the dataset that have faulty values for memory_bytes. 
  Faulty because they exceed the maximum possible memory size assigned by Galaxy.