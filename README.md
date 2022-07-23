# Tool Resource Prediction for Genomic Datasets

## Requirements
Python 3.9 was used

## How to use

- All scripts have to be run with working directory set to "src" directory
- All the data files have to be .txt files except the files for running trainings and the tool configuration file from galaxy. These have to be .yaml files

## How to train a model:
- run "main.py" in the "src" folder with following parameters:
```
  - first parameter: path to the run configuration file
  - (optional) --save: save the training results and the trained model to a file
  - (optional) --baseline: run the galaxy baseline on the given data ()
  - (optional) --remove_outliers: set to remove outliers from the data. Outliers are data points outside of mean +- 2 * standard deviation
```

## Run configuration file
- The run configuration file has to be a yaml-file inside the "run_configurations" folder
- In this file you can define multiple trainings that will be run sequentially in the given order
- The file must have the following format:
```yaml
  <arbitrary unique name>:
  model_type: <"xgb" for Extra Gradient Boosting or "rf" for Random Forest
  dataset_path: <path to the dataset used for the training>
  is_mixed_data: False
  seed: <seed for splitting data into training and test set>
  model_params:
      <parameters for the model you want to use>
```

### Example run configuration (yaml-file)

```yaml
train1:
  model_type: "xgb"
  dataset_path: "../processed_data/most_memory_tools/tools_with_trinity/trinity/2.9.1.txt"
  is_mixed_data: False
  seed: 0
  model_params:
    n_estimators: 200
    random_state: 0
train2:
  model_type: "rf"
  dataset_path: "../processed_data/most_memory_tools/tools_with_trinity/trinity/2.9.1.txt"
  is_mixed_data: False
  seed: 0
  model_params:
    n_estimators: 200
    random_state: 0
      bootstrap: False
```

## Format of the dataset
The data used for the training, baseline or evaluation has to be in the following format (seperated by commas):
```
Tool_id, Filesize, Number_of_files, Slots, Memory_bytes, Create_time
```

## Saved data
By setting "--save" when running the "main.py" the training results and the trained model will be saved.
The trained model is hereby saved as an ONNX-file to the "saved_models" directory inside the "src" folder.
The training results are saved to the "saved_data" directory inside the "src" folder. 

The training results hereby have the following format:
```
Tool name: ...
Dataset_path: ...
Is_mixed_data: ...
Seed: ...
Model_params: ...
Time for training in mins: ...
Feature importance: <feature importances for Filesize, Number_of_files, Slots>
Mean absolute error: ...
Mean squared error: ...
Root mean squared error: ...
R2 Score: ...
<if model type was Random Forest the following metrics are also given>
Mean absolute error with uncertainty: 161.10533273756087
Mean squared error with uncertainty: 71876.79254177918
Root mean squared error with uncertainty: 268.09847545590253
R2 Score with uncertainty: 0.08260334804878877
############################
Filesize, Prediction, Target, Create_time
############################
<here all the data points from the test set are listed with Filesize, Prediction, Target, Create_time> 
```