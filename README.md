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
python main.py ../run_configurations/specific.yaml --baseline --save
```

### Train model on trinity/2.9.1 dataset
```
python main.py ../run_configurations/specific.yaml --save
```

### Evaluate an existing model on trinity/2.9.1 dataset
```
python main.py ../run_configurations/specific.yaml --model=saved_models/model_rf_trinity-2.9.1_2022_07_23-06_19_45_PM.onnx --save
```

## How to train a model:
- run "main.py" in the "src" folder with following parameters:
```
  - first parameter: path to the run configuration file
  - (optional) --save: save the training results and the trained model to a file (default: false)
  - (optional) --baseline: run the galaxy baseline on the given data (default: false)
  - (optional) --remove_outliers: set to remove outliers from the data (default: false). This only works for the training and not evaluation.
    Outliers are data points outside of mean +- 2 * standard deviation
```
The model predicts the memory bytes in GB

### Run configuration file
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

#### Example run configuration (yaml-file)

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

### Format of the dataset
The data used for the training, baseline or evaluation has to be in the following format (seperated by commas):
```
Tool_id, Filesize (in bytes), Number_of_files, Slots, Memory_bytes (in bytes), Create_time
```

### Saved data
By setting "--save" when running the "main.py" the training results and the trained model will be saved.
The trained model is hereby saved as an ONNX-file to the "saved_models" directory inside the "src" folder.
The training results are saved to the "saved_data" directory inside the "src" folder. 

The training results hereby have the following format:
```yaml
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
<if model type was Random Forest and a probability was given as parameter the following metrics are also given>
Mean absolute error with uncertainty: ...
Mean squared error with uncertainty: ...
Root mean squared error with uncertainty: ...
R2 Score with uncertainty: ...
############################
Filesize, Prediction, Target, Create_time
############################
<here all the data points from the test set are listed with Filesize (GB), Prediction (GB), Target (GB), Create_time> 
```

## How to load a model and evaluate on given data

- run "main.py" in the "src" folder with following parameters:
```
  - first parameter: path to the run configuration file. The run configuration file has to be in the same format as mentioned above
  - (optional) --save: save the evaluation results to a file (default: false)
  - (optional) --model: The path to the model (ONNX-file) you want to load and predict with
```

Don't forget to set the "dataset_path" to the actual data you want to use for evaluation.

The model predicts the memory bytes in GB