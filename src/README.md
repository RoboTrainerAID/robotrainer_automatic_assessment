# Machine Learning Pipeline
This project provides an end-to-end pipeline for processing motoric test data and training machine learning models (MLP, SVR, Random Forest, AutoML).

The pipeline covers:
- Preprocessing of sensor data
- Normalization and dimensionality reduction (PCA)
- Flexible input block preparation (row, user, user+path)
- Model training and optimization (Bayesian optimization and AutoML)
- Evaluation and result storage

## How to Use
1. Put .bag files into the `data/` folder
2. The bag files are mounted into `/home/docker/ros_ws/data/` 
3. Build the docker image with `./build_docker.sh`
4. Start the container with `./start_docker.sh`
   - Executing `./start_docker.sh` again in a new terminal will connect to the already running container.
5. In the Docker, start the data preprocessing:
   ```bash
   python src/main.py
   ```

1. **Prepare your input data** <br>
Place your processed dataset(s) as CSV files into `src/data/` <br>
At minimum, the pipeline expects:
    - *.csv → main dataset containing sensor data and ground-truth test results
3. Build the docker image with `./build_docker.sh`
4. Start the container with `./start_docker.sh`
   - Executing `./start_docker.sh` again in a new terminal will connect to the already running container.
5. **Adjust configuration** <br>
Edit `src/config.ini` to set file paths, topics, PCA settings, and model parameters.
6. **Run the pipeline** <br>
Execute: 
    ```bash
    python3 opt/src/main.py
    ```
7. **Results** <br>
After training:
    - Models are trained and optimized
    - Evaluation results (MSE, RMSE, R² per topic + overall) are stored as CSV `opt/src/data/<model>_results.csv`

## Pipeline

The main script `main.py` performs the following steps:

1. **Data Loading** 
    - Reads CSV file (path from `config.ini`)
    - Applies one-hot encoding to categorical variables
2. **Feature and Target Preparation**
    - Splits dataset into features (X) and targets (y)
    - Supports different preparation modes:
        - 'row' → each row is one sample
        - 'user' → all rows per user are aggregated into one sample
        - 'user_path' → all rows per user and path are aggregated into one sample
3. **Normalization**
    - Applies min-max normalization to all numeric columns
4. **Optional PCA**
    - If `use_pca=True`, applies Principal Component Analysis
    - Variance threshold is determined by `pca_variance_threshold`
5. **Model Training**
    - Supported models:
        - `mlp` (Multi-Layer Perceptron)
        - `svr` (Support Vector Regression, Multi-Output)
        - `random_forest` (Random Forest Regressor)
        - `automl` (AutoSklearn)
    - Hyperparameter optimization with Bayesian Search (BayesSearchCV) (except AutoML)
6. **Evaluation**
    - Calculates MSE, RMSE, and R² per topic and overall
    - Saves results to CSV

## Configuration (`config.ini`)
The Pipeline uses a configuration file (`src/config.ini`).

### Sections
- **[DATA PATH]** 
    - `path_to_data`: Path to input CSV
    - `path_to_evaluation`: Path to evaluation file
- **[DATA]**
    - `mode`: Data preparation mode (`row`, `user`, `user_path`)
- **[SETTINGS]**
    - `model`: Model type (`mlp`, `svr`, `random_forest`, `automl`)
    - `use_pca`: `True/False` → whether PCA should be applied
    - `pca_variance_threshold`: Variance threshold for PCA (e.g. 0.95)
    - `kfolds`: Number of cross-validation folds
- **[GROUND TRUTH]**
    - `topics`: List of motoric test results to be predicted

### Example (excerpt from `config.ini`)

```ini
[DATA PATH]
path_to_data = opt/src/data/processed_data.csv
path_to_evaluation = opt/src/data/evaluation.csv

[DATA]
mode = row

[SETTINGS]
model = mlp
use_pca = True
pca_variance_threshold = 0.95
kfolds = 5

[GROUND TRUTH]
topics = Balance Test, Single Leg Stance, Robotrainer Front, Robotrainer Left, Robotrainer Right, Hand Grip Left, Jump & Reach, Tandem Walk, Figure 8 Walk, Jumping Sideways, Throwing Beanbag at Target, Tapping Test, Ruler Drop Test
```

## Output
- Trained models (not persisted, only scores + hyperparameters)
- Evaluation results (CSV) `opt/src/data/<model>.csv`

Contents include:
- Dataset name
- Model type
- Scores per topic
- Overall scores (MSE, RMSE, R²)
- Hyperparameters

