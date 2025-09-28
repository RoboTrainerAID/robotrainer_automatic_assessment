# Data Processing and Aggregation Pipeline

This project is designed for processing sensor data per user and path. It enables:

- Filtering specific topics
- Aggregating data by time intervals
- Filling missing values for certain physiological measurements
- Adding age and gender for each user
- Merging ground-truth data


---
## How to use
1. Put your input CSV files into the `src/data/` folder:
    - main dataset (recorded with RoboTrainer)
    - user information dataset (age, gender, user)
    - motoric test dataset
3. Build the docker image with `./build_docker.sh`
4. Start the container with `./start_docker.sh`
   - Executing `./start_docker.sh` again in a new terminal will connect to the already running container.
5. In the Docker, start the data preprocessing:
   ```bash
   python3 opt/src/main.py
   ```
## Data Preprocessing
The script `main.py`performs the following steps:

1. Topic Filtering 
    - Reads the main dataset form `data_csv`.
    - Keeps only the columns (topics) defined in `config.ini`.
2. Missing Value Handling
    - For `heart_rate`, `hrv`, `ppi` and `ppg` missing values are filled using forward/backward fill per user.
    - Other topics retain `Nan` if no values are available.
3. Time Aggregation 
    - Aggregates each user/path to a fixed sampling rate (`samples_per_second` in config).
    - At least one averaged row is kept per user/path.
    - Numeric values within each interval are averaged.
4. Add User Metadata (optional)
    - If `add_age_gender = True` in `config.ini`, the user’s age and gender are merged in from the user information dataset.
5. Merge Ground Truth 
    - The motoric test data is merged based on the `user` column to enrich the dataset with test results.

    
## Configuration (`config.ini`)
The preprocessing uses `opt/src/config.ini` to define which data to keep and how to process it.

### Sections
- **[DATA]**
    - `data_csv`: Path to main dataset
    - `user_info_csv`: Path to user metadata
    - `ground_truth_csv`: Path to ground-truth file

- **[SETTINGS]**
    - `topics`: Columns to include in the output
    - `add_age_gender`: `True`/`False` → whether age and gender should be added
    - `samples_per_second`: Target sampling rate (e.g. 1 → one row per second)
    - `path_column`: Column representing path
    - `time_column`: Column representing time within path
    - `duration_column`: Column representing total duration of path

### Example (excerpt from `config.ini`)

```ini
[DATA]
data_csv = opt/src/data/KATE_AA_dataset_10Hz.csv
user_info_csv = opt/src/data/test_subjects.csv
ground_truth_csv = opt/src/data/motoric_test.csv

[SETTINGS]
topics = hrv,heart_rate,ppi
add_age_gender = True
samples_per_second = 1
path_column = path
time_column = time
duration_column = total_duration
```

## Output
- Processed dataset (CSV) → `src/data/processed_data_<samples_per_second>_<add_age_gender>_<num_topics>.csv`
- Includes:
    - Filtered topics
    - Aggregated rows (based on sampling rate)
    - User metadata (optional)
    - Ground-truth data