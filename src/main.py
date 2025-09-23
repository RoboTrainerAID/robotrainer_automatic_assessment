import pandas as pd
import configparser
import os
from data_processing import DataProcessor  # deine Klasse

def main():
    # --- Load configuration ---
    config = configparser.ConfigParser()
    config.read("opt/src/config.ini")
    print("Found sections:", config.sections())

    settings = config["SETTINGS"]
    data_path = config["DATA"]

    # --- Load data ---
    data = pd.read_csv(data_path.get("data_csv"))
    user_info = pd.read_csv(data_path.get("user_info_csv"))

    # --- Use DataProcessor ---
    print("Initializing DataProcessor with configuration...")
    processor = DataProcessor()
    processor.config = {
        "topics": [t.strip() for t in settings.get("topics", "").split(",")]
    }

    # --- Filter topics ---
    data = processor.filter_topics(data)
    print("Columns after filtering:", data.columns.tolist())
    print("Duplicate columns:", data.columns[data.columns.duplicated()].tolist())

    # --- Remove duplicate columns ---
    data = data.loc[:, ~data.columns.duplicated()].copy()
    user_info = user_info.loc[:, ~user_info.columns.duplicated()].copy()


    # --- 5) Aggregation nach Samples ---
    samples_per_second = float(settings.get("samples_per_second", 1))
    data = processor.aggregate_by_samples(
        data,
        time_column=settings.get("time_column"),
        path_column=settings.get("path_column"),
        duration_column=settings.get("duration_column"),
        samples_per_unit=samples_per_second
    )

    
    # --- Optional: Add age and gender ---
    if settings.getboolean("add_age_gender", fallback=False):
        data = processor.add_age_and_gender(data, user_info)
    
    # --- Add ground truth ---
    ground_truth_path = data_path.get("ground_truth_csv")
    data = processor.add_ground_truth(data, ground_truth_path, on_column="user")

    # --- Save processed data ---
    output_dir = "opt/src/processed_data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "processed_data.csv")
    data.to_csv(output_path, index=False)

    # --- Show final data ---
    print("\n--- Ergebnis ---")
    print(data)

  

if __name__ == "__main__":
    main()
