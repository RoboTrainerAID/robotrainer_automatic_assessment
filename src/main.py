import pandas as pd
import configparser
from data_processing import DataProcessor  # deine Klasse

def main():
    # --- Konfiguration laden ---
    config = configparser.ConfigParser()
    config.read("opt/src/config.ini")
    print(config.sections())

    settings = config["SETTINGS"]
    data_path = config["DATA"]

    data = pd.read_csv(data_path.get("data_csv"))
    user_info = pd.read_csv(data_path.get("user_info_csv"))


    # --- DataProcessor nutzen ---
    processor = DataProcessor()
    processor.config = {
        "topics": [t.strip() for t in settings.get("topics", "").split(",")]
    }

    

    # 2) Topics filtern
    data = processor.filter_topics(data)
    print("Spaltennamen:", data.columns.tolist())
    print("Doppelte Spalten:", data.columns[data.columns.duplicated()].tolist())
    data = data.loc[:, ~data.columns.duplicated()].copy()
    user_info = user_info.loc[:, ~user_info.columns.duplicated()].copy()


    # 3) Zeitliche Begrenzung (relativ zu total_duration)
    data = processor.limit_data_by_time(
        data,
        time_column=settings.get("time_column"),
        path_column=settings.get("path_column"),
        time_length=int(settings.get("time_length")),
        duration_column=settings.get("duration_column")
    )

    
    # 1) Alter & Geschlecht hinzufügen
    if settings.getboolean("add_age_gender", fallback=False):
        data = processor.add_age_and_gender(data, user_info)

    print("\n--- Ergebnis ---")
    print(data)

  

if __name__ == "__main__":
    main()
