import pandas as pd
import glob

def merge_and_save(file_list, output_name, subjects_df, sep=","):
    if not file_list:
        print(f"No files found for {output_name}")
        return
    
    dfs = [pd.read_csv(f, sep=sep) for f in file_list]
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df = merged_df.merge(subjects_df, on="user", how="left")

    merged_df.to_csv(output_name, index=False, sep=sep)
    print(f"Merged {len(file_list)} files into {output_name}")


def average_by_user_path(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    grouped_df = df.groupby(["user", "path"], as_index=False).mean(numeric_only=True)
    
    non_numeric_cols = df[["user", "path", "age", "gender"]].drop_duplicates()
    final_df = grouped_df.merge(non_numeric_cols, on=["user", "path"], how="left")

    final_df = final_df.drop(columns=["path_index_left", "path_index_front", "path_index_right"], errors="ignore")

    final_df.to_csv(output_csv, index=False)
    print(f"CSV erfolgreich gruppiert und gemittelt: {output_csv}")


def main():
    path = 'opt/src/data/'

    subjects_df = pd.read_csv("opt/src/data/test_subjects.csv", sep=",")

    subjects_df = subjects_df.rename(columns={
        "Alter": "age",
        "Geschlecht": "gender"
    })[["user", "age", "gender"]]

    csv_1Hz = glob.glob(path + '*1Hz*.csv')
    csv_10Hz = glob.glob(path + '*10Hz*.csv')

    merge_and_save(csv_1Hz, 'opt/src/KATE_AA_dataset_1Hz.csv', subjects_df, sep=",")
    merge_and_save(csv_10Hz, 'opt/src/KATE_AA_dataset_10Hz.csv', subjects_df, sep=",")

    average_by_user_path(
    "opt/src/KATE_AA_dataset_1Hz.csv",
    "opt/src/KATE_AA_dataset.csv"
    )


if __name__ == "__main__":
    main()


