import pandas as pd
import numpy as np

class DataProcessor:
    """Processes data based on configuration settings."""

    def add_age_and_gender(self, data, user_info):
        """
        Adds age and gender based on the user mapping.
        Only the columns 'user', 'age', and 'gender' are included.
        """
        user_info_filtered = user_info.rename(columns={"Alter": "age", "Geschlecht": "gender"})[["user", "age", "gender"]]
        return data.merge(user_info_filtered, on="user", how="left")


    def filter_topics(self, data):
        """
        Filters the data based on the topics (column names)
        specified in the configuration.
        """
        topics_to_keep = self.config.get("topics", [])

        # Important base columns (only include them if they actually exist)
        base_cols = [col for col in ["time", "path", "total_duration", "user", "age", "gender"] if col in data.columns]

        # Only keep the desired topics in addition
        topic_cols = [col for col in data.columns if col in topics_to_keep]

        return data[base_cols + topic_cols]


    def aggregate_by_samples(self, data, time_column, path_column, duration_column="total_duration", samples_per_unit=1):
        """
        Aggregates the data per user/path to a desired sample rate.
        Fills missing values for heart_rate, hrv, and ppi with previous or next values of the same user.
        """
        result_list = []

        # Columns to be filled
        cols_to_fill = [c for c in ["heart_rate", "hrv", "ppi"] if c in data.columns]
        data = data.sort_values(["user", path_column, time_column])

        # Forward and backward fill per user
        for col in cols_to_fill:
            data[col] = data.groupby("user")[col].ffill().bfill()

        for (user, path), group in data.groupby(["user", path_column]):
            total_time = group[duration_column].iloc[0]

            n_samples = max(1, int(np.ceil(total_time * samples_per_unit)))
            bins = np.linspace(0, total_time, n_samples + 1)

            group["interval"] = pd.cut(group[time_column], bins=bins, labels=False, include_lowest=True)

            agg = group.groupby("interval").mean(numeric_only=True)

            # Additional information
            agg[path_column] = path
            agg["user"] = user

            result_list.append(agg)

        return pd.concat(result_list, ignore_index=True)


    def add_ground_truth(self, data, ground_truth_path, on_column="user"):
        """
        Adds the ground truth data from a CSV file based on the column `on_column`.
        """
        ground_truth = pd.read_csv(ground_truth_path)
        ground_truth = ground_truth.loc[:, ~ground_truth.columns.duplicated()].copy()
        return data.merge(ground_truth, on=on_column, how="left")