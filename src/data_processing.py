import pandas as pd

class DataProcessor:

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

    def limit_data_by_time(self, data, time_column, path_column, time_length, duration_column="total_duration"):
        """
        Begrenzt die Daten für jede 'path' auf eine bestimmte Zeitlänge
        relativ zur total_duration.
        """
        def clip_group(group):
            max_time = min(group[duration_column].iloc[0], time_length)
            return group[group[time_column] <= max_time]

        # concat statt apply -> vermeidet Index-Duplikate
        groups = [clip_group(g) for _, g in data.groupby(path_column)]
        return pd.concat(groups, ignore_index=True)

    def add_ground_truth(self, data, ground_truth_path, on_column="user"):
        """
        Adds the ground truth data from a CSV file based on the column `on_column`.
        """
        ground_truth = pd.read_csv(ground_truth_path)
        ground_truth = ground_truth.loc[:, ~ground_truth.columns.duplicated()].copy()
        return data.merge(ground_truth, on=on_column, how="left")