import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from config import PipelineConfig
from data_preparation import (data_normalization, apply_pca)
from ai_models import Model


pipeline_config = PipelineConfig('opt/src/config.ini')
print("pipeline_config.topics:", pipeline_config.topics)

print ("DataFrame is loading")
df = pd.read_csv(pipeline_config.path_to_data)
print("DataFrame loaded successfully")
print(df.head())

df = data_normalization(df)
print("DataFrame normalized successfully")
print(df.head())

topics_to_exclude = [col for col in pipeline_config.topics if col in df.columns]
df_for_pca = df.drop(columns=topics_to_exclude)

print("Data prepared for PCA successfully")
print(df_for_pca.head())

df_pca, variance_ratio = apply_pca(df_for_pca, n_components=0.95)
print("PCA applied successfully")
print(df_pca.head())

X = df_pca


y = df[pipeline_config.topics[0]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_model = Model.optimize_model("svr", X_train, y_train)