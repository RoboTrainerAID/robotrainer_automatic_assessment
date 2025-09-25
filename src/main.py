import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split

from config import PipelineConfig
from data_preparation import (data_normalization, apply_pca, prepare_input_blocks, ensure_numeric)
from ai_models import Model


def run_pipeline():
    pipeline_config = PipelineConfig('opt/src/config.ini')
    print("pipeline_config.topics:", pipeline_config.topics)

    print("DataFrame is loading")
    df = pd.read_csv(pipeline_config.path_to_data)
    print("DataFrame loaded successfully")
    print(df.head())

    df = pd.get_dummies(df, drop_first=False) 
    
    X, y = prepare_input_blocks(df, pipeline_config, mode=pipeline_config.mode)
    print(f"Prepared input blocks, shape X: {X.shape}, y: {y.shape}")

    X = ensure_numeric(X)
   
    df = pd.DataFrame(X)

    X_norm = data_normalization(df).values
    print("Data normalized successfully")

    if pipeline_config.use_pca:
        X_pca, variance_ratio = apply_pca(X_norm, n_components=pipeline_config.pca_variance_threshold)
        print("PCA applied successfully")
        X_final = X_pca
    else:
        X_final = X_norm
        print("PCA not applied, using original data")

    best_model = Model.optimize_model(
        model_type=pipeline_config.model,
        k_folds=pipeline_config.kfolds,
        X_train=X_final,
        y_train=y
    )
    
    best_model = Model.optimize_model(pipeline_config.model, pipeline_config.kfolds, X, y)
    return best_model


if __name__ == "__main__":
    run_pipeline()
