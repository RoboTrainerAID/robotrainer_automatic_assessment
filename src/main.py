import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
    print(f"Original DataFrame shape: {df.shape}")
    print(f"Column types:\n{df.dtypes.value_counts()}")
    
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"Categorical columns: {len(categorical_cols)}")
    
    #  Limit get_dummies to avoid matrix explosion
    if len(categorical_cols) > 0:
        # Only for columns with few unique values
        safe_categorical = []
        for col in categorical_cols:
            unique_count = df[col].nunique()
            if unique_count <= 10:  # Threshold can be adjusted
                safe_categorical.append(col)
            else:
                print(f"Warning: Column '{col}' has {unique_count} unique values - skipping")

        if safe_categorical:
            df = pd.get_dummies(df, columns=safe_categorical, drop_first=False)
        else:
            print("No categorical columns found for get_dummies")

    print(f"After get_dummies: {df.shape}")
    
    X, y = prepare_input_blocks(df, pipeline_config, mode=pipeline_config.mode)
    print(f"Prepared input blocks, shape X: {X.shape}, y: {y.shape}")

    X = ensure_numeric(X)
   
    df = pd.DataFrame(X)

    X_norm = data_normalization(df).values
    print("Data normalized successfully")
    print(f"X_norm shape: {X_norm.shape}")

    y_norm = data_normalization(pd.DataFrame(y)).values
    print("Target values normalized successfully")
    print(f"y_norm shape: {y_norm.shape}")

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
        y_train=y_norm
    )
    # Use normalized y-values for evaluation to be consistent
    scores = best_model.evaluate(X_final, y_norm)

    print("Creating prediction plot...")

    # Plot for Topic 0 (in case of multi-output)
    plot_path = os.path.join("opt", "src", "plots")
    os.makedirs(plot_path, exist_ok=True)
    
    plot_file = os.path.join(plot_path, f"{pipeline_config.model}_prediction_curve.png")
    best_model.plot_prediction_curve(X_final, y_norm, topic_index=0, save_path=plot_file)
    dataset_name = os.path.basename(pipeline_config.path_to_data)

    Model.save_results_to_csv(
        scores,
        best_model.model.get_params(),
        pipeline_config.model,
        dataset_name=dataset_name,
        filename=None
    )

    print("\n--- Creating plots for all topics ---")
    plot_dir = os.path.join("opt", "src", "plots")
    best_model.plot_all_topics(X_final, y_norm, save_dir=plot_dir, num_points=300)
    
    return best_model


if __name__ == "__main__":
    run_pipeline()
