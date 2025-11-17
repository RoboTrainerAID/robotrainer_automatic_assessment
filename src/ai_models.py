import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from flaml import automl

from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import clone
from sklearn.model_selection import KFold
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical


class Model:
    def __init__(self, model_type, k_folds, **kwargs):
        self.model_type = model_type
        self.k_folds = k_folds
        self.model = self._initialize_model(model_type, **kwargs)


    @staticmethod
    def _initialize_model(model_type, **kwargs):
        if model_type == "svr":
            base_model = SVR(**kwargs)
            return MultiOutputRegressor(base_model)
        elif model_type == "random_forest":
            return RandomForestRegressor(**kwargs)
        elif model_type == "mlp":
            return MLPRegressor(max_iter=500, early_stopping=True, n_iter_no_change=5, **kwargs)
        elif model_type == "automl":
            return None
        else:
            raise ValueError(f"Unknown model type: {model_type}")


    @staticmethod
    def optimize_model(model_type, k_folds, X_train, y_train):
        if model_type == "svr":
            search_spaces = {
                "estimator__C": Real(1e-3, 1e3, prior="log-uniform"),
                "estimator__gamma": Real(1e-4, 1e0, prior="log-uniform"),
                "estimator__kernel": Categorical(["linear", "rbf", "poly"])
            }
            base_model = MultiOutputRegressor(SVR())

        elif model_type == "random_forest":
            search_spaces = {
                "n_estimators": Integer(100, 400),
                "max_depth": Integer(5, 30),
                "min_samples_split": Integer(2, 10),
                "min_samples_leaf": Integer(1, 5),
                "max_features": Categorical(["sqrt", "log2", None])
            }
            base_model = RandomForestRegressor()

        elif model_type == "mlp":
            search_spaces = {
                "hidden_layer_sizes": Categorical(
                    ["50", "100", "50-50", "100-50", "100-100"]
                ),
                "activation": Categorical(["relu", "tanh"]),
                "solver": Categorical(["adam", "lbfgs"]),
                "alpha": Real(1e-5, 1e-1, prior="log-uniform"),
                "learning_rate_init": Real(1e-4, 1e-2, prior="log-uniform")
            }
            base_model = TupleMLP(max_iter=500)

        elif model_type == "automl":
            print(f"AutoML input shapes: X_train: {X_train.shape}, y_train: {y_train.shape}")
            
            # Multi-output: train AutoML separately for each target variable
            if y_train.ndim > 1 and y_train.shape[1] > 1:
                print(f"Multi-output AutoML: Training {y_train.shape[1]} separate AutoML models...")
                
                # list to store all trained AutoML models
                automl_models = []
                
                for i in range(y_train.shape[1]):
                    print(f"Training AutoML for target variable {i+1}/{y_train.shape[1]}...")
                    
                    automl_instance = automl.AutoML()
                    automl_instance.fit(
                        X_train, 
                        y_train[:, i], 
                        task="regression", 
                        metric="rmse", 
                        time_budget=1400, 
                        n_splits=3, 
                        early_stop=True, 
                        ensemble=True, 
                        verbose=False
                    )
                    
                    automl_models.append(automl_instance)
                    print(f"AutoML {i+1} finished. Best model: {automl_instance.best_estimator}")
                
                # Wrapper class for Multi-Output AutoML
                class MultiOutputAutoML:
                    def __init__(self, models):
                        self.models = models
                    
                    def predict(self, X):
                        predictions = []
                        for model in self.models:
                            pred = model.predict(X)
                            predictions.append(pred)
                        return np.column_stack(predictions)
                    
                    def fit(self, X, y):
                        pass
                    
                    def get_params(self, deep=True):
                        """
                        Returns the parameters of all AutoML models.
                        """
                        params = {}
                        for i, model in enumerate(self.models):
                            try:
                                model_params = model.best_config if hasattr(model, 'best_config') else {}
                                params[f'automl_model_{i}'] = model_params
                            except:
                                params[f'automl_model_{i}'] = str(model.best_estimator) if hasattr(model, 'best_estimator') else 'unknown'
                        
                        # Summary info
                        params['num_models'] = len(self.models)
                        params['model_type'] = 'multi_output_automl'
                        
                        return params
                    
                    def set_params(self, **params):
                        """
                        Placeholder for sklearn compatibility
                        """
                        return self
                
                multi_automl = MultiOutputAutoML(automl_models)
                print("Multi-output AutoML finished training!")
                
                # Create model instance without problematic initialization
                model_instance = Model.__new__(Model)
                model_instance.model_type = model_type
                model_instance.k_folds = k_folds
                model_instance.model = multi_automl
                
                return model_instance
            
            else:
                # Single output: Normal AutoML
                y_flat = y_train.ravel() if y_train.ndim > 1 else y_train
                print(f"Single-output AutoML: X_train: {X_train.shape}, y_train: {y_flat.shape}")
                
                automl_instance = automl.AutoML()
                automl_instance.fit(X_train, y_flat, task="regression", metric="rmse", time_budget=600, n_splits=3, early_stop=True, ensemble=True, verbose=True)
                print("AutoML finished training!")
                print("Best model:", automl_instance.best_estimator)
                
                # Create model instance without problematic initialization
                model_instance = Model.__new__(Model)
                model_instance.model_type = model_type
                model_instance.k_folds = k_folds
                model_instance.model = automl_instance
                
                return model_instance

        else:
            raise ValueError(f"BayesOpt not yet implemented for {model_type}")

        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        opt = BayesSearchCV(
            estimator=base_model,
            search_spaces=search_spaces,
            n_iter=200,
            scoring="neg_mean_squared_error",
            cv=kfold,
            n_jobs=-1,
            verbose=0
        )

        print("Search spaces:", search_spaces)
        opt.fit(X_train, y_train)

        best_params = opt.best_params_
        print("Best hyperparameters:", best_params)
        print("Best MSE:", -opt.best_score_)
        print("(Note: Lower MSE indicates a better model)")

        # return trained model with best parameters
        best_model = clone(base_model).set_params(**best_params)
        best_model.fit(X_train, y_train)

        clean_params = {
            k.replace("estimator__", "") if k.startswith("estimator__") else k: v
            for k, v in best_params.items()
        }

        return Model(model_type, k_folds, **clean_params)._replace_model(best_model)


    # Helper function to set the trained model
    def _replace_model(self, trained_model):
        self.model = trained_model
        return self


    # Training & Prediction
    def train(self, X, y):
        self.model.fit(X, y)


    def predict(self, X):
        return self.model.predict(X)


    # Evaluation
    def evaluate(self, X, y):
        """
        Returns MSE, RMSE and R².
        If y contains multiple topics (multi-output), scores are calculated per topic
        and an overall score averaged across all topics is returned as well.
        """
        predictions = self.predict(X)

        # Single target case
        if len(y.shape) == 1 or y.shape[1] == 1:
            mse = mean_squared_error(y, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y, predictions)
            return {"Overall": {"MSE": mse, "RMSE": rmse, "R2": r2}}

        # Multi-Target case
        scores = {}
        mses, rmses, r2s = [], [], []

        for i in range(y.shape[1]):
            mse = mean_squared_error(y[:, i], predictions[:, i])
            rmse = np.sqrt(mse)
            r2 = r2_score(y[:, i], predictions[:, i])
            scores[f"Topic_{i}"] = {"MSE": mse, "RMSE": rmse, "R2": r2}
            mses.append(mse)
            rmses.append(rmse)
            r2s.append(r2)

        # Calculate total scores
        scores["Overall"] = {
            "MSE": np.mean(mses),
            "RMSE": np.mean(rmses),
            "R2": np.mean(r2s)
        }
        return scores


    # Save results to CSV
    @staticmethod
    def save_results_to_csv(
        scores,
        hyperparams,
        model_name,
        dataset_name,
        filename=None,
        hyperparams_as_json=False
    ):
        """
        Saves scores, hyperparameters and model information to a CSV.
        - Each model type gets its own CSV file.
        - Files are stored by default in opt/src/data/.
        - Appends new results if the file already exists.
        - If new columns appear, the file is rewritten with the extended header.
        """
        

        # Target directory
        save_dir = os.path.join("opt", "src", "data")
        os.makedirs(save_dir, exist_ok=True)

        if filename is None:
            filename = os.path.join(save_dir, f"{model_name}_results.csv")

        rows = []
        for key, vals in scores.items():
            row = {
                "Dataset": dataset_name,
                "Model": model_name,
                "Topic": key
            }
            row.update(vals)

            if hyperparams_as_json:
                try:
                    row["Hyperparameters"] = json.dumps(hyperparams, default=str)
                except Exception:
                    row["Hyperparameters"] = str(hyperparams)
            else:
                for k, v in (hyperparams or {}).items():
                    if isinstance(v, (str, int, float, bool)) or v is None:
                        row[k] = v
                    else:
                        row[k] = str(v)

            rows.append(row)

        df_new = pd.DataFrame(rows)

        front = ["Dataset", "Model", "Topic"]
        front_present = [c for c in front if c in df_new.columns]
        rest = [c for c in df_new.columns if c not in front_present]
        df_new = df_new[front_present + rest]

        if os.path.exists(filename):
            existing_cols = pd.read_csv(filename, nrows=0).columns.tolist()

            if set(df_new.columns).issubset(existing_cols):
                df_new.to_csv(filename, mode="a", header=False, index=False)
            else:
                df_existing = pd.read_csv(filename)
                union = list(dict.fromkeys(existing_cols + df_new.columns.tolist()))
                df_existing = df_existing.reindex(columns=union)
                df_new = df_new.reindex(columns=union)
                df_concat = pd.concat([df_existing, df_new], ignore_index=True)
                df_concat.to_csv(filename, index=False)
        else:
            df_new.to_csv(filename, index=False)

        print(f"Ergebnisse gespeichert in {filename}")
        return df_new


    def plot_prediction_curve(self, X, y, topic_index=0, num_points=300, save_path=None):
        """
        Displays the prediction vs. true value curve for an MLP or other models.
        Supports multi-output via topic_index.

        Args:
            X: Input data (numpy array or pandas DataFrame)
            y: Target values (numpy array or pandas DataFrame)
            topic_index: Index of the target to plot when y is multi-output
            num_points: Number of points to display in the plot (e.g., 300)
            save_path: Optional path to save the plot image
        """
        y_pred = self.predict(X)
 
        # Multi-Output handling
        if len(y.shape) > 1 and y.shape[1] > 1:
            y_true = y[:, topic_index]
            y_pred = y_pred[:, topic_index]
            title = f"Prediction Curve (Topic {topic_index})"
        else:
            y_true = y.ravel()
            y_pred = y_pred.ravel()
            title = "Prediction Curve"
 
        if len(y_true) > num_points:
            y_true = y_true[:num_points]
            y_pred = y_pred[:num_points]
 
        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(y_true, label="True", linewidth=2)
        plt.plot(y_pred, label="Predicted", linestyle="--", linewidth=2)
        plt.title(title)
        plt.xlabel("Sample Index")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
 
        if save_path:
            plt.savefig(save_path)
            print(f"Diagramm gespeichert unter: {save_path}")
        else:
            plt.show()


    def plot_all_topics(self, X, y, save_dir="opt/src/plots", num_points=300):
        """
        Automatically create True-vs-Predicted plots (time-series curve and scatter)
        for all target variables (topics) and save each figure as a PDF.

        Args:
            X: Input data (numpy array or pandas DataFrame)
            y: Target values (numpy array or pandas DataFrame)
            save_dir: Directory where plots will be saved
            num_points: Maximum number of points to display in each plot (for clarity)
        """
        os.makedirs(save_dir, exist_ok=True)
 
        y_pred = self.predict(X)
 
        if len(y.shape) == 1 or y.shape[1] == 1:
            num_topics = 1
        else:
            num_topics = y.shape[1]
 
        print(f"Creating plots for {num_topics} topic(s)...")
 
        for topic_idx in range(num_topics):
            if num_topics == 1:
                y_true = y.ravel()
                y_pred_topic = y_pred.ravel()
            else:
                y_true = y[:, topic_idx]
                y_pred_topic = y_pred[:, topic_idx]
 
            if len(y_true) > num_points:
                y_true = y_true[:num_points]
                y_pred_topic = y_pred_topic[:num_points]
 
            # Time-series: True vs Predicted over sample index (no grid)
            plt.figure(figsize=(8.27, 4))
            plt.plot(y_true, label="True", linewidth=2, color="#9B9B9B")
            plt.plot(y_pred_topic, label="Predicted", linestyle="--", linewidth=2, color="#c00000")
            plt.xlabel("Sample Index", fontsize=12)
            plt.ylabel("Value", fontsize=12)
            plt.legend(fontsize=12)
            plt.tight_layout()
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            curve_path = os.path.join(save_dir, f"topic_{topic_idx}_curve.pdf")
            plt.savefig(curve_path)
            plt.close()

            # Scatter: True vs Predicted values
            plt.figure(figsize=(8.27, 5))
            plt.scatter(y_true, y_pred_topic, alpha=0.6, color="#c00000")
            plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2)
            plt.xlabel("True Values", fontsize=12)
            plt.ylabel("Predicted Values", fontsize=12)
            plt.title(f"Predicted vs True - Topic {topic_idx}", fontsize=12)
            plt.grid(True)
            plt.tight_layout()
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            scatter_path = os.path.join(save_dir, f"topic_{topic_idx}_scatter.pdf")
            plt.savefig(scatter_path)
            plt.close()
 
            print(f"Topic {topic_idx}: Plots saved to:")
            print(f"   {curve_path}")
            print(f"   {scatter_path}")

        print("All topic plots created successfully!")


class TupleMLP(MLPRegressor):
    def set_params(self, **params):
        if "hidden_layer_sizes" in params and isinstance(params["hidden_layer_sizes"], str):
            params["hidden_layer_sizes"] = tuple(
                map(int, params["hidden_layer_sizes"].split("-"))
            )
        return super().set_params(**params)