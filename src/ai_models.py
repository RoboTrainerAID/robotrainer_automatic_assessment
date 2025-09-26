import os
import pandas as pd
import numpy as np
import autosklearn.regression

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

    # --------------------------------------------------------
    # Initialisiere Basismodelle
    # --------------------------------------------------------
    @staticmethod
    def _initialize_model(model_type, **kwargs):
        if model_type == "svr":
            return SVR(**kwargs)
        elif model_type == "random_forest":
            return RandomForestRegressor(**kwargs)
        elif model_type == "mlp":
            return MLPRegressor(max_iter=1000, **kwargs)
        elif model_type == "automl":
            return autosklearn.regression.AutoSklearnRegressor(
                time_left_for_this_task=600,
                per_run_time_limit=60,
                metric=autosklearn.metrics.mean_squared_error,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    # --------------------------------------------------------
    # Bayes. Optimierung & Training
    # --------------------------------------------------------
    @staticmethod
    def optimize_model(model_type, k_folds, X_train, y_train):
        if model_type == "svr":
            search_spaces = {
                "C": Real(1e-3, 1e3, prior="log-uniform"),
                "gamma": Real(1e-4, 1e0, prior="log-uniform"),
                "kernel": Categorical(["linear", "rbf", "poly"])
            }
            base_model = SVR()

        elif model_type == "random_forest":
            search_spaces = {
                "n_estimators": Integer(50, 300),
                "max_depth": Integer(3, 30),
                "min_samples_split": Integer(2, 10)
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
            base_model = TupleMLP(max_iter=1000)

        elif model_type == "automl":
            automl = autosklearn.regression.AutoSklearnRegressor(
                time_left_for_this_task=120,
                per_run_time_limit=60,
                metric=autosklearn.metrics.mean_squared_error,
                n_jobs=-1
            )
            automl.fit(X_train, y_train)
            print("AutoML fertig trainiert")
            print(automl.leaderboard())
            return Model(model_type, k_folds, **{})._replace_model(automl)

        else:
            raise ValueError(f"BayesOpt not yet implemented for {model_type}")

        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        opt = BayesSearchCV(
            estimator=base_model,
            search_spaces=search_spaces,
            n_iter=25,
            scoring="neg_mean_squared_error",
            cv=kfold,
            n_jobs=-1,
            verbose=0
        )

        print("Search spaces:", search_spaces)
        opt.fit(X_train, y_train)

        best_params = opt.best_params_
        print("Beste Hyperparameter:", best_params)
        print("Bestes Score:", -opt.best_score_)

        # trainiertes Modell mit besten Parametern zurückgeben
        best_model = clone(base_model).set_params(**best_params)
        best_model.fit(X_train, y_train)

        return Model(model_type, k_folds, **best_params)._replace_model(best_model)

    # --------------------------------------------------------
    # Hilfsfunktion zum Setzen des trainierten Modells
    # --------------------------------------------------------
    def _replace_model(self, trained_model):
        self.model = trained_model
        return self

    # --------------------------------------------------------
    # Training & Prediction
    # --------------------------------------------------------
    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------
    def evaluate(self, X, y):
        """
        Gibt MSE, RMSE und R² zurück.
        Falls y mehrere Topics hat, werden die Scores pro Topic berechnet
        und zusätzlich ein Gesamtscore über alle Topics hinweg.
        """
        predictions = self.predict(X)

        # Einzel-Target Fall
        if len(y.shape) == 1 or y.shape[1] == 1:
            mse = mean_squared_error(y, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y, predictions)
            return {"Overall": {"MSE": mse, "RMSE": rmse, "R2": r2}}

        # Multi-Target Fall
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

        # Gesamtscores berechnen
        scores["Overall"] = {
            "MSE": np.mean(mses),
            "RMSE": np.mean(rmses),
            "R2": np.mean(r2s)
        }
        return scores

    # --------------------------------------------------------
    # Ergebnisse speichern
    # --------------------------------------------------------
    @staticmethod
    def save_results_to_csv(scores, hyperparams, model_name, dataset_name, filename="results.csv"):
        """
        Speichert Scores + Hyperparameter + Modell in eine CSV.
        Hängt neue Ergebnisse an, falls die Datei bereits existiert.
        """
        rows = []
        for key, vals in scores.items():
            row = {
                "Dataset": dataset_name,
                "Model": model_name,
                "Topic": key
            }
            row.update(vals)          # Scores hinzufügen
            row.update(hyperparams)   # Hyperparameter hinzufügen
            rows.append(row)

        df = pd.DataFrame(rows)

        if os.path.exists(filename):
            df.to_csv(filename, mode="a", header=False, index=False)
        else:
            df.to_csv(filename, index=False)

        print(f"Ergebnisse gespeichert in {filename}")
        return df


# --------------------------------------------------------
# Hilfsklasse für flexible MLP-Struktur (z.B. "50-100")
# --------------------------------------------------------
class TupleMLP(MLPRegressor):
    def set_params(self, **params):
        if "hidden_layer_sizes" in params and isinstance(params["hidden_layer_sizes"], str):
            params["hidden_layer_sizes"] = tuple(
                map(int, params["hidden_layer_sizes"].split("-"))
            )
        return super().set_params(**params)
