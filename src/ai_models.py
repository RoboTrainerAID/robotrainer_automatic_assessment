import autosklearn.regression

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import clone
from sklearn.model_selection import cross_val_score, KFold
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical



class Model:
    def __init__(self, model_type, k_folds, **kwargs):
        self.model_type = model_type
        self.k_folds = k_folds
        self.model = self._initialize_model(model_type, **kwargs)
        

    @staticmethod
    def _initialize_model(model_type, k_folds, **kwargs):
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
                "hidden_layer_sizes": Categorical(["50", "100", "50-50", "100-50", "100-100"]),
                "activation": Categorical(["relu", "tanh"]),
                "solver": Categorical(["adam", "lbfgs"]),
                "alpha": Real(1e-5, 1e-1, prior="log-uniform"),
                "learning_rate_init": Real(1e-4, 1e-2, prior="log-uniform")
            }
            base_model = TupleMLP(max_iter=1000)

        elif model_type == "automl":
            automl = autosklearn.regression.AutoSklearnRegressor(
                time_left_for_this_task=120,   # 10 Minuten Gesamtzeit
                per_run_time_limit=60,         # max. 1 Minute pro Modell
                metric=autosklearn.metrics.mean_squared_error,
                n_jobs=-1
            )
            automl.fit(X_train, y_train)
            print("AutoML fertig trainiert")
            print(automl.leaderboard())
            return automl
        
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

        return best_params

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        return mean_squared_error(y, predictions)
    

class TupleMLP(MLPRegressor):
    def set_params(self, **params):
        if "hidden_layer_sizes" in params and isinstance(params["hidden_layer_sizes"], str):
            params["hidden_layer_sizes"] = tuple(map(int, params["hidden_layer_sizes"].split("-")))
        return super().set_params(**params)