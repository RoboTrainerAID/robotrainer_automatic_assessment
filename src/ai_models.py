from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical


class Model:
    def __init__(self, model_type, **kwargs):
        self.model_type = model_type
        self.model = self._initialize_model(model_type, **kwargs)

    @staticmethod
    def optimize_model(model_type, X_train, y_train):
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

        else:
            raise ValueError(f"BayesOpt not yet implemented for {model_type}")

        opt = BayesSearchCV(
            estimator=base_model,
            search_spaces=search_spaces,
            n_iter=25,
            scoring="neg_mean_squared_error",
            cv=3,
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
