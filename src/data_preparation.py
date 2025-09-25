import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer



def data_normalization(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = (df[numeric_columns] - df[numeric_columns].min()) / (df[numeric_columns].max() - df[numeric_columns].min())
    return df


def apply_pca(X, n_components=0.95):
    """
    Apply PCA to NumPy array X
    """
    X = np.array(X)

    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_imputed)

    X_pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])

    #Scree Plot to visualize explained variance by each principal component 
    plt.plot(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_.cumsum(), marker='o')
    plt.xlabel("Anzahl Komponenten")
    plt.ylabel("Kumulative erklärte Varianz")
    plt.title("Scree Plot")
    plt.show()

    return X_pca_df, pca.explained_variance_ratio_


def prepare_input_blocks(df, pipeline_config, mode="row"):
    """
    Prepares X and y for the model.
    
    mode:
        "row"        -> each row = 1 sample
        "user"       -> all rows per user = 1 sample
        "user_path"  -> all rows per user+path = 1 sample
    """
    topics = pipeline_config.topics

    if mode == "row":
        X = df.drop(columns=topics + ["user", "path"], errors="ignore")
        y = df[topics].values
        return X.values, y

    elif mode == "user":
        groups = df.groupby("user")
    elif mode == "user_path":
        groups = df.groupby(["user", "path"])
    else:
        raise ValueError(f"Unknown mode: {mode}")

    X_blocks = []
    y_blocks = []

    for _, group in groups:
        feature_cols = [c for c in group.columns if c not in topics + ["user", "path"]]
        features = group[feature_cols].values.flatten() 
        X_blocks.append(features)

        y_block = group[topics].iloc[0].values
        y_blocks.append(y_block)

    max_len = max(len(x) for x in X_blocks)
    X_padded = np.array([np.pad(x, (0, max_len - len(x)), 'constant') for x in X_blocks])

    return X_padded, np.array(y_blocks)


def ensure_numeric(X):
    if not np.issubdtype(X.dtype, np.number):
        print("X contains non-numeric values. Converting to numeric values...")
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        X = pd.get_dummies(X, drop_first=True)
        X = X.values  

    X = X.astype(float)
    return X
    