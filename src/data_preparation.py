import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer


def data_normalization(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = (df[numeric_columns] - df[numeric_columns].min()) / (df[numeric_columns].max() - df[numeric_columns].min())
    return df

def apply_pca(df, n_components=0.95):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_columns]

    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    pca = PCA(n_components=n_components)
    df_pca = pca.fit_transform(X_imputed)

    df_pca = pd.DataFrame(df_pca, columns=[f'PC{i+1}' for i in range(df_pca.shape[1])])

    # Scree Plot to visualize explained variance by each principal component 
    plt.plot(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_.cumsum(), marker='o')
    plt.xlabel("Anzahl Komponenten")
    plt.ylabel("Kumulative erklärte Varianz")
    plt.title("Scree Plot")
    plt.show()

    return df_pca, pca.explained_variance_ratio_




    