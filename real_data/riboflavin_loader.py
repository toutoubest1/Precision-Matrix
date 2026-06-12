import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import norm

from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

# Load Riboflavin dataset

def load_riboflavin_data():
    """
    Try several possible OpenML names for the Riboflavin dataset.
    If none works, raise an error and let the user provide a local file.
    """
    candidate_names = [
        "riboflavin",
        "Riboflavin",
        "RiboflavinData",
        "riboflavin_data"
    ]

    last_error = None

    for name in candidate_names:
        try:
            ds = fetch_openml(name=name, version="active", as_frame=True)
            print(f"Loaded dataset from OpenML with name: {name}")
            X_df = ds.data.copy()
            y = ds.target.copy() if ds.target is not None else None
            return X_df, y
        except Exception as e:
            last_error = e

    raise RuntimeError(
        "Could not fetch Riboflavin from OpenML. "
        "Please either:\n"
        "1) check the exact OpenML dataset name/version, or\n"
        "2) download the dataset locally and load it from a CSV file.\n"
        f"Last error: {last_error}"
    )

#  Optional local loader 

def load_riboflavin_from_csv(csv_path, target_col=None):
    """
    Load Riboflavin from a local CSV file.
    If target_col is given, remove it from the feature matrix.
    """
    df = pd.read_csv(csv_path)

    if target_col is not None and target_col in df.columns:
        y = df[target_col].copy()
        X_df = df.drop(columns=[target_col]).copy()
    else:
        y = None
        X_df = df.copy()

    return X_df, y

# Preprocess: keep top variable genes

def preprocess_riboflavin(X_df, top_k=100):
    """
    Keep numeric columns only, remove columns with all missing values,
    and select the top_k most variable features.
    """
    X_num = X_df.select_dtypes(include=[np.number]).copy()

    # Drop columns that are entirely missing
    X_num = X_num.dropna(axis=1, how="all")

    # Median fill only for variance ranking
    X_tmp = X_num.copy()
    medians = X_tmp.median(axis=0)
    X_tmp = X_tmp.fillna(medians)

    # Select top variable genes/features
    variances = X_tmp.var(axis=0)
    top_features = variances.sort_values(ascending=False).head(top_k).index.tolist()
    X_top = X_tmp[top_features].copy()

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_top.values)

    return X_scaled, top_features
