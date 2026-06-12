import numpy as np
import pandas as pd
import warnings

from sklearn.covariance import GraphicalLasso, GraphicalLassoCV
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.exceptions import ConvergenceWarning

from numpy.linalg import inv, norm, eigvalsh, pinv


# Global settings

np.random.seed(123)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)



# Data Generation


def make_sparse_precision(p=50, seed=123):
    rng = np.random.default_rng(seed)
    Theta = np.zeros((p, p))

    for b in range(p // 10):
        start, end = b * 10, (b + 1) * 10
        for i in range(start, end):
            for j in range(i + 1, end):
                if abs(i - j) == 1:
                    Theta[i, j] = Theta[j, i] = 0.3

    for i in range(p):
        for j in range(i + 1, p):
            if Theta[i, j] == 0 and rng.uniform() < 0.03:
                Theta[i, j] = Theta[j, i] = rng.choice([-1, 1]) * 0.15

    for i in range(p):
        Theta[i, i] = np.sum(np.abs(Theta[i, :])) + 0.5

    # Extra safeguard
    min_eval = np.min(eigvalsh(Theta))
    if min_eval <= 0:
        Theta += np.eye(p) * (abs(min_eval) + 0.5)

    return Theta


def generate_corrupted_data(n, Theta_true, miss_rate, contam_rate, seed):
    rng = np.random.default_rng(seed)
    p = Theta_true.shape[0]
    Sigma_true = inv(Theta_true)

    X_clean = rng.multivariate_normal(np.zeros(p), Sigma_true, size=n)

    X_obs = X_clean.copy()
    contam_mask = rng.uniform(size=X_obs.shape) < contam_rate
    X_obs[contam_mask] += rng.normal(0, 8.0, size=np.sum(contam_mask))

    miss_mask = rng.uniform(size=X_obs.shape) < miss_rate
    X_obs[miss_mask] = np.nan

    return X_clean, X_obs
