#Utility Functions


def mask_outliers_adaptive(X):
    X_masked = X.copy()
    p = X.shape[1]

    med = np.nanmedian(X_masked, axis=0)
    mad = np.nanmedian(np.abs(X_masked - med), axis=0)
    mad = np.where(mad < 1e-6, 1e-6, mad)

    thresh = 3.0 + np.log10(p) if p > 1 else 3.5
    z = np.abs((X_masked - med) / (1.4826 * mad))
    outliers = z > thresh

    X_masked[outliers] = np.nan
    return X_masked


def winsorize_data(X, q=0.05):
    Xw = X.copy()
    for j in range(X.shape[1]):
        low = np.nanquantile(X[:, j], q)
        high = np.nanquantile(X[:, j], 1 - q)
        Xw[:, j] = np.clip(X[:, j], low, high)
    return Xw

def robust_impute(X, miss_rate):
    """
    More stable imputation strategy:
    - low/moderate missing: IterativeImputer
    - high missing: median imputation
    """
    if miss_rate <= 0.2:
        imputer = IterativeImputer(max_iter=50, tol=1e-3, random_state=123)
        return imputer.fit_transform(X)
    else:
        return SimpleImputer(strategy="median").fit_transform(X)


def ddc_like_impute_glasso(X_obs, miss_rate, alpha=0.1, random_state=123):
    """
    DDC-inspired baseline:
    1. Detect cellwise outliers using robust coordinate-wise scores.
    2. Treat detected outliers as missing.
    3. Impute missing and masked entries.
    4. Apply Graphical Lasso.

    """
    X_ddc = mask_outliers_adaptive(X_obs)
    X_ddc_imp = robust_impute(X_ddc, miss_rate=miss_rate)

    theta_ddc, alpha_used = safe_glasso_fit(
        X_ddc_imp,
        alpha=alpha,
        use_cv=False,
        random_state=random_state
    )

    return theta_ddc
