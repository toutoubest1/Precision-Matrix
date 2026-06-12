# Controlled perturbation function

def add_missing_and_contamination_real(X, miss_rate=0.1, contam_rate=0.05,
                                       contam_sd=8.0, seed=123):
    """
    Add entrywise contamination and missingness to a real dataset.
    """
    rng = np.random.default_rng(seed)
    X_obs = X.copy()

    # Add contamination
    contam_mask = rng.uniform(size=X_obs.shape) < contam_rate
    X_obs[contam_mask] += rng.normal(0, contam_sd, size=np.sum(contam_mask))

    # Add missing values
    miss_mask = rng.uniform(size=X_obs.shape) < miss_rate
    X_obs[miss_mask] = np.nan

    return X_obs
