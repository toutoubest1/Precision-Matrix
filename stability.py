def fit_stability_selection(X, alpha, n_subsets=10, threshold=0.8, random_state=123):
    """
    Stability selection with safe GraphicalLasso.
    """
    n, p = X.shape
    edge_counts = np.zeros((p, p))
    sub_n = int(0.8 * n)

    rng = np.random.default_rng(random_state)

    for b in range(n_subsets):
        idx = rng.choice(n, sub_n, replace=False)
        prec_sub, _ = safe_glasso_fit(X[idx], alpha=alpha, use_cv=False, random_state=random_state + b)
        edge_counts += (np.abs(prec_sub) > 1e-4).astype(int)

    stable_mask = (edge_counts / n_subsets >= threshold).astype(float)
    np.fill_diagonal(stable_mask, 1.0)
    return stable_mask
