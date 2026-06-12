def safe_glasso_fit(X, alpha=0.1, use_cv=False, random_state=123):
    """
    Safe GraphicalLasso / GraphicalLassoCV fit with fallback.
    Returns precision matrix on original scale.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Tiny jitter for numerical stability
    rng = np.random.default_rng(random_state)
    X_scaled = X_scaled + 1e-6 * rng.normal(size=X_scaled.shape)

    try:
        if use_cv:
            model = GraphicalLassoCV(
                cv=5,
                max_iter=1000,
                tol=1e-3,
                enet_tol=1e-3
            )
        else:
            model = GraphicalLasso(
                alpha=alpha,
                max_iter=1000,
                tol=1e-3
            )

        model.fit(X_scaled)
        precision_scaled = model.precision_
        alpha_used = getattr(model, "alpha_", alpha)

    except Exception:
        try:
            # Fallback 1: stronger regularization
            alpha2 = max(alpha, 0.2)
            model = GraphicalLasso(alpha=alpha2, max_iter=2000, tol=1e-3)
            model.fit(X_scaled)
            precision_scaled = model.precision_
            alpha_used = alpha2
        except Exception:
            # Fallback 2: ridge inverse covariance
            S = np.cov(X_scaled, rowvar=False)
            S = S + 0.2 * np.eye(S.shape[0])
            precision_scaled = pinv(S)
            alpha_used = 0.2

    std_dev = np.sqrt(scaler.var_)
    std_dev = np.where(std_dev < 1e-8, 1e-8, std_dev)
    precision = precision_scaled / np.outer(std_dev, std_dev)

    return precision, alpha_used
