def run_real_riboflavin_study(
    X_clean,
    Theta_reference,
    miss_grid=[0.1, 0.2, 0.3],
    contam_grid=[0.05, 0.10, 0.15],
    n_rep=20
):
    """
    Evaluate methods on Riboflavin with controlled perturbations.
    Metrics are computed against a surrogate reference graph estimated
    from the original unperturbed data.
    """
    results = []

    p = X_clean.shape[1]
    tri = np.triu_indices(p, k=1)

    ref_edges = (np.abs(Theta_reference[tri]) > 1e-4).astype(int)

    for miss in miss_grid:
        for contam in contam_grid:
            print(f"Riboflavin: Missing={miss}, Contamination={contam}")

            for r in range(n_rep):
                seed = 12000 + r

                X_obs = add_missing_and_contamination_real(
                    X_clean,
                    miss_rate=miss,
                    contam_rate=contam,
                    contam_sd=8.0,
                    seed=seed
                )

                # Mean + Glasso
                
                X_mean = SimpleImputer(strategy="mean").fit_transform(X_obs)
                t_mean, _ = safe_glasso_fit(
                    X_mean,
                    alpha=0.1,
                    use_cv=False,
                    random_state=seed + 10
                )

                
                # Winsor + Glasso
                
                X_win = winsorize_data(X_mean)
                t_win, _ = safe_glasso_fit(
                    X_win,
                    alpha=0.1,
                    use_cv=False,
                    random_state=seed + 20
                )

               
                # Mask + Mean
                
                X_mom = mask_outliers_adaptive(X_obs)
                X_mom = SimpleImputer(strategy="mean").fit_transform(X_mom)
                t_mom, _ = safe_glasso_fit(
                    X_mom,
                    alpha=0.1,
                    use_cv=False,
                    random_state=seed + 30
                )

                
                # Mask + CV
                
                X_mcv_raw = mask_outliers_adaptive(X_obs)
                X_mcv_imp = robust_impute(X_mcv_raw, miss_rate=miss)

                use_cv_flag = (miss <= 0.2)
                t_mcv, alpha_used = safe_glasso_fit(
                    X_mcv_imp,
                    alpha=0.15,
                    use_cv=use_cv_flag,
                    random_state=seed + 40
                )

                # DDC-like + Graphical Lasso
                t_ddc = ddc_like_impute_glasso(
                    X_obs,
                    miss_rate=miss,
                    alpha=0.1,
                    random_state=seed + 45
                )
                
                # Improved Robust
               
                stab_threshold = 0.8 if miss <= 0.2 else 0.6
                n_subsets = 10 if miss <= 0.2 else 6

                stable_mask = fit_stability_selection(
                    X_mcv_imp,
                    alpha=alpha_used,
                    n_subsets=n_subsets,
                    threshold=stab_threshold,
                    random_state=seed + 50
                )

                t_robust = t_mcv * stable_mask

                methods = {
                    "Improved_Robust": t_robust,
                    "DDC_like+Glasso": t_ddc,
                    "Mask+CV": t_mcv,
                    "Mask+Mean": t_mom,
                    "Mean+Glasso": t_mean,
                    "Winsor+Glasso": t_win
                }

                for name, th in methods.items():
                    est_edges = (np.abs(th[tri]) > 1e-4).astype(int)

                    f1 = f1_score(ref_edges, est_edges, zero_division=0)
                    err = norm(th - Theta_reference, ord="fro")
                    n_edges = np.sum(est_edges)

                    results.append({
                        "missing": miss,
                        "contam": contam,
                        "method": name,
                        "F1_to_ref": f1,
                        "Fro_Error_to_ref": err,
                        "Edges": n_edges
                    })

    df_results = pd.DataFrame(results)

    summary = (
        df_results
        .groupby(["missing", "contam", "method"])[["F1_to_ref", "Fro_Error_to_ref", "Edges"]]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = [
        f"{c[0]}_{c[1]}" if c[1] else c[0]
        for c in summary.columns.to_flat_index()
    ]

    return df_results, summary
