# Simulation Logic

def run_comprehensive_study(n=200, p=50, n_rep=20):
    results = []

    for miss in [0.1, 0.2, 0.3]:
        for contam in [0.05, 0.1, 0.15]:
            print(f"Processing: Missing={miss}, Contamination={contam}")

            for r in range(n_rep):
                seed = 4000 + r
                Theta_true = make_sparse_precision(p, seed=seed)
                X_clean, X_obs = generate_corrupted_data(n, Theta_true, miss, contam, seed)

               
                # Oracle
                
                t_oracle, _ = safe_glasso_fit(
                    X_clean, alpha=0.1, use_cv=False, random_state=seed
                )

                
                # Mean + Glasso
                
                X_mean = SimpleImputer(strategy="mean").fit_transform(X_obs)
                t_mean, _ = safe_glasso_fit(
                    X_mean, alpha=0.1, use_cv=False, random_state=seed + 10
                )

                
                # Winsor + Glasso
                
                X_win = winsorize_data(X_mean)
                t_win, _ = safe_glasso_fit(
                    X_win, alpha=0.1, use_cv=False, random_state=seed + 20
                )

               
                # Mask + Mean
                
                X_mom = mask_outliers_adaptive(X_obs)
                X_mom = SimpleImputer(strategy="mean").fit_transform(X_mom)
                t_mom, _ = safe_glasso_fit(
                    X_mom, alpha=0.1, use_cv=False, random_state=seed + 30
                )

               
                # Mask + CV / robust imputation
                
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
                    "Oracle": t_oracle,
                    "Mean+Glasso": t_mean,
                    "Winsor+Glasso": t_win,
                    "Mask+Mean": t_mom,
                    "Mask+CV": t_mcv,
                    "DDC_like+Glasso": t_ddc,
                    "Improved_Robust": t_robust
                }

                tri = np.triu_indices(p, k=1)

                for name, th in methods.items():
                    true_edges = (np.abs(Theta_true[tri]) > 1e-4).astype(int)
                    est_edges = (np.abs(th[tri]) > 1e-4).astype(int)

                    f1 = f1_score(true_edges, est_edges, zero_division=0)
                    err = norm(th - Theta_true, ord="fro")
                    n_edges = np.sum(est_edges)   # number of selected off-diagonal edges

                    results.append({
                        "missing": miss,
                        "contam": contam,
                        "method": name,
                        "F1": f1,
                        "Error": err,
                        "Edges": n_edges
                    })

    df_results = pd.DataFrame(results)

    summary = (
        df_results
        .groupby(["missing", "contam", "method"])[["F1", "Error", "Edges"]]
        .agg(["mean", "std"])
        .reset_index()
    )

    # flatten columns
    summary.columns = [
        f"{c[0]}_{c[1]}" if c[1] else c[0]
        for c in summary.columns.to_flat_index()
    ]

    return summary


if __name__ == "__main__":
    final_table = run_comprehensive_study(n_rep=20)
    print("\n Toy data Final Methodology Comparison (Mean & SD) ")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1200)
    print(final_table)
    final_table.to_csv("simulation_with_ddc_like_results.csv", index=False)
