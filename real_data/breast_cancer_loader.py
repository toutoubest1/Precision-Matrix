# Load the breast cancer dataset


bc = load_breast_cancer()
X_real_df = pd.DataFrame(bc.data, columns=bc.feature_names)

# Standardize the original complete data once
scaler_real = StandardScaler()
X_real_clean = scaler_real.fit_transform(X_real_df.values)

print("Breast cancer data shape:", X_real_clean.shape)

# Build a surrogate reference graph from the original data

# Since the true graph is unknown for real data, we use the graph
# estimated from the original unperturbed data as a surrogate reference.

Theta_ref, alpha_ref = safe_glasso_fit(
    X_real_clean,
    alpha=0.1,
    use_cv=True,
    random_state=123
)
