# Grade Cell: Question 9
#
# Task: Implement a bootstrap procedure for OLS coefficients.
#
# Instructions:
# - Implement a function `bootstrap_ols_coefficients` that:
#   * draws B bootstrap samples of the training set (with replacement)
#   * fits OLS on each sample using scaled features
#   * stores coefficient vectors
#   * returns (coef_bootstrap_df, coef_ci_95)
# - Use B=200, RANDOM_STATE


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

coef_bootstrap_df = None
coef_ci_95 = None


def bootstrap_ols_coefficients(X, y, B=200, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    n_samples, n_features = X.shape

    boot_coefs = np.zeros((B, n_features + 1))

    model = LinearRegression()

    indices = np.arange(n_samples)

    for i in range(B):
        boot_idx = rng.choice(indices, size=n_samples, replace=True)

        X_resampled = X[boot_idx]
        y_resampled = y.iloc[boot_idx] if hasattr(y, 'iloc') else y[boot_idx]

        model.fit(X_resampled, y_resampled)
        boot_coefs[i, 0] = model.intercept_
        boot_coefs[i, 1:] = model.coef_

    feature_names = ['intercept'] + [f'feat_{j}' for j in range(n_features)]

    coef_bootstrap_df = pd.DataFrame(boot_coefs, columns=feature_names)

    coef_ci_95 = coef_bootstrap_df.quantile([0.025, 0.975])
    coef_ci_95.index = ["lower", "upper"]

    coef_ci_95 = coef_ci_95.T

    return coef_bootstrap_df, coef_ci_95


# Execution
coef_bootstrap_df, coef_ci_95 = bootstrap_ols_coefficients(
    X_train_scaled,
    y_train,
    B=200,
    random_state=RANDOM_STATE
)