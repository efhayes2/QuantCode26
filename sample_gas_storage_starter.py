# Assuming we have discretized the inventory into 'grid_points' (e.g., 0%, 10%... 100%)
# For each time step t, we need to estimate the value of being in state (S_t, v_t)

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def storage_regression_step(S_paths, Inventory_paths, Realized_Future_Values):
    # S_paths: Simulated gas prices at time t
    # Inventory_paths: Possible fill levels (usually a grid)
    # Realized_Future_Values: Realized values from t+1

    # 1. Create a 2D feature matrix [Price, Inventory]
    X = np.column_stack([S_paths, Inventory_paths])

    # 2. Generate 2D Polynomial features: [1, S, v, S^2, v^2, S*v]
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    # 3. Fit the regression
    model = LinearRegression().fit(X_poly, Realized_Future_Values)

    # 4. The continuation value is now a function of both price and fill-level
    continuation_values = model.predict(X_poly)

    return continuation_values