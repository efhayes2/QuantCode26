import numpy as np
import pandas as pd
from scipy.optimize import minimize


class FTRPortfolioOptimizer:
    def __init__(self, path_returns_df):
        self.returns = path_returns_df
        self.mean_returns = path_returns_df.mean()
        self.cov_matrix = path_returns_df.cov()
        self.num_paths = len(path_returns_df.columns)

    def portfolio_stats(self, weights):
        """Calculates expected return and volatility for a given weight set."""
        weights = np.array(weights)
        port_return = np.sum(self.mean_returns * weights)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        return port_return, port_vol

    def minimize_volatility(self, target_return):
        """Standard Markowitz optimization: Min Vol for a given Target Return."""
        init_guess = [1.0 / self.num_paths] * self.num_paths
        bounds = [(0, 1) for _ in range(self.num_paths)]  # No shorting for this example

        constraints = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum of weights = 1
            {'type': 'eq', 'fun': lambda w: self.portfolio_stats(w)[0] - target_return}
        )

        result = minimize(
            lambda w: self.portfolio_stats(w)[1],  # Objective: Min Vol
            init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        return result


if __name__ == "__main__":
    # 1. Generate Synthetic Daily Payoffs for 4 FTR Paths
    # Path 1 & 2: Highly correlated (same interface)
    # Path 3: Negatively correlated to Path 1 (offsetting congestion)
    # Path 4: High volatility (risky speculative path)
    np.random.seed(42)
    days = 252

    data = {
        'P34_to_PSEG': np.random.normal(0.05, 0.1, days),
        'W_HUB_to_PSEG': np.random.normal(0.04, 0.09, days),
        'NY_to_PSEG': np.random.normal(0.02, 0.05, days),
        'DOM_to_PEPCO': np.random.normal(0.06, 0.2, days)  # High risk/reward
    }

    df_payoffs = pd.DataFrame(data)

    # 2. Run Optimization
    opt = FTRPortfolioOptimizer(df_payoffs)

    # Target a daily return of 4.5%
    target = 0.045
    optimized = opt.minimize_volatility(target)

    if optimized.success:
        print(f"Optimal Weights for Target Return {target:.2%}:")
        for i, path in enumerate(df_payoffs.columns):
            print(f"{path:<15}: {optimized.x[i]:.2%}")

        ret, vol = opt.portfolio_stats(optimized.x)
        print(f"\nPortfolio Return: {ret:.2%}")
        print(f"Portfolio Volatility (Risk): {vol:.2%}")
        print(f"Sharpe Ratio (assuming rf=0): {ret / vol:.2f}")