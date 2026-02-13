import pandas as pd
import numpy as np


class FTRBacktester:
    def __init__(self, source_prices, sink_prices):
        # The 'Basis' is the FTR payoff (Sink - Source)
        self.basis = sink_prices - source_prices

    def generate_signals(self, window=20, z_threshold=1.5):
        """
        Simple Mean Reversion Strategy:
        Buy Basis (Long FTR) when spread is unusually low.
        Sell Basis (Short FTR) when spread is unusually high.
        """
        rolling_mean = self.basis.rolling(window=window).mean()
        rolling_std = self.basis.rolling(window=window).std()

        # Calculate Z-Score
        z_score = (self.basis - rolling_mean) / rolling_std

        # Signals: 1 = Long FTR (Sink-Source), -1 = Short FTR, 0 = Flat
        signals = np.where(z_score < -z_threshold, 1, 0)
        signals = np.where(z_score > z_threshold, -1, signals)

        return pd.Series(signals, index=self.basis.index)

    def calculate_returns(self, signals):
        # Strategy returns: signal * daily change in basis
        # In FTRs, we care about the daily 'mark-to-market' of the congestion spread
        basis_diff = self.basis.diff().shift(-1)
        strategy_returns = signals * basis_diff
        return strategy_returns.cumsum()


if __name__ == "__main__":
    # Generate synthetic hourly Basis data (Mean Reverting)
    # Suppose we found a path via PCA that looks mean-reverting
    np.random.seed(42)
    n = 500
    basis = np.zeros(n)
    theta = 0.1  # Speed of mean reversion
    mu = 5.0  # Long-term average congestion spread ($/MWh)
    sigma = 2.0  # Volatility

    for t in range(1, n):
        basis[t] = basis[t - 1] + theta * (mu - basis[t - 1]) + np.random.normal(0, sigma)

    prices_df = pd.Series(basis)

    # Backtest
    tester = FTRBacktester(source_prices=0, sink_prices=prices_df)  # Source=0 for pure basis study
    signals = tester.generate_signals()
    equity_curve = tester.calculate_returns(signals)

    print(f"Final Strategy Equity: ${equity_curve.iloc[-2]:.2f}")