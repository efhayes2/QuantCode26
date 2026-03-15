import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime


class WhitewaterForwardDashboard:
    def __init__(self, directory='~/data/pngs/'):
        self.directory = os.path.expanduser(directory)
        # Pointing specifically to the forward strategy output
        self.source_file = os.path.join(self.directory, 'whitewater_forward_current.csv')
        self.df = None

    def load_data(self):
        """Loads the static reference file created by the forward strategy model."""
        if not os.path.exists(self.source_file):
            raise FileNotFoundError(f"Missing {self.source_file}. Ensure gbm_strategy_on_forwards.py has run.")

        self.df = pd.read_csv(self.source_file, parse_dates=['day'])
        self.df.set_index('day', inplace=True)
        return self

    def run_analysis(self):
        """Generates the diagnostic dashboard for the Forward market strategy."""
        # Risk Calculation
        returns = self.df['total_daily_pl']
        nav = self.df['nav']
        drawdown = nav - nav.cummax()

        # Monthly Sharpe calculation for the "Tearsheet" feel
        m_returns = self.df['total_daily_pl'].resample('ME').sum()
        monthly_sharpe = np.sqrt(12) * (m_returns.mean() / m_returns.std())

        daily_sharpe = np.sqrt(252) * (returns.mean() / returns.std())
        max_dd = drawdown.min()

        # Visual Setup
        plt.style.use('ggplot')
        fig, axes = plt.subplots(3, 1, figsize=(12, 14), gridspec_kw={'height_ratios': [2, 1, 1]})

        # Plot 1: Net NAV (Forward)
        axes[0].plot(self.df.index, nav, color='darkblue', lw=2.5, label='Forward Strategy Net NAV')
        axes[0].set_title(
            f'WHITEWATER FORWARD PERFORMANCE DASHBOARD\nMonthly Sharpe: {monthly_sharpe:.2f} | Max DD: ${max_dd:,.0f}',
            fontsize=16, fontweight='bold')
        axes[0].set_ylabel('Cumulative P/L ($)')
        axes[0].legend(loc='upper left')

        # Plot 2: Drawdown
        axes[1].fill_between(self.df.index, drawdown, 0, color='crimson', alpha=0.3)
        axes[1].plot(self.df.index, drawdown, color='darkred', lw=1)
        axes[1].set_title('Forward Risk Exposure (Drawdown Profile)', fontsize=12)
        axes[1].set_ylabel('Loss ($)')

        # Plot 3: Forward Alpha Distribution
        axes[2].hist(returns, bins=60, color='steelblue', edgecolor='black', alpha=0.7)
        axes[2].axvline(0, color='black', linestyle='--', lw=1)
        axes[2].set_title('Forward Daily P/L (Alpha Distribution)', fontsize=12)
        axes[2].set_xlabel('Daily P/L ($)')

        plt.tight_layout()

        # Save Logic with "Forward" identifiers
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        hist_png = os.path.join(self.directory, f'whitewater_forward_dashboard_{ts}.png')
        curr_png = os.path.join(self.directory, 'whitewater_forward_dashboard_current.png')

        plt.savefig(hist_png)
        plt.savefig(curr_png)

        print(f"Forward Dashboard Update Complete:")
        print(f" - {os.path.basename(hist_png)}")
        print(f" - whitewater_forward_dashboard_current.png")

        # Output console summary for quick verification
        print(f"\nQuick Metrics:")
        print(f"Daily Sharpe: {daily_sharpe:.2f}")
        print(f"Monthly Sharpe: {monthly_sharpe:.2f}")

        plt.show()


if __name__ == "__main__":
    # Ensure your strategy script saves its output as 'whitewater_forward_current.csv'
    analyzer = WhitewaterForwardDashboard()
    analyzer.load_data().run_analysis()