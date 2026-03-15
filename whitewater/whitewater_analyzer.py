import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime


class WhitewaterDashboard:
    def __init__(self, directory='~/data/pngs/'):
        self.directory = os.path.expanduser(directory)
        self.source_file = os.path.join(self.directory, 'whitewater_current.csv')
        self.df = None

    def load_data(self):
        """Loads the static reference file created by the strategy model."""
        if not os.path.exists(self.source_file):
            raise FileNotFoundError(f"Missing {self.source_file}. Run the strategy model first.")

        self.df = pd.read_csv(self.source_file, parse_dates=['day'])
        self.df.set_index('day', inplace=True)
        return self

    def run_analysis(self, tc, no_HH):
        """Generates the 3-part diagnostic dashboard and saves current/timestamped versions."""
        # Risk Calculation
        returns = self.df['total_daily_pl']
        nav = self.df['nav']
        drawdown = nav - nav.cummax()  # Underwater calculation

        sharpe = np.sqrt(252) * (returns.mean() / returns.std())
        max_dd = drawdown.min()

        # Visual Setup
        plt.style.use('ggplot')
        fig, axes = plt.subplots(3, 1, figsize=(12, 14), gridspec_kw={'height_ratios': [2, 1, 1]})

        # Plot 1: Net NAV
        axes[0].plot(self.df.index, nav, color='navy', lw=2.5, label='Strategy Net NAV')
        axes[0].set_title(f'WHITEWATER PERFORMANCE DASHBOARD\nNet Sharpe: {sharpe:.2f} | Max DD: ${max_dd:,.0f}',
                          fontsize=16, fontweight='bold')
        axes[0].set_ylabel('Cumulative P/L ($)')
        axes[0].legend(loc='upper left')

        # Plot 2: Drawdown (The "Underwater" Plot)
        axes[1].fill_between(self.df.index, drawdown, 0, color='crimson', alpha=0.3)
        axes[1].plot(self.df.index, drawdown, color='darkred', lw=1)
        axes[1].set_title('Risk Exposure (Drawdown Profile)', fontsize=12)
        axes[1].set_ylabel('Loss ($)')

        # Plot 3: P/L Distribution (Fat-Tail Analysis)
        axes[2].hist(returns, bins=60, color='skyblue', edgecolor='black', alpha=0.7)
        axes[2].axvline(0, color='black', linestyle='--', lw=1)
        axes[2].set_title('Daily Profit & Loss Distribution', fontsize=12)
        axes[2].set_xlabel('Daily P/L ($)')

        plt.tight_layout()

        # 2. Determine the HH suffix
        hh_suffix = "_no_HH" if no_HH else ""
        tc_suffix = f"_tc{int(tc * 100)}"

        # 3. Construct the filename
        filename = f"whitewater_dashboard_current{tc_suffix}{hh_suffix}.png"
        curr_png = os.path.join(self.directory, filename)

        # Save Logic (Historical and Current)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        hist_png = os.path.join(self.directory, f'whitewater_dashboard_{ts}.png')
        #curr_png = os.path.join(self.directory, 'whitewater_dashboard_current.png')

        plt.savefig(hist_png)
        plt.savefig(curr_png)

        print(f"Dashboard Update Complete:")
        print(f" - {os.path.basename(hist_png)}")
        print(f" - whitewater_dashboard_current.png")
        plt.show()


if __name__ == "__main__":
    analyzer = WhitewaterDashboard()
    analyzer.load_data().run_analysis()