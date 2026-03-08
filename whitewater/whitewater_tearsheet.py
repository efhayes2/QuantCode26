import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from datetime import datetime


class WhitewaterMasterTearSheet:
    def __init__(self, directory='~/data/pngs/', capital=50000000):
        self.directory = os.path.expanduser(directory)
        self.source_file = os.path.join(self.directory, 'whitewater_current.csv')
        self.capital = capital
        self.df = None

    def load_data(self):
        if not os.path.exists(self.source_file):
            raise FileNotFoundError(f"Missing {self.source_file}. Run model first.")
        self.df = pd.read_csv(self.source_file, parse_dates=['day'])
        self.df.set_index('day', inplace=True)
        self.df.sort_index(inplace=True)
        return self

    def calculate_metrics(self):
        pl = self.df['total_daily_pl']
        nav = self.df['nav']

        # Monthly Resampling
        monthly_pl = pl.resample('ME').sum()
        monthly_ret = (monthly_pl / self.capital) * 100

        # Monthly Returns Grid
        ret_df = monthly_ret.to_frame('ret')
        ret_df['year'] = ret_df.index.year
        ret_df['month'] = ret_df.index.month
        monthly_grid = ret_df.pivot(index='year', columns='month', values='ret').reindex(columns=range(1, 13)).fillna(
            0.0)

        # Monthly Risk Ratios
        m_mean, m_std = monthly_ret.mean(), monthly_ret.std()
        m_sharpe = (m_mean / m_std * np.sqrt(12)) if m_std != 0 else 0
        down_std = monthly_ret[monthly_ret < 0].std()
        m_sortino = (m_mean / down_std * np.sqrt(12)) if (not pd.isna(down_std) and down_std != 0) else 0

        # Monthly Drawdown Logic
        max_monthly_dd_pct = monthly_ret.min()  # The worst single-month return %
        max_monthly_dd_val = (max_monthly_dd_pct / 100) * self.capital  # Dollar equivalent

        # Calmar Ratio
        max_dd_val = (nav - nav.cummax()).min()
        calmar = abs((m_mean * 12) / ((max_dd_val / self.capital) * 100)) if max_dd_val != 0 else 0

        # Distribution Stats
        skew, exc_kurt = monthly_ret.skew(), monthly_ret.kurt()

        # Signal Bias Table
        self.df['year'] = self.df.index.year
        ls_breakdown = self.df.groupby('year')['sig_target_waha_katy'].value_counts().unstack(fill_value=0)
        ls_breakdown = ls_breakdown.rename(columns={1: 'Short Waha', -1: 'Long Waha'})

        def get_regime_note(row):
            total = row['Short Waha'] + row['Long Waha']
            if total == 0: return "N/A"
            spct = row['Short Waha'] / total
            if spct > 0.80: return "Dominant Short Bias"
            if spct > 0.60: return "Short Bias (Widening)"
            if spct < 0.20: return "Dominant Long Bias"
            if spct < 0.40: return "Long Bias (Narrowing)"
            return "Neutral / Mean Reverting"

        ls_breakdown['Regime Commentary'] = ls_breakdown.apply(get_regime_note, axis=1)

        # Cluster Logic
        spreads = ['target_waha_hh', 'target_waha_katy', 'target_waha_hsc']
        market_corr = self.df[spreads].corr()
        strategy_corr = self.df[[f'daily_pl_{s}' for s in spreads]].corr()

        cluster_data = [
            ["HH / Katy", f"{market_corr.iloc[0, 1]:.2f}", f"{strategy_corr.iloc[0, 1]:.2f}"],
            ["Katy / HSC", f"{market_corr.iloc[1, 2]:.2f}", f"{strategy_corr.iloc[1, 2]:.2f}"],
            ["HH / HSC", f"{market_corr.iloc[0, 2]:.2f}", f"{strategy_corr.iloc[0, 2]:.2f}"]
        ]
        waha_corr = self.df[[f'daily_pl_{s}' for s in spreads]].corrwith(self.df['Waha']).mean()

        return {
            "monthly_grid": monthly_grid,
            "m_sharpe": m_sharpe,
            "m_sortino": m_sortino,
            "calmar": calmar,
            "skew": skew,
            "exc_kurt": exc_kurt,
            "ls_breakdown": ls_breakdown,
            "max_dd": max_dd_val,
            "max_monthly_dd": max_monthly_dd_val,  # Add this
            "max_monthly_dd_pct": max_monthly_dd_pct,  # Add this
            "win_rate_months": (monthly_ret > 0).sum() / len(monthly_ret),  # Added this line back
            "total_pl": pl.sum(),
            "avg_waha_corr": waha_corr,
            "cluster_data": cluster_data,
            "top_5_m_dd": monthly_ret.sort_values().head(5)
        }

    def generate_pdf(self):
        d = self.calculate_metrics()
        pdf_path = os.path.join(self.directory, 'whitewater_tearsheet_current.pdf')

        with PdfPages(pdf_path) as pdf:
            # --- PAGE 1: EXECUTIVE SUMMARY ---
            fig1, ax1 = plt.subplots(figsize=(8.5, 11));
            ax1.axis('off')
            plt.text(0.5, 0.95, 'WHITEWATER STRATEGY: EXECUTIVE SUMMARY', fontsize=18, weight='bold', ha='center')
            # --- PAGE 1: EXECUTIVE SUMMARY ---
            sum_data = [
                ["Capital Base", f"${self.capital:,.0f}"],
                ["Total Net Profit", f"${d['total_pl']:,.0f}"],
                ["Max Daily Drawdown", f"${d['max_dd']:,.0f}"],
                ["Max Monthly Drawdown", f"${d['max_monthly_dd']:,.0f} ({d['max_monthly_dd_pct']:.2f}%)"],  # New Row
                ["Monthly Sharpe", f"{d['m_sharpe']:.2f}"],
                ["Monthly Sortino", f"{d['m_sortino']:.2f}"],
                ["Calmar Ratio", f"{d['calmar']:.2f}"],
                ["Profitable Months", f"{d['win_rate_months']:.1%}"]
            ]
            ax1.table(cellText=sum_data, colLabels=['Metric', 'Value'], loc='center',
                      bbox=[0.1, 0.52, 0.8, 0.35]).set_fontsize(11)

            ax_nav = fig1.add_axes([0.1, 0.1, 0.8, 0.32])
            ax_nav.plot(self.df.index, self.df['nav'], color='#1f77b4', lw=2)
            ax_nav.set_title("Cumulative Net Performance ($)")
            ax_nav.grid(True, alpha=0.3)
            pdf.savefig(fig1);
            plt.close()

            # --- PAGE 2: PERFORMANCE & RISK DEEP-DIVE ---
            fig2, ax2 = plt.subplots(figsize=(8.5, 11));
            ax2.axis('off')
            plt.text(0.5, 0.96, 'MONTHLY PERFORMANCE & RISK DIAGNOSTICS', fontsize=16, weight='bold', ha='center')

            # 1. Monthly Grid (FIXED: Reset Index + colWidths for Year)
            plt.text(0.05, 0.92, 'Monthly Returns (%)', weight='bold', fontsize=11)
            grid_data = d['monthly_grid'].reset_index()
            grid_data.columns = ['Year'] + ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov',
                                            'Dec']
            widths = [0.12] + [0.07] * 12
            tab_grid = ax2.table(cellText=grid_data.round(2).values, colLabels=grid_data.columns,
                                 loc='center', cellLoc='center', bbox=[0.02, 0.75, 0.96, 0.16], colWidths=widths)
            tab_grid.set_fontsize(8)

            # 2. Key Ratios Box
            txt = (f"Sharpe: {d['m_sharpe']:.2f}  |  Sortino: {d['m_sortino']:.2f}  |  Calmar: {d['calmar']:.2f}\n"
                   f"Skewness: {d['skew']:.2f}  |  Excess Kurtosis: {d['exc_kurt']:.2f}  |  Corr to Waha Price: {d['avg_waha_corr']:.2f}")
            plt.text(0.5, 0.70, txt, fontsize=10, weight='bold', color='navy', ha='center',
                     bbox=dict(facecolor='white', edgecolor='navy', alpha=0.1, pad=5))

            # 3. Waha Signal Bias (Fixed: Reset Index + colWidths)
            plt.text(0.05, 0.61, 'Waha Signal Bias & Regime Analysis', weight='bold', fontsize=11)
            ls_data = d['ls_breakdown'].reset_index()
            ls_data.columns = ['Year', 'Short Waha', 'Long Waha', 'Regime Commentary']
            ls_widths = [0.12, 0.15, 0.15, 0.50]
            ls_tab = ax2.table(cellText=ls_data.values, colLabels=ls_data.columns,
                               loc='center', cellLoc='center', bbox=[0.05, 0.44, 0.9, 0.15], colWidths=ls_widths)
            ls_tab.set_fontsize(9)

            # 4. Correlation Cluster (Gap Created)
            plt.text(0.05, 0.38, 'Correlation Cluster: Market vs Strategy P/L', weight='bold', fontsize=11)
            cluster_tab = ax2.table(cellText=d['cluster_data'],
                                    colLabels=['Hub Pair', 'Market Corr', 'Strategy P/L Corr'],
                                    loc='center', cellLoc='center', bbox=[0.1, 0.22, 0.8, 0.13])
            cluster_tab.set_fontsize(9)

            # 5. Top 5 Drawdowns (Anchored at Bottom)
            plt.text(0.05, 0.16, 'Top 5 Worst Monthly Returns (%)', weight='bold', fontsize=11)
            dd_data = [[d.strftime('%b %Y'), f"{v:.2f}%"] for d, v in d['top_5_m_dd'].items()]
            ax2.table(cellText=dd_data, colLabels=['Month', 'Return'], loc='center', cellLoc='center',
                      bbox=[0.1, 0.02, 0.8, 0.11]).set_fontsize(9)

            pdf.savefig(fig2);
            plt.close()
        print(f"--- Master Tear Sheet Generated Successfully ---")


if __name__ == "__main__":
    WhitewaterMasterTearSheet(capital=50000000).load_data().generate_pdf()