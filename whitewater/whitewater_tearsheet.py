import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from datetime import datetime


class WhitewaterMasterTearSheet:
    def __init__(self, directory='~/data/pngs/', capital=10000000, no_HH=False, tc_per_mmbtu=0.05):
        self.directory = os.path.expanduser(directory)
        self.source_file = os.path.join(self.directory, 'whitewater_current.csv')
        self.capital = capital
        self.no_HH = no_HH
        self.tc_per_mmbtu = tc_per_mmbtu
        self.df = None

    def load_data(self):
        if not os.path.exists(self.source_file):
            raise FileNotFoundError(f"Missing {self.source_file}. Run model first.")
        self.df = pd.read_csv(self.source_file, parse_dates=['day'])
        self.df.set_index('day', inplace=True)
        self.df.sort_index(inplace=True)
        return self

    def calculate_metrics(self):
        # Ensure total_daily_pl is used to calculate the actual terminal value
        total_profit = self.df['total_daily_pl'].sum()
        terminal_value = self.capital + total_profit

        # Calculate precise year count
        days_active = (self.df.index[-1] - self.df.index[0]).days
        years = days_active / 365.25

        # Corrected Annualized Return (Geometric/CAGR)
        # Growth factor is (Ending Value / Starting Capital)
        ann_return_pct = ((terminal_value / self.capital) ** (1 / years) - 1) * 100

        # Monthly Resampling for Risk Ratios
        monthly_ret = (self.df['total_daily_pl'].resample('ME').sum() / self.capital) * 100
        m_mean, m_std = monthly_ret.mean(), monthly_ret.std()

        # Risk Ratios
        m_sharpe = (m_mean / m_std * np.sqrt(12)) if m_std != 0 else 0
        down_std = monthly_ret[monthly_ret < 0].std()
        m_sortino = (m_mean / down_std * np.sqrt(12)) if (not pd.isna(down_std) and down_std != 0) else 0

        # Drawdown Logic
        nav_series = self.capital + self.df['total_daily_pl'].cumsum()
        max_dd_val = (nav_series - nav_series.cummax()).min()
        max_dd_pct = abs((max_dd_val / self.capital) * 100)

        # Calmar Ratio using the corrected Annualized Return
        calmar = (ann_return_pct / max_dd_pct) if max_dd_pct != 0 else 0

        # Signal Bias Table (Using Katy as Proxy)
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

        # Correlation Clusters
        if self.no_HH:
            spreads = ['target_waha_katy', 'target_waha_hsc']
            market_corr = self.df[spreads].corr()
            strategy_corr = self.df[[f'daily_pl_{s}' for s in spreads]].corr()
            cluster_data = [
                ["Katy / HSC", f"{market_corr.iloc[0, 1]:.2f}", f"{strategy_corr.iloc[0, 1]:.2f}"],
                ["N/A (HH Excl)", "-", "-"],
                ["N/A (HH Excl)", "-", "-"]
            ]
        else:
            spreads = ['target_waha_hh', 'target_waha_katy', 'target_waha_hsc']
            market_corr = self.df[spreads].corr()
            strategy_corr = self.df[[f'daily_pl_{s}' for s in spreads]].corr()
            cluster_data = [
                ["HH / Katy", f"{market_corr.iloc[0, 1]:.2f}", f"{strategy_corr.iloc[0, 1]:.2f}"],
                ["Katy / HSC", f"{market_corr.iloc[1, 2]:.2f}", f"{strategy_corr.iloc[1, 2]:.2f}"],
                ["HH / HSC", f"{market_corr.iloc[0, 2]:.2f}", f"{strategy_corr.iloc[0, 2]:.2f}"]
            ]

        return {
            "monthly_grid": monthly_ret.to_frame('ret'),
            "ann_return_pct": ann_return_pct,
            "m_sharpe": m_sharpe,
            "m_sortino": m_sortino,
            "calmar": calmar,
            "skew": monthly_ret.skew(),
            "exc_kurt": monthly_ret.kurt(),
            "ls_breakdown": ls_breakdown,
            "max_dd": max_dd_val,
            "max_monthly_dd_pct": monthly_ret.min(),
            "win_rate_months": (monthly_ret > 0).sum() / len(monthly_ret),
            "total_pl": total_profit,
            "avg_waha_corr": self.df[[f'daily_pl_{s}' for s in spreads]].corrwith(self.df['Waha']).mean(),
            "cluster_data": cluster_data,
            "top_5_m_dd": monthly_ret.sort_values().head(5)
        }

    def generate_pdf(self):
        d = self.calculate_metrics()
        tc_suffix = f"_tc{int(self.tc_per_mmbtu * 100)}"
        hh_suffix = "_no_HH" if self.no_HH else ""
        pdf_name = f'whitewater_tearsheet_current{tc_suffix}{hh_suffix}.pdf'
        pdf_path = os.path.join(self.directory, pdf_name)

        with PdfPages(pdf_path) as pdf:
            # --- PAGE 1: EXECUTIVE SUMMARY ---
            fig1, ax1 = plt.subplots(figsize=(8.5, 11))
            ax1.axis('off')
            title = 'WHITEWATER STRATEGY: EXECUTIVE SUMMARY' + (' (Excl. HH)' if self.no_HH else '')
            plt.text(0.5, 0.95, title, fontsize=18, weight='bold', ha='center')

            sum_data = [
                ["Capital Base", f"${self.capital:,.0f}"],
                ["Trans. Costs (TC)", f"${self.tc_per_mmbtu:.3f} / MMBtu"],
                ["Total Net Profit", f"${d['total_pl']:,.0f}"],
                ["Annualized Return (CAGR)", f"{d['ann_return_pct']:.2f}%"],
                ["Max Daily Drawdown", f"${d['max_dd']:,.0f}"],
                ["Max Monthly Drawdown %", f"{d['max_monthly_dd_pct']:.2f}%"],
                ["Monthly Sharpe", f"{d['m_sharpe']:.2f}"],
                ["Monthly Sortino", f"{d['m_sortino']:.2f}"],
                ["Calmar Ratio", f"{d['calmar']:.2f}"],
                ["Profitable Months", f"{d['win_rate_months']:.1%}"]
            ]
            ax1.table(cellText=sum_data, colLabels=['Metric', 'Value'], loc='center',
                      bbox=[0.1, 0.48, 0.8, 0.42]).set_fontsize(11)

            ax_nav = fig1.add_axes([0.1, 0.1, 0.8, 0.32])
            ax_nav.plot(self.df.index, self.capital + self.df['total_daily_pl'].cumsum(), color='#1f77b4', lw=2)
            ax_nav.set_title("Total Account Value (Capital + Net P&L)")
            ax_nav.grid(True, alpha=0.3)
            pdf.savefig(fig1)
            plt.close()

            # --- PAGE 2: PERFORMANCE & RISK DEEP-DIVE ---
            fig2, ax2 = plt.subplots(figsize=(8.5, 11))
            ax2.axis('off')
            plt.text(0.5, 0.96, 'MONTHLY PERFORMANCE & RISK DIAGNOSTICS', fontsize=16, weight='bold', ha='center')

            # Monthly Grid
            grid_data = d['monthly_grid'].reset_index()
            grid_data['year'] = grid_data['day'].dt.year
            grid_data['month'] = grid_data['day'].dt.month
            pivot = grid_data.pivot(index='year', columns='month', values='ret').reindex(columns=range(1, 13)).fillna(
                0.0)
            pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

            ax2.table(cellText=pivot.round(2).reset_index().values, colLabels=['Year'] + list(pivot.columns),
                      loc='center', cellLoc='center', bbox=[0.02, 0.75, 0.96, 0.16],
                      colWidths=[0.12] + [0.07] * 12).set_fontsize(8)

            txt = (f"Sharpe: {d['m_sharpe']:.2f}  |  Sortino: {d['m_sortino']:.2f}  |  Calmar: {d['calmar']:.2f}\n"
                   f"Skewness: {d['skew']:.2f}  |  Excess Kurtosis: {d['exc_kurt']:.2f}  |  Corr to Waha Price: {d['avg_waha_corr']:.2f}")
            plt.text(0.5, 0.70, txt, fontsize=10, weight='bold', color='navy', ha='center',
                     bbox=dict(facecolor='white', edgecolor='navy', alpha=0.1, pad=5))

            # Regime Analysis
            ls_data = d['ls_breakdown'].reset_index()
            ls_data.columns = ['Year', 'Short Waha', 'Long Waha', 'Regime Commentary']
            ax2.table(cellText=ls_data.values, colLabels=ls_data.columns,
                      loc='center', cellLoc='center', bbox=[0.05, 0.44, 0.9, 0.15],
                      colWidths=[0.12, 0.15, 0.15, 0.50]).set_fontsize(9)

            # Correlation Cluster
            ax2.table(cellText=d['cluster_data'], colLabels=['Hub Pair', 'Market Corr', 'Strategy P/L Corr'],
                      loc='center', cellLoc='center', bbox=[0.1, 0.22, 0.8, 0.13]).set_fontsize(9)

            # Top 5 DD
            dd_data = [[dt.strftime('%b %Y'), f"{v:.2f}%"] for dt, v in d['top_5_m_dd'].items()]
            ax2.table(cellText=dd_data, colLabels=['Month', 'Return'], loc='center', cellLoc='center',
                      bbox=[0.1, 0.02, 0.8, 0.11]).set_fontsize(9)

            pdf.savefig(fig2)
            plt.close()

        print(f"--- Tearsheet Generated: {pdf_name} ---")


if __name__ == "__main__":
    WhitewaterMasterTearSheet(capital=10000000, no_HH=True, tc_per_mmbtu=0.05).load_data().generate_pdf()