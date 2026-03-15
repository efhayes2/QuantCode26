import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages


class WhitewaterForwardTearsheet:
    def __init__(self, directory='~/data/pngs/', capital=50_000_000):
        self.directory = os.path.expanduser(directory)
        self.source_file = os.path.join(self.directory, 'whitewater_forward_current.csv')
        self.capital = capital
        self.df = None

    def load_data(self):
        if not os.path.exists(self.source_file):
            raise FileNotFoundError(f"Missing {self.source_file}. Run the strategy first.")
        self.df = pd.read_csv(self.source_file, parse_dates=['day'])
        self.df.set_index('day', inplace=True)
        return self

    def _calculate_metrics(self):
        df = self.df
        monthly_returns = df['total_daily_pl'].resample('ME').sum() / self.capital

        nav = df['nav']
        max_dd_dollars = (nav - nav.cummax()).min()
        max_dd_pct = (max_dd_dollars / self.capital) * 100

        # Monthly DD logic
        m_nav = monthly_returns.cumsum()
        max_m_dd = (m_nav - m_nav.cummax()).min()

        m_sharpe = (monthly_returns.mean() / monthly_returns.std()) * np.sqrt(12)
        downside_std = monthly_returns[monthly_returns < 0].std() * np.sqrt(12)
        m_sortino = (monthly_returns.mean() * 12) / downside_std if downside_std > 0 else 0

        ann_return = monthly_returns.mean() * 12
        calmar = ann_return / abs(max_dd_pct / 100) if max_dd_pct != 0 else 0

        return {
            "Capital Base": f"${self.capital:,.0f}",
            "Total Net Profit": f"${df['total_daily_pl'].sum():,.0f}",
            "Max Daily Drawdown": f"${max_dd_dollars:,.0f} ({max_dd_pct:.2f}%)",
            "Monthly Sharpe": f"{m_sharpe:.2f}",
            "Monthly Sortino": f"{m_sortino:.2f}",
            "Calmar Ratio": f"{calmar:.2f}",
            "Profitable Months": f"{(monthly_returns > 0).sum() / len(monthly_returns) * 100:.1f}%"
        }

    def generate_pdf_tearsheet(self):
        """Generates a combined 2-page PDF report."""
        metrics = self._calculate_metrics()
        ts_date = datetime.now().strftime("%Y%m%d")
        pdf_path = os.path.join(self.directory, f"whitewater_forward_tearsheet_{ts_date}.pdf")

        with PdfPages(pdf_path) as pdf:
            # --- PAGE 1: EXECUTIVE SUMMARY ---
            fig1 = plt.figure(figsize=(11, 8.5))
            plt.suptitle(f"WHITEWATER FORWARD STRATEGY: EXECUTIVE SUMMARY", fontsize=16, fontweight='bold', y=0.95)

            gs = GridSpec(2, 1, height_ratios=[1, 1.5], hspace=0.3)

            ax_table = fig1.add_subplot(gs[0])
            ax_table.axis('off')
            table_data = [[k, v] for k, v in metrics.items()]
            table = ax_table.table(cellText=table_data, colWidths=[0.4, 0.4], loc='center', cellLoc='left',
                                   edges='horizontal')
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1, 2)

            ax_curve = fig1.add_subplot(gs[1])
            ax_curve.plot(self.df.index, self.df['nav'], color='navy', lw=2)
            ax_curve.set_title("Cumulative Net Performance ($)", loc='left', fontsize=12)
            ax_curve.grid(True, alpha=0.3)

            pdf.savefig(fig1)
            plt.close()

            # --- PAGE 2: MONTHLY HEATMAP ---
            m_returns = self.df['total_daily_pl'].resample('ME').sum() / self.capital * 100
            heatmap_df = m_returns.to_frame('returns')
            heatmap_df['year'] = heatmap_df.index.year
            heatmap_df['month'] = heatmap_df.index.month
            pivot = heatmap_df.pivot(index='year', columns='month', values='returns').fillna(0)

            fig2, ax = plt.subplots(figsize=(12, 6))
            im = ax.imshow(pivot, cmap='RdYlGn', aspect='auto', vmin=-5, vmax=5)

            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    ax.text(j, i, f'{pivot.iloc[i, j]:.2f}', ha='center', va='center', color='black', fontsize=9)

            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][
                               :len(pivot.columns)])
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_title("MONTHLY PERFORMANCE & RISK DIAGNOSTICS (Returns %)", fontweight='bold', pad=20)

            plt.colorbar(im, label='Monthly Return %')

            pdf.savefig(fig2)
            plt.close()

        print(f"Combined Forward Tearsheet PDF Generated: {os.path.basename(pdf_path)}")


if __name__ == "__main__":
    tearsheet = WhitewaterForwardTearsheet(capital=50_000_000)
    tearsheet.load_data().generate_pdf_tearsheet()