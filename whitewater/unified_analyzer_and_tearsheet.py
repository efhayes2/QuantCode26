import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import os
from datetime import datetime
from pathlib import Path
from matplotlib.gridspec import GridSpec


class WhitewaterForwardStrategy:
    def __init__(self, db_path, base_lots=15, high_vol_lots=50, tc_per_mmbtu=0.01):
        self.db_path = Path(os.path.expanduser(db_path))
        self.base_lots = base_lots
        self.high_vol_lots = high_vol_lots
        self.contract_size = 10000
        self.tc_per_mmbtu = tc_per_mmbtu
        self.spreads = ['target_waha_hh', 'target_waha_katy', 'target_waha_hsc']
        self.full_df = None
        self.results_df = None

    def load_and_prep(self):
        conn = sqlite3.connect(self.db_path)
        # Features
        price_df = pd.read_sql("SELECT * FROM phys_prices", conn).rename(columns={'date': 'day'})
        weather_raw = pd.read_sql("SELECT * FROM phys_weather", conn).rename(columns={'date': 'day'})
        lng_df = pd.read_sql("SELECT * FROM phys_lng", conn).rename(columns={'date': 'day'})
        # Financial Targets
        swap_hubs = {'WA': 'swap_wa', 'HH': 'swap_hh', 'KT': 'swap_kt', 'HS': 'swap_hs'}
        swap_list = []
        for hub, table in swap_hubs.items():
            s_df = pd.read_sql(f"SELECT date as day, M1 as {hub}_M1 FROM {table}", conn)
            s_df['day'] = pd.to_datetime(s_df['day'])
            s_df.set_index('day', inplace=True)
            swap_list.append(s_df)
        conn.close()

        for df_item in [price_df, weather_raw, lng_df]:
            df_item['day'] = pd.to_datetime(df_item['day'])
            df_item.set_index('day', inplace=True)
            df_item.sort_index(inplace=True)
            df_item.columns = [c.replace(' ', '_') for c in df_item.columns]

        weather_pivoted = weather_raw.pivot_table(index='day', columns='station', values='min_temp_f').add_prefix(
            'min_temp_f_')
        df = price_df.join([weather_pivoted, lng_df] + swap_list, how='inner').ffill()

        df['target_waha_hh'] = df['HH_M1'] - df['WA_M1']
        df['target_waha_katy'] = df['KT_M1'] - df['WA_M1']
        df['target_waha_hsc'] = df['HS_M1'] - df['WA_M1']

        for s in self.spreads:
            for i in range(1, 4): df[f'{s}_lag{i}'] = df[s].shift(i)

        df['tomorrow_min_temp_MAF'] = df['min_temp_f_MAF']
        self.full_df = df.drop(columns=['Agua_Dulce', 'min_temp_f_PEQ'], errors='ignore').dropna().sort_index()
        return self

    def run_backtest(self):
        final_results = []
        feature_cols = [c for c in self.full_df.columns if 'target' not in c and '_M1' not in c]
        for trade_year in [2022, 2023, 2024]:
            train = self.full_df.loc[:f"{trade_year - 1}-12-31"]
            test = self.full_df.loc[f"{trade_year}-01-01":f"{trade_year}-12-31"]
            if train.empty or test.empty: continue
            year_outcomes = test.copy()
            for s in self.spreads:
                model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05)
                model.fit(train[feature_cols], train[s])
                year_outcomes[f'pred_{s}'] = model.predict(test[feature_cols])
            final_results.append(year_outcomes)
        self.results_df = pd.concat(final_results)
        return self

    def simulate_forward_logic(self):
        df = self.results_df
        df['current_lots'] = np.where(df['tomorrow_min_temp_MAF'] < 35, self.high_vol_lots, self.base_lots)
        for s in self.spreads:
            df[f'sig_{s}'] = np.where(df[f'pred_{s}'] > df[s], 1, -1)
            df[f'daily_pl_{s}'] = df[s].diff().shift(-1) * df[f'sig_{s}'] * df['current_lots'] * self.contract_size
            df[f'tc_daily_{s}'] = df['current_lots'] * self.contract_size * self.tc_per_mmbtu
            df[f'daily_pl_{s}'] -= df[f'tc_daily_{s}']
        df['total_daily_pl'] = df[[f'daily_pl_{s}' for s in self.spreads]].sum(axis=1)
        df['nav'] = df['total_daily_pl'].fillna(0).cumsum()
        return self


class StrategyAnalyzer:
    def __init__(self, results_df, use_forward=False, capital=50_000_000):
        self.df = results_df.copy()
        self.use_forward = use_forward
        self.capital = capital
        self.strategy_label = "FORWARD BASIS" if use_forward else "SPOT BASIS"
        self.save_dir = os.path.expanduser('~/data/pngs/')
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)

    def generate_dashboard(self):
        """Generates the high-density 3-panel dashboard from your PNG."""
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=fig)

        # 1. Main Equity Curve
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.df.index, self.df['nav'], color='#2c3e50', lw=2)
        ax1.set_title(f"WHITEWATER {self.strategy_label}: CUMULATIVE NET PERFORMANCE", loc='left', fontweight='bold')
        ax1.grid(True, alpha=0.2)

        # 2. Rolling Sharpe
        ax2 = fig.add_subplot(gs[1, 0])
        returns = self.df['total_daily_pl'] / self.capital
        rolling = (returns.rolling(60).mean() / returns.rolling(60).std()) * np.sqrt(252)
        ax2.plot(rolling, color='#3498db' if self.use_forward else '#2ecc71')
        ax2.set_title("ROLLING 60-DAY SHARPE", loc='left')
        ax2.grid(True, alpha=0.2)

        # 3. Drawdown
        ax3 = fig.add_subplot(gs[1, 1])
        dd = self.df['nav'] - self.df['nav'].cummax()
        ax3.fill_between(dd.index, dd, 0, color='#e74c3c', alpha=0.3)
        ax3.set_title("DRAWDOWN EXPOSURE ($)", loc='left')
        ax3.grid(True, alpha=0.2)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"dashboard_{self.strategy_label.lower().replace(' ', '_')}.png"))
        plt.close()

    def generate_tearsheet(self):
        """Generates the professional 2-page report from your PDF."""
        # --- Page 1: Executive Summary ---
        metrics = self._calculate_metrics()

        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        plt.text(0.05, 0.95, f"WHITEWATER {self.strategy_label}: EXECUTIVE SUMMARY", fontsize=16, fontweight='bold')

        # Metrics Table (Matching your PDF layout)
        table_data = [[k, v] for k, v in metrics.items()]
        the_table = plt.table(cellText=table_data, colWidths=[0.3, 0.3], loc='center left', cellLoc='left',
                              edges='horizontal')
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(10)
        the_table.scale(1, 1.8)

        plt.savefig(os.path.join(self.save_dir, f"tearsheet_p1_{self.strategy_label.lower().replace(' ', '_')}.png"))
        plt.close()

        # --- Page 2: Monthly Returns Grid ---
        self._plot_monthly_heatmap()

    def _calculate_metrics(self):
        df = self.df
        returns = df['total_daily_pl'] / self.capital
        m_returns = df['total_daily_pl'].resample('ME').sum() / self.capital

        nav = df['nav']
        max_dd = (nav - nav.cummax()).min()

        sharpe_m = (m_returns.mean() / m_returns.std()) * np.sqrt(12)
        downside_std = m_returns[m_returns < 0].std() * np.sqrt(12)
        sortino = (m_returns.mean() * 12) / downside_std if downside_std != 0 else 0

        return {
            "Capital Base": f"${self.capital:,.0f}",
            "Total Net Profit": f"${df['total_daily_pl'].sum():,.0f}",
            "Max Daily Drawdown": f"${max_dd:,.0f}",
            "Monthly Sharpe": round(sharpe_m, 2),
            "Monthly Sortino": round(sortino, 2),
            "Calmar Ratio": round((m_returns.mean() * 12) / abs(max_dd / self.capital), 2) if max_dd != 0 else 0,
            "Profitable Months": f"{(m_returns > 0).sum() / len(m_returns) * 100:.1f}%"
        }

    def _plot_monthly_heatmap(self):
        m_returns = self.df['total_daily_pl'].resample('ME').sum() / self.capital * 100
        heatmap_data = m_returns.to_frame('ret')
        heatmap_data['year'] = heatmap_data.index.year
        heatmap_data['month'] = heatmap_data.index.month
        pivot = heatmap_data.pivot(index='year', columns='month', values='ret')

        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(pivot, cmap='RdYlGn', aspect='auto')
        ax.set_title("MONTHLY PERFORMANCE & RISK DIAGNOSTICS (%)", fontweight='bold')
        plt.colorbar(im)
        plt.savefig(os.path.join(self.save_dir, f"tearsheet_p2_{self.strategy_label.lower().replace(' ', '_')}.png"))
        plt.close()


if __name__ == "__main__":
    DB_PATH = '~/PyCharmProjects/QuantCode26/whitewater/whitewater.db'
    use_fwd = True

    strategy = WhitewaterForwardStrategy(DB_PATH)
    strategy.load_and_prep().run_backtest().simulate_forward_logic()

    analyzer = StrategyAnalyzer(strategy.results_df, use_forward=use_fwd)
    analyzer.generate_dashboard()
    analyzer.generate_tearsheet()
    print(f"Reports generated in ~/data/pngs/ for {analyzer.strategy_label}")