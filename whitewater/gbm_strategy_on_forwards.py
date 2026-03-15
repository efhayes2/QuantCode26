import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import os
from datetime import datetime
from pathlib import Path


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
        """Loads everything directly from SQLite."""
        conn = sqlite3.connect(self.db_path)

        # 1. Physical Features
        price_df = pd.read_sql("SELECT * FROM phys_prices", conn).rename(columns={'date': 'day'})
        weather_raw = pd.read_sql("SELECT * FROM phys_weather", conn).rename(columns={'date': 'day'})
        lng_df = pd.read_sql("SELECT * FROM phys_lng", conn).rename(columns={'date': 'day'})

        # 2. Financial Targets (M1 Swaps)
        swap_hubs = {'WA': 'swap_wa', 'HH': 'swap_hh', 'KT': 'swap_kt', 'HS': 'swap_hs'}
        swap_list = []
        for hub, table in swap_hubs.items():
            s_df = pd.read_sql(f"SELECT date as day, M1 as {hub}_M1 FROM {table}", conn)
            s_df['day'] = pd.to_datetime(s_df['day'])
            s_df.set_index('day', inplace=True)
            swap_list.append(s_df)
        conn.close()

        # 3. Standardize Physical Data
        for df_item in [price_df, weather_raw, lng_df]:
            df_item['day'] = pd.to_datetime(df_item['day'])
            df_item.set_index('day', inplace=True)
            df_item.sort_index(inplace=True)
            df_item.columns = [c.replace(' ', '_') for c in df_item.columns]

        # 4. Process Weather & Join
        weather_pivoted = weather_raw.pivot_table(index='day', columns='station', values='min_temp_f').add_prefix(
            'min_temp_f_')
        df = price_df.join([weather_pivoted, lng_df] + swap_list, how='inner').ffill()

        # 5. Financial Feature Engineering
        df['target_waha_hh'] = df['HH_M1'] - df['WA_M1']
        df['target_waha_katy'] = df['KT_M1'] - df['WA_M1']
        df['target_waha_hsc'] = df['HS_M1'] - df['WA_M1']

        for s in self.spreads:
            for i in range(1, 4):
                df[f'{s}_lag{i}'] = df[s].shift(i)

        df['tomorrow_min_temp_MAF'] = df['min_temp_f_MAF']

        self.full_df = df.drop(columns=['Agua_Dulce', 'min_temp_f_PEQ'], errors='ignore').dropna().sort_index()
        print(
            f"DB Load Complete. Rows: {len(self.full_df)} | Range: {self.full_df.index.min().date()} to {self.full_df.index.max().date()}")
        return self

    def run_backtest(self):
        final_results = []
        feature_cols = [c for c in self.full_df.columns if 'target' not in c and '_M1' not in c]

        # Running 2022-2024 to ensure swap coverage
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
            # Daily P&L based on M1 price movement
            df[f'daily_pl_{s}'] = df[s].diff().shift(-1) * df[f'sig_{s}'] * df['current_lots'] * self.contract_size
            df[f'tc_daily_{s}'] = df['current_lots'] * self.contract_size * self.tc_per_mmbtu
            df[f'daily_pl_{s}'] -= df[f'tc_daily_{s}']

        df['total_daily_pl'] = df[[f'daily_pl_{s}' for s in self.spreads]].sum(axis=1)
        df['nav'] = df['total_daily_pl'].fillna(0).cumsum()
        return self

    def generate_rolling_sharpe(self, window=60):
        capital = 1_000_000
        daily_returns = self.results_df['total_daily_pl'] / capital
        rolling_sharpe = (daily_returns.rolling(window).mean() / daily_returns.rolling(window).std()) * np.sqrt(252)

        plt.figure(figsize=(12, 5))
        rolling_sharpe.plot(color='#3498db', lw=2, label='60-Day Forward Sharpe')
        plt.axhline(y=rolling_sharpe.mean(), color='red', linestyle='--')
        plt.title('Forward Strategy Stability: Rolling 60-Day Sharpe')

        save_path = os.path.expanduser('~/data/pngs/forward_rolling_sharpe.png')
        plt.savefig(save_path)
        print(f"Forward Sharpe plot saved to: {save_path}")
        return self

    def export(self):
        # Match the directory expected by the dashboard
        save_dir = os.path.expanduser('~/data/pngs/')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save timestamped version for history
        self.results_df.to_csv(os.path.join(save_dir, f"whitewater_forward_hist_{ts}.csv"))

        # Save the STATIC reference file the dashboard script looks for
        self.results_df.to_csv(os.path.join(save_dir, "whitewater_forward_current.csv"))

        print(f"Export complete. Dashboard source 'whitewater_forward_current.csv' updated.")
        return self

if __name__ == "__main__":
    DB_PATH = '~/PyCharmProjects/QuantCode26/whitewater/whitewater.db'
    strategy = WhitewaterForwardStrategy(DB_PATH)
    strategy.load_and_prep() \
        .run_backtest() \
        .simulate_forward_logic() \
        .generate_rolling_sharpe() \
        .export()