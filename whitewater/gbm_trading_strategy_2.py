import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import os
from datetime import datetime


class WhitewaterVariableStrategy:
    def __init__(self, excel_path, base_lots=10, high_vol_lots=40, tc_per_mmbtu=0.02):
        """
        Whitewater Variable Sizing Strategy
        :param base_lots: Number of lots for normal weather days (e.g., 10)
        :param high_vol_lots: Number of lots for cold forecast days (e.g., 40)
        :param tc_per_mmbtu: Transaction cost in dollars per MMBtu (e.g., 0.02)
        """
        self.excel_path = os.path.expanduser(excel_path)
        self.base_lots = base_lots
        self.high_vol_lots = high_vol_lots
        self.contract_size = 10000
        self.tc_per_mmbtu = tc_per_mmbtu
        self.spreads = ['target_waha_hh', 'target_waha_katy', 'target_waha_hsc']
        self.full_df = None
        self.results_df = None

    def load_data(self):
        print(f"Loading data from {self.excel_path}...")
        tabs = pd.read_excel(self.excel_path, sheet_name=None)
        price_df = tabs['prices'].rename(columns={'date': 'day'})
        weather_raw = tabs['weather'].rename(columns={'date': 'day'})
        lng_df = tabs['lng'].rename(columns={'date': 'day'})
        for df in [price_df, weather_raw, lng_df]: df['day'] = pd.to_datetime(df['day'])
        weather_pivoted = weather_raw.pivot_table(index='day', columns='station', values='min_temp_f').add_prefix(
            'min_temp_f_')
        price_df.set_index('day', inplace=True);
        lng_df.set_index('day', inplace=True)
        self.full_df = price_df.join([weather_pivoted, lng_df], how='inner')
        return self

    def add_features(self):
        print("Engineering features and lags...")
        df = self.full_df
        df['target_waha_hh'] = df['Henry Hub'] - df['Waha']
        df['target_waha_katy'] = df['Katy'] - df['Waha']
        df['target_waha_hsc'] = df['HSC'] - df['Waha']
        for s in self.spreads:
            df[f'{s}_lag1'] = df[s].shift(1)
            df[f'{s}_lag2'] = df[s].shift(2)
            df[f'{s}_lag3'] = df[s].shift(3)
        df['tomorrow_min_temp_MAF'] = df['min_temp_f_MAF'].shift(-1)

        # Remove columns identified in diagnostic with massive gaps
        cols_to_drop = ['Agua Dulce', 'min_temp_f_PEQ']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

        self.full_df = df.dropna().sort_index()
        print(f"Dataset size after cleaning: {len(self.full_df)} days.")
        return self

    def run_backtest(self):
        final_results = []
        feature_cols = [c for c in self.full_df.columns if 'target' not in c]
        for trade_year in [2021, 2022, 2023, 2024]:
            print(f"Backtesting Year: {trade_year}...")
            train = self.full_df.loc[f"{trade_year - 5}-01-01":f"{trade_year - 1}-12-31"]
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

    def simulate_strategy(self):
        print(
            f"Simulating: Base {self.base_lots} lots | Cold {self.high_vol_lots} lots | {self.tc_per_mmbtu * 100}c TC...")
        df = self.results_df

        # Weather Filter: Size up to high_vol_lots if Midland forecast < 35F
        df['current_lots'] = np.where(df['tomorrow_min_temp_MAF'] < 35, self.high_vol_lots, self.base_lots)

        # Uri Mask (Feb 2021)
        uri_mask = (df.index >= '2021-02-01') & (df.index <= '2021-02-28')

        for s in self.spreads:
            df[f'sig_{s}'] = np.where(df[f'pred_{s}'] > df[s], 1, -1)
            df[f'pos_{s}'] = df[f'sig_{s}'] * df['current_lots'] * self.contract_size

            # Turnover-based TC (Correctly handling scalar abs)
            first_pos = df[f'pos_{s}'].iloc[0]
            turnover_mmbtu = df[f'pos_{s}'].diff().abs().fillna(abs(first_pos))
            df[f'tc_daily_{s}'] = turnover_mmbtu * self.tc_per_mmbtu

            # P/L logic
            gross_pl = df[s].diff().shift(-1) * df[f'sig_{s}'] * df['current_lots'] * self.contract_size
            df[f'daily_pl_{s}'] = gross_pl - df[f'tc_daily_{s}']

            # Zero out Uri
            df.loc[uri_mask, f'daily_pl_{s}'] = 0.0

        df['total_daily_pl'] = df[[f'daily_pl_{s}' for s in self.spreads]].sum(axis=1)
        df['nav'] = df['total_daily_pl'].fillna(0).cumsum()
        self.results_df = df
        return self

    def export_and_plot(self):
        """Step 5: Output results and save PNG to disk."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.expanduser('~/data/pngs/')
        if not os.path.exists(save_dir): os.makedirs(save_dir)

        csv_name = f"whitewater_levered_{self.base_lots}_{self.high_vol_lots}_{ts}.csv"
        png_name = f"whitewater_levered_{self.base_lots}_{self.high_vol_lots}_{ts}.png"

        csv_path = os.path.join(save_dir, csv_name)
        png_path = os.path.join(save_dir, png_name)

        # Save CSV
        self.results_df.to_csv(csv_path)
        print(f"Results CSV saved to: {csv_path}")

        # Plotting
        plt.figure(figsize=(12, 7))
        plt.plot(self.results_df.index, self.results_df['nav'], color='navy', lw=2)
        plt.title(f'Whitewater Levered NAV ({self.base_lots}/{self.high_vol_lots} Lots | 2c TC)', fontsize=14)
        plt.ylabel('Net Cumulative P/L ($)')
        plt.grid(True, alpha=0.3)

        # SAVE THE PNG
        plt.savefig(png_path)
        print(f"NAV Plot saved to: {png_path}")

        plt.show()

# ==========================================
# SET YOUR PARAMETERS HERE
# ==========================================
if __name__ == "__main__":
    DATA_PATH = '~/PyCharmProjects/QuantCode26/whitewater/data/trading_data.xlsx'

    BASE_LOTS = 10  # Standard daily position
    COLD_LOTS = 40  # Position size when forecast < 35F
    TRANS_COST = 0.02  # $0.02/MMBtu

    strategy = WhitewaterVariableStrategy(
        excel_path=DATA_PATH,
        base_lots=BASE_LOTS,
        high_vol_lots=COLD_LOTS,
        tc_per_mmbtu=TRANS_COST
    )

    (strategy.load_data()
     .add_features()
     .run_backtest()
     .simulate_strategy()
     .export_and_plot())