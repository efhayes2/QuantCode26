import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import os
from datetime import datetime
from pathlib import Path

from whitewater.whitewater_analyzer import WhitewaterDashboard
from whitewater.whitewater_tearsheet import WhitewaterMasterTearSheet


class WhitewaterSpotStrategy:
    def __init__(self, excel_path, base_lots, high_vol_lots, tc_per_mmbtu, exclude_hh):
        self.excel_path = os.path.expanduser(excel_path)
        self.base_lots = base_lots
        self.high_vol_lots = high_vol_lots
        self.contract_size = 10000
        self.tc_per_mmbtu = tc_per_mmbtu

        # Determine active trading spreads
        self.exclude_hh = exclude_hh
        if exclude_hh:
            self.spreads = ['target_waha_katy', 'target_waha_hsc']
        else:
            self.spreads = ['target_waha_hh', 'target_waha_katy', 'target_waha_hsc']

        self.full_df = None
        self.results_df = None

    def load_and_prep(self):
        db_path = Path(self.excel_path).parent.parent / "whitewater.db"
        use_sqlite = False

        if db_path.exists():
            try:
                conn = sqlite3.connect(db_path)
                price_df = pd.read_sql("SELECT * FROM phys_prices", conn).rename(columns={'date': 'day'})
                weather_raw = pd.read_sql("SELECT * FROM phys_weather", conn).rename(columns={'date': 'day'})
                lng_df = pd.read_sql("SELECT * FROM phys_lng", conn).rename(columns={'date': 'day'})
                conn.close()
                use_sqlite = True
            except Exception as e:
                print(f"SQL Load failed: {e}")

        if not use_sqlite:
            tabs = pd.read_excel(self.excel_path, sheet_name=None)
            price_df = tabs['prices'].rename(columns={'date': 'day'})
            weather_raw = tabs['weather'].rename(columns={'date': 'day'})
            lng_df = tabs['lng'].rename(columns={'date': 'day'})

        for df_item in [price_df, weather_raw, lng_df]:
            df_item['day'] = pd.to_datetime(df_item['day'])
            df_item.set_index('day', inplace=True)
            df_item.sort_index(inplace=True)
            df_item.columns = [c.replace(' ', '_') for c in df_item.columns]

        weather_pivoted = weather_raw.pivot_table(
            index='day', columns='station', values='min_temp_f'
        ).add_prefix('min_temp_f_')

        df = price_df.join([weather_pivoted, lng_df], how='left').ffill()

        # Define targets (Always calculate for features, but only trade 'self.spreads')
        df['target_waha_hh'] = df['Henry_Hub'] - df['Waha']
        df['target_waha_katy'] = df['Katy'] - df['Waha']
        df['target_waha_hsc'] = df['HSC'] - df['Waha']

        # Only create lags for the spreads we are actually analyzing/trading
        for s in self.spreads:
            for i in range(1, 4):
                df[f'{s}_lag{i}'] = df[s].shift(i)

        df['tomorrow_min_temp_MAF'] = df['min_temp_f_MAF']
        drop_cols = ['Agua_Dulce', 'min_temp_f_PEQ']
        self.full_df = df.drop(columns=drop_cols, errors='ignore').dropna().sort_index()

        print(f"Data ready. Exclude HH: {self.exclude_hh} | Trading: {len(self.spreads)} hubs")
        return self

    def run_backtest(self):
        final_results = []
        feature_cols = [c for c in self.full_df.columns if 'target' not in c]
        for trade_year in [2021, 2022, 2023, 2024]: #, 2025]:
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

    def simulate_spot_logic(self):
        df = self.results_df
        df['current_lots'] = np.where(df['tomorrow_min_temp_MAF'] < 35, self.high_vol_lots, self.base_lots)
        uri_mask = (df.index >= '2021-02-01') & (df.index <= '2021-02-28')

        for s in self.spreads:
            df[f'sig_{s}'] = np.where(df[f'pred_{s}'] > df[s], 1, -1)
            df[f'pos_{s}'] = df[f'sig_{s}'] * df['current_lots'] * self.contract_size
            df[f'tc_daily_{s}'] = df[f'pos_{s}'].abs() * self.tc_per_mmbtu
            gross_pl = df[s].diff().shift(-1) * df[f'sig_{s}'] * df['current_lots'] * self.contract_size
            df[f'daily_pl_{s}'] = gross_pl - df[f'tc_daily_{s}']
            df.loc[uri_mask, f'daily_pl_{s}'] = 0.0

        # Sum active spreads and calculate NAV
        pl_cols = [f'daily_pl_{s}' for s in self.spreads]
        df['total_daily_pl'] = df[pl_cols].sum(axis=1)
        df['nav'] = df['total_daily_pl'].fillna(0).cumsum()
        return self

    def export(self):
        save_dir = os.path.expanduser('~/data/pngs/')
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Keep standardized names for downstream script compatibility
        self.results_df.to_csv(os.path.join(save_dir, f"whitewater_hist_{ts}.csv"))
        self.results_df.to_csv(os.path.join(save_dir, "whitewater_current.csv"))
        print(f"Export complete: whitewater_current.csv (Exclude HH: {self.exclude_hh})")
        return self


if __name__ == "__main__":
    XLSX = '~/PyCharmProjects/QuantCode26/whitewater/data/trading_data.xlsx'

    # --- Configuration ---
    XLSX = '~/PyCharmProjects/QuantCode26/whitewater/data/trading_data.xlsx'
    tc_list = [0.02, 0.04, 0.05, 0.06]
    exclude_hh_list = [True, False]

    base_lots = 3
    high_vol_lots = 10
    capital = 10000000

    # --- Double Loop Execution ---
    for tc_val in tc_list:
        for hh_val in exclude_hh_list:
            print(f"\n>>> Running Scenario: TC={tc_val}, Exclude_HH={hh_val}")

            # 1. Run Strategy Simulation
            # We re-init to ensure parameters are applied to the simulation
            strategy = WhitewaterSpotStrategy(XLSX,
                                              base_lots=base_lots,
                                              high_vol_lots=high_vol_lots,
                                              tc_per_mmbtu=tc_val,
                                              exclude_hh=hh_val)

            # Standardize: Load -> Backtest (Model) -> Simulate (P&L) -> Export (CSV)
            strategy.load_and_prep() \
                .run_backtest() \
                .simulate_spot_logic() \
                .export()

            # 2. Run Dashboard (PNG Generation)
            analyzer = WhitewaterDashboard()
            analyzer.load_data().run_analysis(tc=tc_val, no_HH=hh_val)

            # 3. Run Master Tear Sheet (PDF Generation)
            # Pass the same capital, no_HH, and tc to ensure the report matches the run
            WhitewaterMasterTearSheet(capital=capital,
                                      no_HH=hh_val,
                                      tc_per_mmbtu=tc_val).load_data().generate_pdf()

    print("\n--- All Scenarios Complete ---")

    # Configuration
    # tc_per_mmbtu = 0.02
    # exclude_hh = True  # Set to True to prune the lagging HH leg
    # base_lots = 3
    # high_vol_lots = 10
    # capital = 10000000
    # strategy = WhitewaterSpotStrategy(XLSX,
    #                                   base_lots=base_lots,
    #                                   high_vol_lots=high_vol_lots,
    #                                   tc_per_mmbtu=tc_per_mmbtu,
    #                                   exclude_hh=exclude_hh)
    #
    # strategy.load_and_prep().run_backtest().simulate_spot_logic().export()
    #
    # analyzer = WhitewaterDashboard()
    # analyzer.load_data().run_analysis(tc=tc_per_mmbtu, no_HH=exclude_hh)
    # WhitewaterMasterTearSheet(capital=capital, no_HH=exclude_hh,
    #                           tc_per_mmbtu=tc_per_mmbtu).load_data().generate_pdf()