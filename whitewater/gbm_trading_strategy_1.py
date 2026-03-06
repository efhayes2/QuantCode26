# import pandas as pd
# import numpy as np
# import xgboost as xgb
# import matplotlib.pyplot as plt
# import os
#
#
# def main():
#     # 1. FILE PATH & DATA LOADING
#     file_path = os.path.expanduser('~/PyCharmProjects/QuantCode26/whitewater/data/trading_data.xlsx')
#
#     if not os.path.exists(file_path):
#         print(f"Error: File not found at {file_path}")
#         return
#
#     all_tabs = pd.read_excel(file_path, sheet_name=None)
#
#     price_df = all_tabs['prices'].rename(columns={'date': 'day'})
#     weather_raw = all_tabs['weather'].rename(columns={'date': 'day'})
#     lng_df = all_tabs['lng'].rename(columns={'date': 'day'})
#
#     for df in [price_df, weather_raw, lng_df]:
#         df['day'] = pd.to_datetime(df['day'])
#
#     # 2. WEATHER PIVOTING
#     weather_pivoted = weather_raw.pivot_table(
#         index='day',
#         columns='station',
#         values='min_temp_f'
#     ).add_prefix('min_temp_f_')
#
#     price_df.set_index('day', inplace=True)
#     lng_df.set_index('day', inplace=True)
#
#     # 3. JOIN & FEATURE ENGINEERING
#     full_df = price_df.join([weather_pivoted, lng_df], how='inner')
#
#     # Define Spread Targets
#     full_df['target_waha_hh'] = full_df['Henry Hub'] - full_df['Waha']
#     full_df['target_waha_katy'] = full_df['Katy'] - full_df['Waha']
#     full_df['target_waha_hsc'] = full_df['HSC'] - full_df['Waha']
#
#     # --- ADDING LAGGED SPREADS (t-1) ---
#     # This gives the model a 'starting point' based on yesterday's market
#     full_df['lag_waha_hh'] = full_df['target_waha_hh'].shift(1)
#     full_df['lag_waha_katy'] = full_df['target_waha_katy'].shift(1)
#     full_df['lag_waha_hsc'] = full_df['target_waha_hsc'].shift(1)
#
#     # --- Look-ahead Bias (t+1) Features ---
#     full_df['tomorrow_min_temp_MAF'] = full_df['min_temp_f_MAF'].shift(-1)
#     full_df['tomorrow_freeport_lng'] = full_df['Freeport'].shift(-1)
#
#     # Drop NAs created by shifts
#     full_df = full_df.dropna()
#
#     # 4. GBM MODEL & WALK-FORWARD BACKTEST
#     final_results = []
#
#     for trade_year in [2021, 2022, 2023, 2024]:
#         print(f"Processing Year: {trade_year}...")
#
#         train_start, train_end = f"{trade_year - 5}-01-01", f"{trade_year - 1}-12-31"
#         test_start, test_end = f"{trade_year}-01-01", f"{trade_year}-12-31"
#
#         train_set = full_df[train_start:train_end]
#         test_set = full_df[test_start:test_end]
#
#         if train_set.empty or test_set.empty:
#             continue
#
#         # Exclude targets from features, but KEEP the lags
#         feature_cols = [c for c in full_df.columns if 'target' not in c]
#         year_outcomes = test_set.copy()
#
#         for spread in ['target_waha_hh', 'target_waha_katy', 'target_waha_hsc']:
#             model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05)
#             model.fit(train_set[feature_cols], train_set[spread])
#             year_outcomes[f'pred_{spread}'] = model.predict(test_set[feature_cols])
#
#         final_results.append(year_outcomes)
#
#     backtest_df = pd.concat(final_results)
#
#     # 5. INSPECTION
#     print("\n--- Model with Lagged Spreads (Tail) ---")
#     inspect_cols = ['target_waha_katy', 'pred_target_waha_katy', 'lag_waha_katy', 'min_temp_f_MAF']
#     print(backtest_df[inspect_cols].tail(10))
#
#     backtest_df[['target_waha_katy', 'pred_target_waha_katy']].plot(figsize=(12, 6))
#     plt.title('Katy-Waha Basis with t-1 Lag and t+1 Weather')
#     plt.show()
#
#
# if __name__ == "__main__":
#     main()