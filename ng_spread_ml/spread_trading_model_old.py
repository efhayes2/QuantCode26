import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier


# 1. Define your Relative Month Mapping
def get_relative_contract_map(start_month_code):
    codes = ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z']
    start_idx = codes.index(start_month_code)
    return {code: (start_idx - i) % 12 + 1 for i, code in enumerate(codes)}


# 2. Setup Data and Features
df = pd.read_csv('prompt_spreads.csv', parse_dates=['Date'])
v_map = get_relative_contract_map('V')  # Position relative to October

# Feature Engineering
df['Month_Code'] = df['Front_Contract'].str[2:3]
df['Relative_Month'] = df['Month_Code'].map(v_map)
df['DayOfYear'] = df['Date'].dt.dayofyear

# Fundamental Proxies:
# (In a production version, we would merge actual EIA CSVs here)
# We calculate a rolling "Spread Momentum" to act as a proxy for inventory tightness
df['Spread_Vel'] = df['Spread'].diff(5)
df['Inventory_Proxy'] = df['Spread'].rolling(20).mean() - df['Spread'].rolling(100).mean()

# 3. Thursday Strategy with Stop-Loss
thursday_df = df[df['Date'].dt.dayofweek == 3].copy()
thursday_df['Target_Change'] = thursday_df['Spread'].shift(-1) - thursday_df['Spread']
thursday_df['Target'] = (thursday_df['Target_Change'] > 0).astype(int)
thursday_df = thursday_df.dropna()

features = ['Spread', 'Relative_Month', 'DayOfYear', 'Spread_Vel', 'Inventory_Proxy', 'Front_Position']

# Backtest Settings
multiplier, contracts, stop_loss = 10000, 100, 0.20
results = []

for trade_year in range(2014, 2021):
    train_start = 2008 if trade_year == 2020 else trade_year - 10

    train = thursday_df[(thursday_df['Date'].dt.year >= train_start) & (thursday_df['Date'].dt.year < trade_year)]
    test = thursday_df[thursday_df['Date'].dt.year == trade_year].copy()

    model = GradientBoostingClassifier(n_estimators=50, random_state=42)
    model.fit(train[features], train['Target'])

    test['Pred'] = model.predict(test[features])
    test['Signal'] = test['Pred'].map({1: 1, 0: -1})

    # Apply logic: If weekly move against signal > stop_loss, cap loss at $200k
    test['Raw_PnL'] = test['Signal'] * test['Target_Change'] * contracts * multiplier
    test['Final_PnL'] = test['Raw_PnL'].clip(lower=-stop_loss * contracts * multiplier)

    results.append(test)

final_df = pd.concat(results)
print(f"Enhanced Model PnL: ${final_df['Final_PnL'].sum():,.2f}")
print(f"Enhanced Accuracy: {len(final_df[final_df['Pred'] == final_df['Target']]) / len(final_df):.2%}")