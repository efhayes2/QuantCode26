import pandas as pd
import os
from pathlib import Path


def main():
    home = Path.home()
    data_dir = home / "data"

    # Load and Fix Date Parsing for Excel Files
    def load_eia_excel(path):
        df = pd.read_excel(path)
        # Handle 6-digit (YYYYMM) and 8-digit (YYYYMMDD) strings
        df['date'] = df['date'].astype(str).apply(
            lambda x: pd.to_datetime(x + '01', format='%Y%m%d') if len(x) == 6 else pd.to_datetime(x)
        )
        return df.sort_values('date')

    print("Loading datasets...")
    df_hh = pd.read_csv(data_dir / "HH_daily.csv")
    df_hh['date'] = pd.to_datetime(df_hh['date'])
    df_hh = df_hh.sort_values('date')

    df_supply = load_eia_excel(data_dir / "supply.xlsx").rename(columns={'value': 'tx_production'})
    df_demand = load_eia_excel(data_dir / "demand.xlsx").rename(columns={'value': 'lng_exports'})
    df_inv = load_eia_excel(data_dir / "inventory.xlsx").rename(columns={'value': 'salt_storage'})

    # Perform Merges
    # 'nearest' is safer for backtesting when series have different start dates
    print("Aligning...")
    master = pd.merge_asof(df_hh, df_supply, on='date', direction='nearest')
    master = pd.merge_asof(master, df_demand, on='date', direction='nearest')
    master = pd.merge_asof(master, df_inv, on='date', direction='nearest')

    # Calculate Features
    master['market_tightness'] = master['tx_production'] - master['lng_exports']
    master['storage_wow'] = master['salt_storage'].diff(5)

    # Instead of dropna(), let's see where the data starts
    initial_count = len(master)
    master = master.dropna()
    final_count = len(master)

    print(f"Merged {initial_count} rows. Retained {final_count} rows after cleaning.")

    if final_count > 0:
        master.to_csv(data_dir / "master_backtest.csv", index=False)
        print(f"Success! Latest date: {master['date'].max()}")
    else:
        print("WARNING: Master DataFrame is still empty. Check date overlaps.")
        print(f"HH Start: {df_hh['date'].min()}, Supply Start: {df_supply['date'].min()}")


if __name__ == "__main__":
    main()