import pandas as pd
import numpy as np
import os


def engineer_physical_features(df):
    """
    Applies the Master Basis Strategy physical overlays.
    """
    # 1. Seasonal Thermal Efficiency (STE)
    # Using 65F as the pivot point where pipeline compressor efficiency begins to degrade
    df['STE'] = 1.0 - (0.002 * (df['min_temp_f_MAF'] - 65).clip(lower=0))

    # 2. Basis Pressure (The 'Normalized' Spread)
    # This identifies when spreads are widening due to physical friction
    df['bp_hh'] = df['target_waha_hh'] / df['STE']
    df['bp_katy'] = df['target_waha_katy'] / df['STE']
    df['bp_hsc'] = df['target_waha_hsc'] / df['STE']

    # 3. Cross-Hub Divergence (HH vs Katy)
    # A high divergence flags a localized bottleneck vs. a regional Waha blowout
    df['div_hh_katy'] = df['target_waha_hh'] - df['target_waha_katy']

    # 4. Blowout Risk Index
    # Binary trigger: High Temp (Low STE) + High Spread Level
    df['blowout_risk'] = ((df['STE'] < 0.96) & (df['target_waha_katy'] > 1.25)).astype(int)

    return df


if __name__ == "__main__":
    # 1. Setup paths
    data_dir = os.path.expanduser('~/data/pngs/')
    source_file = os.path.join(data_dir, 'whitewater_current.csv')

    if not os.path.exists(source_file):
        print(f"Error: Could not find {source_file}")
    else:
        # 2. Load Data
        print(f"Loading data from {source_file}...")
        df = pd.read_csv(source_file, parse_dates=['day'])
        df.set_index('day', inplace=True)

        # 3. Apply Engineering
        print("Applying Seasonal Thermal Efficiency and Basis Pressure logic...")
        df = engineer_physical_features(df)

        # 4. Validation Output
        print("\n--- Physical Feature Preview ---")
        cols_to_show = ['min_temp_f_MAF', 'STE', 'target_waha_katy', 'bp_katy', 'blowout_risk']
        # Show a slice of a hot period to see STE in action
        hot_days = df[df['min_temp_f_MAF'] > 85][cols_to_show].head(10)

        if not hot_days.empty:
            print(hot_days)
        else:
            print(df[cols_to_show].tail(10))

        # 5. Save for ww_strat_3 preview
        output_path = os.path.join(data_dir, 'whitewater_physics_ready.csv')
        df.to_csv(output_path)
        print(f"\nSaved physics-ready data to: {output_path}")