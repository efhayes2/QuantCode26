import pandas as pd
import os


def main():
    # 1. Define the file path
    file_path = os.path.join('data', 'weather_data.csv')

    # 2. Load the data
    # parse_dates converts the 'day' column from a string to a datetime object
    df = pd.read_csv(file_path, parse_dates=['day'])

    # 3. Basic Inspection
    print("--- Dataset Overview ---")
    print(df.info())

    print("\n--- First 5 Rows ---")
    print(df.head())

    # 4. Filter for Freeze-Off risk (Midland < 25°F)
    # This helps you see immediately if your 'Waha supply risk' data is in there
    maf_freezes = df[(df['station'] == 'MAF') & (df['min_temp_f'] < 25)]

    print(f"\n--- Found {len(maf_freezes)} days with Midland Min Temp < 25°F ---")
    if not maf_freezes.empty:
        print(maf_freezes[['day', 'min_temp_f']].head())


if __name__ == "__main__":
    main()