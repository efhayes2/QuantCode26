import pandas as pd
import os
from mssql_database import MSSQLDatabase
from utility import get_month_map


# 1. Data Acquisition Layer
def get_raw_data(db: MSSQLDatabase = None, csv_path: str = None) -> pd.DataFrame:
    """
    Fetches raw price data from either a database or a CSV file.
    """
    if csv_path and os.path.exists(csv_path):
        print(f"Loading data from CSV: {csv_path}")
        # CSV structure includes Expiry, so we parse Date and Expiry as dates
        df = pd.read_csv(csv_path, parse_dates=['Date', 'Expiry'])
    elif db:
        print("Loading data from MSSQL Database...")
        query = """
                SELECT Date, Contract, Price, MonthCode, Year
                FROM NG_Prices
                WHERE Commodity = 'NG' \
                """
        df = db.query(query, parse_dates=['Date'])
    else:
        raise ValueError("No valid data source provided. Provide a DB instance or a CSV path.")

    return df


# 2. Corrected Processing Layer
def extract_spread_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies all March-April pairs and calculates their position in the strip.
    """
    # Ensure Expiry exists and is a datetime object
    if 'Expiry' not in df.columns or df['Expiry'].dtype != 'datetime64[ns]':
        month_map = get_month_map()
        df['Expiry'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['MonthCode'].map(month_map) + '-01')

    result_rows = []
    grouped = df.groupby('Date')

    for trade_date, group in grouped:
        # Sort current date strip by Expiry to determine relative positions
        sorted_group = group.sort_values('Expiry')

        # Get all March ('H') and April ('J') contracts for this date
        marches = group[group['MonthCode'] == 'H']
        aprils = group[group['MonthCode'] == 'J']

        # Match March and April contracts by year
        for _, march_row in marches.iterrows():
            m_year = march_row['Year']
            m_expiry = march_row['Expiry']

            # Find the corresponding April contract for the same year
            april_match = aprils[aprils['Year'] == m_year]

            if not april_match.empty:
                a_row = april_match.iloc[0]

                # Find the 1-based position of this March contract in the strip
                # index() returns 0-based, so we add 1
                pos = sorted_group['Expiry'].tolist().index(m_expiry) + 1

                # Include all spreads within the range (up to position 13)
                if 1 <= pos <= 13:
                    result_rows.append({
                        'Date': trade_date,
                        'March_Contract': march_row['Contract'],
                        'April_Contract': a_row['Contract'],
                        'March_Price': march_row['Price'],
                        'April_Price': a_row['Price'],
                        'Spread': march_row['Price'] - a_row['Price'],
                        'March_Position': pos
                    })

    return pd.DataFrame(result_rows)


# === MAIN ===
if __name__ == "__main__":
    # CONFIGURATION
    USE_CSV = True
    # Relative path as requested
    CSV_FILE_PATH = os.path.join("data", "all_natgas_prices.csv")

    db_instance = None
    try:
        if USE_CSV:
            raw_df = get_raw_data(csv_path=CSV_FILE_PATH)
        else:
            db_instance = MSSQLDatabase()
            raw_df = get_raw_data(db=db_instance)

        # Extraction logic
        spread_df = extract_spread_prices(raw_df)

        if spread_df.empty:
            print("No valid March-April pairs found.")
        else:
            # Filter only when March is exactly at the 13th position
            filtered = spread_df[spread_df['March_Position'] == 13]
            print(f"Found {len(filtered)} rows where March is the 13th contract.")

            # From each of those start dates, follow the contract until it becomes prompt
            final_rows = []
            for start_date in filtered['Date'].unique():
                # Identify the specific contract name that was at position 13 on this date
                march_contract = filtered.loc[filtered['Date'] == start_date, 'March_Contract'].values[0]

                # Filter spread_df for that specific contract from that date forward
                march_period = spread_df[
                    (spread_df['March_Contract'] == march_contract) &
                    (spread_df['Date'] >= start_date)
                    ].copy()

                final_rows.append(march_period)

            if final_rows:
                final_df = pd.concat(final_rows).sort_values('Date').drop_duplicates()
                # Save to CSV
                final_df.to_csv("mar_apr_spread_filtered.csv", index=False)
                print("Successfully saved mar_apr_spread_filtered.csv")
            else:
                print("Process completed but no final rows matched the criteria.")

    finally:
        if db_instance:
            db_instance.dispose()