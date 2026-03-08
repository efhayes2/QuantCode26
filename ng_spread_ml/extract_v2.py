import pandas as pd
import os
from mssql_database import MSSQLDatabase
from utility import get_month_map, get_month_labels


# 1. Data Acquisition Layer
def get_raw_data(db: MSSQLDatabase = None, csv_path: str = None) -> pd.DataFrame:
    """
    Fetches raw price data from either a database or a CSV file.
    """
    if csv_path and os.path.exists(csv_path):
        print(f"Loading data from CSV: {csv_path}")
        df = pd.read_csv(csv_path, parse_dates=['Date', 'Expiry'])
    elif db:
        print("Loading data from MSSQL Database...")
        query = """
                SELECT Date, Contract, Price, MonthCode, Year
                FROM NG_Prices
                WHERE Commodity = 'NG'
                """
        df = db.query(query, parse_dates=['Date'])
    else:
        raise ValueError("No valid data source provided.")
    return df


# 2. Generalized Processing Layer
def extract_spread_prices(df: pd.DataFrame, front_code: str, back_code: str) -> pd.DataFrame:
    """
    Identifies all front-back pairs and calculates their position in the strip.
    """
    if 'Expiry' not in df.columns or df['Expiry'].dtype != 'datetime64[ns]':
        month_map = get_month_map()
        df['Expiry'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['MonthCode'].map(month_map) + '-01')

    result_rows = []
    grouped = df.groupby('Date')

    for trade_date, group in grouped:
        sorted_group = group.sort_values('Expiry')

        # Use the dynamic codes provided
        front_contracts = group[group['MonthCode'] == front_code]
        back_contracts = group[group['MonthCode'] == back_code]

        for _, f_row in front_contracts.iterrows():
            f_year = f_row['Year']
            f_expiry = f_row['Expiry']

            # Match the back contract (usually same year, or next if Dec/Jan)
            # For NG, H/J are same year.
            b_match = back_contracts[back_contracts['Year'] == f_year]

            if not b_match.empty:
                b_row = b_match.iloc[0]
                pos = sorted_group['Expiry'].tolist().index(f_expiry) + 1

                if 1 <= pos <= 13:
                    result_rows.append({
                        'Date': trade_date,
                        'Front_Contract': f_row['Contract'],
                        'Back_Contract': b_row['Contract'],
                        'Front_Price': f_row['Price'],
                        'Back_Price': b_row['Price'],
                        'Spread': f_row['Price'] - b_row['Price'],
                        'Front_Position': pos
                    })

    return pd.DataFrame(result_rows)


# === MAIN ===
if __name__ == "__main__":
    # USER INPUTS
    # 1. Provide only the codes
    front_code = 'Q'
    back_code = 'U'

    # 2. Automatically derive the labels
    front_label, back_label = get_month_labels(front_code, back_code)

    print(f"Processing Spread: {front_label} ({front_code}) vs {back_label} ({back_code})")
    USE_CSV = True
    CSV_FILE_PATH = os.path.join("data", "all_natgas_prices.csv")

    db_instance = None
    try:
        raw_df = get_raw_data(csv_path=CSV_FILE_PATH) if USE_CSV else get_raw_data(db=MSSQLDatabase())

        # Pass the codes into the extraction logic
        spread_df = extract_spread_prices(raw_df, front_code, back_code)

        if spread_df.empty:
            print(f"No valid {front_label}-{back_label} pairs found.")
        else:
            # Filter for when the front contract is the 13th in the strip
            filtered = spread_df[spread_df['Front_Position'] == 13]
            print(f"Found {len(filtered)} instances where {front_label} is at position 13.")

            final_rows = []
            for start_date in filtered['Date'].unique():
                # Identify the specific contract name that was at position 13
                target_contract = filtered.loc[filtered['Date'] == start_date, 'Front_Contract'].values[0]

                # Track that specific contract from that date forward
                contract_lifecycle = spread_df[
                    (spread_df['Front_Contract'] == target_contract) &
                    (spread_df['Date'] >= start_date)
                    ].copy()

                final_rows.append(contract_lifecycle)

            if final_rows:
                final_df = pd.concat(final_rows).sort_values('Date').drop_duplicates()
                # output_name = f"{front_label.lower()}_{back_label.lower()}_spread_filtered.csv"
                output_name = f'prompt_spreads_' + front_code + back_code + '.csv'
                output_name = 'data\\' + output_name if not output_name.startswith('data\\') else output_name
                final_df.to_csv(output_name, index=False)
                print(f"Successfully saved {output_name}")

    finally:
        if db_instance:
            db_instance.dispose()