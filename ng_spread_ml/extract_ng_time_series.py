import pandas as pd
from mssql_database import MSSQLDatabase
from utility import get_month_map


# Pull and filter March-April spread data with 13th-contract criterion
def extract_spread_prices(db: MSSQLDatabase, front_month='H',
                          back_month='J') -> pd.DataFrame:
    # Pull all NG futures data
    query = """
        SELECT Date, Contract, Price, MonthCode, Year
        FROM NG_Prices
        WHERE Commodity = 'NG'
    """
    df = db.query(query, parse_dates=['Date'])

    # Create an expiry date
    month_map = get_month_map()
    df['Expiry'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['MonthCode'].map(month_map) + '-01')

    # For each trading day, identify March as 13th contract and get corresponding April
    result_rows = []
    grouped = df.groupby('Date')

    for trade_date, group in grouped:
        sorted_group = group.sort_values('Expiry')  # Closest contract first
        front_row = group[group['MonthCode'] == 'H']
        back_row = group[group['MonthCode'] == 'J']

        # Make sure both March and April are in the list
        if not front_row.empty and not back_row.empty:
            march_expiry = front_row['Expiry'].values[0]
            position_of_march = sorted_group['Expiry'].tolist().index(march_expiry) + 1  # +1 for 1-based index

            if 1 <= position_of_march <= 13:
                result_rows.append({
                    'Date': trade_date,
                    'March_Contract': front_row['Contract'].values[0],
                    'April_Contract': back_row['Contract'].values[0],
                    'March_Price': front_row['Price'].values[0],
                    'April_Price': back_row['Price'].values[0],
                    'Spread': front_row['Price'].values[0] - back_row['Price'].values[0],
                    'March_Position': position_of_march
                })

    spread_df1 = pd.DataFrame(result_rows)
    return spread_df1


# === MAIN ===
if __name__ == "__main__":
    db_ = MSSQLDatabase()
    spread_df = extract_spread_prices(db_)

    # Filter only when March is the 13th contract
    filtered = spread_df[spread_df['March_Position'] == 13]

    # From each of those start dates, forward-fill until March becomes prompt (1st contract)
    final_rows = []

    for start_date in filtered['Date']:
        march_contract = filtered.loc[filtered['Date'] == start_date, 'March_Contract'].values[0]

        march_period = spread_df[
            (spread_df['March_Contract'] == march_contract) &
            (spread_df['Date'] >= start_date)
            ].copy()

        final_rows.append(march_period)

    final_df = pd.concat(final_rows).sort_values('Date').drop_duplicates()

    # Save to CSV
    final_df.to_csv("mar_apr_spread_filtered.csv", index=False)
    print("Saved mar_apr_spread_filtered.csv")

    db_.dispose()
