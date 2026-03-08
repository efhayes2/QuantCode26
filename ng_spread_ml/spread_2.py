import pandas as pd

from mssql_database import MSSQLDatabase
from utility import get_month_map, get_march_position_map, get_relative_contract_map


def extract_spread(db, front_contract='H', back_contract='J'):
    # Step 1: Pull all relevant data
    query = """
        SELECT Date, Contract, Price, MonthCode, Year
        FROM NG_prices
        WHERE Commodity = 'NG'
        ORDER BY Date, Year, MonthCode
    """
    df = db.query(query, parse_dates=['Date'])

    # Step 2: Create an expiry date column
    month_map = get_month_map()
    df['Expiry'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['MonthCode'].map(month_map) + '-01')

    # Step 3: Map from the first month to front position
    contract_position_map = get_relative_contract_map(front_contract)
        #get_march_position_map())

    # Step 4: Initialize lists
    prompt_rows = []
    next_rows = []

    # Step 5: Group by trading date
    for trade_date, group in df.groupby('Date'):
        group_sorted = group.sort_values('Expiry')
        first_contract = group_sorted.iloc[0]
        first_month_code = first_contract['MonthCode']

        front_contract_rows = group_sorted[group_sorted['MonthCode'] == front_contract]
        back_contract_rows = group_sorted[group_sorted['MonthCode'] == back_contract]

        front_contract_position = contract_position_map.get(first_month_code, None)

        if not front_contract_rows.empty and not back_contract_rows.empty:
            if first_month_code != front_contract:
                # Not prompt yet, use the current H and J with mapped position
                prompt_rows.append({
                    'Date': trade_date,
                    'Front_Contract': front_contract_rows.iloc[0]['Contract'],
                    'Back_Contract': back_contract_rows.iloc[0]['Contract'],
                    'Front_Price': front_contract_rows.iloc[0]['Price'],
                    'Back_Price': back_contract_rows.iloc[0]['Price'],
                    'Spread': front_contract_rows.iloc[0]['Price'] - back_contract_rows.iloc[0]['Price'],
                    'Front_Position': front_contract_position
                })

            else:
                # March is the prompt, extract both prompt and next contracts
                if len(front_contract_rows) >= 2 and len(back_contract_rows) >= 2:
                    c1 = front_contract_rows.iloc[0]
                    c2 = back_contract_rows.iloc[0]

                    prompt_rows.append({
                        'Date': trade_date,
                        'Front_Contract': c1['Contract'],
                        'Back_Contract': c2['Contract'],
                        'Front_Price': c1['Price'],
                        'Back_Price': c2['Price'],
                        'Spread': c1['Price'] - c2['Price'],
                        'Front_Position': 1
                    })

                    c1 = front_contract_rows.iloc[1]
                    c2 = back_contract_rows.iloc[1]

                    next_rows.append({
                        'Date': trade_date,
                        'Front_Contract': c1['Contract'],
                        'Back_Contract': c2['Contract'],
                        'Front_Price': c1['Price'],
                        'Back_Price': c2['Price'],
                        'Spread': c1['Price'] - c2['Price'],
                        'Front_Position': 13
                    })

    # Step 6: Convert lists to DataFrames
    prompt_df = pd.DataFrame(prompt_rows)
    next_df = pd.DataFrame(next_rows)

    # Optional: Save to CSV
    prompt_df.to_csv("prompt_spreads.csv", index=False)
    next_df.to_csv("next_spreads.csv", index=False)

    print("Saved prompt_spreads.csv and next_spreads.csv")
    return prompt_df, next_df

# === MAIN ===
if __name__ == "__main__":
    db = MSSQLDatabase()
    first = 'V'
    second = 'F'
    spread_df = extract_spread(db, first, second)

    # Step 1: Find the first date where March is the 13th contract
    start_dates = (
        spread_df[spread_df['March_Position'] == 13]
        .groupby('March_Contract')['Date']
        .min()
        .reset_index()
        .rename(columns={'Date': 'Start_Date'})
    )

    # Step 2: Find the first date where March becomes prompt (1st contract)
    end_dates = (
        spread_df[spread_df['March_Position'] == 1]
        .groupby('March_Contract')['Date']
        .min()
        .reset_index()
        .rename(columns={'Date': 'End_Date'})
    )

    # Step 3: Merge and build the windows
    periods = pd.merge(start_dates, end_dates, on='March_Contract', how='inner')

    final_rows = []
    for _, row in periods.iterrows():
        contract = row['March_Contract']
        start = row['Start_Date']
        end = row['End_Date']

        window = spread_df[
            (spread_df['March_Contract'] == contract) &
            (spread_df['Date'] >= start) &
            (spread_df['Date'] <= end)
        ].copy()

        final_rows.append(window)

    final_df = pd.concat(final_rows).sort_values('Date').drop_duplicates()

    # Save to CSV
    final_df.to_csv("mar_apr_spread_filtered_v2.csv", index=False)
    print("Saved mar_apr_spread_filtered_v2.csv")

    db.dispose()
