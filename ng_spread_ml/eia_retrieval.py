import requests
import pandas as pd


def get_eia_history(api_key):
    # This 'seriesid' route is the most reliable way to get
    # the full Lower 48 history (NW2_EPG0_SWO_R48_BCFW)
    url = f"https://api.eia.gov/v2/seriesid/NW2_EPG0_SWO_R48_BCFW/data?api_key={api_key}"

    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return

    # Parse data
    data = response.json()['response']['data']
    df = pd.DataFrame(data)

    # Standardize columns for our model
    df = df[['period', 'value']].rename(columns={
        'period': 'Date',
        'value': 'Storage_Bcf'
    })

    df['Date'] = pd.to_datetime(df['Date'])
    df['Storage_Bcf'] = pd.to_numeric(df['Storage_Bcf'])

    # Sort chronologically
    df = df.sort_values('Date')

    df.to_csv('eia_storage_v2.csv', index=False)
    print("Successfully saved eia_storage_v2.csv. Please upload it!")


# Use your key
get_eia_history('JnArFK2Bm0HHBAavKrhpfSEj11ubK131qL3JF5ls')