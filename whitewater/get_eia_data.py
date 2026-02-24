import os
import requests
import pandas as pd
from dotenv import load_dotenv

# Load the .env file
load_dotenv()
API_KEY = os.getenv('EIA_API_KEY')


def get_hh_data():
    # Series ID for Henry Hub Natural Gas Spot Price (Daily)
    series_id = 'NG.RNGAS.D'
    url = (f"https://api.eia.gov/v2/natural-gas/pri/fut/data/"
           f"?api_key={API_KEY}&frequency=daily&data[0]=value"
           f"&facets[series][]={series_id}&sort[0][column]=period&sort[0][direction]=desc")

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        df = pd.DataFrame(data['response']['data'])
        df = df[['period', 'value']].rename(columns={'period': 'date', 'value': 'hh_price'})
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


# Execute
df_hh = get_hh_data()
if df_hh is not None:
    print("Henry Hub Data Loaded Successfully:")
    print(df_hh.head())