import requests
import pandas as pd
import numpy as np
import time

# 1. Coordinates for the same major Gas Hubs
cities = {
    "New York": {"lat": 40.71, "lon": -74.01},
    "Chicago": {"lat": 41.85, "lon": -87.65},
    "Houston": {"lat": 29.76, "lon": -95.36},
    "Atlanta": {"lat": 33.75, "lon": -84.39},
    "Columbus": {"lat": 39.96, "lon": -83.00}
}


def get_historical_forecast_data(cities_dict, start_date, end_date):
    """
    Fetches what the models were PREDICTING (Historical Forecast) for each city.
    Dates must be in YYYY-MM-DD format.
    """
    all_city_dfs = []

    for city, coords in cities_dict.items():
        print(f"Fetching historical forecasts for {city}...")

        # Open-Meteo Historical Forecast API Endpoint
        url = "https://historical-forecast-api.open-meteo.com/v1/forecast"

        params = {
            "latitude": coords["lat"],
            "longitude": coords["lon"],
            "start_date": start_date,
            "end_date": end_date,
            "daily": "temperature_2m_mean",
            "temperature_unit": "fahrenheit",
            "timezone": "America/New_York"
        }

        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame({
                    "Date": data["daily"]["time"],
                    "Forecast_Mean_Temp": data["daily"]["temperature_2m_mean"],
                    "City": city
                })
                # Calculate HDD for this city's forecast
                df["HDD"] = df["Forecast_Mean_Temp"].apply(lambda x: max(0, 65 - x))
                all_city_dfs.append(df)
            else:
                print(f"Failed for {city}: {response.status_code}")

            # Respect API rate limits
            time.sleep(1)

        except Exception as e:
            print(f"Error fetching {city}: {e}")

    if not all_city_dfs:
        return None

    # 2. Averaging the cities to create the National Proxy
    full_data = pd.concat(all_city_dfs)
    national_proxy = full_data.groupby("Date")["HDD"].mean().reset_index()
    national_proxy.rename(columns={"HDD": "National_Forecast_HDD"}, inplace=True)

    return national_proxy


# --- EXECUTION ---
# Set your range (Note: Open-Meteo free tier historical forecast
# typically goes back to mid-2021)
START = "2022-01-01"
END = "2023-12-31"

historical_hdds = get_historical_forecast_data(cities, START, END)

if historical_hdds is not None:
    # 3. Write to CSV
    filename = 'historical_forecast_hdds.csv'
    historical_hdds.to_csv(filename, index=False)
    print(f"\nSuccess! File saved as: {filename}")
    print(historical_hdds.head())
else:
    print("No data was acquired.")