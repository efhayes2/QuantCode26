import requests
import pandas as pd
import numpy as np
import time

# 1. Major Hubs (Weighted toward high gas consumption and power gen)
cities = {
    "New York": {"lat": 40.71, "lon": -74.01},
    "Chicago": {"lat": 41.85, "lon": -87.65},
    "Houston": {"lat": 29.76, "lon": -95.36},  # Heavy CDD impact
    "Atlanta": {"lat": 33.75, "lon": -84.39},
    "Columbus": {"lat": 39.96, "lon": -83.00}
}


def get_historical_forecast_weather(cities_dict, start_date, end_date):
    all_city_dfs = []

    for city, coords in cities_dict.items():
        print(f"Fetching historical forecasts for {city}...")
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

                # --- HDD Calculation (Heating) ---
                df["HDD"] = df["Forecast_Mean_Temp"].apply(lambda x: max(0, 65 - x))

                # --- CDD Calculation (Cooling) ---
                df["CDD"] = df["Forecast_Mean_Temp"].apply(lambda x: max(0, x - 65))

                all_city_dfs.append(df)
            else:
                print(f"Failed for {city}: {response.status_code}")

            time.sleep(1)  # Rate limit protection

        except Exception as e:
            print(f"Error fetching {city}: {e}")

    if not all_city_dfs:
        return None

    full_data = pd.concat(all_city_dfs)

    # 2. National Proxy Averaging
    national_proxy = full_data.groupby("Date").agg({
        "HDD": "mean",
        "CDD": "mean"
    }).reset_index()

    # Calculate Total Degree Days (TDD) - The "Whole Market" metric
    national_proxy["Forecast_TDD"] = national_proxy["HDD"] + national_proxy["CDD"]

    return national_proxy


# --- EXECUTION ---
START = "2022-01-01"
END = "2023-12-31"

historical_weather = get_historical_forecast_weather(cities, START, END)

if historical_weather is not None:
    filename = 'historical_forecast_weather_full.csv'
    historical_weather.to_csv(filename, index=False)
    print(f"\nSuccess! File saved as: {filename}")
    print(historical_weather.tail())