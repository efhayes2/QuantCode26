import requests
import pandas as pd

# 1. Coordinates for major Gas Hubs (Proxy for US Lower 48 demand)
cities = {
    "New York": {"lat": 40.71, "lon": -74.01},
    "Chicago": {"lat": 41.85, "lon": -87.65},
    "Houston": {"lat": 29.76, "lon": -95.36},
    "Atlanta": {"lat": 33.75, "lon": -84.39},
    "Columbus": {"lat": 39.96, "lon": -83.00}
}


def get_forecast_hdds(cities_dict):
    all_data = []

    for city, coords in cities_dict.items():
        # Open-Meteo API (Free for non-commercial use)
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": coords["lat"],
            "longitude": coords["lon"],
            "daily": "temperature_2m_mean",
            "temperature_unit": "fahrenheit",
            "timezone": "America/New_York",
            "forecast_days": 14
        }

        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame({
                "Date": data["daily"]["time"],
                "Mean_Temp": data["daily"]["temperature_2m_mean"],
                "City": city
            })
            # HDD Formula: max(0, 65 - Mean_Temp)
            df["HDD"] = df["Mean_Temp"].apply(lambda x: max(0, 65 - x))
            all_data.append(df)

    if not all_data:
        return None

    full_df = pd.concat(all_data)

    # Create a simple Population-Weighted Proxy (Averaging these hubs)
    national_proxy = full_df.groupby("Date")["HDD"].mean().reset_index()
    national_proxy.rename(columns={"HDD": "Forecast_HDD"}, inplace=True)
    return national_proxy


# Execute and Save
forecast_hdds = get_forecast_hdds(cities)
if forecast_hdds is not None:
    forecast_hdds.to_csv('forecast_weather_hdds.csv', index=False)
    print("Success: 'forecast_weather_hdds.csv' has been created.")
    print(forecast_hdds.head())