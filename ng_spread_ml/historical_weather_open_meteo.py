import requests
import pandas as pd
import time

# Same 5 Gas Hubs
cities = {
    "New York": {"lat": 40.71, "lon": -74.01},
    "Chicago": {"lat": 41.85, "lon": -87.65},
    "Houston": {"lat": 29.76, "lon": -95.36},
    "Atlanta": {"lat": 33.75, "lon": -84.39},
    "Columbus": {"lat": 39.96, "lon": -83.00}
}


def pull_full_history(cities_dict, start_year, end_year):
    all_chunks = []

    # Loop through each year to keep requests manageable and avoid timeouts
    for year in range(start_year, end_year + 1):
        print(f"--- Processing Year: {year} ---")
        year_start = f"{year}-01-01"
        year_end = f"{year}-12-31"

        for city, coords in cities_dict.items():
            # Use the ARCHIVE API for deep history
            url = "https://archive-api.open-meteo.com/v1/archive"
            params = {
                "latitude": coords["lat"],
                "longitude": coords["lon"],
                "start_date": year_start,
                "end_date": year_end,
                "daily": "temperature_2m_mean",
                "temperature_unit": "fahrenheit",
                "timezone": "America/New_York"
            }

            try:
                r = requests.get(url, params=params)
                if r.status_code == 200:
                    data = r.json()
                    temp_df = pd.DataFrame({
                        "Date": data["daily"]["time"],
                        "Actual_Mean_Temp": data["daily"]["temperature_2m_mean"],
                        "City": city
                    })
                    # Calculate HDDs and CDDs
                    temp_df["HDD"] = temp_df["Actual_Mean_Temp"].apply(lambda x: max(0, 65 - x))
                    temp_df["CDD"] = temp_df["Actual_Mean_Temp"].apply(lambda x: max(0, x - 65))
                    all_chunks.append(temp_df)
                else:
                    print(f"Error {r.status_code} for {city} in {year}")

                time.sleep(0.5)  # Small pause to be a good API citizen

            except Exception as e:
                print(f"Request failed for {city}: {e}")

    if not all_chunks: return None

    # Aggregate and average into the National Proxy
    full_df = pd.concat(all_chunks)
    national_history = full_df.groupby("Date").agg({"HDD": "mean", "CDD": "mean"}).reset_index()
    return national_history


# EXECUTION: Get the full 20 years for your prompt_spreads.csv
history_df = pull_full_history(cities, 2004, 2024)

if history_df is not None:
    history_df.to_csv("full_weather_history_2004_2024.csv", index=False)
    print("Full 20-year weather file created.")