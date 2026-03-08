import pandas as pd


def get_storage_from_html():
    # This is the direct link to the historical table page
    url = "https://www.eia.gov/dnav/ng/hist/nw2_epg0_swo_r48_bcfw.htm"

    try:
        # read_html returns a list of all tables on the page
        tables = pd.read_html(url)

        # The data is usually in the second or third table
        # We look for the table that has 'Year' and 'Jan' etc.
        for df in tables:
            if 'Year' in df.columns:
                # This is a 'grid' format, we need to melt it into a time series
                # Melt months into rows
                df = df.melt(id_vars='Year', var_name='Month', value_name='Storage_Bcf')

                # Create a date column
                # Note: EIA uses '15-Jan' style or similar
                df = df.dropna()
                # Clean up: remove row headers like 'Week Ending' if they exist
                df = df[df['Storage_Bcf'].apply(lambda x: str(x).isdigit())]

                print("Successfully scraped data from HTML.")
                df.to_csv('eia_storage_v2.csv', index=False)
                return df

        # If the grid parsing is too complex, we try the API one last time
        # with a different library-agnostic URL construction
        print("Could not parse HTML grid. Trying one last API method...")
    except Exception as e:
        print(f"HTML Scrape failed: {e}")


get_storage_from_html()