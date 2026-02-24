import json
import pandas as pd
import os
from pathlib import Path


def parse_eia_date(date_str):
    """
    Handles EIA's specific date formats:
    - YYYYMMDD (8 digits: Daily/Weekly)
    - YYYYMM (6 digits: Monthly)
    - YYYY (4 digits: Annual)
    """
    ds = str(date_str)
    try:
        if len(ds) == 8:
            return pd.to_datetime(ds, format='%Y%m%d')
        elif len(ds) == 6:
            return pd.to_datetime(ds + '01', format='%Y%m%d')
        elif len(ds) == 4:
            return pd.to_datetime(ds + '0101', format='%Y%m%d')
        else:
            return pd.to_datetime(ds)
    except:
        return pd.NaT


def extract_series(source_path, target_id, cutoff_date='2016-02-01'):
    """Parses source for a specific ID and returns a filtered DataFrame."""
    data_points = []
    if not os.path.exists(source_path):
        print(f"Error: {source_path} not found.")
        return None

    with open(source_path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                if record.get("series_id") == target_id:
                    data_points = record.get("data", [])
                    break
            except json.JSONDecodeError:
                continue

    if not data_points:
        return None

    df = pd.DataFrame(data_points, columns=['date', 'value'])
    df['date'] = df['date'].apply(parse_eia_date)

    # Apply Cutoff: February 1, 2016
    df = df[df['date'] >= cutoff_date]
    return df.sort_values('date')


def main():
    # Adjust to your local PyCharm/User data path
    home = Path.home()
    data_dir = home / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Ensure this matches your file name (case-sensitive on some systems)
    source_file = data_dir / "ng.txt"

    # Mapping for all four files
    series_map = {
        "NG.RNGWHHD.D": "HH_daily.xlsx",  # Henry Hub Daily
        "NG.N9050TX2.M": "supply.xlsx",  # TX Marketed Production
        "NG.NGM_EPG0_ENG_YSPL-Z00_MMCF.M": "demand.xlsx",  # Sabine Pass LNG Exports
        "NG.NW2_EPG0_SSO_R33_BCF.W": "inventory.xlsx"  # South Central Salt Storage
    }

    dataframes = {}
    for sid, filename in series_map.items():
        print(f"Processing {sid}...")
        df = extract_series(source_file, sid)
        if df is not None and not df.empty:
            output_path = data_dir / filename
            df.to_excel(output_path, index=False)
            dataframes[sid] = df
            print(f"  -> Saved {len(df)} records to {filename}")
        else:
            print(f"  -> Warning: No data found for {sid} after 2016-02-01.")

    # Create master_aligned.xlsx with separate sheets
    if dataframes:
        master_path = data_dir / "master_aligned.xlsx"
        with pd.ExcelWriter(master_path) as writer:
            for sid, df in dataframes.items():
                # Name the sheet after the filename (e.g., 'HH_daily')
                sheet_name = series_map[sid].split('.')[0]
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"\nMaster workbook created at: {master_path}")
    else:
        print("\nProcess complete: No fundamental data was found to save.")


if __name__ == "__main__":
    main()