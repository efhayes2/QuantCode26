import json
import pandas as pd
import os
from pathlib import Path


def parse_eia_date(date_str):
    date_str = str(date_str)
    try:
        if len(date_str) == 8:  # Daily/Weekly (YYYYMMDD)
            return pd.to_datetime(date_str, format='%Y%m%d')
        elif len(date_str) == 6:  # Monthly (YYYYMM)
            return pd.to_datetime(date_str + '01', format='%Y%m%d')
        return pd.to_datetime(date_str)
    except:
        return pd.NaT


def extract_to_excel(source_path, target_series_id, output_path):
    data_points = []
    if not os.path.exists(source_path):
        return f"Error: Source {source_path} not found."

    with open(source_path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                if record.get("series_id") == target_series_id:
                    data_points = record.get("data", [])
                    break
            except json.JSONDecodeError:
                continue

    if data_points:
        df = pd.DataFrame(data_points, columns=['date', 'value'])
        df['date'] = df['date'].apply(parse_eia_date)
        df = df.dropna(subset=['date']).sort_values('date')
        df.to_excel(output_path, index=False)
        return f"Saved to {output_path}"
    return f"Error: {target_series_id} not found."


def main():
    # Adjusted to your environment paths from the traceback
    home = Path.home()
    data_dir = home / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    source = data_dir / "ng.txt"

    series_map = {
        "NG.N9050TX2.M": "supply.xlsx",
        "NG.NGM_EPG0_ENG_YSPL-Z00_MMCF.M": "demand.xlsx",
        "NG.NW2_EPG0_SSO_R33_BCF.W": "inventory.xlsx"
    }

    for sid, filename in series_map.items():
        print(extract_to_excel(source, sid, data_dir / filename))


if __name__ == "__main__":
    main()