import json
import pandas as pd
from pathlib import Path


def parse_eia_date(date_str):
    """Standard EIA date parser for YYYYMM (Monthly) or YYYYMMDD (Daily)."""
    date_str = str(date_str)
    if len(date_str) == 6:
        return pd.to_datetime(date_str, format='%Y%m')
    elif len(date_str) == 8:
        return pd.to_datetime(date_str, format='%Y%m%d')
    return pd.to_datetime(date_str, errors='coerce')


def extract_to_excel():
    # 1. Setup Project Paths
    data_dir = Path("~/PyCharmProjects/QuantCode26/whitewater/data").expanduser()
    source_file = data_dir / "NG.txt"
    output_file = data_dir / "demand_pull.xlsx"

    if not source_file.exists():
        print(f"Error: NG.txt not found at {source_file}")
        return

    # 2. Define the Texas LNG Hubs (Demand Pull)
    series_map = {
        "Sabine_Pass": "NG.NGM_EPG0_ENG_YSPL-Z00_MMCF.M",
        "Freeport": "NG.NGM_EPG0_ENG_YFPT-Z00_MMCF.M",
        "Corpus_Christi": "NG.NGM_EPG0_ENG_YCRP-Z00_MMCF.M"
    }

    print("Extracting multi-terminal demand data...")

    data_frames = {}

    # Stream the EIA file once to find all requested series
    with open(source_file, 'r') as f:
        for line in f:
            try:
                record = json.loads(line)
                sid = record.get('series_id')
                # Match the series ID to our friendly names
                for name, target_sid in series_map.items():
                    if sid == target_sid:
                        df = pd.DataFrame(record['data'], columns=['date', 'value'])
                        df['date'] = df['date'].apply(parse_eia_date)
                        # Filter for post-Jan 2016 era
                        df = df[df['date'] > '2016-01-31'].dropna().sort_values('date')
                        data_frames[name] = df
            except:
                continue

    # 3. Create the Combined Total (Sheet1)
    # We merge all terminals to ensure we sum them correctly on the same dates
    combined_df = None
    for name, df in data_frames.items():
        temp_df = df.rename(columns={'value': name})
        if combined_df is None:
            combined_df = temp_df
        else:
            combined_df = pd.merge(combined_df, temp_df, on='date', how='outer')

    # Fill NaNs with 0 (for years before a terminal was online)
    combined_df = combined_df.fillna(0)

    # Sum the columns for the 'Total Demand Pull'
    terminal_names = list(data_frames.keys())
    combined_df['value'] = combined_df[terminal_names].sum(axis=1)

    # 4. Save to Workbook with individual tabs
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        # Combined Total (Legacy requirement for other scripts)
        combined_df[['date', 'value']].to_excel(writer, sheet_name='Sheet1', index=False)

        # Individual Hub tabs for the Data Memo/Analysis
        for name, df in data_frames.items():
            df.to_excel(writer, sheet_name=name, index=False)

    print(f"Success! Integrated demand workbook saved to: {output_file}")


if __name__ == "__main__":
    extract_to_excel()