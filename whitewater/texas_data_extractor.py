import json
import pandas as pd
from pathlib import Path
import os


def extract_to_excel(source_path, series_id_map, output_dir):
    """
    Scans the EIA TXT file for specific hub IDs and saves them to Excel.
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    if not source_path.exists():
        print(f"ERROR: Could not find file at: {source_path}")
        print(f"Current Working Directory: {os.getcwd()}")
        return

    print(f"SUCCESS: Found {source_path.name}. Starting extraction...")
    found_count = 0

    with open(source_path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                sid = record.get("series_id")

                if sid in series_id_map:
                    name = series_id_map[sid]
                    data = record.get("data", [])

                    if not data:
                        print(f"  [EMPTY] {sid} ({name}) has no data points.")
                        continue

                    df = pd.DataFrame(data, columns=['date', 'value'])

                    # EIA dates are YYYYMMDD strings
                    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
                    df = df.dropna(subset=['date']).sort_values('date')

                    # Filter for the 'Modern Regime' (Post-2016)
                    df = df[df['date'] >= '2016-02-01']

                    if df.empty:
                        print(f"  [EMPTY] {name} has no data after 2016-02-01.")
                        continue

                    output_file = output_dir / f"{name}.xlsx"
                    df.to_excel(output_file, index=False)
                    print(f"  [SAVED] {sid} -> {output_file.name} ({len(df)} rows)")
                    found_count += 1

            except json.JSONDecodeError:
                continue

    if found_count == 0:
        print("\nWARNING: Extraction finished but no matching Series IDs were found.")
        print("Check if the Series IDs in the script match the IDs in your NG.txt file.")
    else:
        print(f"\nFinished. Extracted {found_count} hub datasets to {output_dir}")


def main():
    # .expanduser() converts '~' to your Mac's home path (e.g., /Users/efh2)
    source_file = Path("~/data/NG.txt").expanduser()
    output_dir = Path("~/data").expanduser()

    # The Specific EIA Series IDs for Texas Spot Hubs (Daily)
    # Waha: Permian Supply Push
    # Katy: Houston Demand/Hub Center
    # Agua Dulce: South Texas/LNG/Mexico Gate
    # HH: The Global Benchmark
    hubs = {
        "NG.RNGCWHP.D": "waha_spot",
        "NG.RNGCKTP.D": "katy_spot",
        "NG.RNGCADP.D": "agua_dulce_spot",
        "NG.RNGWHHD.D": "hh_spot"
    }

    extract_to_excel(source_file, hubs, output_dir)


if __name__ == "__main__":
    main()