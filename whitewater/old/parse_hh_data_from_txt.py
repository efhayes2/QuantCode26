import json
import csv
import os
from pathlib import Path


def parse_ng_file(source_path, output_path):
    """
    Parses a multi-JSON text file for the Henry Hub Daily Price series
    and writes it to a clean CSV.
    """
    # The specific series_id for Henry Hub Daily Spot Price
    TARGET_SERIES = "NG.RNGWHHD.D"

    if not os.path.exists(source_path):
        print(f"Error: Source file {source_path} not found.")
        return

    extracted_data = []

    print(f"Scanning {source_path} for {TARGET_SERIES}...")

    with open(source_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                # Each line is an independent JSON object
                record = json.loads(line)

                if record.get("series_id") == TARGET_SERIES:
                    # Data is stored as a list of [date, value] pairs
                    # e.g., ["20260217", 3.13]
                    extracted_data = record.get("data", [])
                    print(f"Found {len(extracted_data)} data points for {TARGET_SERIES}.")
                    break  # Stop once we find the target series

            except json.JSONDecodeError:
                # Skip lines that aren't valid JSON (like header/footer fragments)
                continue

    if extracted_data:
        # Sort data by date ascending (EIA often provides newest first)
        extracted_data.sort(key=lambda x: x[0])

        # Write to CSV
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['date', 'hh_price'])  # Header
            writer.writerows(extracted_data)

        print(f"Successfully saved data to {output_path}")
    else:
        print(f"Error: Could not find series {TARGET_SERIES} in file.")


def main():
    # Define paths using home directory expansion
    data_dir = Path.home() / "data"
    source_file = data_dir / "ng.txt"
    output_file = data_dir / "HH_daily.csv"

    # Ensure the directory exists
    data_dir.mkdir(parents=True, exist_ok=True)

    parse_ng_file(source_file, output_file)


if __name__ == "__main__":
    main()