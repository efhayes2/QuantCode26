import sqlite3
import pandas as pd
import os
from pathlib import Path

# --- 1. PATH CONFIGURATION (Mac/efh2) ---
# Using Path.home() to resolve to /Users/efh2/
base_project_dir = Path.home() / "PyCharmProjects" / "QuantCode26" / "whitewater"
data_dir = base_project_dir / "data" / "fixed_float_swap_curves"
db_path = base_project_dir / "whitewater.db"
scripts_dir = base_project_dir / "sqlite_scripts"

# Ensure the scripts directory exists
scripts_dir.mkdir(parents=True, exist_ok=True)

# The Hubs to process
hubs = ['AE', 'EP', 'HH', 'HS', 'KT', 'SE', 'SU', 'WA']


def load_fixed_float_swaps():
    # Connect to the SQLite DB
    conn = sqlite3.connect(db_path)
    print(f"Database: {db_path}")

    for hub in hubs:
        # File naming: FFWA.csv, FFHH.csv, etc.
        file_name = f"FF{hub}.csv"
        file_path = data_dir / file_name

        if not file_path.exists():
            print(f"Skipping {hub}: File not found at {file_path}")
            continue

        try:
            # 2. READ & PREP
            # We assume Col 1 is the Date, followed by M1, M2, M3, M4, M5
            df = pd.read_csv(file_path)

            # Standardize names: [date, M1, M2, M3, M4, M5]
            # This handles both 5 and 6 month files gracefully
            num_months = len(df.columns) - 1
            new_cols = ['date'] + [f'M{i}' for i in range(1, num_months + 1)]
            df.columns = new_cols

            # Format date to ISO string for proper SQL sorting/filtering
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

            # 3. UPLOAD TO SQL
            table_name = f"swap_{hub.lower()}"
            df.to_sql(table_name, conn, if_exists='replace', index=False)

            # 4. INDEX FOR PERFORMANCE
            # This makes joins between hubs (e.g. WA vs HH) instant
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_date ON {table_name}(date)")

            print(f"Loaded {table_name}: {len(df)} rows")

        except Exception as e:
            print(f"Error processing {hub}: {e}")

    conn.close()
    print("\n--- Data Upload Complete ---")


if __name__ == "__main__":
    load_fixed_float_swaps()