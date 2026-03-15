import sqlite3
import pandas as pd
from pathlib import Path

# --- 1. PATH CONFIGURATION ---
base_dir = Path.home() / "PyCharmProjects" / "QuantCode26" / "whitewater"
xlsx_path = base_dir / "data" / "trading_data.xlsx"
db_path = base_dir / "whitewater.db"


def load_trading_xlsx():
    # Connect to the DB
    conn = sqlite3.connect(db_path)
    print(f"Opening Excel: {xlsx_path}")

    try:
        # 2. LOAD ALL SHEETS
        # sheet_name=None returns a dictionary of {sheet_name: dataframe}
        excel_data = pd.read_excel(xlsx_path, sheet_name=None)

        for sheet_name, df in excel_data.items():
            # Clean table name (remove spaces, lowercase)
            table_name = f"phys_{sheet_name.lower().replace(' ', '_')}"

            # Clean Column Names (SQL doesn't like dots or spaces)
            df.columns = [c.strip().replace(' ', '_').replace('.', '_') for c in df.columns]

            # Ensure date column is formatted correctly if it exists
            # Looking for common date column names
            date_col = next((c for c in df.columns if 'date' in c.lower() or 'day' in c.lower()), None)
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col]).dt.strftime('%Y-%m-%d')
                print(f"Processing sheet '{sheet_name}' with date column '{date_col}'")

            # 3. UPLOAD
            df.to_sql(table_name, conn, if_exists='replace', index=False)

            # 4. INDEX ON DATE
            if date_col:
                conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_date ON {table_name}({date_col})")

            print(f"Loaded '{sheet_name}' into table: {table_name} ({len(df)} rows)")

    except Exception as e:
        print(f"Error loading Excel: {e}")
    finally:
        conn.close()
        print("\n--- Excel Migration Complete ---")


if __name__ == "__main__":
    load_trading_xlsx()