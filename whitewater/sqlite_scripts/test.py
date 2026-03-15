import sqlite3
import pandas as pd
from pathlib import Path

# Path to your DB
db_path = Path.home() / "PyCharmProjects" / "QuantCode26" / "whitewater" / "whitewater.db"


def inspect_db():
    conn = sqlite3.connect(db_path)

    # 1. List all tables in the database
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
    print("Tables in Database:")
    print(tables['name'].tolist())
    print("-" * 30)

    # 2. Display the latest 5 rows for Waha (swap_wa)
    # We sort by date descending to see the most recent data first
    try:
        query = "SELECT * FROM swap_wa ORDER BY date ASC LIMIT 5"
        df_waha = pd.read_sql(query, conn)
        print("Latest Waha Swap Quotes (M1-M5):")
        print(df_waha)
    except Exception as e:
        print(f"Could not read swap_wa: {e}")

    conn.close()


if __name__ == "__main__":
    inspect_db()