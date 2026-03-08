from sqlalchemy import create_engine, text
import pandas as pd

class MSSQLDatabase:
    def __init__(self, server: str = r'QUANTUM-PC1\SQLEXPRESS',
                 database: str ='Uconn_HFT',
                 driver: str = 'ODBC Driver 17 for SQL Server'):
        self.connection_string = (
            f"mssql+pyodbc://{server}/{database}"
            f"?driver={driver.replace(' ', '+')}"
            f"&trusted_connection=yes"
        )
        self.engine = create_engine(self.connection_string, fast_executemany=True)

    def query(self, sql: str, parse_dates=None) -> pd.DataFrame:
        with self.engine.connect() as conn:
            return pd.read_sql(text(sql), conn, parse_dates=parse_dates)

    def dispose(self):
        self.engine.dispose()



def run():
    db = MSSQLDatabase()

    query = """
        SELECT Date, Contract, Price, Commodity, MonthCode, Year
        FROM NG_Prices
        WHERE Commodity = 'NG'
    """
    df = db.query(query, parse_dates=['Date'])
    print(df.head())

    db.close()

if __name__ == "__main__":
    run()

