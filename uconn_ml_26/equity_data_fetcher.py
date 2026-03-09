import pandas as pd
import yfinance as yf
from pathlib import Path


def fetch_equity_sector_data():
    # The Original 9 Select Sector SPDR ETFs (Launched 1998)
    # This allows for a clean backtest to the early 2000s
    tickers = {
        'XLK': 'Technology',
        'XLF': 'Financials',
        'XLV': 'Healthcare',
        'XLY': 'Cons_Disc',
        'XLI': 'Industrials',
        'XLP': 'Cons_Staples',
        'XLE': 'Energy',
        'XLU': 'Utilities',
        'XLB': 'Materials'
    }

    print(f"Fetching 20+ years of data for {len(tickers)} sectors...")

    # auto_adjust=True ensures we get dividend/split adjusted prices in 'Close'
    raw_data = yf.download(
        list(tickers.keys()),
        start="2002-01-01",
        end="2026-12-31",
        auto_adjust=True
    )

    # Select the adjusted Close prices
    data = raw_data['Close']

    # Rename columns to Sector Names
    data = data.rename(columns=tickers)

    # Drop missing values to align all series
    df = data.dropna()

    # Save to your data directory
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "equity_data.csv"

    df.to_csv(output_path)
    print(f"Saved {len(df)} rows of equity data starting from {df.index.min().date()}")


if __name__ == "__main__":
    fetch_equity_sector_data()