import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TreasuryConfig:
    """Configuration for Treasury Data extraction and analysis."""
    api_key: str = field(default_factory=lambda: os.getenv('FRED_API_KEY'))
    start_date: str = "2000-01-01"
    end_date: str = "2026-12-31"
    data_dir: Path = Path(__file__).parent / "data"
    filename: str = "us_treasury_data.csv"

    tenors: dict = field(default_factory=lambda: {
        "DGS1MO": "1M", "DGS3MO": "3M", "DGS1": "1Y", "DGS2": "2Y",
        "DGS3": "3Y", "DGS5": "5Y", "DGS7": "7Y", "DGS10": "10Y",
        "DGS20": "20Y", "DGS30": "30Y"
    })

    def __post_init__(self):
        if not self.api_key:
            raise ValueError("FRED_API_KEY not found in environment or .env file.")
        self.data_dir.mkdir(exist_ok=True)
