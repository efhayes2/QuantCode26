import pandas as pd
from pathlib import Path
from uconn_ml_26.base_analyzer import BasePCAAnalyzer


class EquityAnalyzer(BasePCAAnalyzer):
    """PCA Analyzer for Equity Indices using Sector ETFs."""

    def prepare_changes(self) -> pd.DataFrame:
        """
        Equities require percentage changes (returns) to normalize
        for different absolute price levels.
        """
        return self.filtered_df.pct_change().dropna()


if __name__ == "__main__":
    base_path = Path(__file__).parent
    data_path = base_path / "data" / "equity_data.csv"
    plot_dir = base_path / "plots"
    plot_dir.mkdir(exist_ok=True)

    if data_path.exists():
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)

        # Factor names tailored for Equity Market Theory
        analyzer = EquityAnalyzer(
            df,
            factor_names=['Market (Beta)', 'Value vs Growth', 'Cyclical vs Defensive']
        )

        # 5-Year Rolling Windows (2005 - 2025)
        for start in [2005, 2010, 2015, 2020]:
            end = start + 5
            analyzer.run_analysis(start, end)

            # Filenames for the two PNG types
            scree_file = plot_dir / f"Equity_{start}_{end}_Scree.png"
            loadings_file = plot_dir / f"Equity_{start}_{end}_Loadings.png"

            # Generate and save both
            analyzer.plot_scree(title_prefix="S&P 500 Sector", save_path=scree_file)
            analyzer.plot_results(title_prefix="S&P 500 Sector", save_path=loadings_file)

        print(f"\nSuccess: Equity plots generated in {plot_dir}")
    else:
        print(f"Error: Data not found at {data_path}")