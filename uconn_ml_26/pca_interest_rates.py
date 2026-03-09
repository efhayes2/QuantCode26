import pandas as pd
from pathlib import Path
from uconn_ml_26.base_analyzer import BasePCAAnalyzer
from dotenv import load_dotenv


class InterestRateAnalyzer(BasePCAAnalyzer):
    """PCA Analyzer for Treasury Yield Curves (Constant Maturity Rates)."""

    def prepare_changes(self) -> pd.DataFrame:
        """
        Interest rates are analyzed using daily differences (Basis Points)
        to capture the term structure of volatility.
        """
        return self.filtered_df.diff().dropna()


if __name__ == "__main__":
    load_dotenv()

    # Setup Paths
    base_path = Path(__file__).parent
    data_path = base_path / "data" / "us_treasury_data.csv"
    plot_dir = base_path / "plots"
    plot_dir.mkdir(exist_ok=True)

    if data_path.exists():
        # Load the cached data from your earlier extraction
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)

        # Initialize subclass
        analyzer = InterestRateAnalyzer(
            df,
            factor_names=['Level', 'Slope', 'Curvature']
        )

        # Execute 5-year rolling analysis
        for start in [2000, 2005, 2010, 2015, 2020]:
            end = start + 5
            analyzer.run_analysis(start, end)

            # Define distinct filenames
            scree_file = plot_dir / f"Rates_{start}_{end}_Scree.png"
            loadings_file = plot_dir / f"Rates_{start}_{end}_Loadings.png"

            # Generate and save both diagnostic charts
            # plot_scree is now available via inheritance from BasePCAAnalyzer
            analyzer.plot_scree(title_prefix="US Treasury", save_path=scree_file)
            analyzer.plot_results(title_prefix="US Treasury", save_path=loadings_file)

        print("\nAll Treasury PCA plots have been saved to the /plots directory.")
    else:
        print(f"CRITICAL: Data file not found at {data_path}.")
        print("Please run your data extraction script (pca_fred_v2.py) first.")