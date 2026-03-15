import os
import ssl
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from sklearn.decomposition import PCA

# --- IMPORT CONFIG FROM YOUR MODULE ---
# from uconn_ml_26.data_classes import TreasuryConfig
from data_classes import TreasuryConfig

# --- SSL FIX & PLOT SETUP ---
if not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
    ssl._create_default_https_context = ssl._create_unverified_context

try:
    matplotlib.use('TkAgg')
except:
    pass

#e = np.exp(1)

class TreasuryDataManager:
    """Handles data fetching from FRED and local persistence."""

    def __init__(self, config: TreasuryConfig):
        self.config = config
        from fredapi import Fred
        self.fred = Fred(api_key=self.config.api_key)
        self.file_path = self.config.data_dir / self.config.filename

    def get_data(self) -> pd.DataFrame:
        """Retrieves data from local cache or extracts from FRED."""
        if self.file_path.exists():
            print(f"Loading cached data from {self.file_path}...")
            return pd.read_csv(self.file_path, index_col=0, parse_dates=True)

        print("Cache not found. Starting extraction from FRED...")
        df_dict = {}
        for symbol, label in self.config.tenors.items():
            try:
                print(f"  Downloading {label}...")
                series = self.fred.get_series(symbol, observation_start=self.config.start_date)
                df_dict[label] = series
                time.sleep(0.1)
            except Exception as e:
                print(f"    Error on {label}: {e}")

        df = pd.DataFrame(df_dict).dropna()
        df.to_csv(self.file_path)
        print(f"Data saved to {self.file_path}")
        return df


class PCAAnalyzer:
    """Performs PCA and generates visualizations for financial analysis."""

    def __init__(self, data: pd.DataFrame):
        self.raw_data = data
        self.pca = None
        self.loadings = None
        self.filtered_df = None
        self.analysis_years = ""

    def run_analysis(self, start_year: int, end_year: int):
        """Filters data and performs PCA on daily changes (basis points)."""
        self.analysis_years = f"{start_year}_{end_year}"
        mask = (self.raw_data.index.year >= start_year) & (self.raw_data.index.year <= end_year)
        self.filtered_df = self.raw_data.loc[mask]

        if self.filtered_df.empty:
            print(f"Warning: No data found for {start_year}-{end_year}")
            return

        yield_changes = self.filtered_df.diff().dropna()
        n_components = min(len(self.filtered_df.columns), 10)
        self.pca = PCA(n_components=n_components)
        self.pca.fit(yield_changes)

        self.loadings = pd.DataFrame(
            self.pca.components_[:3].T,
            columns=['Level', 'Slope', 'Curvature'],
            index=self.filtered_df.columns
        )

        if self.loadings['Level'].mean() < 0:
            self.loadings *= -1

        print(f"\nAnalysis Results for {start_year}-{end_year}:")
        for i, name in enumerate(['Level', 'Slope', 'Curvature']):
            print(f"  {name}: {self.pca.explained_variance_ratio_[i]:.2%}")

    def plot_results(self, base_filename: str = None):
        """Generates and saves Scree plot and 3-frame loading bar charts."""
        if self.pca is None: return

        plot_dir = Path(__file__).parent / "plots"
        plot_dir.mkdir(exist_ok=True)

        # --- GRAPH 1: SCREE PLOT ---
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        exp_var = self.pca.explained_variance_ratio_
        pc_labels = [f'PC{i + 1}' for i in range(len(exp_var))]

        ax1.bar(pc_labels, exp_var, color='teal', alpha=0.6)
        ax1.set_title(f"Scree Plot: Explained Variance ({self.analysis_years})")
        ax1.set_ylabel("Variance Contribution")

        for i, v in enumerate(exp_var):
            ax1.text(i, v + 0.005, f'{v:.1%}', ha='center', fontweight='bold', fontsize=9)

        if base_filename:
            fig1.savefig(plot_dir / f"{base_filename}_Scree.png", dpi=300, bbox_inches='tight')
            print(f"Saved Scree Plot: {base_filename}_Scree.png")

        # --- GRAPH 2: 3-FRAME LOADINGS BAR CHART ---
        fig2, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        fig2.suptitle(f"PCA Factor Weights (Loadings)\nData Period: {self.analysis_years}",
                      fontsize=14, fontweight='bold')

        colors = ['#2ecc71', '#e67e22', '#9b59b6']
        factors = ['Level', 'Slope', 'Curvature']

        for i, (ax, factor, color) in enumerate(zip(axes, factors, colors)):
            ax.bar(self.loadings.index, self.loadings[factor], color=color, alpha=0.8)
            ax.axhline(0, color='black', lw=1, alpha=0.5)
            ax.set_ylabel(f"{factor} Weight")
            ax.set_title(f"Factor {i + 1}: {factor}", loc='left', fontsize=10, fontstyle='italic')
            ax.grid(axis='y', alpha=0.3)

        plt.xlabel("Tenor (Yield Curve Node Points)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if base_filename:
            fig2.savefig(plot_dir / f"{base_filename}_Loadings.png", dpi=300, bbox_inches='tight')
            print(f"Saved Loading Chart: {base_filename}_Loadings.png")

        plt.show()


# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    load_dotenv()

    config = TreasuryConfig()
    manager = TreasuryDataManager(config)
    raw_df = manager.get_data()

    analyzer = PCAAnalyzer(raw_df)

    # 5-Year Window Loop
    for start_year_ in [2000, 2005, 2010, 2015, 2020]:
        end_year_ = start_year_ + 5
        # Clean naming for files
        filename_ = f"PCA_{start_year_}_{end_year_}"

        analyzer.run_analysis(start_year=start_year_, end_year=end_year_)
        analyzer.plot_results(base_filename=filename_)

