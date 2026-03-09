import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from abc import ABC, abstractmethod
from pathlib import Path


class BasePCAAnalyzer(ABC):
    """
    Abstract base class for performing and visualizing PCA on financial data.
    Provides shared logic for variance analysis, factor loading visualization,
    and Scree plot generation.
    """

    def __init__(self, data: pd.DataFrame, factor_names: list = None):
        """
        Initialize with a DataFrame and custom names for the top 3 factors.
        """
        self.raw_data = data
        self.factor_names = factor_names or ['PC1', 'PC2', 'PC3']
        self.pca = None
        self.loadings = None
        self.filtered_df = None
        self.analysis_years = ""

    @abstractmethod
    def prepare_changes(self) -> pd.DataFrame:
        """
        Subclasses must implement this to define how they calculate
        daily changes (e.g., diff() for Rates, pct_change() for Equities).
        """
        pass

    def run_analysis(self, start_year: int, end_year: int):
        """
        Filters data for a specific regime, calculates changes,
        and fits the PCA model.
        """
        self.analysis_years = f"{start_year}_{end_year}"
        mask = (self.raw_data.index.year >= start_year) & (self.raw_data.index.year <= end_year)
        self.filtered_df = self.raw_data.loc[mask]

        if self.filtered_df.empty:
            print(f"Warning: No data found for period {self.analysis_years}")
            return

        # Prepare stationary data (internal centering is handled by PCA)
        changes = self.prepare_changes()

        # Fit PCA on the covariance matrix (default behavior)
        n_components = min(len(self.filtered_df.columns), 10)
        self.pca = PCA(n_components=n_components)
        self.pca.fit(changes)

        # Extract the Loadings for the first 3 components
        self.loadings = pd.DataFrame(
            self.pca.components_[:3].T,
            columns=self.factor_names,
            index=self.filtered_df.columns
        )

        self._standardize_signs()

        print(f"\nAnalysis Results: {self.analysis_years}")
        for i, name in enumerate(self.factor_names):
            print(f"  {name}: {self.pca.explained_variance_ratio_[i]:.2%}")

    def _standardize_signs(self):
        """
        Standardizes signs so the first PC (Market/Level) is generally
        represented as a positive move for financial clarity.
        """
        if self.loadings.iloc[:, 0].mean() < 0:
            self.loadings *= -1

    def plot_scree(self, title_prefix: str, save_path: Path = None):
        """
        Generates and optionally saves a Scree Plot with percentage labels.
        """
        if self.pca is None: return

        fig, ax = plt.subplots(figsize=(10, 4))
        exp_var = self.pca.explained_variance_ratio_
        pc_labels = [f'PC{i + 1}' for i in range(len(exp_var))]

        bars = ax.bar(pc_labels, exp_var, color='teal', alpha=0.6)
        ax.set_title(f"{title_prefix} Scree Plot: Explained Variance ({self.analysis_years})")
        ax.set_ylabel("Variance Contribution")

        # Add text labels above bars
        for i, bar in enumerate(bars):
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.005,
                    f'{yval:.1%}', ha='center', fontweight='bold', fontsize=9)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Scree plot saved: {save_path.name}")

        plt.show()

    def plot_results(self, title_prefix: str, save_path: Path = None):
        """
        Generates and optionally saves a 3-frame loading bar chart.
        """
        if self.pca is None: return

        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        fig.suptitle(f"{title_prefix} Factor Weights (Loadings)\nPeriod: {self.analysis_years}",
                     fontsize=14, fontweight='bold')

        # Distinct colors for Level, Slope, and Curvature
        colors = ['#2ecc71', '#e67e22', '#3498db']

        for i, (ax, factor, color) in enumerate(zip(axes, self.factor_names, colors)):
            ax.bar(self.loadings.index, self.loadings[factor], color=color, alpha=0.8)
            ax.axhline(0, color='black', lw=1, alpha=0.5)
            ax.set_ylabel(f"{factor} Weight")
            ax.set_title(f"Principal Component {i + 1}: {factor}", loc='left', fontsize=10, fontstyle='italic')
            ax.grid(axis='y', alpha=0.3)

        plt.xlabel("Constituents / Tenors")
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Loading chart saved: {save_path.name}")

        plt.show()