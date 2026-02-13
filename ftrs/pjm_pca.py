import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class PJMPCAQuant:
    def __init__(self, n_components=3):
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()

    def prepare_data(self, df):
        # PCA is sensitive to scale; always standardize
        scaled_data = self.scaler.fit_transform(df)
        return scaled_data

    def analyze_market_structure(self, df):
        scaled_data = self.prepare_data(df)
        self.pca.fit(scaled_data)

        # Output Explained Variance
        explained_var = self.pca.explained_variance_ratio_
        print(f"{'Component':<12} | {'Explained Variance':<20}")
        print("-" * 35)
        for i, var in enumerate(explained_var):
            print(f"PC{i + 1:<10} | {var:.2%}")

        return self.pca.components_


if __name__ == "__main__":
    # 1. Create Synthetic PJM LMP Data
    # Let's simulate 1000 hourly observations for 4 nodes
    np.random.seed(42)
    t = np.linspace(0, 100, 1000)

    # Common Factor (System Demand)
    system_load = 30 + 10 * np.sin(t) + np.random.normal(0, 2, 1000)

    # Nodes with varying sensitivities to the system and local congestion
    # Node_A (Cheap West), Node_B (Load Center), Node_C (Constraint East)
    data = {
        'WESTERN_HUB': system_load + np.random.normal(0, 1, 1000),
        'PSEG': system_load + 5 + (2 * np.sin(t / 2)) + np.random.normal(0, 2, 1000),
        'PECO': system_load + 3 + (1.5 * np.sin(t / 2)) + np.random.normal(0, 1, 1000),
        'BGE': system_load + 10 + np.random.normal(0, 5, 1000)
    }

    df_lmp = pd.DataFrame(data)

    # 2. Run PCA
    quant = PJMPCAQuant()
    loadings = quant.analyze_market_structure(df_lmp)

    # 3. Interpret Loadings
    # Loadings show how much each node contributes to a PC
    loadings_df = pd.DataFrame(
        loadings.T,
        columns=['PC1 (System)', 'PC2 (Congestion)', 'PC3 (Residual)'],
        index=df_lmp.columns
    )

    print("\n--- Feature Loadings (Eigenvectors) ---")
    print(loadings_df.round(3))