import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.decomposition import PCA
import time

import matplotlib
# Use 'Qt5Agg' or 'TkAgg' to force a popup window
# Note: This must be called BEFORE importing matplotlib.pyplot
matplotlib.use('TkAgg')



# 1. Expanded Ticker List for a Granular Curve
# ^IRX (3M), ^FVX (5Y), ^TNX (10Y), ^TYX (30Y)
# Adding: ^IRX (13-week), SHY (approx 2Y), ^VXG (7Y), TLT (20Y+ proxy)
# For reliability, we'll stick to the CBOE yield tickers:
tickers = {
    "^IRX": "3M",
    "^FVX": "5Y",
    "^TNX": "10Y",
    "^TYX": "30Y"
}
# Note: yfinance can be spotty with ^GS2 (2Y) and ^GS7 (7Y),
# so we will use the core 4 for the most robust classroom demo.

price_data = {}

print("Downloading Treasury data...")
for t, label in tickers.items():
    time.sleep(0.5)
    df_ticker = yf.download(t, start="2022-01-01", progress=False)

    if not df_ticker.empty:
        # Extract Close prices
        if isinstance(df_ticker.columns, pd.MultiIndex):
            price_data[label] = df_ticker['Close'][t]
        else:
            price_data[label] = df_ticker['Close']

df = pd.DataFrame(price_data).dropna()

# 2. Graph 1: Yield Curve Evolution (Time Series)
plt.figure(figsize=(12, 6))
for col in df.columns:
    plt.plot(df[col], label=col)

plt.title("US Treasury Yield Curve Evolution (2022 - Present)")
plt.ylabel("Yield (%)")
plt.xlabel("Date")
plt.legend(title="Tenors", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 3. PCA Implementation
yield_changes = df.diff().dropna()

if not yield_changes.empty:
    pca = PCA(n_components=3)
    pca.fit(yield_changes)

    # 4. Graph 2: PCA Weights (Loadings)
    # We plot the 'components_' which show how much each tenor
    # contributes to each Principal Component.
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=['PC1: Level', 'PC2: Slope', 'PC3: Curvature'],
        index=df.columns
    )

    plt.figure(figsize=(10, 6))
    plt.plot(loadings['PC1: Level'], marker='o', lw=2, label='PC1: Level')
    plt.plot(loadings['PC2: Slope'], marker='o', lw=2, label='PC2: Slope')
    plt.plot(loadings['PC3: Curvature'], marker='o', lw=2, label='PC3: Curvature')

    plt.axhline(0, color='black', lw=1, alpha=0.5)
    plt.title("PCA Weights: Level, Slope, and Curvature Factors")
    plt.xlabel("Tenor (Node Points)")
    plt.ylabel("Weighting (Sensitivity)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Final Summary for the Class
    exp_var = pca.explained_variance_ratio_
    print(f"\nVariance Explained:")
    for i, name in enumerate(['Level', 'Slope', 'Curvature']):
        print(f"{name}: {exp_var[i]:.2%}")