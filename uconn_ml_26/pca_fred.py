import pandas as pd
import numpy as np
import ssl
import os
from dotenv import load_dotenv
from fredapi import Fred
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
import time

# --- 1. SETUP & SECURITY ---
load_dotenv()  # Loads variables from .env
FRED_API_KEY = os.getenv('FRED_API_KEY')
matplotlib.use('TkAgg')

# SSL Fix for Mac
if not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
    ssl._create_default_https_context = ssl._create_unverified_context

# --- 2. DATA PULL ---
tenors = {
    "DGS1MO": "1M", "DGS3MO": "3M", "DGS1": "1Y", "DGS2": "2Y", "DGS3": "3Y",
    "DGS5": "5Y", "DGS7": "7Y", "DGS10": "10Y", "DGS20": "20Y", "DGS30": "30Y"
}

fred = Fred(api_key=FRED_API_KEY)
df_dict = {}

print("Fetching Treasury Data via fredapi...")
for symbol, label in tenors.items():
    try:
        series = fred.get_series(symbol, observation_start='2010-01-01')
        df_dict[label] = series
        time.sleep(0.1)
    except Exception as e:
        print(f"    Error on {label}: {e}")

if not df_dict:
    print("No data found. Check your .env file and FRED_API_KEY.")
else:
    df = pd.DataFrame(df_dict).dropna()
    yield_changes = df.diff().dropna()

    # --- 3. PCA FIT ---
    pca = PCA(n_components=3)
    # Fit and transform to get Factor Scores (the time series of components)
    scores = pca.fit_transform(yield_changes)
    scores_df = pd.DataFrame(scores, index=yield_changes.index, columns=['Level', 'Slope', 'Curvature'])

    # --- 4. GRAPH 1: SCREE PLOT ---
    plt.figure(figsize=(10, 5))
    exp_var = pca.explained_variance_ratio_
    bars = plt.bar(['PC1', 'PC2', 'PC3'], exp_var, color='skyblue')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{yval:.2%}', ha='center', va='bottom')
    plt.title("Scree Plot: Explained Variance")
    plt.show()

    # --- 5. GRAPH 2: FACTOR SCORES (Time Series) ---
    # This shows how the 'Level' or 'Slope' moved over time
    plt.figure(figsize=(12, 6))
    plt.plot(scores_df['Level'].cumsum(), label='Cumulative Level Change', alpha=0.8)
    plt.plot(scores_df['Slope'], label='Slope Factor (Twist)', alpha=0.6)
    plt.axhline(0, color='black', lw=1)
    plt.title("Evolution of PCA Factor Scores (2010 - 2026)")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()

    # --- 6. GRAPH 3: PCA LOADINGS ---
    loadings = pd.DataFrame(pca.components_.T, columns=['Level', 'Slope', 'Curvature'], index=df.columns)
    if loadings['Level'].mean() < 0: loadings *= -1

    plt.figure(figsize=(10, 5))
    plt.plot(loadings, marker='o')
    plt.legend(loadings.columns)
    plt.title("PCA Factor Weights (Loadings)")
    plt.axhline(0, color='black', alpha=0.3)
    plt.show()

    print(f"\nCaptured Variance:\n{pca.explained_variance_ratio_}")