import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


def main():
    # 1. Setup paths
    home = Path.home()
    data_dir = home / "data"
    # If you are running this inside your PyCharm project,
    # you can change this to Path(".") or a relative path
    master_path = data_dir / "master_aligned.xlsx"

    if not master_path.exists():
        print(f"Error: {master_path} not found.")
        print("Please ensure the Excel file is in your ~/data folder.")
        return

    # 2. Load Sheets from the Excel Master
    print(f"Loading data from {master_path}...")
    try:
        # Loading the specific sheets we created in the previous step
        df_hh = pd.read_excel(master_path, sheet_name='HH_daily')
        df_supply = pd.read_excel(master_path, sheet_name='supply')
        df_inv = pd.read_excel(master_path, sheet_name='inventory')
    except Exception as e:
        print(f"Error reading sheets: {e}")
        return

    # 3. Pre-processing: Standardize Dates and Sort
    print("Aligning time series...")
    for df in [df_hh, df_supply, df_inv]:
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values('date', inplace=True)

    # 4. Synchronize (Align) Data
    # We use the daily Henry Hub price as the 'spine'
    # 'backward' merge ensures we use the most recent available data for each day
    df = pd.merge_asof(df_hh.rename(columns={'value': 'price'}),
                       df_inv.rename(columns={'value': 'inventory'}),
                       on='date', direction='backward')

    df = pd.merge_asof(df,
                       df_supply.rename(columns={'value': 'production'}),
                       on='date', direction='backward')

    # 5. Calculate 90-Day Rolling Correlations
    # We identify regime shifts by watching these correlations break down
    df['corr_inv'] = df['price'].rolling(window=90).corr(df['inventory'])
    df['corr_prod'] = df['price'].rolling(window=90).corr(df['production'])

    # 6. Professional Visualization (Fitted for 3 Panels)
    plt.style.use('ggplot')
    # Large vertical canvas to ensure nothing is cut off
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 22))

    def format_panel(ax, data, title, color, label, is_price=False):
        if is_price:
            ax.plot(df['date'], data, color=color, linewidth=1.5)
            ax.set_ylabel('$/MMBtu', fontweight='bold')
        else:
            ax.plot(df['date'], data, color=color, linewidth=2, label=label)
            ax.axhline(0, color='black', linewidth=1, alpha=0.5)
            # Highlight Decorrelation/Regime Shift Zones
            ax.fill_between(df['date'], -1, 1, where=(data.abs() < 0.3),
                            color='red', alpha=0.1, label='Regime Shift Zone')
            ax.set_ylabel('Correlation', fontweight='bold')
            ax.set_ylim(-1.1, 1.1)
            ax.legend(loc='lower left')

        ax.set_title(title, fontsize=16, fontweight='bold', pad=25)

        # Sideways Quarterly Labels on EVERY plot
        ax.tick_params(axis='x', labelbottom=True, labelsize=10, rotation=90)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.grid(True, linestyle='--', alpha=0.6)

    # Panel 1: Henry Hub Price Action
    format_panel(ax1, df['price'], 'Henry Hub Daily Spot Price', '#2c3e50', 'Price', is_price=True)

    # Panel 2: Price vs. Inventory Correlation
    format_panel(ax2, df['corr_inv'], 'Price-Inventory Relationship (90D Rolling Corr)', '#e67e22', 'Inv Corr')

    # Panel 3: Price vs. Production Correlation
    format_panel(ax3, df['corr_prod'], 'Price-Production Relationship (90D Rolling Corr)', '#27ae60', 'Prod Corr')

    # Final Adjustment for Fit
    plt.tight_layout()
    # Ensure space for vertical labels between plots
    plt.subplots_adjust(hspace=0.7)

    # Save to ~/data
    output_path = data_dir / "master_regime_analysis_fitted.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    print(f"\nSuccess! Aligned data analyzed.")
    print(f"Visualization saved to: {output_path}")


if __name__ == "__main__":
    main()