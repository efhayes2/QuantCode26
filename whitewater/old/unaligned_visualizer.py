import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


def main():
    # 1. Setup paths based on your environment
    home = Path.home()
    data_dir = home / "data"

    # 2. Load the datasets
    print("Loading datasets from ~/data...")
    df_hh = pd.read_excel(data_dir / "HH_daily.xlsx")
    df_supply = pd.read_excel(data_dir / "supply.xlsx")
    df_demand = pd.read_excel(data_dir / "demand.xlsx")
    df_inv = pd.read_excel(data_dir / "inventory.xlsx")

    # 3. Pre-processing: Ensure datetime and sort
    for df in [df_hh, df_supply, df_demand, df_inv]:
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values('date', inplace=True)

    # 4. Set professional styling
    plt.style.use('ggplot')
    # We avoid sharex=True to ensure independent control over each panel's labels
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 22))

    def apply_formatting(ax, title, ylabel, color='black'):
        ax.set_title(title, fontsize=14, fontweight='bold', pad=25)
        ax.set_ylabel(ylabel, fontweight='bold', color=color)

        # Force x-axis labels to show for THIS panel
        ax.tick_params(axis='x', labelbottom=True, labelsize=9)
        ax.tick_params(axis='y', labelcolor=color)

        # Force Quarterly Timing (Jan, Apr, Jul, Oct)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

        # Set the sideways (vertical) rotation
        plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
        ax.grid(True, linestyle='--', alpha=0.6)

    # Panel 1: Henry Hub Daily Prices
    ax1.plot(df_hh['date'], df_hh['value'], color='#2c3e50', linewidth=1)
    apply_formatting(ax1, 'Market Spine: Henry Hub Daily Spot Price', 'Price ($/MMBtu)')

    # Panel 2: Dual-Axis Fundamentals (Production vs. Exports)
    # Left Axis: TX Production
    ax2.plot(df_supply['date'], df_supply['value'], color='#27ae60', linewidth=2, label='TX Production')
    apply_formatting(ax2, 'Regional Balance: Production (Left) vs. LNG Exports (Right)', 'TX Production (MMcf)',
                     color='#27ae60')

    # Right Axis: Sabine Pass Exports
    ax2_twin = ax2.twinx()
    ax2_twin.plot(df_demand['date'], df_demand['value'], color='#e74c3c', linewidth=2, label='Sabine Pass Exports',
                  linestyle='--')
    ax2_twin.set_ylabel('Sabine Pass Exports (MMcf)', color='#e74c3c', fontweight='bold')
    ax2_twin.tick_params(axis='y', labelcolor='#e74c3c')
    ax2_twin.grid(False)  # Keep secondary grid off to avoid visual clutter

    # Combine legends for the dual-axis panel
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left', frameon=True)

    # Panel 3: South Central Salt Storage
    ax3.fill_between(df_inv['date'], df_inv['value'], color='#f39c12', alpha=0.3)
    ax3.plot(df_inv['date'], df_inv['value'], color='#d35400', linewidth=1.5)
    apply_formatting(ax3, 'The Waha Shock Absorber: Salt Region Storage', 'Inventory (Bcf)')

    # Synchronize the x-axis limits manually to maintain alignment
    min_date = pd.to_datetime('2016-02-01')
    max_date = df_hh['date'].max()
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(min_date, max_date)

    # Final Layout Tweaks
    plt.tight_layout()
    # Add significant vertical space for the sideways labels and titles
    plt.subplots_adjust(hspace=0.8)

    # 5. Save the graphic
    output_path = data_dir / "dual_axis_quarterly_sideways.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    print(f"Final visualization saved to: {output_path}")


if __name__ == "__main__":
    main()