import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


def main():
    # 1. Setup path to the master Excel file
    # Change Path.home() / "data" to your specific path if it's different
    master_path = Path.home() / "data" / "master_aligned.xlsx"

    if not master_path.exists():
        print(f"Error: {master_path} not found.")
        return

    # 2. Load the specific sheets directly
    print("Loading data from master_aligned.xlsx...")
    try:
        df_hh = pd.read_excel(master_path, sheet_name='HH_daily').rename(columns={'value': 'price'})
        df_inv = pd.read_excel(master_path, sheet_name='inventory').rename(columns={'value': 'inventory'})
    except Exception as e:
        print(f"Error reading Excel sheets: {e}")
        return

    # 3. Standardize and Sort
    for df in [df_hh, df_inv]:
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values('date', inplace=True)

    # 4. Synchronize (Align)
    # We create the daily spine and fill in the latest inventory
    df = pd.merge_asof(df_hh, df_inv, on='date', direction='backward')

    # --- THE SIMULATION ---
    # Event: A physical constraint (Linepack Max) causes a price shock at Waha
    # We'll set this to a recent date in your data
    event_date = pd.to_datetime('2024-01-01')

    # We simulate the 'Pre-See' (The SCADA event):
    # The price collapses because molecules are trapped, breaking the correlation with storage
    df['synthetic_price'] = df['price'].copy()
    df.loc[df['date'] >= event_date, 'synthetic_price'] = df.loc[df['date'] >= event_date, 'price'] * 0.4

    # 5. The 'Post-Detect' (90-Day Rolling Correlation used by outside quants)
    df['rolling_corr'] = df['synthetic_price'].rolling(window=90).corr(df['inventory'])

    # 6. Calculate the Alpha Window (Detection Lag)
    # Statistical models usually flag a break when abs correlation drops below 0.3
    shift_detected = df[(df['date'] >= event_date) & (df['rolling_corr'].abs() < 0.3)]

    if not shift_detected.empty:
        detection_date = shift_detected.iloc[0]['date']
        latency_days = (detection_date - event_date).days
    else:
        latency_days = "N/A"

    # 7. Visualization
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 14), sharex=True)

    # Panel 1: The "Pre-See" (WWM SCADA View)
    ax1.plot(df['date'], df['synthetic_price'], color='#c0392b', label='Waha Spot Price')
    ax1.axvline(event_date, color='black', linestyle='--', linewidth=2, label='SCADA: Max Linepack (T-0)')
    ax1.set_title(
        f'SCADA Lead-Time Simulation\nPhysical Event: {event_date.date()} | Quant Detection Lag: {latency_days} Days',
        fontsize=16, fontweight='bold')
    ax1.set_ylabel('Price ($/MMBtu)', fontweight='bold')
    ax1.legend(loc='upper right')

    # Panel 2: The "Post-Detect" (Citadel/Quant View)
    ax2.plot(df['date'], df['rolling_corr'], color='#2980b9', linewidth=2, label='90D Correlation (Quant Model)')
    ax2.axvline(event_date, color='black', linestyle='--', linewidth=2)

    if not shift_detected.empty:
        ax2.axvline(detection_date, color='green', linestyle=':', linewidth=3, label='Statistical Detection')
        # Highlight the Alpha Capture Window
        ax2.fill_between(df['date'], event_date, detection_date, color='yellow', alpha=0.3,
                         label='WWM Alpha Window (Pre-See)')

    ax2.set_ylabel('Correlation Coeff', fontweight='bold')
    ax2.set_ylim(-1.1, 1.1)
    ax2.axhline(0.3, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(-0.3, color='red', linestyle='--', alpha=0.5)

    # Quarterly sideways formatting
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax2.get_xticklabels(), rotation=90, ha='center')
    ax2.legend(loc='lower left')

    plt.tight_layout()
    # Save to the same directory as the master file
    output_path = master_path.parent / "scada_latency_simulation.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    print(f"\nSimulation Successful.")
    print(f"Physical Event: {event_date.date()}")
    print(f"Quant Detection: {latency_days} days later.")
    print(f"Visualization saved to: {output_path}")


if __name__ == "__main__":
    main()