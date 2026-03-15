import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


def extract_series(file_path, sid):
    """Streams the EIA file to extract a specific Series ID."""
    if not file_path.exists():
        return None
    data_list = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                if record.get("series_id") == sid:
                    data_list = record.get("data", [])
                    break
            except:
                continue

    if not data_list:
        return None

    df = pd.DataFrame(data_list, columns=['date', 'value'])
    # Convert EIA YYYYMM format to datetime
    df['date'] = pd.to_datetime(df['date'], format='%Y%m', errors='coerce')
    return df.sort_values('date')


def main():
    # 1. Setup specific project paths
    # Resolves to /Users/efh2/PyCharmProjects/QuantCode26/whitewater/data
    data_dir = Path("/whitewater/data").expanduser()
    source_file = data_dir / "NG.txt"

    if not source_file.exists():
        print(f"ERROR: NG.txt not found at {source_file}")
        return

    # 2. Define the Physical Levers
    proxies = {
        "supply": "NG.N9050TX2.M",  # TX Marketed Production
        "storage": "NG.N5030TX2.M",  # TX Total Storage Volume
        "freeport": "NG.NGM_EPG0_ENG_YFPT-Z00_MMCF.M",
        "corpus": "NG.NGM_EPG0_ENG_YCRP-Z00_MMCF.M",
        "sabine": "NG.NGM_EPG0_ENG_YSPL-Z00_MMCF.M"
    }

    print(f"Extracting physical data from {source_file.name}...")
    dfs = {}
    for name, sid in proxies.items():
        dfs[name] = extract_series(source_file, sid)

    # 3. Combine and Align Data
    if dfs['supply'] is None:
        print("Critical Error: Could not find Production data.")
        return

    df_master = dfs['supply'].rename(columns={'value': 'supply'})

    for name in ['storage', 'freeport', 'corpus', 'sabine']:
        if dfs[name] is not None:
            df_master = pd.merge(df_master, dfs[name].rename(columns={'value': name}), on='date', how='left')

    df_master = df_master.fillna(0)

    # --- CRITICAL FIX: HARD DATE FILTER ---
    # This ensures the dataframe ONLY contains data from Feb 2016 onwards
    df_master = df_master[df_master['date'] >= '2016-02-01'].copy()

    # 4. Calculate Basis Pressure
    df_master['storage_change'] = df_master['storage'].diff()
    df_master['total_lng'] = df_master['freeport'] + df_master['corpus'] + df_master['sabine']
    df_master['basis_pressure'] = df_master['supply'] - (df_master['total_lng'] + df_master['storage_change'])

    # Standardize
    df_master['pressure_std'] = (df_master['basis_pressure'] - df_master['basis_pressure'].mean()) / df_master[
        'basis_pressure'].std()

    # 5. Visualization
    plt.style.use('ggplot')
    fig, ax1 = plt.subplots(figsize=(16, 10))

    ax1.plot(df_master['date'], df_master['pressure_std'], color='#8e44ad', linewidth=2.5,
             label='Texas Basis Pressure Index')
    ax1.axhline(0, color='black', linestyle='--', alpha=0.5)

    # Highlight High Pressure Zones
    ax1.fill_between(df_master['date'], 1.5, df_master['pressure_std'],
                     where=(df_master['pressure_std'] > 1.5),
                     color='red', alpha=0.4, label='Basis Blowout Risk')

    ax1.set_title('Texas "Basis Pressure" Index: Physical Proxy for Waha Constraints\n(Post-Jan 2016 Export Regime)',
                  fontsize=18, fontweight='bold', pad=30)
    ax1.set_ylabel('Standardized Constraint Level (Sigma)', fontsize=12, fontweight='bold')

    # --- CRITICAL FIX: FORCED ROTATION ---
    # Set quarterly ticks
    ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    # Force 90-degree rotation on the axis object
    plt.setp(ax1.get_xticklabels(), rotation=90, ha='center')

    # Set the x-limit strictly to the data range starting Feb 2016
    ax1.set_xlim(df_master['date'].min(), df_master['date'].max())

    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper left', frameon=True, facecolor='white')

    plt.tight_layout()

    # 6. Save Outputs
    output_img = data_dir / "basis_pressure_index.png"
    output_csv = data_dir / "basis_pressure_data.csv"

    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    df_master.to_csv(output_csv, index=False)

    print(f"\nSuccess! Graph starts at: {df_master['date'].min().date()}")
    print(f"Files saved to: {data_dir}")


if __name__ == "__main__":
    main()