import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import numpy as np


def extract_series(file_path, sid):
    """Streams the EIA file to extract a specific Series ID."""
    if not file_path.exists(): return None
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
    if not data_list: return None
    df = pd.DataFrame(data_list, columns=['date', 'value'])

    sample_date = str(data_list[0][0])
    if len(sample_date) == 8:
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
    else:
        df['date'] = pd.to_datetime(df['date'], format='%Y%m', errors='coerce')

    return df.dropna(subset=['date']).sort_values('date')


def main():
    # --- STRATEGY PARAMETERS ---
    # Adjusting this threshold changes the sensitivity of 'Blowout Risk' alerts.
    # 1.5 is high conviction; 1.3 captures 'Shoulder' risks and winter shadows.
    risk_threshold = 1.3

    # 1. Setup specific project paths
    data_dir = Path("~/PyCharmProjects/QuantCode26/whitewater/data").expanduser()
    source_file = data_dir / "NG.txt"

    # 2. Extract Data
    print(f"Beginning extraction with Risk Threshold set to: {risk_threshold}")
    proxies = {
        "supply": "NG.N9050TX2.M", "storage": "NG.N5030TX2.M",
        "freeport": "NG.NGM_EPG0_ENG_YFPT-Z00_MMCF.M",
        "corpus": "NG.NGM_EPG0_ENG_YCRP-Z00_MMCF.M",
        "sabine": "NG.NGM_EPG0_ENG_YSPL-Z00_MMCF.M"
    }

    dfs = {name: extract_series(source_file, sid) for name, sid in proxies.items()}

    # 3. Process Henry Hub
    hh_raw = extract_series(source_file, "NG.RNGWHHD.D")
    if hh_raw is not None:
        hh_raw['value'] = pd.to_numeric(hh_raw['value'], errors='coerce')
        hh_monthly = hh_raw.groupby(pd.Grouper(key='date', freq='MS'))['value'].mean().reset_index()
        hh_monthly = hh_monthly.rename(columns={'value': 'hh_price'})
    else:
        hh_monthly = pd.DataFrame(columns=['date', 'hh_price'])

    # 4. Combine and Filter
    df = dfs['supply'].rename(columns={'value': 'supply'})
    for name in ['storage', 'freeport', 'corpus', 'sabine']:
        if dfs[name] is not None:
            df = pd.merge(df, dfs[name].rename(columns={'value': name}), on='date', how='left')

    df = pd.merge(df, hh_monthly, on='date', how='left')
    df = df.fillna(0)
    df = df[df['date'] >= '2016-02-01'].copy()

    # 5. Physics & Thermal Logic
    temp_norms = {1: 58, 2: 61, 3: 67, 4: 73, 5: 79, 6: 84, 7: 86, 8: 86, 9: 82, 10: 75, 11: 67, 12: 60}
    df['month'] = df['date'].dt.month
    df['thermal_eff'] = df['month'].map(lambda m: 1.0 - (max(0, temp_norms[m] - 65) * 0.0094))

    df['eff_lng'] = (df['freeport'] + df['corpus'] + df['sabine']) * df['thermal_eff']
    df['storage_change'] = df['storage'].diff()
    df['basis_pressure'] = df['supply'] - (df['eff_lng'] + df['storage_change'])
    df['pressure_std'] = (df['basis_pressure'] - df['basis_pressure'].mean()) / df['basis_pressure'].std()

    # Divergence Detection
    df['p_delta'] = df['pressure_std'].diff()
    df['h_delta'] = df['hh_price'].diff()
    df['sig_divergence'] = ((df['p_delta'] * df['h_delta']) < 0) & (df['p_delta'].abs() > 0.2)
    divergent_points = df[df['sig_divergence']]

    # 6. Visualization
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 16))

    # PANEL 1: Financial vs Physical Pressure
    p_line = ax1.plot(df['date'], df['pressure_std'], color='#8e44ad', linewidth=3,
                      label='Thermal Basis Pressure (Sigma)')

    # Use the variable threshold for the pink region
    ax1.fill_between(df['date'], risk_threshold, df['pressure_std'],
                     where=(df['pressure_std'] > risk_threshold),
                     color='red', alpha=0.3, label=f'Blowout Risk Zone (>{risk_threshold}σ)')

    ax1.scatter(divergent_points['date'], divergent_points['pressure_std'],
                color='#f1c40f', s=100, marker='D', edgecolor='black', zorder=5, label='Divergence Signal')

    ax1.set_ylabel('Constraint Intensity (Sigma)', fontweight='bold', color='#8e44ad')
    ax1.set_title(f'Master Basis Strategy (Threshold: {risk_threshold}σ)', fontsize=18, fontweight='bold')

    ax1_twin = ax1.twinx()
    h_line = ax1_twin.plot(df['date'], df['hh_price'], color='#2c3e50', linestyle='--', linewidth=2, alpha=0.7,
                           label='Henry Hub ($/MMBtu)')
    ax1_twin.set_ylabel('Henry Hub Price ($/MMBtu)', fontweight='bold', color='#2c3e50')
    ax1_twin.grid(False)

    # Legend
    lines = p_line + h_line
    ax1.legend(lines + [plt.Line2D([0], [0], color='red', alpha=0.3, lw=4),
                        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#f1c40f', markersize=10)],
               ['Basis Pressure', 'HH Price', 'Blowout Risk', 'Divergence Signal'], loc='upper left')

    # PANEL 2: Thermal Efficiency
    ax2.plot(df['date'], df['thermal_eff'], color='#e67e22', linewidth=2.5, label='LNG Intake Efficiency %')
    ax2.set_title('Seasonal Thermal Efficiency (Mechanical Constraint)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Efficiency Factor', fontweight='bold')
    ax2.set_ylim(0.7, 1.05)
    ax2.legend(loc='lower left')

    # Forced Time Labeling on Both
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.setp(ax.get_xticklabels(), rotation=90, ha='center', visible=True)
        ax.set_xlim(df['date'].min(), df['date'].max())

    plt.tight_layout()
    plt.savefig(data_dir / "master_basis_strategy.png", dpi=300, bbox_inches='tight')
    df.to_csv(data_dir / "master_basis_data.csv", index=False)

    print(f"\nModel Updated. Threshold: {risk_threshold} sigma.")
    print(f"Graph starts at: {df['date'].min().date()}")


if __name__ == "__main__":
    main()