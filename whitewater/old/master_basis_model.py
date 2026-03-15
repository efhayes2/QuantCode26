import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import numpy as np


def extract_series(file_path, sid):
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


def main(risk_threshold):
    # 1. Setup specific project paths
    data_dir = Path("~/PyCharmProjects/QuantCode26/whitewater/data").expanduser()
    source_file = data_dir / "NG.txt"

    # 2. Extract Data
    proxies = {
        "supply": "NG.N9050TX2.M", "storage": "NG.N5030TX2.M",
        "freeport": "NG.NGM_EPG0_ENG_YFPT-Z00_MMCF.M",
        "corpus": "NG.NGM_EPG0_ENG_YCRP-Z00_MMCF.M",
        "sabine": "NG.NGM_EPG0_ENG_YSPL-Z00_MMCF.M"
    }

    dfs = {name: extract_series(source_file, sid) for name, sid in proxies.items()}
    hh_raw = extract_series(source_file, "NG.RNGWHHD.D")
    hh_raw['value'] = pd.to_numeric(hh_raw['value'], errors='coerce')
    hh_monthly = hh_raw.groupby(pd.Grouper(key='date', freq='MS'))['value'].mean().reset_index().rename(
        columns={'value': 'hh_price'})

    # 3. Merging & Adaptive Logic
    df = dfs['supply'].rename(columns={'value': 'supply'})
    for name in ['storage', 'freeport', 'corpus', 'sabine']:
        if dfs[name] is not None:
            df = pd.merge(df, dfs[name].rename(columns={'value': name}), on='date', how='left')
    df = pd.merge(df, hh_monthly, on='date', how='left')
    df = df.fillna(0)
    df = df[df['date'] >= '2016-02-01'].copy()

    # 4. Thermal Logic & Adaptive Z-Score
    temp_norms = {1: 58, 2: 61, 3: 67, 4: 73, 5: 79, 6: 84, 7: 86, 8: 86, 9: 82, 10: 75, 11: 67, 12: 60}
    df['month'] = df['date'].dt.month
    df['thermal_eff'] = df['month'].map(lambda m: 1.0 - (max(0, temp_norms[m] - 65) * 0.0094))
    df['eff_lng'] = (df['freeport'] + df['corpus'] + df['sabine']) * df['thermal_eff']
    df['storage_change'] = df['storage'].diff()

    # Relative Imbalance as % of Supply
    df['raw_imbalance'] = df['supply'] - (df['eff_lng'] + df['storage_change'])
    df['rel_imbalance_pct'] = (df['raw_imbalance'] / df['supply']) * 100

    # ADAPTIVE Z-SCORE: Rolling 36-Month Window
    df['rolling_mean'] = df['rel_imbalance_pct'].rolling(window=36, min_periods=12).mean()
    df['rolling_std'] = df['rel_imbalance_pct'].rolling(window=36, min_periods=12).std()
    df['z_score'] = (df['rel_imbalance_pct'] - df['rolling_mean']) / df['rolling_std']

    # --- UPDATED DIVERGENCE LOGIC ---
    # We lower the sensitivity to 0.1 to account for the smoother rolling line
    df['p_delta'] = df['z_score'].diff()
    df['h_delta'] = df['hh_price'].diff()
    df['sig_divergence'] = ((df['p_delta'] * df['h_delta']) < 0) & (df['p_delta'].abs() > 0.1)
    divergent_points = df[df['sig_divergence']]

    # --- CUMULATIVE DURATION CALCULATION ---
    df['is_risk'] = df['z_score'] > risk_threshold
    total_risk_months = df['is_risk'].sum()
    total_months = len(df)
    years_in_data = total_months / 12
    avg_months_per_year = total_risk_months / years_in_data

    print("\n" + "=" * 40)
    print("STRATEGIC RISK DURATION REPORT")
    print(f"Threshold Sensitivity: {risk_threshold} Sigma")
    print("-" * 40)
    print(f"Total Months in Red Zone: {total_risk_months}")
    print(f"Actionable Window: {avg_months_per_year:.1f} months/year")
    print(f"Exposure Rate: {(avg_months_per_year / 12) * 100:.1f}%")
    print(f"Trade Signals (Divergence Diamonds): {len(divergent_points)}")
    print("=" * 40 + "\n")

    # 5. Presentation Visualization
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 18))

    # PANEL 1: Adaptive Z-Score & Reintroduced Diamonds
    p_line = ax1.plot(df['date'], df['z_score'], color='#8e44ad', linewidth=3, label='Basis Pressure (Z-Score)',
                      zorder=4)
    ax1.fill_between(df['date'], risk_threshold, df['z_score'], where=(df['z_score'] > risk_threshold), color='red',
                     alpha=0.3, zorder=3)

    # Plotting the Diamonds
    ax1.scatter(divergent_points['date'], divergent_points['z_score'],
                color='#f1c40f', s=100, marker='D', edgecolor='black', zorder=6, label='Divergence Signal')

    # Transformed Thermal Overlay
    z_min, z_max = df['z_score'].min(), df['z_score'].max()
    z_range = z_max - z_min
    eff_min, eff_max = df['thermal_eff'].min(), df['thermal_eff'].max()
    target_min, target_max = z_min + (0.25 * z_range), z_min + (0.75 * z_range)
    df['thermal_overlay'] = target_min + (df['thermal_eff'] - eff_min) * (target_max - target_min) / (eff_max - eff_min)
    t_line = ax1.plot(df['date'], df['thermal_overlay'], color='#e67e22', linewidth=2, alpha=0.6,
                      label='Thermal Efficiency Overlay', zorder=2)

    ax1_twin = ax1.twinx()
    h_line = ax1_twin.plot(df['date'], df['hh_price'], color='#2c3e50', linestyle='--', linewidth=2, alpha=0.7,
                           label='Henry Hub ($/MMBtu)')
    ax1_twin.set_ylabel('Henry Hub Price ($/MMBtu)', fontweight='bold', color='#2c3e50')
    ax1_twin.grid(False)

    ax1.set_ylabel('Constraint Intensity (Z-Score)', fontweight='bold', color='#8e44ad')
    ax1.set_title(f'Master Basis Strategy: Adaptive Stress Model (Threshold: {risk_threshold}σ)', fontsize=18,
                  fontweight='bold')

    lines = p_line + h_line + t_line
    ax1.legend(lines + [plt.Line2D([0], [0], color='red', alpha=0.3, lw=4),
                        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#f1c40f', markersize=10)],
               ['Basis Pressure (Z-Score)', 'HH Price', 'Thermal Efficiency Overlay', 'Blowout Risk',
                'Divergence Signal'], loc='upper left')

    ax2.plot(df['date'], df['thermal_eff'], color='#e67e22', linewidth=2.5)
    ax2.set_title('Seasonal Thermal Efficiency (Mechanical Constraint)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Efficiency Factor', fontweight='bold')
    ax2.set_ylim(0.7, 1.05)

    for ax in [ax1, ax2]:
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.setp(ax.get_xticklabels(), rotation=90, ha='center', visible=True)
        ax.set_xlim(df['date'].min(), df['date'].max())

    plt.tight_layout()
    plt.savefig(data_dir / "master_basis_strategy_overlay.png", dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main(0.5)