import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def extract_series(file_path, sid):
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
    if not data_list: return None
    df = pd.DataFrame(data_list, columns=['date', 'value'])
    df['date'] = pd.to_datetime(df['date'], format='%Y%m', errors='coerce')
    return df.sort_values('date')


def main():
    # Setup project paths
    data_dir = Path("/whitewater/data").expanduser()
    source_file = data_dir / "NG.txt"

    if not source_file.exists():
        print(f"Error: NG.txt not found at {source_file}")
        return

    # Define Levers
    proxies = {
        "supply": "NG.N9050TX2.M",
        "storage": "NG.N5030TX2.M",
        "freeport": "NG.NGM_EPG0_ENG_YFPT-Z00_MMCF.M",
        "corpus": "NG.NGM_EPG0_ENG_YCRP-Z00_MMCF.M",
        "sabine": "NG.NGM_EPG0_ENG_YSPL-Z00_MMCF.M"
    }

    dfs = {name: extract_series(source_file, sid) for name, sid in proxies.items()}

    # Merge and Align
    df_master = dfs['supply'].rename(columns={'value': 'supply'})
    for name in ['storage', 'freeport', 'corpus', 'sabine']:
        if dfs[name] is not None:
            df_master = pd.merge(df_master, dfs[name].rename(columns={'value': name}), on='date', how='left')

    df_master = df_master.fillna(0)
    df_master = df_master[df_master['date'] >= '2016-02-01'].copy()

    # Calculate Pressure
    df_master['storage_change'] = df_master['storage'].diff()
    df_master['total_lng'] = df_master['freeport'] + df_master['corpus'] + df_master['sabine']
    df_master['basis_pressure'] = df_master['supply'] - (df_master['total_lng'] + df_master['storage_change'])
    df_master['pressure_std'] = (df_master['basis_pressure'] - df_master['basis_pressure'].mean()) / df_master[
        'basis_pressure'].std()

    # Prep Heatmap Data
    df_master['year'] = df_master['date'].dt.year
    df_master['month'] = df_master['date'].dt.month
    heatmap_data = df_master.pivot(index="year", columns="month", values="pressure_std")

    # Visualization
    plt.style.use('ggplot')
    plt.figure(figsize=(14, 10))
    sns.heatmap(heatmap_data, annot=True, cmap="RdYlGn_r", center=0,
                cbar_kws={'label': 'Basis Pressure (Sigma)'})

    plt.title("Texas Basis Pressure Seasonality (Post-2016)\nRed = High Blowout Risk (Physical Constraint)",
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel("Month (Jan - Dec)", fontsize=12, fontweight='bold')
    plt.ylabel("Year", fontsize=12, fontweight='bold')

    # Rotate Y-axis labels if needed, though Seaborn handles months well
    plt.yticks(rotation=0)

    output_path = data_dir / "basis_seasonality_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {output_path}")


if __name__ == "__main__":
    main()