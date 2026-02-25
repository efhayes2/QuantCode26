import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


def main():
    # 1. Setup paths
    data_dir = Path("~/PyCharmProjects/QuantCode26/whitewater/data").expanduser()
    source_file = data_dir / "basis_pressure_data.csv"

    if not source_file.exists():
        print("Error: Please run basis_pressure_generator.py first to create the source data.")
        return

    # 2. Load the Physical Data
    df = pd.read_csv(source_file)
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month

    # 3. Define the "Thermal Drag" Profile
    # Average monthly temperatures for the Texas Coast (Corpus Christi proxy)
    # Jan: 58, Feb: 61, Mar: 67, Apr: 73, May: 79, Jun: 84, Jul: 86, Aug: 86, Sep: 82, Oct: 75, Nov: 67, Dec: 60
    temp_normals = {
        1: 58, 2: 61, 3: 67, 4: 73, 5: 79, 6: 84,
        7: 86, 8: 86, 9: 82, 10: 75, 11: 67, 12: 60
    }

    # Calculate Drag: 0.94% reduction per degree above 65°F
    def calculate_drag(month):
        temp = temp_normals[month]
        if temp <= 65:
            return 1.0  # No derate in cool weather
        else:
            degrees_above = temp - 65
            efficiency = 1.0 - (degrees_above * 0.0094)
            return efficiency

    df['thermal_efficiency'] = df['month'].map(calculate_drag)

    # 4. Apply Thermal Drag to LNG Exports
    # We reduce the 'Effective Exit Capacity' based on the heat
    df['effective_lng'] = df['total_lng'] * df['thermal_efficiency']
    df['thermal_loss_mmcf'] = df['total_lng'] - df['effective_lng']

    # 5. Recalculate "Thermal Pressure"
    # This shows the pressure caused specifically by heat-related bottlenecks
    df['thermal_pressure'] = df['supply'] - (df['effective_lng'] + df['storage_change'])
    df['thermal_pressure_std'] = (df['thermal_pressure'] - df['thermal_pressure'].mean()) / df['thermal_pressure'].std()

    # 6. Visualization
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 14), sharex=True)

    # Panel 1: The Thermal Drag Factor
    ax1.plot(df['date'], df['thermal_efficiency'], color='#e67e22', linewidth=2, label='LNG Thermal Efficiency')
    ax1.set_title('The "Summer Squeeze": LNG Thermal Efficiency Derate (TX Coast)', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Efficiency (% of Nameplate)')
    ax1.set_ylim(0.7, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Panel 2: Thermal Adjusted Pressure
    ax2.plot(df['date'], df['thermal_pressure_std'], color='#c0392b', linewidth=2, label='Thermal Adjusted Pressure')
    ax2.fill_between(df['date'], 1.5, df['thermal_pressure_std'],
                     where=(df['thermal_pressure_std'] > 1.5), color='red', alpha=0.3)
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.set_title('Basis Pressure Index (Weather Adjusted)', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Standardized Squeeze (Sigma)')

    # Sideways Labels
    ax2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax2.get_xticklabels(), rotation=90)

    plt.tight_layout()
    plt.savefig(data_dir / "thermal_basis_analysis.png", dpi=300)
    df.to_csv(data_dir / "thermal_basis_data.csv", index=False)

    print(f"Thermal Analysis Complete. Q3 Drag identified. Files saved to {data_dir}")


if __name__ == "__main__":
    main()