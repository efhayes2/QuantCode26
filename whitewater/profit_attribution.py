import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def run_profit_attribution(csv_path):
    # Load the current spot results
    df = pd.read_csv(os.path.expanduser(csv_path), parse_dates=['day'])
    df.set_index('day', inplace=True)

    hubs = ['target_waha_hh', 'target_waha_katy', 'target_waha_hsc']
    attribution = {}

    plt.figure(figsize=(12, 7))

    for s in hubs:
        # Isolate the P&L for just this hub
        pl_col = f'daily_pl_{s}'
        if pl_col not in df.columns:
            continue

        cum_pl = df[pl_col].cumsum()

        # Calculate Hub-Specific Metrics
        total_profit = df[pl_col].sum()
        win_rate = (df[pl_col] > 0).sum() / len(df)
        # Ratio of Avg Win vs Avg Loss
        profit_factor = abs(df[pl_col][df[pl_col] > 0].mean() / df[pl_col][df[pl_col] < 0].mean())

        attribution[s] = {
            "Total P&L": f"${total_profit:,.0f}",
            "Win Rate": f"{win_rate * 100:.1f}%",
            "Profit Factor": round(profit_factor, 2)
        }

        plt.plot(cum_pl, label=f"{s.replace('target_waha_', '').upper()} Attribution")

    plt.title("WHITWATER SPOT STRATEGY: PROFIT ATTRIBUTION BY HUB ($0.05 TC)", fontsize=14, fontweight='bold')
    plt.ylabel("Cumulative Net P&L ($)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Save the attribution plot
    save_path = os.path.expanduser('~/data/pngs/hub_attribution.png')
    plt.savefig(save_path)

    # Display the summary table
    print("\nHUB RESILIENCE SUMMARY ($0.05 TC per MMBtu):")
    print("-" * 50)
    attr_df = pd.DataFrame(attribution).T
    print(attr_df)
    print("-" * 50)
    print(f"Attribution plot saved to: {save_path}")


if __name__ == "__main__":
    # Point to the specific spot result file that yielded the 1.26 Sharpe
    run_profit_attribution('~/data/pngs/whitewater_forward_current.csv')