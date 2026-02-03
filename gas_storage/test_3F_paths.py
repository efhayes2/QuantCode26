import numpy as np
from storage_helper import MarketEnvironment3F


def test_paths():
    mkt = MarketEnvironment3F(M=5, T_days=365, sigma_chi=0.0315, sigma_xi=0.01, rho=0.3)

    # Generate the paths from your helper
    S, chi, xi = mkt.generate_paths_3f()

    print("=" * 60)
    print("3-FACTOR PATH DIAGNOSTICS")
    print("=" * 60)
    print(f"Short-term Factor (Chi) Std Dev: {np.std(chi):.4f}")
    print(f"Long-term Factor (Xi) Std Dev:   {np.std(xi):.4f}")

    # Verify correlation between the two factors across all paths
    actual_rho = np.corrcoef(chi.flatten(), xi.flatten())[0, 1]
    print(f"Target Rho: {mkt.rho} | Observed Rho: {actual_rho:.4f}")
    print("-" * 60)

    # Snapshot of Day 216 (The winter jump)
    day = 216
    print(f"Day {day} Snapshots (Path 0):")
    print(f"  Price S:   ${S[day, 0]:.2f}")
    print(f"  Chi (Dev): {chi[day, 0]:.4f}")
    print(f"  Xi (Base): {xi[day, 0]:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    test_paths()