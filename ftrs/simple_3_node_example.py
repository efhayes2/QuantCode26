from dataclasses import dataclass
from typing import List, Dict


@dataclass
class GridState:
    base_price: float  # Cost of the "System" marginal unit (Node A)
    shadow_price: float  # The cost of the congestion on Line A-B
    # PTDFs for each node relative to the constrained line A-B
    ptdfs: Dict[str, float]


class ShadowPriceSimulator:
    def __init__(self, state: GridState):
        self.state = state
        self.lmps = self._calculate_lmps()

    def _calculate_lmps(self) -> Dict[str, float]:
        # LMP = Base Price + (PTDF * Shadow Price)
        return {
            node: self.state.base_price + (ptdf * self.state.shadow_price)
            for node, ptdf in self.state.ptdfs.items()
        }

    def get_ftr_payoff(self, src: str, snk: str, mw: float) -> float:
        return (self.lmps[snk] - self.lmps[src]) * mw


if __name__ == "__main__":
    # Case: Line A-B is congested.
    # To send power from A to B, 80% of it must go over the A-B line.
    # To send power from C to B, only 20% goes over the A-B line.

    congestion_context = GridState(
        base_price=20.0,
        shadow_price=50.0,  # Relieving the line saves $50/MWh
        ptdfs={'A': 0.0, 'B': 1.0, 'C': 0.4}
    )

    sim = ShadowPriceSimulator(congestion_context)

    print("Computed LMPs based on Shadow Price:")
    for node, price in sim.lmps.items():
        print(f"Node {node}: ${price:.2f}")

    # Calculate FTR Payoffs
    positions = [('A', 'B', 50), ('C', 'B', 20)]
    print("\nFTR Results:")
    for src, snk, mw in positions:
        payoff = sim.get_ftr_payoff(src, snk, mw)
        print(f"FTR {src}->{snk} ({mw}MW): ${payoff:.2f}")