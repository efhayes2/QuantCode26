import numpy as np
import time

from storage_helper import get_fwd_curve
from storage_helper import MarketEnvironment, StorageContract


def generate_paths(mkt):
    """Generates price paths using the daily mean-reversion logic."""
    fwd = get_fwd_curve(mkt.T_days)
    S = np.zeros((mkt.T_days + 1, mkt.M))
    x = np.zeros((mkt.T_days + 1, mkt.M))

    x[0] = np.log(fwd[0])
    S[0] = fwd[0]
    phi = np.exp(-mkt.kappa)
    stdev = np.sqrt((mkt.sigma ** 2 / (2 * mkt.kappa)) * (1 - np.exp(-2 * mkt.kappa)))

    for t in range(1, mkt.T_days + 1):
        m = np.log(fwd[t]) - (mkt.sigma ** 2 / (2 * mkt.kappa))
        x[t] = x[t - 1] * phi + m * (1 - phi) + stdev * np.random.standard_normal(mkt.M)
        S[t] = np.exp(x[t])
    return S


def get_poly_basis(prices):
    """Constructs the design matrix for degree-3 polynomial regression."""
    return np.column_stack([np.ones(len(prices)), prices, prices ** 2, prices ** 3])


def fast_lstsq_coeffs(X_poly, y):
    """Direct solver for polynomial coefficients."""
    return np.linalg.lstsq(X_poly, y, rcond=None)[0]


def run_backward_training(mkt, con):
    """Training phase: Solves for the optimal policy coefficients."""
    S = generate_paths(mkt)
    fwd = get_fwd_curve(mkt.T_days)
    grid = con.inventory_grid
    num_v = len(grid)

    V = np.zeros((num_v, mkt.M))
    all_coeffs = [[None for _ in range(num_v)] for _ in range(mkt.T_days)]

    # Terminal Value: (Inventory - Target) * Terminal Price
    for i, v in enumerate(grid):
        V[i, :] = (v - con.target_vol) * con.terminal_price

    for t in range(mkt.T_days - 1, -1, -1):
        if t % 50 == 0: print(f"  [Backward] Day {t:3} | Fwd: {fwd[t]:5.2f}")

        X_poly = get_poly_basis(S[t, :])
        cv_surface = np.zeros((num_v, mkt.M))

        for i in range(num_v):
            coeffs = fast_lstsq_coeffs(X_poly, V[i, :])
            all_coeffs[t][i] = coeffs
            cv_surface[i, :] = X_poly @ coeffs

        new_V = np.zeros_like(V)
        for i, v_start in enumerate(grid):
            # Admissible: inject 1, withdraw 3
            next_idx = np.arange(max(0, i - 3), min(num_v, i + 2))
            best_val = np.full(mkt.M, -1e15)

            for idx in next_idx:
                u = grid[idx] - v_start
                cash = -u * S[t, :]
                best_val = np.maximum(best_val, cash + cv_surface[idx, :])
            new_V[i, :] = best_val
        V = new_V

    training_val = np.mean(V[int(con.initial_vol / con.grid_step), :])
    return training_val, all_coeffs


def run_forward_validation(mkt, con, all_coeffs):
    """Validation phase: Applies the frozen policy to fresh paths."""
    S_new = generate_paths(mkt)
    grid = con.inventory_grid
    num_v = len(grid)

    current_idx = np.full(mkt.M, int(con.initial_vol / con.grid_step))
    total_cash = np.zeros(mkt.M)

    for t in range(mkt.T_days):
        if t % 50 == 0: print(f"  [Forward] Day {t:3}")

        X_poly = get_poly_basis(S_new[t, :])
        new_idx = np.zeros(mkt.M, dtype=int)

        # Grouping paths by inventory level for vectorization
        for v_i in range(num_v):
            mask = (current_idx == v_i)
            if not np.any(mask): continue

            admissible = np.arange(max(0, v_i - 3), min(num_v, v_i + 2))
            path_prices = S_new[t, mask]
            path_X_poly = X_poly[mask]

            best_score = np.full(np.sum(mask), -1e15)
            best_move_idx = np.full(np.sum(mask), v_i, dtype=int)

            for next_v in admissible:
                u = grid[next_v] - grid[v_i]
                # Decision Score = Current Cash + (Training Coeffs * Current Basis)
                score = (-u * path_prices) + (path_X_poly @ all_coeffs[t][next_v])

                better = score > best_score
                best_score[better] = score[better]
                best_move_idx[better] = next_v

            total_cash[mask] -= (grid[best_move_idx] - grid[v_i]) * path_prices
            new_idx[mask] = best_move_idx

        current_idx = new_idx

    final_v = grid[current_idx]
    total_cash += (final_v - con.target_vol) * con.terminal_price
    return np.mean(total_cash)


if __name__ == "__main__":
    mkt = MarketEnvironment(sigma=0.0945, M=2000)
    con = StorageContract()

    print("=" * 60)
    print(f"STORAGE VALUATION: Training on {mkt.M} paths...")

    t0 = time.time()
    train_val, coefs = run_backward_training(mkt, con)
    t1 = time.time()
    valid_val = run_forward_validation(mkt, con, coefs)
    t2 = time.time()

    print("\n" + "=" * 60)
    print(f"BACKWARD VALUE (UB): ${train_val:,.2f} | Time: {t1 - t0:.2f}s")
    print(f"FORWARD VALUE  (LB): ${valid_val:,.2f} | Time: {t2 - t1:.2f}s")
    print(f"LSMC BIAS: {((train_val - valid_val) / valid_val) * 100:.4f}%")
    print("=" * 60)