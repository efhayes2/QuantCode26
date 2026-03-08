import numpy as np
import matplotlib.pyplot as plt


class ExposureSimulator:
    def __init__(self, s0, drift, vol_term_structure, t_max, steps, n_paths=1000):
        self.s0 = s0
        self.drift = drift
        self.vols = np.array(vol_term_structure)
        self.t_max = t_max
        self.steps = steps
        self.n_paths = n_paths
        self.dt = t_max / steps
        self.time_grid = np.linspace(0, t_max, steps + 1)

    def simulate_paths(self):
        z = np.random.standard_normal((self.n_paths, self.steps))

        # Log-returns incorporating term structure
        log_returns = (self.drift - 0.5 * self.vols ** 2) * self.dt + \
                      (self.vols * np.sqrt(self.dt) * z)

        cumulative_log_returns = np.cumsum(log_returns, axis=1)
        full_log_returns = np.hstack([np.zeros((self.n_paths, 1)), cumulative_log_returns])
        self.paths = self.s0 * np.exp(full_log_returns)
        return self.paths

    def calculate_and_plot_metrics(self, strike):
        exposures = np.maximum(self.paths - strike, 0)

        # Calculate Expected Exposure (Mean)
        ee_profile = np.mean(exposures, axis=0)

        # Calculate PFE 95th Percentile (Value at Risk for credit)
        pfe_95 = np.percentile(exposures, 95, axis=0)

        # Visualization
        plt.figure(figsize=(12, 7))
        plt.plot(self.time_grid, self.paths[:100].T, color='gray', alpha=0.2, label='Sample Price Paths')

        plt.plot(self.time_grid, ee_profile, color='blue', linewidth=3, label='Expected Exposure (EE)')
        plt.plot(self.time_grid, pfe_95, color='red', linestyle='--', linewidth=2, label='PFE (95th percentile)')

        plt.title('Counterparty Credit Risk: EE vs PFE95')
        plt.xlabel('Time (Years)')
        plt.ylabel('Exposure Value')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.show()

    def calculate_and_save_metrics(self, strike, file_path):
        exposures = np.maximum(self.paths - strike, 0)

        # Calculate Expected Exposure (Mean)
        ee_profile = np.mean(exposures, axis=0)

        # Calculate PFE 95th Percentile
        pfe_95 = np.percentile(exposures, 95, axis=0)

        # Visualization
        plt.figure(figsize=(12, 7))
        plt.plot(self.time_grid, self.paths[:100].T, color='gray', alpha=0.15)
        plt.plot(self.time_grid, ee_profile, color='blue', linewidth=3, label='EE')
        plt.plot(self.time_grid, pfe_95, color='red', linestyle='--', linewidth=2, label='PFE 95')

        plt.title('Counterparty Credit Risk: EE vs PFE95')
        plt.xlabel('Time (Years)')
        plt.ylabel('Exposure Value')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # SAVE TO PDF
        # Ensure the directory C:\data exists before running this
        try:
            plt.savefig(file_path)
            print(f"Successfully saved plot to: {file_path}")
        except Exception as e:
            print(f"Error saving file: {e}")

        plt.close()  # Good practice to close the plot to free memory

# --- Execution ---
steps = 252  # Daily steps for one year
vol_ts = np.linspace(0.15, 0.35, steps)  # Vol smile/term structure

sim = ExposureSimulator(s0=100, drift=0.02, vol_term_structure=vol_ts, t_max=1.0, steps=steps)
sim.simulate_paths()
sim.calculate_and_plot_metrics(strike=102)

sim.calculate_and_save_metrics(102, r'c:\data\cva_test.png')