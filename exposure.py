"""
exposure.py
-----------
Pathwise exposure aggregation and netting engine for the Quant Desk Toolkit.

Takes the raw stochastic paths from simulator.py, calculates the Mark-to-Future 
(MTM) of every trade at every time step, applies netting rules, and generates 
the Expected Exposure (EE), Expected Negative Exposure (ENE), and Potential 
Future Exposure (PFE) profiles.
"""

import numpy as np
from typing import List, Callable, Dict
from dataclasses import dataclass

@dataclass
class NettingSet:
    """A collection of trades netted under a single ISDA Master Agreement."""
    name: str
    trade_valuators: List[Callable[[np.ndarray, int], np.ndarray]]
    threshold: float = 0.0
    minimum_transfer_amount: float = 0.0
    mpor: float = 10.0 / 252.0 

class ExposureEngine:
    """Computes pathwise Mark-to-Future (MTM) and aggregates exposures."""
    def __init__(self, mc_results: dict, hw_sim):
        self.time_grid = mc_results["time_grid"]
        self.r_paths = mc_results["r_paths"]
        self.path_dfs = mc_results["path_dfs"]
        self.hw_sim = hw_sim # Corrected: Saved for pathwise pricing

    def compute_mtm_matrix_vectorized(self, valuator: Callable[[np.ndarray, int], np.ndarray]) -> np.ndarray:
        """Returns a shape (n_paths, n_steps) matrix of future PVs."""
        n_paths, n_steps = self.r_paths.shape[0], len(self.time_grid)
        mtm = np.zeros((n_paths, n_steps))
        for j in range(n_steps):
            mtm[:, j] = valuator(self.r_paths[:, j], j) # Passes 1D slice
        return mtm

    def netting_set_mtm(self, netting_set: NettingSet) -> np.ndarray:
        """Aggregates the pathwise MTM of all trades in a netting set."""
        net_mtm = np.zeros((self.r_paths.shape[0], len(self.time_grid)))
        for valuator in netting_set.trade_valuators:
            net_mtm += self.compute_mtm_matrix_vectorized(valuator)
        return net_mtm

    def exposure_summary(self, netting_set: NettingSet, use_mpor: bool = False) -> dict:
        """Calculates standard exposure metrics (EE, ENE, Peak PFE, EPE)."""
        net_mtm = self.netting_set_mtm(netting_set)
        EE = np.mean(np.maximum(net_mtm, 0.0), axis=0)
        ENE = np.mean(np.minimum(net_mtm, 0.0), axis=0)
        PFE_paths = np.percentile(np.maximum(net_mtm, 0.0), 95, axis=0)
        
        horizon = self.time_grid[-1]
        EPE = np.sum(0.5 * (EE[:-1] + EE[1:]) * np.diff(self.time_grid)) / horizon if horizon > 0 else 0.0

        return {"EE": EE, "ENE": ENE, "PFE_95": PFE_paths, "peak_PFE": np.max(PFE_paths), "EPE": EPE, "mtm_paths": net_mtm}

    def irs_valuator(self, fixed_rate: float, payment_dates: np.ndarray, notional: float, payer: bool = True):
        """Creates a vectorized fast-pricing closure for an IRS."""
        def valuator(r_t: np.ndarray, t_idx: int) -> np.ndarray:
            t = self.time_grid[t_idx]
            annuity = np.zeros_like(r_t)
            for T_i in payment_dates:
                if T_i > t:
                    # FIX: Pass 1D 'r_t', not the full matrix
                    annuity += 0.5 * self.hw_sim.path_discount_factor(r_t, self.time_grid, t_idx, T_i)
            
            pv_fixed = fixed_rate * notional * annuity
            maturity = payment_dates[-1] if len(payment_dates) > 0 else 0.0
            pv_float = notional * (1.0 - self.hw_sim.path_discount_factor(r_t, self.time_grid, t_idx, maturity)) if maturity > t else np.zeros_like(r_t)
                
            return (pv_float - pv_fixed) if payer else (pv_fixed - pv_float)
        return valuator