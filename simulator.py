"""
simulator.py
------------
Monte Carlo path generation engine for the Quant Desk Toolkit.

Implements:
  - HullWhiteSimulator : Exact discretization of the Hull-White one-factor
                         short rate model, calibrated to the initial OIS curve.
                         Includes affine analytical pricing for pathwise valuation.
  - GBMSimulator       : Geometric Brownian Motion for equity / FX underlyings.
  - MonteCarloEngine   : Combined engine with correlated path generation.
"""

import numpy as np
from typing import Dict, List, Optional
from curve_factory import Curve

# Fallback for antithetic variates if math_helpers is unavailable
try:
    from common_utils.math_helpers import generate_antithetic_normals
except ImportError:
    def generate_antithetic_normals(shape: tuple) -> np.ndarray:
        n_paths = shape[-1]
        half_paths = n_paths // 2
        draw_shape = list(shape)
        draw_shape[-1] = half_paths
        Z = np.random.standard_normal(draw_shape)
        return np.concatenate([Z, -Z], axis=-1)

class HullWhiteSimulator:
    """1-Factor Hull-White short rate model."""
    def __init__(self, a: float, sigma: float, curve: Curve):
        self.a = a
        self.sigma = sigma
        self.curve = curve

    def alpha(self, t: float, dt_epsilon: float = 1e-4) -> float:
        """Deterministic shift alpha(t) for perfect curve calibration."""
        f_0_t = self.curve.forward_rate(t, t + dt_epsilon)
        vol_adj = (self.sigma**2 / (2.0 * self.a**2)) * (1.0 - np.exp(-self.a * t))**2
        return f_0_t + vol_adj

    def step_exact_x(self, x_t: np.ndarray, dt: float, z: np.ndarray) -> np.ndarray:
        """Exact analytical step for the underlying OU process x(t)."""
        decay = np.exp(-self.a * dt)
        variance = (self.sigma**2 / (2.0 * self.a)) * (1.0 - np.exp(-2.0 * self.a * dt))
        return x_t * decay + np.sqrt(variance) * z

    def path_discount_factor(self, r_t: np.ndarray, time_grid: np.ndarray, t_idx: int, T: float) -> np.ndarray:
        """
        Calculates the zero-coupon bond price P(t, T) given the simulated 
        short rate r(t) at simulation time step t.
        Uses the affine yield structure: P(t, T) = A(t, T) * exp(-B(t, T) * r(t))
        """
        t = time_grid[t_idx]
        if t >= T:
            return np.ones_like(r_t)
            
        tau = T - t
        B = (1.0 - np.exp(-self.a * tau)) / self.a
        
        # Extract initial curve parameters
        P_0_t = self.curve.df(t)
        P_0_T = self.curve.df(T)
        f_0_t = self.curve.forward_rate(t, t + 1e-4)
        
        # Compute affine A(t, T) parameter
        A = (P_0_T / P_0_t) * np.exp(
            B * f_0_t - (self.sigma**2 / (4 * self.a)) * (1.0 - np.exp(-2.0 * self.a * t)) * (B**2)
        )
        
        return A * np.exp(-B * r_t)

class GBMSimulator:
    """Geometric Brownian Motion driven by the stochastic short rate."""
    def __init__(self, name: str, spot: float, vol: float):
        self.name = name
        self.spot = spot
        self.vol = vol

    def step_exact(self, s_t: np.ndarray, r_avg_dt: np.ndarray, dt: float, z: np.ndarray) -> np.ndarray:
        drift = r_avg_dt - 0.5 * (self.vol**2) * dt
        diffusion = self.vol * np.sqrt(dt) * z
        return s_t * np.exp(drift + diffusion)

class MonteCarloEngine:
    """Correlated Monte Carlo path generator."""
    def __init__(self, hw_simulator: HullWhiteSimulator, gbm_simulators: Optional[List[GBMSimulator]] = None, correlation_matrix: Optional[np.ndarray] = None):
        self.hw = hw_simulator
        self.gbms = gbm_simulators or []
        n_assets = 1 + len(self.gbms)
        if correlation_matrix is not None:
            self.chol = np.linalg.cholesky(correlation_matrix)
        else:
            self.chol = np.eye(n_assets)

    def run(self, time_grid: np.ndarray, n_paths: int = 1000, use_antithetic: bool = True) -> Dict[str, np.ndarray]:
        n_steps = len(time_grid) - 1
        n_assets = 1 + len(self.gbms)
        r_paths = np.zeros((n_paths, len(time_grid)))
        df_paths = np.ones((n_paths, len(time_grid)))
        x_t = np.zeros(n_paths) 
        gbm_paths = {gbm.name: np.zeros((n_paths, len(time_grid))) for gbm in self.gbms}
        
        r_paths[:, 0] = self.hw.alpha(0.0)
        for gbm in self.gbms:
            gbm_paths[gbm.name][:, 0] = gbm.spot
            
        for i in range(n_steps):
            dt = time_grid[i+1] - time_grid[i]
            Z_indep = generate_antithetic_normals((n_assets, n_paths)) if use_antithetic else np.random.standard_normal((n_assets, n_paths))
            Z_corr = self.chol @ Z_indep 
            Z_hw = Z_corr[0, :]
            x_t = self.hw.step_exact_x(x_t, dt, Z_hw)
            r_paths[:, i+1] = x_t + self.hw.alpha(time_grid[i+1])
            r_avg_dt = 0.5 * (r_paths[:, i] + r_paths[:, i+1]) * dt
            df_paths[:, i+1] = df_paths[:, i] * np.exp(-r_avg_dt)
            for j, gbm in enumerate(self.gbms):
                gbm_paths[gbm.name][:, i+1] = gbm.step_exact(gbm_paths[gbm.name][:, i], r_avg_dt, dt, Z_corr[j+1, :])
                
        return {"time_grid": time_grid, "r_paths": r_paths, "path_dfs": df_paths, "gbm_paths": gbm_paths}