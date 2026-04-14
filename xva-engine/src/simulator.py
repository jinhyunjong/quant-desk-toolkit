"""
simulator.py
------------
Monte Carlo path generation engine for the Quant Desk Toolkit.

Implements:
  - HullWhiteSimulator : Exact discretization of the Hull-White one-factor
                         short rate model, calibrated to the initial OIS curve
  - GBMSimulator       : Geometric Brownian Motion for equity / FX underlyings,
                         driven by the stochastic short rate
  - MonteCarloEngine   : Combined engine with correlated path generation via
                         Cholesky decomposition

Design principles
-----------------
- Hull-White uses the EXACT discretization (no Euler-Maruyama error) via the
  Ornstein-Uhlenbeck analytical solution. This is critical for XVA accuracy
  since discretization error in short rates compounds into discount factor
  errors across long simulation horizons.
- Variance reduction via antithetic variates (from math_helpers) is applied
  by default to halve Monte Carlo standard error at no additional model cost.
- All simulators output shape (n_paths, n_steps+1) arrays where index 0
  corresponds to t=0, consistent with the time_grid passed in.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from curve_factory import Curve
from common_utils.math_helpers import antithetic_variates


# =============================================================================
# HULL-WHITE ONE-FACTOR SHORT RATE SIMULATOR
# =============================================================================

class HullWhiteSimulator:
    """
    Exact discretization of the Hull-White one-factor model.

    Model specification
    -------------------
    The short rate r(t) is decomposed as:

        r(t) = x(t) + α(t)

    where x(t) is the zero-mean Ornstein-Uhlenbeck (O-U) process:

        dx(t) = -a * x(t) dt + σ * dW(t)

    and α(t) is the time-dependent drift calibrated to fit the initial
    OIS discount curve exactly:

        α(t) = f(0, t) + σ² / (2a²) * (1 - e^{-at})²

    where f(0, t) is the instantaneous forward rate at t from the OIS curve.

    Exact discretization
    --------------------
    The O-U process has an analytical solution, giving the exact transition:

        x(t + Δt) = x(t) * e^{-aΔt}
                    + σ * sqrt((1 - e^{-2aΔt}) / (2a)) * Z

    where Z ~ N(0, 1). No discretization error.

    The short rate at each step:

        r(t + Δt) = (r(t) - α(t)) * e^{-aΔt} + α(t + Δt)
                    + σ * sqrt((1 - e^{-2aΔt}) / (2a)) * Z

    Stochastic discount factors
    ---------------------------
    The Hull-White model admits closed-form bond prices conditional on r(t):

        P(t, T | r(t)) = A(t, T) * exp(-B(t, T) * r(t))

    where:
        B(t, T)      = (1 - e^{-a(T-t)}) / a
        ln A(t, T)   = ln(P(0,T) / P(0,t))
                       + B(t,T) * f(0,t)
                       - σ² / (4a) * B(t,T)² * (1 - e^{-2at})

    Parameters
    ----------
    a : float
        Mean reversion speed. Typical range: 0.01 – 0.15.
        Higher a → faster reversion to α(t), less long-rate volatility.
    sigma : float
        Short rate volatility. Typical range: 0.005 – 0.02.
    curve : Curve
        Initial OIS discount curve used for α(t) calibration and
        instantaneous forward rate f(0, t).
    """

    def __init__(self, a: float, sigma: float, curve: Curve) -> None:
        if a <= 0:
            raise ValueError(f"Mean reversion speed a must be positive, got {a}.")
        if sigma <= 0:
            raise ValueError(f"Volatility sigma must be positive, got {sigma}.")
        self.a     = a
        self.sigma = sigma
        self.curve = curve

    # -------------------------------------------------------------------------
    # Calibration helpers
    # -------------------------------------------------------------------------

    def instantaneous_forward_rate(self, t: float, dt: float = 1e-5) -> float:
        """
        Instantaneous forward rate f(0, t) derived numerically from the
        OIS discount curve via:

            f(0, t) = -d/dt ln P(0, t)
                    ≈ -[ln P(0, t+dt) - ln P(0, t-dt)] / (2*dt)
        """
        t  = max(t, dt)
        p1 = float(self.curve.df(t + dt))
        p0 = float(self.curve.df(max(t - dt, 1e-8)))
        return float(-(np.log(p1) - np.log(p0)) / (2 * dt))

    def alpha(self, t: float) -> float:
        """
        Time-dependent drift α(t) calibrated to fit the OIS curve.

            α(t) = f(0, t) + σ² / (2a²) * (1 - e^{-at})²
        """
        f0t = self.instantaneous_forward_rate(t)
        convexity = (self.sigma ** 2 / (2 * self.a ** 2)) * (1 - np.exp(-self.a * t)) ** 2
        return float(f0t + convexity)

    # -------------------------------------------------------------------------
    # Path simulation
    # -------------------------------------------------------------------------

    def simulate(
        self,
        time_grid: np.ndarray,
        n_paths: int,
        use_antithetic: bool = True,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate short rate paths on the given time grid.

        Returns
        -------
        r_paths : np.ndarray, shape (n_paths, len(time_grid))
            Simulated short rate paths.
        alpha_grid : np.ndarray, shape (len(time_grid),)
            Drift α(t) at each time grid point.
        z : np.ndarray, shape (n_paths, n_steps)
            Brownian increments used to generate r_paths. Passed to GBMSimulator 
            to construct correlated equity/FX paths via Cholesky decomposition.
        """
        time_grid = np.asarray(time_grid, dtype=float)
        if time_grid[0] != 0.0:
            raise ValueError("time_grid must start at 0.")
        if np.any(np.diff(time_grid) <= 0):
            raise ValueError("time_grid must be strictly increasing.")

        n_steps    = len(time_grid) - 1
        rng        = np.random.default_rng(seed)

        # Pre-compute alpha and instantaneous forward rates on grid
        alpha_grid = np.array([self.alpha(t) for t in time_grid])
        r0         = alpha_grid[0]   # initial short rate from curve

        # Number of base paths (antithetic doubles them)
        n_base = n_paths // 2 if use_antithetic else n_paths

        # Standard normal draws: shape (n_base, n_steps)
        z_base = rng.standard_normal((n_base, n_steps))
        z      = antithetic_variates(z_base) if use_antithetic else z_base

        # Initialise short rate array
        r_paths      = np.zeros((n_paths, n_steps + 1))
        r_paths[:, 0] = r0

        for i in range(n_steps):
            t      = time_grid[i]
            t_next = time_grid[i + 1]
            dt     = t_next - t

            # Exact O-U discretization standard deviation
            std_ou = self.sigma * np.sqrt((1 - np.exp(-2 * self.a * dt)) / (2 * self.a))

            # Exact transition
            x_t           = r_paths[:, i] - alpha_grid[i]
            x_next        = x_t * np.exp(-self.a * dt) + std_ou * z[:, i]
            r_paths[:, i + 1] = x_next + alpha_grid[i + 1]

        return r_paths, alpha_grid, z

    # -------------------------------------------------------------------------
    # Stochastic discount factors
    # -------------------------------------------------------------------------

    def path_discount_factor(
        self,
        r_paths: np.ndarray,
        time_grid: np.ndarray,
        t_idx: int,
        T: float,
    ) -> np.ndarray:
        """
        Compute the stochastic discount factor P(t, T | r(t)) for each path
        at simulation time t = time_grid[t_idx], for maturity T > t.
        """
        t = time_grid[t_idx]
        if T <= t:
            raise ValueError(f"T ({T:.4f}) must be greater than t ({t:.4f}).")

        tau = T - t
        a, sigma = self.a, self.sigma

        # B(t, T)
        B = (1 - np.exp(-a * tau)) / a

        # ln A(t, T)
        p0T   = float(self.curve.df(T))
        p0t   = float(self.curve.df(t)) if t > 0 else 1.0
        f0t   = self.instantaneous_forward_rate(t)

        ln_A = (
            np.log(p0T / p0t)
            + B * f0t
            - (sigma ** 2 / (4 * a)) * B ** 2 * (1 - np.exp(-2 * a * t))
        )

        # P(t, T | r(t)) = exp(ln_A - B * r(t))
        r_t = r_paths[:, t_idx]
        return np.exp(ln_A - B * r_t)

    def numeraire_rebased_df(
        self,
        r_paths: np.ndarray,
        time_grid: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the path-wise money market account (numeraire) discount factor
        B(0, t) = exp(-integral_0^t r(s) ds) via trapezoidal integration.
        """
        n_paths, n_steps_plus1 = r_paths.shape
        path_dfs = np.ones((n_paths, n_steps_plus1))

        for i in range(1, n_steps_plus1):
            dt = time_grid[i] - time_grid[i - 1]
            avg_r = 0.5 * (r_paths[:, i - 1] + r_paths[:, i])
            path_dfs[:, i] = path_dfs[:, i - 1] * np.exp(-avg_r * dt)

        return path_dfs

    def diagnostics(
        self,
        r_paths: np.ndarray,
        time_grid: np.ndarray,
        alpha_grid: np.ndarray,
    ) -> dict:
        """Compute calibration diagnostics vs theoretical Hull-White moments."""
        sim_mean = r_paths.mean(axis=0)
        sim_std  = r_paths.std(axis=0)

        theo_var = (self.sigma ** 2 / (2 * self.a)) * (
            1 - np.exp(-2 * self.a * time_grid)
        )
        theo_std = np.sqrt(theo_var)

        return {
            "time_grid"        : time_grid,
            "simulated_mean"   : sim_mean,
            "theoretical_mean" : alpha_grid,
            "simulated_std"    : sim_std,
            "theoretical_std"  : theo_std,
            "mean_error"       : np.abs(sim_mean - alpha_grid),
        }


# =============================================================================
# GBM SIMULATOR (EQUITY / FX)
# =============================================================================

class GBMSimulator:
    """
    Geometric Brownian Motion simulator for equity and FX underlyings.
    """

    def __init__(
        self,
        S0: float,
        sigma_S: float,
        rho: float = 0.0,
    ) -> None:
        if S0 <= 0:
            raise ValueError("S0 must be positive.")
        if sigma_S <= 0:
            raise ValueError("sigma_S must be positive.")
        if not -1.0 <= rho <= 1.0:
            raise ValueError("rho must be in [-1, 1].")

        self.S0      = S0
        self.sigma_S = sigma_S
        self.rho     = rho

    def simulate(
        self,
        time_grid: np.ndarray,
        r_paths: np.ndarray,
        z_ir: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Simulate asset price paths correlated with short rate paths.
        """
        time_grid = np.asarray(time_grid, dtype=float)
        n_paths, n_steps_plus1 = r_paths.shape
        n_steps = n_steps_plus1 - 1

        rng = np.random.default_rng(seed)
        z_indep = rng.standard_normal((n_paths, n_steps))

        if z_ir is not None:
            # Cholesky decomposition for correlated normals
            z_asset = self.rho * z_ir + np.sqrt(1 - self.rho ** 2) * z_indep
        else:
            z_asset = z_indep

        S_paths       = np.zeros((n_paths, n_steps_plus1))
        S_paths[:, 0] = self.S0

        for i in range(n_steps):
            dt   = time_grid[i + 1] - time_grid[i]
            r_t  = r_paths[:, i]
            drift = (r_t - 0.5 * self.sigma_S ** 2) * dt
            diffusion = self.sigma_S * np.sqrt(dt) * z_asset[:, i]
            S_paths[:, i + 1] = S_paths[:, i] * np.exp(drift + diffusion)

        return S_paths


# =============================================================================
# COMBINED MONTE CARLO ENGINE
# =============================================================================

@dataclass
class MonteCarloEngine:
    """
    Combined Monte Carlo engine for joint simulation of interest rates
    and equity/FX underlyings with full correlation structure.
    """

    hw_simulator   : HullWhiteSimulator
    gbm_simulators : list = field(default_factory=list)

    def run(
        self,
        time_grid: np.ndarray,
        n_paths: int,
        use_antithetic: bool = True,
        seed: Optional[int] = None,
    ) -> Dict:
        """
        Run the full Monte Carlo simulation and capture paths.
        """
        # Simulate short rates — capture z_ir for downstream correlation wiring
        r_paths, alpha_grid, z_ir = self.hw_simulator.simulate(
            time_grid,
            n_paths,
            use_antithetic=use_antithetic,
            seed=seed,
        )

        # Path-wise numeraire discount factors
        path_dfs = self.hw_simulator.numeraire_rebased_df(r_paths, time_grid)

        # Simulate correlated equity/FX paths — pass z
