"""
curve_factory.py
----------------
Multi-curve construction for the Quant Desk Toolkit.

Implements:
  - Base Curve class with discount factor and zero rate interpolation
  - OIS discount curve bootstrapped from swap rates (SOFR/Fed Funds)
  - SOFR projection curve for floating cash flow projection
  - Forward rate computation across both curves

Design follows the post-LIBOR multi-curve framework:
  - Discounting : OIS curve (risk-free, CSA-consistent)
  - Projection  : SOFR curve (floating leg forward rates)
"""

import numpy as np
from typing import Optional, Dict
from common_utils.math_helpers import (
    log_linear_interp,
    cubic_spline_interp,
    brent_solver,
)


# =============================================================================
# BASE CURVE
# =============================================================================

class Curve:
    """
    Base discount curve built from a tenor grid and discount factors.

    Supports:
      - Discount factor retrieval via log-linear interpolation
      - Zero rate retrieval via cubic spline interpolation
      - Instantaneous and period forward rate computation

    Parameters
    ----------
    tenors : array-like
        Time grid in years, strictly increasing, starting at 0.
    discount_factors : array-like
        Discount factors P(0, T) at each tenor. P(0, 0) = 1.0.
    label : str, optional
        Descriptive label for the curve (e.g. 'OIS', 'SOFR').
    """

    def __init__(
        self,
        tenors: np.ndarray,
        discount_factors: np.ndarray,
        label: str = "Curve",
    ) -> None:
        tenors = np.asarray(tenors, dtype=float)
        discount_factors = np.asarray(discount_factors, dtype=float)

        if tenors.shape != discount_factors.shape:
            raise ValueError("tenors and discount_factors must have the same length.")
        if np.any(np.diff(tenors) <= 0):
            raise ValueError("tenors must be strictly increasing.")
        if np.any(discount_factors <= 0) or np.any(discount_factors > 1.0):
            raise ValueError("Discount factors must be in (0, 1].")
        if not np.isclose(discount_factors[0], 1.0, atol=1e-6):
            raise ValueError("P(0,0) must equal 1.0.")

        self.tenors = tenors
        self.discount_factors = discount_factors
        self.label = label

        # Derive continuously compounded zero rates: r(T) = -ln(P(0,T)) / T
        # Avoid division by zero at T=0 using np.errstate
        with np.errstate(divide="ignore", invalid="ignore"):
            self.zero_rates = np.where(
                tenors > 0,
                -np.log(discount_factors) / tenors,
                0.0,
            )
            # Extrapolate T=0 rate from the first positive tenor to avoid a hard zero
            if len(tenors) > 1:
                self.zero_rates[0] = self.zero_rates[1]

    # -------------------------------------------------------------------------
    # Core retrieval methods
    # -------------------------------------------------------------------------

    def df(self, t: float | np.ndarray) -> float | np.ndarray:
        """
        Return discount factor P(0, t) via log-linear interpolation.

        Parameters
        ----------
        t : float or np.ndarray
            Time(s) in years. Must be >= 0.

        Returns
        -------
        float or np.ndarray
            Discount factor(s) P(0, t).
        """
        t = np.asarray(t, dtype=float)
        if np.any(t < 0):
            raise ValueError("t must be non-negative.")
        return log_linear_interp(t, self.tenors, self.discount_factors)

    def zero_rate(self, t: float | np.ndarray) -> float | np.ndarray:
        """
        Return continuously compounded zero rate r(0, t) via cubic spline.

        Parameters
        ----------
        t : float or np.ndarray
            Time(s) in years. Must be > 0.

        Returns
        -------
        float or np.ndarray
            Zero rate(s) at t.
        """
        t = np.asarray(t, dtype=float)
        if np.any(t <= 0):
            raise ValueError("t must be strictly positive for zero rate retrieval.")
        return cubic_spline_interp(t, self.tenors, self.zero_rates)

    def forward_rate(
        self,
        t1: float | np.ndarray,
        t2: float | np.ndarray,
        compounding: str = "continuous",
    ) -> float | np.ndarray:
        """
        Return the forward rate f(t1, t2).

        Parameters
        ----------
        t1 : float or np.ndarray
            Start of the forward period (years). Must be >= 0.
        t2 : float or np.ndarray
            End of the forward period (years). Must be > t1.
        compounding : str
            'continuous' or 'simple'. SOFR/OIS projections typically use 'simple'.

        Returns
        -------
        float or np.ndarray
            Forward rate over [t1, t2].
        """
        t1 = np.asarray(t1, dtype=float)
        t2 = np.asarray(t2, dtype=float)
        
        if np.any(t2 <= t1):
            raise ValueError("t2 must be strictly greater than t1.")
            
        df1 = self.df(t1)
        df2 = self.df(t2)
        dt = t2 - t1
        
        if compounding == "continuous":
            return -np.log(df2 / df1) / dt
        elif compounding == "simple":
            return (df1 / df2 - 1.0) / dt
        else:
            raise ValueError(f"Unknown compounding method: {compounding}")

    def annuity(self, payment_times: np.ndarray) -> float:
        """
        Compute the annuity (PV01) as sum of discount factors over a schedule.
        """
        return float(np.sum(self.df(payment_times)))

    def __repr__(self) -> str:
        return (
            f"Curve(label='{self.label}', "
            f"tenors=[{self.tenors[0]:.2f}, ..., {self.tenors[-1]:.2f}], "
            f"n_pillars={len(self.tenors)})"
        )


# =============================================================================
# OIS DISCOUNT CURVE BOOTSTRAP
# =============================================================================

def bootstrap_ois_curve(
    deposit_tenors: np.ndarray,
    deposit_rates: np.ndarray,
    swap_tenors: np.ndarray,
    swap_rates: np.ndarray,
    day_count: float = 1.0,
) -> Curve:
    """
    Bootstrap an OIS discount curve analytically from deposits and par swaps.
    """
    deposit_tenors = np.asarray(deposit_tenors, dtype=float)
    deposit_rates  = np.asarray(deposit_rates,  dtype=float)
    swap_tenors    = np.asarray(swap_tenors,    dtype=float)
    swap_rates     = np.asarray(swap_rates,     dtype=float)

    pillar_tenors = [0.0]
    pillar_dfs    = [1.0]

    # --- Short end: deposits on simple interest ---
    for T, r in zip(deposit_tenors, deposit_rates):
        df = 1.0 / (1.0 + r * T * day_count)
        pillar_tenors.append(T)
        pillar_dfs.append(df)

    # --- Long end: OIS swap sequential bootstrap ---
    for T_n, K in zip(swap_tenors, swap_rates):
        tmp_curve = Curve(
            np.array(pillar_tenors),
            np.array(pillar_dfs),
            label="Bootstrap_tmp",
        )

        existing_payments = np.array([t for t in pillar_tenors if 0 < t < T_n])

        if len(existing_payments) == 0:
            df_n = 1.0 / (1.0 + K * T_n * day_count)
        else:
            all_times   = np.append(existing_payments, T_n)
            prev_times  = np.insert(existing_payments, 0, 0.0)
            tau         = all_times - prev_times

            intermediate_dfs = tmp_curve.df(existing_payments)
            annuity_so_far   = float(np.sum(tau[:-1] * intermediate_dfs * day_count))

            tau_n = tau[-1]
            df_n  = (1.0 - K * annuity_so_far) / (1.0 + K * tau_n * day_count)

        if df_n <= 0:
            raise RuntimeError(f"Bootstrap produced non-positive DF at T={T_n:.2f}.")

        pillar_tenors.append(T_n)
        pillar_dfs.append(df_n)

    return Curve(np.array(pillar_tenors), np.array(pillar_dfs), label="OIS")


# =============================================================================
# SOFR PROJECTION CURVE
# =============================================================================

def build_sofr_projection_curve(
    ois_curve: Curve,
    sofr_swap_tenors: np.ndarray,
    sofr_swap_rates: np.ndarray,
    day_count: float = 1.0,
) -> Curve:
    """
    Build a SOFR projection curve analytically using an OIS discount curve.
    """
    sofr_swap_tenors = np.asarray(sofr_swap_tenors, dtype=float)
    sofr_swap_rates  = np.asarray(sofr_swap_rates,  dtype=float)

    pillar_tenors = [0.0]
    pillar_dfs    = [1.0]

    for T_n, K in zip(sofr_swap_tenors, sofr_swap_rates):
        tmp_sofr = Curve(
            np.array(pillar_tenors),
            np.array(pillar_dfs),
            label="SOFR_tmp",
        )

        all_times  = np.array([t for t in pillar_tenors if 0 < t <= T_n] + [T_n])
        all_times  = np.unique(all_times)
        prev_times = np.insert(all_times[:-1], 0, 0.0)
        tau        = (all_times - prev_times) * day_count

        ois_dfs = ois_curve.df(all_times)
        pv_fixed = K * float(np.sum(tau * ois_dfs))

        pv_float_known = 0.0
        for i, t_i in enumerate(all_times[:-1]):
            t_prev   = prev_times[i]
            # Use simple compounding for the floating leg forward projection
            fwd_sofr = tmp_sofr.forward_rate(t_prev, t_i, compounding="simple") if t_i > 0 else 0.0
            pv_float_known += tau[i] * fwd_sofr * ois_dfs[i]

        T_prev = all_times[-2] if len(all_times) > 1 else 0.0
        df_sofr_prev = tmp_sofr.df(T_prev)
        ois_df_n     = float(ois_curve.df(T_n))
        
        residual  = pv_fixed - pv_float_known
        df_sofr_n = float(df_sofr_prev) * ois_df_n / (ois_df_n + residual)

        if df_sofr_n <= 0:
            raise RuntimeError(f"SOFR projection failed at T={T_n:.2f}.")

        if T_n not in pillar_tenors:
            pillar_tenors.append(T_n)
            pillar_dfs.append(df_sofr_n)

    return Curve(np.array(pillar_tenors), np.array(pillar_dfs), label="SOFR")


# =============================================================================
# CONVENIENCE BUILDER
# =============================================================================

def build_multi_curves(
    deposit_tenors: np.ndarray,
    deposit_rates: np.ndarray,
    ois_swap_tenors: np.ndarray,
    ois_swap_rates: np.ndarray,
    sofr_swap_tenors: Optional[np.ndarray] = None,
    sofr_swap_rates: Optional[np.ndarray] = None,
) -> Dict[str, Curve]:
    """
    Build a full multi-curve environment in one call.
    """
    ois_curve = bootstrap_ois_curve(
        deposit_tenors, deposit_rates,
        ois_swap_tenors, ois_swap_rates,
    )

    result = {"ois": ois_curve}

    if sofr_swap_tenors is not None and sofr_swap_rates is not None:
        sofr_curve = build_sofr_projection_curve(
            ois_curve,
            sofr_swap_tenors,
            sofr_swap_rates,
        )
        result["sofr"] = sofr_curve

    return result
