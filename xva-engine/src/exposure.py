"""
exposure.py
-----------
Counterparty exposure computation for the Quant Desk Toolkit.

Implements:
  - Portfolio revaluation across Monte Carlo paths and time steps
  - Netting set aggregation with bilateral close-out
  - Collateral and margin offset (VM and IM) with exact MPoR lagging
  - Expected Exposure (EE), Expected Positive Exposure (EPE),
    Potential Future Exposure (PFE), and Effective EPE profiles
  - Peak exposure and exposure-at-default summary metrics

Design
------
Exposure is computed in three stages:

  1. Revalue each instrument on each path at each simulation date
     using the Hull-White closed-form bond price formula from simulator.py.

  2. Aggregate instrument MTMs within netting sets, applying collateral
     offsets and the bilateral close-out netting convention.

  3. Summarise across paths to produce EE, EPE, PFE profiles consumed
     by xva.py for CVA/DVA/FVA pricing.

All exposures are expressed as present values discounted to t=0
using the path-wise numeraire (money market account) from simulator.py.

Collateral treatment
--------------------
Two collateral modes are supported:

  *Instantaneous CSA (use_mpor=False)*
      VM(t) = max(V_net(t) - threshold, 0)
      Residual = V_net(t) - VM(t) - IA

  *MPoR-lagged CSA (use_mpor=True)*
      C(t) = max(V_net(t - MPoR) - threshold, 0) + IA   [path-by-path]
      Exposure(t) = max(V_net(t) - C(t), 0)

  The MPoR-lagged mode is the correct treatment for margined netting sets
  under BCBS/ISDA SIMM. It captures the "gap risk" that arises because
  the collateral in transit at the time of default reflects the portfolio
  value MPoR days earlier, not the current value.
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Optional
from common_utils.math_helpers import trapezoidal_integration
from simulator import HullWhiteSimulator


# =============================================================================
# NETTING SET
# =============================================================================

@dataclass
class NettingSet:
    """
    A netting set groups trades that can be offset against each other
    upon counterparty default under a master netting agreement (e.g. ISDA).

    In the event of counterparty default, the close-out amount is the
    net MTM across all trades in the netting set:

        V_net(t) = sum_i V_i(t)

    Positive V_net → we are owed money (we have exposure).
    Negative V_net → we owe money (counterparty has exposure to us).

    Parameters
    ----------
    name : str
        Label for the netting set (e.g. counterparty name or LEI).
    trade_valuators : list of Callable
        List of functions f(r_at_t, t_idx) → np.ndarray of shape (n_paths,),
        each returning the MTM of one trade on all paths at a given time step.
    threshold : float
        Collateral threshold below which no VM is exchanged. Default 0
        (full two-way CSA).
    minimum_transfer_amount : float
        Minimum VM transfer size. Default 0.
        Applied symmetrically: VM is only called if VM_gross >= MTA.
    independent_amount : float
        Fixed independent amount (initial margin) posted by counterparty.
        Reduces exposure directly regardless of MPoR. Default 0.
    mpor : float
        Margin period of risk in years (e.g. 10/252 for 10 business days).
        Used when computing MPoR-lagged collateral in apply_collateral().
        Default 10/252.
    """
    name                    : str
    trade_valuators         : List[Callable]
    threshold               : float = 0.0
    minimum_transfer_amount : float = 0.0
    independent_amount      : float = 0.0
    mpor                    : float = 10 / 252


# =============================================================================
# PORTFOLIO REVALUATION ENGINE
# =============================================================================

class ExposureEngine:
    """
    Computes counterparty exposure profiles from Monte Carlo simulation output.

    Parameters
    ----------
    mc_results : dict
        Output dict from MonteCarloEngine.run(), containing:
            time_grid   : np.ndarray, shape (n_steps+1,)
            r_paths     : np.ndarray, shape (n_paths, n_steps+1)
            path_dfs    : np.ndarray, shape (n_paths, n_steps+1)
    hw_simulator : HullWhiteSimulator
        Calibrated Hull-White simulator, used for path discount factor
        computation when revaluing fixed income instruments.
    """

    def __init__(self, mc_results: dict, hw_simulator: HullWhiteSimulator) -> None:
        self.time_grid    = mc_results["time_grid"]
        self.r_paths      = mc_results["r_paths"]
        self.path_dfs     = mc_results["path_dfs"]
        self.hw_simulator = hw_simulator
        self.n_paths      = self.r_paths.shape[0]
        self.n_steps      = len(self.time_grid) - 1

    # -------------------------------------------------------------------------
    # Core MTM matrix computation
    # -------------------------------------------------------------------------

    def compute_mtm_matrix_vectorized(
        self,
        valuator: Callable[[np.ndarray, int], np.ndarray],
    ) -> np.ndarray:
        """
        Vectorized MTM computation — preferred for performance.

        Parameters
        ----------
        valuator : Callable[[np.ndarray, int], np.ndarray]
            Function f(r_at_t, t_idx) → np.ndarray of shape (n_paths,)
            returning MTM for all paths at time step t_idx simultaneously.

        Returns
        -------
        np.ndarray, shape (n_paths, n_steps+1)
        """
        mtm = np.zeros((self.n_paths, self.n_steps + 1))
        for j in range(self.n_steps + 1):
            mtm[:, j] = valuator(self.r_paths[:, j], j)
        return mtm

    # -------------------------------------------------------------------------
    # Netting set aggregation
    # -------------------------------------------------------------------------

    def netting_set_mtm(self, netting_set: NettingSet) -> np.ndarray:
        """
        Aggregate trade MTMs within a netting set.
        Net MTM = sum of individual trade MTMs (bilateral close-out netting).

        Returns
        -------
        np.ndarray, shape (n_paths, n_steps+1)
        """
        if not netting_set.trade_valuators:
            raise ValueError("Netting set must contain at least one trade valuator.")
            
        net_mtm = np.zeros((self.n_paths, self.n_steps + 1))
        for valuator in netting_set.trade_valuators:
            net_mtm += self.compute_mtm_matrix_vectorized(valuator)
            
        return net_mtm

    # -------------------------------------------------------------------------
    # Collateral offset
    # -------------------------------------------------------------------------

    def apply_collateral(
        self,
        net_mtm: np.ndarray,
        netting_set: NettingSet,
        use_mpor: bool = True,
    ) -> np.ndarray:
        """
        Apply variation margin (VM) and independent amount (IA) offsets
        to the net netting set MTM, returning the collateral-adjusted MTM.
        """
        threshold = netting_set.threshold
        mta       = netting_set.minimum_transfer_amount
        ia        = netting_set.independent_amount
        mpor      = netting_set.mpor

        if not use_mpor:
            # ----------------------------------------------------------------
            # Instantaneous collateral (zero MPoR)
            # ----------------------------------------------------------------
            vm_gross = np.maximum(net_mtm - threshold, 0.0)
            vm = np.where(vm_gross >= mta, vm_gross, 0.0)
            return net_mtm - vm - ia

        else:
            # ----------------------------------------------------------------
            # MPoR-lagged collateral — path-by-path
            # ----------------------------------------------------------------
            C = np.full_like(net_mtm, ia)  # initialise to IA everywhere

            for j in range(self.n_steps + 1):
                t      = self.time_grid[j]
                t_lag  = t - mpor

                if t_lag <= 0.0:
                    C[:, j] = ia
                    continue

                j_lag = int(np.searchsorted(self.time_grid, t_lag, side="right")) - 1
                j_lag = max(j_lag, 0)

                vm_gross = np.maximum(net_mtm[:, j_lag] - threshold, 0.0)
                vm       = np.where(vm_gross >= mta, vm_gross, 0.0)
                C[:, j]  = vm + ia

            return net_mtm - C

    # -------------------------------------------------------------------------
    # Exposure profiles
    # -------------------------------------------------------------------------

    def exposure_profile(
        self,
        net_mtm: np.ndarray,
        discount_to_t0: bool = True,
    ) -> np.ndarray:
        """Compute the path-wise positive exposure profile E(path, t) = max(V(path, t), 0)."""
        positive_exp = np.maximum(net_mtm, 0.0)
        if discount_to_t0:
            return positive_exp * self.path_dfs
        return positive_exp

    def expected_exposure(
        self,
        net_mtm: np.ndarray,
        discount_to_t0: bool = True,
    ) -> np.ndarray:
        """Expected Exposure (EE): cross-sectional mean of positive exposure."""
        exp_profile = self.exposure_profile(net_mtm, discount_to_t0)
        return exp_profile.mean(axis=0)

    def expected_positive_exposure(
        self,
        net_mtm: np.ndarray,
        discount_to_t0: bool = True,
    ) -> float:
        """Expected Positive Exposure (EPE): time-averaged EE."""
        ee = self.expected_exposure(net_mtm, discount_to_t0)
        T  = self.time_grid[-1]
        if T == 0:
            return float(ee[0])
        return float(trapezoidal_integration(self.time_grid, ee) / T)

    def effective_epe(
        self,
        net_mtm: np.ndarray,
        discount_to_t0: bool = True,
    ) -> np.ndarray:
        """Effective EE: non-decreasing version of EE."""
        ee     = self.expected_exposure(net_mtm, discount_to_t0)
        eff_ee = ee.copy()
        for i in range(1, len(eff_ee)):
            eff_ee[i] = max(eff_ee[i], eff_ee[i - 1])
        return eff_ee

    def effective_epe_scalar(
        self,
        net_mtm: np.ndarray,
        horizon: float = 1.0,
        discount_to_t0: bool = True,
    ) -> float:
        """Effective EPE scalar: time-average of Effective EE over horizon."""
        eff_ee   = self.effective_epe(net_mtm, discount_to_t0)
        mask     = self.time_grid <= horizon + 1e-9
        t_trunc  = self.time_grid[mask]
        ee_trunc = eff_ee[mask]
        
        if len(t_trunc) < 2:
            return float(ee_trunc[0]) if len(ee_trunc) > 0 else 0.0
            
        return float(trapezoidal_integration(t_trunc, ee_trunc) / horizon)

    def potential_future_exposure(
        self,
        net_mtm: np.ndarray,
        confidence: float = 0.95,
        discount_to_t0: bool = False,
    ) -> np.ndarray:
        """Potential Future Exposure (PFE) at a given confidence level."""
        exp_profile = self.exposure_profile(net_mtm, discount_to_t0)
        return np.quantile(exp_profile, confidence, axis=0)

    def negative_exposure(
        self,
        net_mtm: np.ndarray,
        discount_to_t0: bool = True,
    ) -> np.ndarray:
        """Expected Negative Exposure (ENE) profile used to compute DVA."""
        neg_exp = np.minimum(net_mtm, 0.0)
        if discount_to_t0:
            neg_exp = neg_exp * self.path_dfs
        return neg_exp.mean(axis=0)

    # -------------------------------------------------------------------------
    # Summary metrics
    # -------------------------------------------------------------------------

    def exposure_summary(
        self,
        netting_set: NettingSet,
        use_mpor: bool = True,
        pfe_confidence: float = 0.95,
    ) -> dict:
        """Compute the full suite of exposure metrics in one call."""
        net_mtm = self.netting_set_mtm(netting_set)
        
        # Apply collateral logic before extracting profiles
        collateralized_mtm = self.apply_collateral(net_mtm, netting_set, use_mpor=use_mpor)

        ee      = self.expected_exposure(collateralized_mtm)
        epe     = self.expected_positive_exposure(collateralized_mtm)
        eff_ee  = self.effective_epe(collateralized_mtm)
        eff_epe = self.effective_epe_scalar(collateralized_mtm)
        pfe     = self.potential_future_exposure(collateralized_mtm, confidence=pfe_confidence)
        
        # DVA is based on our exposure to the counterparty (we owe them).
        # Standard convention: DVA assumes the counterparty has NO recourse to the 
        # collateral we hold against them in a default, so we evaluate ENE on raw MTM.
        ene     = self.negative_exposure(net_mtm)

        return {
            "time_grid"     : self.time_grid,
            "EE"            : ee,
            "EPE"           : epe,
            "Effective_EE"  : eff_ee,
            "Effective_EPE" : eff_epe,
            "PFE"           : pfe,
            "ENE"           : ene,
            "peak_EE"       : float(np.max(ee)),
            "peak_PFE"      : float(np.max(pfe)),
            "n_paths"       : self.n_paths,
            "n_steps"       : self.n_steps,
        }

    # -------------------------------------------------------------------------
    # IRS vectorized valuator
    # -------------------------------------------------------------------------

    def irs_valuator(
        self,
        fixed_rate: float,
        payment_times: np.ndarray,
        notional: float,
        payer: bool,
    ) -> Callable[[np.ndarray, int], np.ndarray]:
        """
        Factory method: returns a vectorized valuator for an IRS position.
        """
        r_paths   = self.r_paths
        time_grid = self.time_grid
        hw_sim    = self.hw_simulator

        def valuator(r_at_t: np.ndarray, t_idx: int) -> np.ndarray:
            t = time_grid[t_idx]

            remaining = payment_times[payment_times > t + 1e-9]
            if len(remaining) == 0:
                return np.zeros(len(r_at_t))

            annuity = np.zeros(len(r_at_t))
            prev_t = t
            for T_i in remaining:
                df_ti = hw_sim.path_discount_factor(r_paths, time_grid, t_idx, T_i)
                tau = T_i - prev_t
                annuity += tau * df_ti
                prev_t = T_i

            T_n      = remaining[-1]
            p_tn     = hw_sim.path_discount_factor(r_paths, time_grid, t_idx, T_n)
            pv_float = notional * (1.0 - p_tn)
            pv_fixed = notional * fixed_rate * annuity

            sign = 1.0 if payer else -1.0
            return sign * (pv_float - pv_fixed)

        return valuator
