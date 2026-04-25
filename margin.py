"""
margin.py
---------
Margin and collateral mechanics for the Quant Desk Toolkit.
Covers ISDA SIMM (Initial Margin) and VM mechanics.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# SIMM v2.5 Constants
SIMM_IR_TENORS = ["2w", "1m", "3m", "6m", "1y", "2y", "3y", "5y", "10y", "15y", "20y", "30y"]
SIMM_IR_THETA = 0.003
SIMM_IR_FLOOR = 0.10
SIMM_IR_CROSS_CCY_RHO = 0.22

@dataclass
class CSATerms:
    """Credit Support Annex (CSA) terms governing VM and IM exchange."""
    threshold_we: float = 0.0
    threshold_them: float = 0.0
    mta_we: float = 0.0
    mta_them: float = 0.0
    rounding_we: float = 0.0
    rounding_them: float = 0.0
    independent_amount_we: float = 0.0
    independent_amount_them: float = 0.0
    settlement_lag: int = 1
    eligible_collateral: List[str] = field(default_factory=lambda: ["USD_cash"])

def simm_fx_delta(net_sensitivities: Dict[str, float]) -> float:
    """SIMM FX Delta Margin (RW = 7.4% per SIMM v2.5)."""
    FX_RW = 0.074
    return float(np.sqrt(sum((FX_RW * s) ** 2 for s in net_sensitivities.values())))

class VMEngine:
    """Variation Margin call simulator."""
    def __init__(self, csa: CSATerms):
        self.csa = csa

    def vm_required(self, net_mtm: float) -> Tuple[float, float]:
        vm_them = max(net_mtm - self.csa.threshold_them, 0.0)
        vm_us = max(-net_mtm - self.csa.threshold_we, 0.0)
        return vm_them, vm_us

    def simulate(self, net_mtm_series: np.ndarray, time_grid: np.ndarray) -> dict:
        n = len(time_grid)
        vm_net = np.zeros(n)
        for i in range(1, n):
            vm_them_req, vm_us_req = self.vm_required(net_mtm_series[i])
            vm_target = vm_them_req - vm_us_req
            increment = vm_target - vm_net[i-1]
            mta = max(self.csa.mta_them, self.csa.mta_we)
            if abs(increment) >= mta:
                vm_net[i] = vm_net[i-1] + increment
            else:
                vm_net[i] = vm_net[i-1]
        return {"vm_net": vm_net, "time_grid": time_grid}

class MarginEngine:
    """Combined VM + IM engine."""
    def __init__(self, csa: CSATerms, schedule_im_amount: float = 0.0):
        self.csa = csa
        self.schedule_im = float(schedule_im_amount)
        self.vm_engine = VMEngine(csa)

    def im_profile(self, time_grid: np.ndarray, decay: str = "linear", portfolio_maturity: float = 10.0) -> np.ndarray:
        tau = np.maximum(portfolio_maturity - time_grid, 0.0) / portfolio_maturity
        return self.schedule_im * (tau if decay == "linear" else np.sqrt(tau))

    def compute(self, net_mtm_series: np.ndarray, time_grid: np.ndarray, portfolio_maturity: float = 10.0, im_decay: str = "linear") -> dict:
        im_prof = self.im_profile(time_grid, im_decay, portfolio_maturity)
        vm_res = self.vm_engine.simulate(net_mtm_series, time_grid)
        coll_us = np.maximum(vm_res["vm_net"], 0.0) + self.csa.independent_amount_them + im_prof
        return {"IM_initial": self.schedule_im, "IM_profile": im_prof, "vm_result": vm_res, "total_collateral_us": coll_us}