
"""
counterparty.py
---------------
Counterparty-level aggregation layer for the Quant Desk Toolkit.

This is the top-level module in the dependency chain. It sits above all
other modules and pulls together:

    Exposure   ← exposure.py    (EE, EPE, PFE per netting set)
    XVA        ← xva.py         (CVA, DVA, FVA)
    SA-CCR     ← sa_ccr.py      (EAD, AddOns, RC)
    Capital    ← capital_rwa.py (RWA_CCR, RWA_CVA, minimum capital)

and adds the counterparty-level concepts that cut across all of them:

    Credit limit monitoring  — utilisation vs. approved limits
    Wrong-way risk (WWR)     — correlation between exposure and PD
    Portfolio aggregation    — cross-counterparty concentration metrics

Dependency graph
----------------
    math_helpers.py
         ↓
    curve_factory.py
         ↓
    instruments.py      simulator.py
              ↓         ↓
            exposure.py
              ↓         ↓
            xva.py    sa_ccr.py
                  ↓   ↓
              capital_rwa.py
                    ↓
              counterparty.py   ← you are here

Design
------
The central object is `Counterparty`, which is a pure data container.
`CounterpartyRiskEngine` is the stateful calculator — it takes a
`Counterparty` together with pre-computed `ExposureEngine` and
`SACCREngine` results and assembles the full risk picture in one call.

`Portfolio` aggregates multiple `CounterpartyRiskEngine` results and
provides concentration metrics used in credit portfolio management.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from capital_rwa import (
    CounterpartyCapitalInput,
    CapitalEngine,
)


# =============================================================================
# CREDIT LIMITS
# =============================================================================

@dataclass
class CreditLimits:
    """
    Approved credit limits for a counterparty.

    Limits are typically set by the Credit Risk function and expressed
    in notional or present-value terms. Each limit type targets a
    different risk metric from the exposure profile.

    Parameters
    ----------
    pfe_limit : float
        Maximum allowable Potential Future Exposure (95% quantile) at
        any point in the simulation horizon. Expressed in currency units.
        Breached when PFE(t) > pfe_limit for any t.
    ee_limit : float
        Maximum allowable Expected Exposure at any time step.
    ead_limit : float
        Maximum SA-CCR EAD (post alpha=1.4). Regulatory-facing limit.
    notional_limit : float
        Gross notional limit across all instruments in all netting sets.
        Blunt but simple; used as a first-pass check before full MC.
    cva_limit : float
        Maximum allowable CVA charge attributable to this counterparty.
        Set by the XVA desk in agreement with the trading desk.
    """
    pfe_limit      : float = np.inf
    ee_limit       : float = np.inf
    ead_limit      : float = np.inf
    notional_limit : float = np.inf
    cva_limit      : float = np.inf


@dataclass
class LimitUtilisation:
    """
    Current utilisation of each credit limit.

    Returned by CounterpartyRiskEngine.check_limits().
    """
    pfe_peak       : float = 0.0
    ee_peak        : float = 0.0
    ead            : float = 0.0
    gross_notional : float = 0.0
    cva            : float = 0.0

    pfe_utilisation      : float = 0.0  # pfe_peak / pfe_limit
    ee_utilisation       : float = 0.0
    ead_utilisation      : float = 0.0
    notional_utilisation : float = 0.0
    cva_utilisation      : float = 0.0

    pfe_breached      : bool = False
    ee_breached       : bool = False
    ead_breached      : bool = False
    notional_breached : bool = False
    cva_breached      : bool = False

    any_breach        : bool = False


# =============================================================================
# WRONG-WAY RISK
# =============================================================================

@dataclass
class WrongWayRiskIndicator:
    """
    Wrong-way risk (WWR) assessment for a counterparty.

    WWR arises when the exposure to a counterparty increases at the same
    time as their probability of default — the two worst outcomes occur
    together. This is particularly acute for:

      - Energy producers with commodity derivatives (rising commodity
        prices increase derivative MTM AND credit quality of the producer)
      - Sovereign counterparties with FX derivatives (currency weakness
        signals credit stress AND increases exposure on FX trades)
      - Correlation between equity of the counterparty and equity
        underlying of a derivatives position

    WWR types
    ---------
    General WWR:  Macroeconomic factors drive both exposure and PD.
                  Example: recession increases IR exposure AND defaults.
                  Hard to model directly; flagged qualitatively.

    Specific WWR: Legal / contractual link between the trade and the
                  counterparty's credit quality.
                  Example: credit-linked note referencing the counterparty.
                  Must be modelled explicitly (add-on to EAD under SA-CCR).

    Parameters
    ----------
    has_specific_wwr : bool
        True if there is a direct contractual link. Requires regulatory
        add-on per BCBS 279 §58. Default False.
    has_general_wwr : bool
        True if qualitative assessment flags general WWR. Default False.
    exposure_pd_correlation : float
        Estimated correlation between EE and counterparty PD, computed
        empirically from historical data or stress scenarios.
        Range: [-1, 1]. Positive → WWR. Negative → right-way risk.
    wwr_sector : str
        Sector driving the WWR (e.g. 'Energy', 'Sovereign', 'Financial').
    wwr_notes : str
        Free-text description of the WWR driver.
    sa_ccr_addon_multiplier : float
        Regulatory WWR multiplier for SA-CCR EAD when specific WWR
        applies (BCBS 279 §58 — supervisor-specified, typically 2.0×).
        Default 1.0 (no add-on).
    """
    has_specific_wwr        : bool  = False
    has_general_wwr         : bool  = False
    exposure_pd_correlation : float = 0.0
    wwr_sector              : str   = ""
    wwr_notes               : str   = ""
    sa_ccr_addon_multiplier : float = 1.0


def compute_exposure_pd_correlation(
    ee_profile: np.ndarray,
    pd_proxy: np.ndarray,
) -> float:
    """
    Estimate the empirical correlation between the EE profile and a
    time series proxy for the counterparty's default intensity.

    In a simulation context, pd_proxy can be the path-wise default
    intensity (e.g. from a CIR model for the credit spread), or a
    historical credit spread time series normalised to the simulation
    time grid.

    Parameters
    ----------
    ee_profile : np.ndarray, shape (n_steps+1,)
        Expected Exposure at each time step.
    pd_proxy : np.ndarray, shape (n_steps+1,)
        Proxy for PD or credit spread at each time step. Must be the
        same length as ee_profile.

    Returns
    -------
    float
        Pearson correlation coefficient ∈ [-1, 1].
        Positive → exposure rises with credit stress (WWR).
        Negative → exposure falls with credit stress (right-way risk).
    """
    if len(ee_profile) != len(pd_proxy):
        raise ValueError("ee_profile and pd_proxy must have the same length.")
    if np.std(ee_profile) < 1e-10 or np.std(pd_proxy) < 1e-10:
        return 0.0
    return float(np.corrcoef(ee_profile, pd_proxy)[0, 1])


# =============================================================================
# COUNTERPARTY
# =============================================================================

@dataclass
class Counterparty:
    """
    Data container for one bilateral counterparty relationship.

    Parameters
    ----------
    name : str
        Counterparty name or internal identifier.
    lei : str
        Legal Entity Identifier (20-character ISO 17442 code).
        Used for regulatory reporting.
    credit_quality : str
        Internal or external rating bucket: 'AAA', 'AA', 'A', 'BBB',
        'BB', 'B', 'CCC', 'NR'.
    pd : float
        1-year probability of default (decimal). Sourced from internal
        credit models, external ratings mapping, or CDS-implied PD.
    lgd : float
        Loss Given Default. Supervisory default: 0.45 (senior unsecured),
        0.75 (subordinated). Foundation IRB allows own estimates.
    recovery_rate : float
        1 - LGD. Used in XVA (CVA/DVA) and WWR calculations.
    maturity : float
        Effective maturity of the relationship in years. For multi-netting-
        set counterparties, use the longest remaining maturity.
    is_financial : bool
        True for regulated financial institutions. Triggers 1.25×
        asset correlation scalar in the IRB capital formula.
    netting_set_names : list of str
        Labels for the netting sets with this counterparty. Actual
        netting set objects live in exposure.py; here we store labels only.
    limits : CreditLimits
        Approved credit limits.
    wwr : WrongWayRiskIndicator
        Wrong-way risk assessment.
    sector : str
        Industry sector (e.g. 'Energy', 'Financial', 'Technology').
        Used for concentration analysis in Portfolio.
    region : str
        Geographic region (e.g. 'North America', 'Europe', 'Asia').
    """
    name              : str
    lei               : str              = ""
    credit_quality    : str              = "BBB"
    pd                : float            = 0.01
    lgd               : float            = 0.45
    recovery_rate     : float            = 0.40
    maturity          : float            = 5.0
    is_financial      : bool             = False
    netting_set_names : List[str]        = field(default_factory=list)
    limits            : CreditLimits     = field(default_factory=CreditLimits)
    wwr               : WrongWayRiskIndicator = field(default_factory=WrongWayRiskIndicator)
    sector            : str              = "Corporate"
    region            : str              = "North America"


# =============================================================================
# COUNTERPARTY RISK ENGINE
# =============================================================================

class CounterpartyRiskEngine:
    """
    Assembles the full risk picture for a single counterparty.

    Takes pre-computed outputs from ExposureEngine, SACCREngine, and
    XVAEngine and combines them with counterparty metadata to produce
    a unified risk summary used by the trading desk, credit risk,
    and capital management.

    Parameters
    ----------
    counterparty : Counterparty
    exposure_summary : dict
        Output of ExposureEngine.exposure_summary(). Must contain:
        time_grid, EE, EPE, Effective_EE, Effective_EPE, PFE, ENE.
    sa_ccr_result : dict
        Output of SACCREngine.compute(). Must contain: EAD, RC,
        PFE_addon, aggregate_addon, IR_addon, FX_addon, Equity_addon.
    xva_result : dict
        Output of XVAEngine.compute(). Must contain: CVA, DVA, FVA,
        BCVA, total_XVA, CS01, HR01.
    gross_notional : float
        Sum of absolute notionals across all instruments in all netting
        sets with this counterparty. Used for notional limit check.
    pd_proxy : np.ndarray, optional
        Time series proxy for PD / credit spread on the simulation grid.
        If provided, used to compute the empirical exposure-PD correlation.
    """

    def __init__(
        self,
        counterparty     : Counterparty,
        exposure_summary : dict,
        sa_ccr_result    : dict,
        xva_result       : dict,
        gross_notional   : float = 0.0,
        pd_proxy         : Optional[np.ndarray] = None,
    ) -> None:
        self.cp               = counterparty
        self.exposure_summary = exposure_summary
        self.sa_ccr_result    = sa_ccr_result
        self.xva_result       = xva_result
        self.gross_notional   = float(gross_notional)
        self.pd_proxy         = pd_proxy

    # -------------------------------------------------------------------------
    # Credit limit check
    # -------------------------------------------------------------------------

    def check_limits(self) -> LimitUtilisation:
        """
        Compare current exposure metrics against approved credit limits.

        Returns
        -------
        LimitUtilisation
            Utilisation ratios and breach flags for each limit type.
        """
        lim = self.cp.limits
        ee  = self.exposure_summary.get("EE",  np.array([0.0]))
        pfe = self.exposure_summary.get("PFE", np.array([0.0]))
        ead = self.sa_ccr_result.get("EAD",  0.0)
        cva = self.xva_result.get("CVA",    0.0)

        pfe_peak = float(np.max(pfe))
        ee_peak  = float(np.max(ee))

        def util(current, limit):
            return current / limit if limit > 0 and not np.isinf(limit) else 0.0

        return LimitUtilisation(
            pfe_peak             = pfe_peak,
            ee_peak              = ee_peak,
            ead                  = ead,
            gross_notional       = self.gross_notional,
            cva                  = cva,
            pfe_utilisation      = util(pfe_peak,            lim.pfe_limit),
            ee_utilisation       = util(ee_peak,             lim.ee_limit),
            ead_utilisation      = util(ead,                 lim.ead_limit),
            notional_utilisation = util(self.gross_notional, lim.notional_limit),
            cva_utilisation      = util(cva,                 lim.cva_limit),
            pfe_breached         = pfe_peak            > lim.pfe_limit,
            ee_breached          = ee_peak             > lim.ee_limit,
            ead_breached         = ead                 > lim.ead_limit,
            notional_breached    = self.gross_notional > lim.notional_limit,
            cva_breached         = cva                 > lim.cva_limit,
            any_breach           = any([
                pfe_peak            > lim.pfe_limit,
                ee_peak             > lim.ee_limit,
                ead                 > lim.ead_limit,
                self.gross_notional > lim.notional_limit,
                cva                 > lim.cva_limit,
            ]),
        )

    # -------------------------------------------------------------------------
    # Wrong-way risk
    # -------------------------------------------------------------------------

    def assess_wwr(self) -> WrongWayRiskIndicator:
        """
        Update the WWR indicator with the empirical exposure-PD correlation
        if a pd_proxy was provided, and apply the SA-CCR EAD multiplier
        for specific WWR.

        Returns
        -------
        WrongWayRiskIndicator
            Updated WWR indicator (does not mutate self.cp.wwr in place).
        """
        wwr = WrongWayRiskIndicator(
            has_specific_wwr        = self.cp.wwr.has_specific_wwr,
            has_general_wwr         = self.cp.wwr.has_general_wwr,
            wwr_sector              = self.cp.wwr.wwr_sector,
            wwr_notes               = self.cp.wwr.wwr_notes,
            sa_ccr_addon_multiplier = self.cp.wwr.sa_ccr_addon_multiplier,
        )

        if self.pd_proxy is not None:
            ee = self.exposure_summary.get("EE", np.array([0.0]))
            wwr.exposure_pd_correlation = compute_exposure_pd_correlation(
                ee, self.pd_proxy
            )
            # Heuristic: flag general WWR if correlation > 0.30
            if wwr.exposure_pd_correlation > 0.30 and not wwr.has_general_wwr:
                wwr.has_general_wwr = True
                wwr.wwr_notes += (
                    f" [Auto-flagged: exposure-PD correlation = "
                    f"{wwr.exposure_pd_correlation:.3f}]"
                )

        return wwr

    # -------------------------------------------------------------------------
    # Capital inputs
    # -------------------------------------------------------------------------

    def _capital_input(self) -> CounterpartyCapitalInput:
        """Build a CounterpartyCapitalInput from the current state."""
        ead = self.sa_ccr_result.get("EAD", 0.0)
        # Apply specific WWR multiplier to EAD before capital calc
        if self.cp.wwr.has_specific_wwr:
            ead *= self.cp.wwr.sa_ccr_addon_multiplier
            
        return CounterpartyCapitalInput(
            name              = self.cp.name,
            ead               = ead,
            pd                = self.cp.pd,
            lgd               = self.cp.lgd,
            maturity          = self.cp.maturity,
            credit_quality    = self.cp.credit_quality,
            counterparty_type = (
                "Bank" if self.cp.is_financial else "Corporate_" + self.cp.credit_quality
            ),
            is_financial      = self.cp.is_financial,
        )

    # -------------------------------------------------------------------------
    # Full risk summary
    # -------------------------------------------------------------------------

    def compute(self, use_irb: bool = True) -> dict:
        """
        Assemble the complete counterparty risk summary.

        Parameters
        ----------
        use_irb : bool
            True → IRB capital formula. False → SA risk weights.

        Returns
        -------
        dict with keys:
            name            : str
            lei             : str
            credit_quality  : str
            pd              : float
            lgd             : float
            sector          : str
            region          : str

            exposure        : dict  — EE, EPE, Effective_EPE, PFE (peak), ENE (peak)
            xva             : dict  — CVA, DVA, FVA, BCVA, total_XVA, CS01, HR01
            sa_ccr          : dict  — EAD, RC, aggregate_addon, multiplier, PFE_addon
            capital         : dict  — RWA_CCR, K, EL; computed via CapitalEngine
            limits          : LimitUtilisation
            wwr             : WrongWayRiskIndicator
        """
        wwr    = self.assess_wwr()
        limits = self.check_limits()
        cap_in = self._capital_input()

        cap_engine  = CapitalEngine(
            counterparties = [cap_in],
            use_irb        = use_irb,
            use_sa_cva     = False,
        )
        cap_result  = cap_engine.compute()
        ccr_capital = cap_result["CCR"].get(self.cp.name, {})

        ee  = self.exposure_summary
        xva = self.xva_result
        sac = self.sa_ccr_result

        return {
            "name"          : self.cp.name,
            "lei"           : self.cp.lei,
            "credit_quality": self.cp.credit_quality,
            "pd"            : self.cp.pd,
            "lgd"           : self.cp.lgd,
            "sector"        : self.cp.sector,
            "region"        : self.cp.region,

            "exposure"      : {
                "EE"            : ee.get("EE"),
                "EPE"           : ee.get("EPE"),
                "Effective_EPE" : ee.get("Effective_EPE"),
                "peak_EE"       : ee.get("peak_EE"),
                "peak_PFE"      : ee.get("peak_PFE"),
                "PFE"           : ee.get("PFE"),
                "ENE"           : ee.get("ENE"),
                "time_grid"     : ee.get("time_grid"),
            },

            "xva"           : {
                "CVA"       : xva.get("CVA"),
                "DVA"       : xva.get("DVA"),
                "FVA"       : xva.get("FVA"),
                "BCVA"      : xva.get("BCVA"),
                "total_XVA" : xva.get("total_XVA"),
                "CS01"      : xva.get("CS01"),
                "HR01"      : xva.get("HR01"),
            },

            "sa_ccr"        : {
                "EAD"            : sac.get("EAD"),
                "RC"             : sac.get("RC"),
                "aggregate_addon": sac.get("aggregate_addon"),
                "IR_addon"       : sac.get("IR_addon"),
                "FX_addon"       : sac.get("FX_addon"),
                "Equity_addon"   : sac.get("Equity_addon"),
                "multiplier"     : sac.get("multiplier"),
                "PFE_addon"      : sac.get("PFE_addon"),
            },

            "capital"       : {
                "RWA_CCR"        : ccr_capital.get("RWA_CCR"),
                "K"              : ccr_capital.get("K"),
                "EL"             : ccr_capital.get("EL"),
                "RWA_CVA"        : cap_result.get("RWA_CVA"),
                "RWA_total"      : cap_result.get("RWA_total"),
                "min_capital"    : cap_result["capital_min"].get("Total_min"),
            },

            "limits"        : limits,
            "wwr"           : wwr,
        }


# =============================================================================
# PORTFOLIO AGGREGATION
# =============================================================================

class Portfolio:
    """
    Aggregate risk metrics across a portfolio of counterparties.

    Takes a list of pre-computed counterparty risk summaries (output of
    CounterpartyRiskEngine.compute()) and produces portfolio-level views
    used in credit portfolio management, concentration risk, and regulatory
    reporting.

    Parameters
    ----------
    counterparty_summaries : list of dict
        Each dict is the output of CounterpartyRiskEngine.compute().
    """

    def __init__(self, counterparty_summaries: List[dict]) -> None:
        self.summaries = counterparty_summaries

    # -------------------------------------------------------------------------
    # Aggregation helpers
    # -------------------------------------------------------------------------

    def _get(self, key_path: List[str], default=0.0):
        """Extract a nested key from all summaries, with a default."""
        values = []
        for s in self.summaries:
            obj = s
            for k in key_path:
                obj = obj.get(k, {}) if isinstance(obj, dict) else default
            values.append(obj if isinstance(obj, (int, float)) else default)
        return values

    # -------------------------------------------------------------------------
    # Portfolio totals
    # -------------------------------------------------------------------------

    def totals(self) -> dict:
        """
        Scalar totals across the portfolio.

        Returns
        -------
        dict with keys:
            total_CVA       : float
            total_DVA       : float
            total_FVA       : float
            total_XVA       : float
            total_EAD       : float
            total_RWA_CCR   : float
            total_RWA_CVA   : float
            total_RWA       : float
            total_EL        : float
            n_counterparties: int
            n_breaches      : int   — number of counterparties with any limit breach
        """
        return {
            "total_CVA"        : sum(self._get(["xva", "CVA"])),
            "total_DVA"        : sum(self._get(["xva", "DVA"])),
            "total_FVA"        : sum(self._get(["xva", "FVA"])),
            "total_XVA"        : sum(self._get(["xva", "total_XVA"])),
            "total_EAD"        : sum(self._get(["sa_ccr", "EAD"])),
            "total_RWA_CCR"    : sum(self._get(["capital", "RWA_CCR"])),
            "total_RWA_CVA"    : sum(self._get(["capital", "RWA_CVA"])),
            "total_RWA"        : sum(self._get(["capital", "RWA_total"])),
            "total_EL"         : sum(self._get(["capital", "EL"])),
            "n_counterparties" : len(self.summaries),
            "n_breaches"       : sum(
                1 for s in self.summaries
                if s.get("limits", LimitUtilisation()).any_breach
            ),
        }

    # -------------------------------------------------------------------------
    # Concentration metrics
    # -------------------------------------------------------------------------

    def concentration(self, metric: str = "EAD") -> dict:
        """
        Compute concentration metrics across counterparties.

        Parameters
        ----------
        metric : str
            Risk metric to measure concentration on. One of:
            'EAD', 'CVA', 'RWA_CCR', 'RWA_total'.

        Returns
        -------
        dict with keys:
            metric          : str
            values          : dict  — {name: metric value}
            shares          : dict  — {name: fraction of total}
            hhi             : float — Herfindahl-Hirschman Index ∈ [0, 1]
                                      HHI = Σ share_i². HHI → 1 = full concentration.
            top1_share      : float — Largest single counterparty share
            top5_share      : float — Top 5 counterparty share
            top5_names      : list  — Names of top 5 counterparties
        """
        path_map = {
            "EAD"      : ["sa_ccr",  "EAD"],
            "CVA"      : ["xva",     "CVA"],
            "RWA_CCR"  : ["capital", "RWA_CCR"],
            "RWA_total": ["capital", "RWA_total"],
        }
        if metric not in path_map:
            raise ValueError(f"metric must be one of {list(path_map)}.")

        raw    = {s["name"]: float(self._get(path_map[metric])[i] or 0.0)
                  for i, s in enumerate(self.summaries)}
        total  = sum(raw.values())
        if total <= 0:
            shares = {k: 0.0 for k in raw}
        else:
            shares = {k: v / total for k, v in raw.items()}

        sorted_names = sorted(shares, key=shares.get, reverse=True)
        hhi          = sum(v ** 2 for v in shares.values())
        top5         = sorted_names[:5]

        return {
            "metric"    : metric,
            "values"    : raw,
            "shares"    : shares,
            "hhi"       : hhi,
            "top1_share": shares.get(sorted_names[0], 0.0) if sorted_names else 0.0,
            "top5_share": sum(shares.get(n, 0.0) for n in top5),
            "top5_names": top5,
        }

    # -------------------------------------------------------------------------
    # Sector and region breakdown
    # -------------------------------------------------------------------------

    def breakdown_by(self, dimension: str = "sector", metric: str = "EAD") -> Dict[str, float]:
        """
        Aggregate a risk metric by a categorical dimension.

        Parameters
        ----------
        dimension : str
            'sector' or 'region'.
        metric : str
            'EAD', 'CVA', 'RWA_CCR', or 'RWA_total'.

        Returns
        -------
        dict
            {dimension_value: sum of metric}
        """
        if dimension not in ("sector", "region"):
            raise ValueError("dimension must be 'sector' or 'region'.")

        path_map = {
            "EAD"      : ["sa_ccr",  "EAD"],
            "CVA"      : ["xva",     "CVA"],
            "RWA_CCR"  : ["capital", "RWA_CCR"],
            "RWA_total": ["capital", "RWA_total"],
        }
        if metric not in path_map:
            raise ValueError(f"metric must be one of {list(path_map)}.")

        result: Dict[str, float] = {}
        for i, s in enumerate(self.summaries):
            dim_val    = s.get(dimension, "Unknown")
            metric_val = float(self._get(path_map[metric])[i] or 0.0)
            result[dim_val] = result.get(dim_val, 0.0) + metric_val

        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    # -------------------------------------------------------------------------
    # Breach report
    # -------------------------------------------------------------------------

    def breach_report(self) -> List[dict]:
        """
        Return a list of counterparties with any active limit breach.

        Returns
        -------
        list of dict, each with keys:
            name, credit_quality, breach details (which limits breached
            and the utilisation ratios).
        """
        breaches = []
        for s in self.summaries:
            lim = s.get("limits")
            if not isinstance(lim, LimitUtilisation):
                continue
            if lim.any_breach:
                breaches.append({
                    "name"                 : s["name"],
                    "credit_quality"       : s.get("credit_quality", "NR"),
                    "pfe_breached"         : lim.pfe_breached,
                    "ee_breached"          : lim.ee_breached,
                    "ead_breached"         : lim.ead_breached,
                    "notional_breached"    : lim.notional_breached,
                    "cva_breached"         : lim.cva_breached,
                    "pfe_utilisation"      : lim.pfe_utilisation,
                    "ead_utilisation"      : lim.ead_utilisation,
                    "cva_utilisation"      : lim.cva_utilisation,
                })
        return breaches

    # -------------------------------------------------------------------------
    # WWR report
    # -------------------------------------------------------------------------

    def wwr_report(self) -> List[dict]:
        """
        Return a list of counterparties with specific or general WWR flags.

        Returns
        -------
        list of dict, one entry per WWR-flagged counterparty.
        """
        flagged = []
        for s in self.summaries:
            wwr = s.get("wwr")
            if not isinstance(wwr, WrongWayRiskIndicator):
                continue
            if wwr.has_specific_wwr or wwr.has_general_wwr:
                flagged.append({
                    "name"                   : s["name"],
                    "has_specific_wwr"       : wwr.has_specific_wwr,
                    "has_general_wwr"        : wwr.has_general_wwr,
                    "exposure_pd_correlation": wwr.exposure_pd_correlation,
                    "sector"                 : wwr.wwr_sector,
                    "notes"                  : wwr.wwr_notes,
                    "ead_multiplier"         : wwr.sa_ccr_addon_multiplier,
                    "EAD"                    : s.get("sa_ccr", {}).get("EAD"),
                    "CVA"                    : s.get("xva",    {}).get("CVA"),
                })
        return sorted(flagged, key=lambda x: abs(x["exposure_pd_correlation"]), reverse=True)

    # -------------------------------------------------------------------------
    # Full portfolio summary
    # -------------------------------------------------------------------------

    def summary(self) -> dict:
        """
        Full portfolio risk summary in one call.

        Returns
        -------
        dict with keys:
            totals              : dict  — scalar portfolio totals
            ead_concentration   : dict  — HHI and top-5 by EAD
            cva_concentration   : dict  — HHI and top-5 by CVA
            sector_ead          : dict  — EAD by sector
            region_ead          : dict  — EAD by region
            breach_report       : list  — limit-breached counterparties
            wwr_report          : list  — WWR-flagged counterparties
        """
        return {
            "totals"           : self.totals(),
            "ead_concentration": self.concentration("EAD"),
            "cva_concentration": self.concentration("CVA"),
            "sector_ead"       : self.breakdown_by("sector", "EAD"),
            "region_ead"       : self.breakdown_by("region", "EAD"),
            "breach_report"    : self.breach_report(),
            "wwr_report"       : self.wwr_report(),
        }