"""
margin.py
---------
Margin and collateral mechanics for the Quant Desk Toolkit.

Implements two complementary layers of the margining framework:

  1. Initial Margin (IM) — computed either via the ISDA SIMM sensitivity-
     based model or the simpler Schedule-based fallback.

  2. Variation Margin (VM) — daily MTM-based margin calls, including
     threshold, minimum transfer amount, and settlement lag mechanics.

Together these feed into the MPoR-lagged collateral treatment in
exposure.py and the RC computation in sa_ccr.py.

Framework references
--------------------
ISDA SIMM:
    ISDA (2021). "ISDA Standard Initial Margin Model: Overview."
    ISDA (2022). "ISDA SIMM Methodology, Version 2.5."

Schedule-based IM:
    BCBS-IOSCO (2015). "Margin requirements for non-centrally cleared
    derivatives." Final Framework, Table 1.

VM mechanics:
    ISDA (2016). "Credit Support Annex for Variation Margin (VM CSA)."

Architecture
------------
ISDA SIMM covers six risk classes:
    IR, Credit (qualifying), Credit (non-qualifying), Equity, Commodity, FX.

This module implements full delta-level SIMM for IR and Credit (the two
dominant risk classes for a rates/credit derivatives book), and a scalar
formula for Equity, Commodity, and FX. Vega and curvature risk are noted
but not implemented — they are secondary for vanilla books.

Delta margin aggregation follows the SIMM prescribed formula:
    DeltaMargin_class = sqrt( Σ_b Σ_b' ρ_bb' × WS_b × WS_b'
                              + Σ_b CR_b² × WS_b² × (1 - ρ_bb'²) )
    [simplified to the bucket-level formula used in production SIMM]

Cross-risk-class aggregation uses the SIMM γ correlation matrix.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# =============================================================================
# ISDA SIMM CONSTANTS  (SIMM v2.5)
# =============================================================================

# IR risk weights by currency volatility group and tenor (% of sensitivity)
# Currency groups:
#   "regular"  — USD, EUR, GBP, AUD, CAD, SEK, CHF
#   "low_vol"  — JPY, DKK, NOK, NZD, SGD, HKD, KRW, TWD
#   "high_vol" — all others (EM currencies)
#
# Source: ISDA SIMM v2.5, Table 43
SIMM_IR_RISK_WEIGHT: Dict[str, Dict[str, float]] = {
    "regular": {
        "2w" : 0.0077, "1m" : 0.0077, "3m" : 0.0074, "6m" : 0.0062,
        "1y" : 0.0056, "2y" : 0.0049, "3y" : 0.0047, "5y" : 0.0047,
        "10y": 0.0045, "15y": 0.0045, "20y": 0.0048, "30y": 0.0056,
    },
    "low_vol": {
        "2w" : 0.0010, "1m" : 0.0010, "3m" : 0.0010, "6m" : 0.0008,
        "1y" : 0.0008, "2y" : 0.0007, "3y" : 0.0007, "5y" : 0.0006,
        "10y": 0.0006, "15y": 0.0007, "20y": 0.0007, "30y": 0.0009,
    },
    "high_vol": {
        "2w" : 0.0141, "1m" : 0.0141, "3m" : 0.0141, "6m" : 0.0113,
        "1y" : 0.0095, "2y" : 0.0083, "3y" : 0.0083, "5y" : 0.0083,
        "10y": 0.0083, "15y": 0.0083, "20y": 0.0083, "30y": 0.0083,
    },
}

# IR tenor order (used to build the intra-currency correlation matrix)
SIMM_IR_TENORS: List[str] = [
    "2w", "1m", "3m", "6m", "1y", "2y", "3y", "5y", "10y", "15y", "20y", "30y"
]

# IR intra-currency tenor-tenor correlation: ρ = max(e^{-θ|i-j|}, floor)
# SIMM v2.5, §D.1.1: α = 0.003 (parameter), floor = 0.10
SIMM_IR_THETA  = 0.003          # controls correlation decay by tenor index distance
SIMM_IR_FLOOR  = 0.10           # minimum tenor correlation

# IR inter-currency correlation (two buckets in different currencies)
SIMM_IR_CROSS_CCY_RHO = 0.22   # SIMM v2.5, §D.1.2

# Credit qualifying risk weights by tenor (% of sensitivity)
# Source: ISDA SIMM v2.5, Table 50
SIMM_CR_RISK_WEIGHT: Dict[str, float] = {
    "1y": 0.0075, "2y": 0.0063, "3y": 0.0056,
    "5y": 0.0048, "10y": 0.0043,
}

# Credit intra-issuer tenor-tenor correlation
SIMM_CR_INTRA_TENOR_RHO = 0.65  # SIMM v2.5, §D.2.1

# Credit inter-issuer correlations by rating bucket pair
# (simplified: same-bucket vs cross-bucket)
SIMM_CR_INTRA_BUCKET_RHO  = 0.35   # same credit quality bucket
SIMM_CR_INTER_BUCKET_RHO  = 0.27   # different credit quality buckets

# Cross-risk-class correlation matrix (γ) — SIMM v2.5, Table 1
# Order: IR, Credit_Q, Credit_NQ, Equity, Commodity, FX
SIMM_CROSS_CLASS_GAMMA = np.array([
    [1.00, 0.27, 0.27, 0.27, 0.27, 0.27],
    [0.27, 1.00, 0.42, 0.63, 0.45, 0.27],
    [0.27, 0.42, 1.00, 0.25, 0.25, 0.27],
    [0.27, 0.63, 0.25, 1.00, 0.45, 0.27],
    [0.27, 0.45, 0.25, 0.45, 1.00, 0.27],
    [0.27, 0.27, 0.27, 0.27, 0.27, 1.00],
])
SIMM_RISK_CLASS_ORDER = ["IR", "Credit_Q", "Credit_NQ", "Equity", "Commodity", "FX"]

# Concentration thresholds (USD mm) — for concentration risk factor CR
# Source: ISDA SIMM v2.5, Table 44 (IR), Table 51 (Credit)
SIMM_IR_CONCENTRATION_THRESHOLD: Dict[str, float] = {
    "regular" : 230e6,
    "low_vol" : 28e6,
    "high_vol": 8.5e6,
}
SIMM_CR_CONCENTRATION_THRESHOLD = 0.95e6   # per issuer


# =============================================================================
# SCHEDULE-BASED INITIAL MARGIN
# =============================================================================

# Schedule-based IM percentages by asset class and remaining maturity
# Source: BCBS-IOSCO (2015), Table 1
SCHEDULE_IM_PCT: Dict[str, Dict[str, float]] = {
    "IR_FX_Gold"  : {"<2y": 0.01, "2y-5y": 0.02, ">5y": 0.04},
    "Credit_IG"   : {"<2y": 0.02, "2y-5y": 0.05, ">5y": 0.10},
    "Credit_HY"   : {"<2y": 0.05, "2y-5y": 0.10, ">5y": 0.15},
    "Equity"      : {"<2y": 0.06, "2y-5y": 0.08, ">5y": 0.10},
    "Commodity"   : {"<2y": 0.10, "2y-5y": 0.12, ">5y": 0.15},
    "Other"       : {"<2y": 0.15, "2y-5y": 0.15, ">5y": 0.15},
}


def schedule_im(
    notional           : float,
    asset_class        : str,
    remaining_maturity : float,
    gross_notional     : Optional[float] = None,
    net_to_gross       : Optional[float] = None,
) -> float:
    """
    Schedule-based Initial Margin (BCBS-IOSCO, Table 1).

    A simple alternative to SIMM for smaller portfolios or as a fallback.
    The gross IM is the notional multiplied by a supervisory percentage
    that depends on asset class and remaining maturity. A net-to-gross
    ratio adjustment reduces IM to reflect netting:

        IM_schedule = notional × pct × (0.4 + 0.6 × NGR)

    where NGR = net_replacement_cost / gross_replacement_cost ∈ [0, 1].
    When NGR = 1 (no netting benefit), IM = notional × pct.
    When NGR = 0 (full netting), IM = 0.4 × notional × pct.

    Parameters
    ----------
    notional : float
        Trade notional.
    asset_class : str
        One of: 'IR_FX_Gold', 'Credit_IG', 'Credit_HY', 'Equity',
        'Commodity', 'Other'.
    remaining_maturity : float
        Remaining maturity in years.
    gross_notional : float, optional
        Gross notional across netting set. Required for NGR adjustment.
    net_to_gross : float, optional
        NGR = net MTM / gross MTM ∈ [0, 1]. If None, NGR assumed = 1.

    Returns
    -------
    float
        Schedule-based IM in currency units.
    """
    if asset_class not in SCHEDULE_IM_PCT:
        asset_class = "Other"
    schedule = SCHEDULE_IM_PCT[asset_class]

    if remaining_maturity < 2.0:
        pct = schedule["<2y"]
    elif remaining_maturity <= 5.0:
        pct = schedule["2y-5y"]
    else:
        pct = schedule[">5y"]

    gross_im = notional * pct

    if net_to_gross is not None:
        ngr       = max(0.0, min(net_to_gross, 1.0))
        gross_im *= (0.4 + 0.6 * ngr)

    return gross_im


# =============================================================================
# ISDA SIMM — BUILDING BLOCKS
# =============================================================================

def _ir_intra_currency_correlation(n_tenors: int) -> np.ndarray:
    """
    Build the SIMM IR intra-currency tenor correlation matrix.

    ρ(i, j) = max(exp(-θ × |i - j|), floor)

    where i, j are tenor indices and θ = SIMM_IR_THETA.

    Parameters
    ----------
    n_tenors : int
        Number of tenor buckets (typically 12 for the full SIMM grid).

    Returns
    -------
    np.ndarray, shape (n_tenors, n_tenors)
    """
    idx  = np.arange(n_tenors)
    dist = np.abs(idx[:, None] - idx[None, :]).astype(float)
    rho  = np.maximum(np.exp(-SIMM_IR_THETA * dist), SIMM_IR_FLOOR)
    np.fill_diagonal(rho, 1.0)
    return rho


def _concentration_risk_factor(
    net_sensitivity: float,
    threshold: float,
) -> float:
    """
    SIMM concentration risk factor CR.

        CR = max(1, sqrt(|s| / T))

    Amplifies the risk weight for exposures that exceed the concentration
    threshold T, reflecting the additional market impact of large positions.

    Parameters
    ----------
    net_sensitivity : float
        Net sensitivity in currency units (e.g. DV01 in USD).
    threshold : float
        Concentration threshold for this risk class / currency group.

    Returns
    -------
    float
        CR ≥ 1.
    """
    return max(1.0, np.sqrt(abs(net_sensitivity) / threshold))


# =============================================================================
# ISDA SIMM — IR DELTA MARGIN
# =============================================================================

def simm_ir_delta(
    sensitivities: Dict[str, Dict[str, float]],
    currency_groups: Optional[Dict[str, str]] = None,
) -> dict:
    """
    Compute ISDA SIMM IR Delta Margin.

    Parameters
    ----------
    sensitivities : dict
        Nested dict: {currency: {tenor: net_delta_sensitivity}}
        Sensitivities must be in currency units (e.g. USD DV01 in USD).
        Example:
            {
                'USD': {'1y': -50_000, '5y': -120_000, '10y': -80_000},
                'EUR': {'5y': -30_000},
            }
        Missing tenors are treated as zero.
    currency_groups : dict, optional
        Maps currency → volatility group ('regular', 'low_vol', 'high_vol').
        Defaults to 'regular' for any unlisted currency.

    Returns
    -------
    dict with keys:
        DeltaMargin_IR  : float  — Total IR delta IM
        by_currency     : dict   — {currency: DeltaMargin}
        WS              : dict   — {currency: {tenor: weighted sensitivity}}
        CR              : dict   — {currency: {tenor: concentration factor}}
    """
    currency_groups = currency_groups or {}
    all_tenors      = SIMM_IR_TENORS
    n_tenors        = len(all_tenors)
    rho             = _ir_intra_currency_correlation(n_tenors)

    by_ccy_margin: Dict[str, float] = {}
    by_ccy_ws:     Dict[str, Dict[str, float]] = {}
    by_ccy_cr:     Dict[str, Dict[str, float]] = {}

    for ccy, tenor_sens in sensitivities.items():
        group     = currency_groups.get(ccy, "regular")
        rw_table  = SIMM_IR_RISK_WEIGHT[group]
        threshold = SIMM_IR_CONCENTRATION_THRESHOLD[group]

        # Net sensitivity and risk weight per tenor
        s_vec  = np.array([tenor_sens.get(t, 0.0) for t in all_tenors])
        rw_vec = np.array([rw_table.get(t, rw_table["10y"]) for t in all_tenors])

        # Concentration risk factors
        cr_vec = np.array([
            _concentration_risk_factor(s_vec[i], threshold)
            for i in range(n_tenors)
        ])

        # Effective weighted sensitivity: WS_k = RW_k × CR_k × s_k
        ws_vec = rw_vec * cr_vec * s_vec

        # Intra-currency aggregation: DM = sqrt(WS^T × ρ × WS)
        dm_ccy = float(np.sqrt(max(ws_vec @ rho @ ws_vec, 0.0)))

        by_ccy_margin[ccy] = dm_ccy
        by_ccy_ws[ccy]     = {t: float(ws_vec[i]) for i, t in enumerate(all_tenors)}
        by_ccy_cr[ccy]     = {t: float(cr_vec[i]) for i, t in enumerate(all_tenors)}

    # Cross-currency aggregation:
    # DM_IR = sqrt( Σ_c DM_c² + Σ_c≠c' ρ_cross × DM_c × DM_c' )
    # = sqrt( WS_agg^T × Rho_cross × WS_agg )
    # where Rho_cross has 1 on diagonal and ρ_cross off-diagonal.
    ccys   = list(by_ccy_margin.keys())
    dm_vec = np.array([by_ccy_margin[c] for c in ccys])
    n_c    = len(ccys)

    if n_c == 0:
        total_ir_dm = 0.0
    elif n_c == 1:
        total_ir_dm = float(dm_vec[0])
    else:
        rho_cross   = np.full((n_c, n_c), SIMM_IR_CROSS_CCY_RHO)
        np.fill_diagonal(rho_cross, 1.0)
        total_ir_dm = float(np.sqrt(max(dm_vec @ rho_cross @ dm_vec, 0.0)))

    return {
        "DeltaMargin_IR": total_ir_dm,
        "by_currency"   : by_ccy_margin,
        "WS"            : by_ccy_ws,
        "CR"            : by_ccy_cr,
    }


# =============================================================================
# ISDA SIMM — CREDIT QUALIFYING DELTA MARGIN
# =============================================================================

def simm_credit_delta(
    sensitivities: Dict[str, Dict[str, float]],
    credit_quality: Optional[Dict[str, str]] = None,
) -> dict:
    """
    Compute ISDA SIMM Credit Qualifying Delta Margin.

    Parameters
    ----------
    sensitivities : dict
        {issuer_name: {tenor: net_cs01_sensitivity}}
        CS01 sensitivities in currency units (e.g. USD per 1 bp spread shift).
        Example:
            {
                'Apple Inc': {'1y': -5_000, '5y': -12_000},
                'Ford Motor': {'5y': -8_000},
            }
    credit_quality : dict, optional
        {issuer_name: rating_bucket} — 'AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC'.
        Used to assign issuers to SIMM credit buckets (1=AAA-AA, 2=A, 3=BBB,
        4=BB, 5=B, 6=CCC). Defaults to bucket 3 (BBB) for unlisted issuers.

    Returns
    -------
    dict with keys:
        DeltaMargin_Credit  : float
        by_issuer           : dict  — {issuer: intra-issuer weighted margin}
        WS                  : dict  — {issuer: {tenor: WS}}
    """
    credit_quality = credit_quality or {}
    all_tenors     = list(SIMM_CR_RISK_WEIGHT.keys())

    # SIMM credit bucket assignment by rating
    _BUCKET_MAP: Dict[str, int] = {
        "AAA": 1, "AA": 1, "A": 2, "BBB": 3, "BB": 4, "B": 5, "CCC": 6, "NR": 3,
    }

    # Step 1: Intra-issuer aggregation
    issuer_margins: Dict[str, float] = {}
    issuer_ws:      Dict[str, Dict[str, float]] = {}
    issuer_buckets: Dict[str, int]   = {}

    for issuer, tenor_sens in sensitivities.items():
        rating  = credit_quality.get(issuer, "BBB").upper()
        bucket  = _BUCKET_MAP.get(rating, 3)
        issuer_buckets[issuer] = bucket

        s_vec  = np.array([tenor_sens.get(t, 0.0) for t in all_tenors])
        rw_vec = np.array([SIMM_CR_RISK_WEIGHT.get(t, 0.0043) for t in all_tenors])

        # Concentration risk factor (per issuer, summed across tenors)
        total_sens = float(np.sum(np.abs(s_vec)))
        cr         = _concentration_risk_factor(total_sens, SIMM_CR_CONCENTRATION_THRESHOLD)
        ws_vec     = rw_vec * cr * s_vec

        # Intra-issuer tenor correlation
        n_t    = len(all_tenors)
        rho_t  = np.full((n_t, n_t), SIMM_CR_INTRA_TENOR_RHO)
        np.fill_diagonal(rho_t, 1.0)
        dm_iss = float(np.sqrt(max(ws_vec @ rho_t @ ws_vec, 0.0)))

        issuer_margins[issuer] = dm_iss
        issuer_ws[issuer]      = {t: float(ws_vec[i]) for i, t in enumerate(all_tenors)}

    # Step 2: Inter-issuer aggregation within and across buckets
    # ρ_ij = SIMM_CR_INTRA_BUCKET_RHO if same bucket, else SIMM_CR_INTER_BUCKET_RHO
    issuers  = list(issuer_margins.keys())
    dm_vec   = np.array([issuer_margins[iss] for iss in issuers])
    n_iss    = len(issuers)

    if n_iss == 0:
        total_cr_dm = 0.0
    elif n_iss == 1:
        total_cr_dm = float(dm_vec[0])
    else:
        rho_iss = np.zeros((n_iss, n_iss))
        for i, iss_i in enumerate(issuers):
            for j, iss_j in enumerate(issuers):
                if i == j:
                    rho_iss[i, j] = 1.0
                elif issuer_buckets[iss_i] == issuer_buckets[iss_j]:
                    rho_iss[i, j] = SIMM_CR_INTRA_BUCKET_RHO
                else:
                    rho_iss[i, j] = SIMM_CR_INTER_BUCKET_RHO
        total_cr_dm = float(np.sqrt(max(dm_vec @ rho_iss @ dm_vec, 0.0)))

    return {
        "DeltaMargin_Credit": total_cr_dm,
        "by_issuer"         : issuer_margins,
        "WS"                : issuer_ws,
    }


# =============================================================================
# ISDA SIMM — SCALAR MARGINS (Equity, Commodity, FX)
# =============================================================================

def simm_equity_delta(
    net_sensitivities: Dict[str, float],
    is_index: Optional[Dict[str, bool]] = None,
) -> float:
    """
    Simplified SIMM Equity Delta Margin.

    Uses SIMM v2.5 supervisory risk weights (Table 60):
        Large-cap, developed: 20% × |delta|
        Index: 15% × |delta|
        Other: 25% × |delta|

    Aggregates across issuers with intra-bucket ρ = 0.15 (large-cap)
    and γ = 0.75 (index-index). Here we use a simplified conservative
    scalar: DM = sqrt(Σ_i (RW_i × s_i)²) (zero cross-issuer correlation).

    Parameters
    ----------
    net_sensitivities : dict
        {equity_name: net delta sensitivity in currency units}
    is_index : dict, optional
        {equity_name: True if index}. Default: all treated as large-cap.

    Returns
    -------
    float
        Equity delta IM.
    """
    is_index = is_index or {}
    ws_sq_sum = 0.0
    for name, sens in net_sensitivities.items():
        rw         = 0.15 if is_index.get(name, False) else 0.20
        ws_sq_sum += (rw * sens) ** 2
    return float(np.sqrt(ws_sq_sum))


def simm_fx_delta(
    net_sensitivities: Dict[str, float],
) -> float:
    """
    SIMM FX Delta Margin.

    Each currency pair has RW = 7.4% (SIMM v2.5, Table 66).
    Currency pairs aggregate with zero correlation across pairs:

        DM_FX = sqrt(Σ_ccy (0.074 × s_ccy)²)

    Parameters
    ----------
    net_sensitivities : dict
        {currency_pair: net FX delta in currency units}

    Returns
    -------
    float
        FX delta IM.
    """
    FX_RW = 0.074
    return float(np.sqrt(sum((FX_RW * s) ** 2 for s in net_sensitivities.values())))


def simm_commodity_delta(
    net_sensitivities: Dict[str, float],
) -> float:
    """
    SIMM Commodity Delta Margin.

    Uses simplified RW = 18% for energy/metals, 16% for agriculture
    (SIMM v2.5, Table 63). Assumes zero cross-commodity correlation.

    Parameters
    ----------
    net_sensitivities : dict
        {commodity_type: net delta in currency units}

    Returns
    -------
    float
        Commodity delta IM.
    """
    COMM_RW = 0.18   # conservative — energy/metals RW
    return float(np.sqrt(sum((COMM_RW * s) ** 2 for s in net_sensitivities.values())))


# =============================================================================
# ISDA SIMM — FULL AGGREGATION
# =============================================================================

class SIMMEngine:
    """
    Full ISDA SIMM Initial Margin calculator (delta only).

    Computes delta IM for each risk class, then aggregates across
    risk classes using the SIMM cross-class correlation matrix γ.

    Parameters
    ----------
    ir_sensitivities : dict, optional
        {currency: {tenor: delta}} — passed to simm_ir_delta().
    credit_sensitivities : dict, optional
        {issuer: {tenor: cs01}} — passed to simm_credit_delta().
    equity_sensitivities : dict, optional
        {equity: delta} — passed to simm_equity_delta().
    fx_sensitivities : dict, optional
        {ccy_pair: delta} — passed to simm_fx_delta().
    commodity_sensitivities : dict, optional
        {commodity: delta} — passed to simm_commodity_delta().
    currency_groups : dict, optional
        {currency: 'regular' | 'low_vol' | 'high_vol'}
    credit_quality : dict, optional
        {issuer: rating_bucket} — passed to simm_credit_delta().
    equity_is_index : dict, optional
        {equity: bool} — passed to simm_equity_delta().
    add_on_multiplier : float
        Regulatory add-on multiplier. Default 1.0 (no add-on).
        Set to > 1.0 for specific WWR trades per BCBS 279 §58.
    """

    def __init__(
        self,
        ir_sensitivities        : Optional[Dict] = None,
        credit_sensitivities    : Optional[Dict] = None,
        equity_sensitivities    : Optional[Dict] = None,
        fx_sensitivities        : Optional[Dict] = None,
        commodity_sensitivities : Optional[Dict] = None,
        currency_groups         : Optional[Dict[str, str]] = None,
        credit_quality          : Optional[Dict[str, str]] = None,
        equity_is_index         : Optional[Dict[str, bool]] = None,
        add_on_multiplier       : float = 1.0,
    ) -> None:
        self.ir_sens    = ir_sensitivities        or {}
        self.cr_sens    = credit_sensitivities    or {}
        self.eq_sens    = equity_sensitivities    or {}
        self.fx_sens    = fx_sensitivities        or {}
        self.co_sens    = commodity_sensitivities or {}
        self.ccy_groups = currency_groups         or {}
        self.cr_quality = credit_quality          or {}
        self.eq_index   = equity_is_index         or {}
        self.add_on     = float(add_on_multiplier)

    def compute(self) -> dict:
        """
        Compute total SIMM IM and per-class breakdown.

        Returns
        -------
        dict with keys:
            IM_total        : float — Total SIMM IM (post add-on multiplier)
            IM_pre_addon    : float — Before add-on multiplier
            by_class        : dict  — {risk_class: DeltaMargin}
            IR_detail       : dict  — Full output of simm_ir_delta()
            Credit_detail   : dict  — Full output of simm_credit_delta()
            add_on_multiplier: float
        """
        # Per-class delta margins
        ir_result  = simm_ir_delta(self.ir_sens, self.ccy_groups)
        cr_result  = simm_credit_delta(self.cr_sens, self.cr_quality)
        dm_ir      = ir_result["DeltaMargin_IR"]
        dm_cr_q    = cr_result["DeltaMargin_Credit"]
        dm_cr_nq   = 0.0   # non-qualifying credit — placeholder
        dm_eq      = simm_equity_delta(self.eq_sens, self.eq_index)
        dm_co      = simm_commodity_delta(self.co_sens)
        dm_fx      = simm_fx_delta(self.fx_
