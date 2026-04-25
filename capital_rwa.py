"""
capital_rwa.py
--------------
Basel III regulatory capital computation for the Quant Desk Toolkit.

Covers the two capital charges that XVA desks are responsible for:

    1. CCR Capital  — capital against counterparty default risk
    2. CVA Capital  — capital against mark-to-market volatility of CVA

This module sits downstream of sa_ccr.py (which produces EAD) and
xva.py (which produces CVA). It converts those outputs into RWA and
minimum capital requirements under Basel III / Basel IV rules.

Framework references
--------------------
CCR Capital (IRB):
    BCBS (2006). "International Convergence of Capital Measurement and
    Capital Standards." Annex 5 — IRB formula for corporate exposures.

CVA Capital — BA-CVA and SA-CVA:
    BCBS (2017). "Finalization of post-crisis reforms." d424, Chapter 7.
    Also known as "Basel IV" CVA framework, effective January 2023 (phased).

Capital adequacy:
    BCBS (2010). "Basel III: A global regulatory framework." CET1 ≥ 4.5%,
    Tier 1 ≥ 6.0%, Total Capital ≥ 8.0% of RWA. Conservation buffer: +2.5%.

Architecture
------------
Inputs:
    EAD           from SACCREngine.compute() in sa_ccr.py
    PD, LGD, M    from internal credit models or external ratings
    CVA           from XVAEngine.compute() in xva.py (for context; not
                  used directly in the regulatory formulas here)

Outputs:
    RWA_CCR       — Risk-weighted assets for counterparty default risk
    K_CVA         — CVA capital requirement (currency units)
    RWA_CVA       — CVA risk-weighted assets (= 12.5 × K_CVA)
    RWA_total     — RWA_CCR + RWA_CVA
    Capital_min   — 8% × RWA_total (Pillar 1 minimum)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from scipy.stats import norm


# =============================================================================
# CONSTANTS
# =============================================================================

# Minimum capital ratios (Basel III, post-conservation buffer)
CET1_RATIO_MIN         = 0.045  # 4.5% CET1
CET1_RATIO_WITH_BUFFER = 0.070  # 4.5% + 2.5% conservation buffer
TIER1_RATIO_MIN        = 0.060  # 6.0% Tier 1
TOTAL_CAPITAL_MIN      = 0.080  # 8.0% Total Capital

RWA_SCALAR             = 12.5   # = 1 / 8%

# BA-CVA inter-counterparty correlation
BA_CVA_RHO             = 0.50

# BA-CVA supervisory risk weights by credit quality (BCBS d424, §20.26)
BA_CVA_RISK_WEIGHT: Dict[str, float] = {
    "AAA" : 0.004,
    "AA"  : 0.004,
    "A"   : 0.005,
    "BBB" : 0.010,
    "BB"  : 0.020,
    "B"   : 0.030,
    "CCC" : 0.100,
    "NR"  : 0.020,   # unrated — same as BB per BCBS
}

# SA standardised risk weights for CCR (non-IRB banks) by counterparty type
# Source: BCBS 2017, Table 1 (CCR standardised approach)
SA_RISK_WEIGHT: Dict[str, float] = {
    "Sovereign"             : 0.00,   # AAA–AA sovereigns
    "Sovereign_lower"       : 0.50,   # A–BBB sovereigns
    "Bank"                  : 0.20,   # OECD banks
    "Corporate_AAA_AA"      : 0.20,
    "Corporate_A"           : 0.50,
    "Corporate_BBB"         : 1.00,
    "Corporate_BB"          : 1.00,
    "Corporate_B_and_below" : 1.50,
    "Corporate_NR"          : 1.00,
    "Retail"                : 0.75,
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CounterpartyCapitalInput:
    """
    Inputs for one counterparty's capital calculation.

    Parameters
    ----------
    name : str
        Counterparty identifier.
    ead : float
        Exposure at Default from SA-CCR (post alpha=1.4 scaling).
    pd : float
        1-year probability of default (decimal, e.g. 0.005 for 50 bps).
    lgd : float
        Loss Given Default (decimal). Supervisory LGD = 0.45 for senior
        unsecured under Foundation IRB; 0.40 for financial institutions
        under the CCR IRB formula (BCBS 2006, §272).
    maturity : float
        Effective maturity in years. For netting sets without a fixed
        end date, use min(remaining maturity, 5) per BCBS §320.
    credit_quality : str
        Rating bucket for BA-CVA risk weight lookup. One of:
        'AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'NR'.
    counterparty_type : str
        For SA (non-IRB) risk weight lookup. One of the keys in
        SA_RISK_WEIGHT.
    is_financial : bool
        True for regulated financial institutions and their subsidiaries.
        Triggers the 1.25× asset correlation scalar in the IRB formula.
    """
    name              : str
    ead               : float
    pd                : float
    lgd               : float = 0.45
    maturity          : float = 1.0
    credit_quality    : str   = "BBB"
    counterparty_type : str   = "Corporate_BBB"
    is_financial      : bool  = False


# =============================================================================
# CCR CAPITAL — IRB FORMULA
# =============================================================================

def asset_correlation(pd: float, is_financial: bool = False) -> float:
    """
    Basel III supervisory asset correlation R for the IRB capital formula.

    For corporate counterparties (BCBS §272):

        R = 0.12 × (1 - e^{-50 PD}) / (1 - e^{-50})
          + 0.24 × [1 - (1 - e^{-50 PD}) / (1 - e^{-50})]

    For regulated financial institutions and their unregulated affiliates
    with assets ≥ $100bn, R is scaled up by 1.25× (BCBS §272(i)):

        R_fin = 1.25 × R_corporate

    The correlation formula gives R ∈ [12%, 24%]. Higher PD → lower R
    (distressed firms are more idiosyncratic). Lower PD → higher R
    (high-quality firms are more exposed to systematic risk).
    """
    pd = max(pd, 1e-8)
    e50   = np.exp(-50.0)
    exp_t = np.exp(-50.0 * pd)
    w     = (1.0 - exp_t) / (1.0 - e50)
    R     = 0.12 * w + 0.24 * (1.0 - w)
    
    if is_financial:
        R *= 1.25
        
    return R


def maturity_adjustment(pd: float, maturity: float) -> float:
    """
    Basel III maturity adjustment for the IRB formula (BCBS §272):

        b = (0.11852 - 0.05478 × ln(PD))²
        MA = (1 + (M - 2.5) × b) / (1 - 1.5 × b)

    The adjustment is > 1 for M > 2.5 years and < 1 for M < 2.5 years,
    penalising longer-maturity exposures for refinancing / rollover risk.
    """
    pd       = max(pd, 1e-8)
    maturity = min(max(maturity, 1.0), 5.0)   # [1, 5] per BCBS §320
    b        = (0.11852 - 0.05478 * np.log(pd)) ** 2
    
    return (1.0 + (maturity - 2.5) * b) / (1.0 - 1.5 * b)


def irb_capital_requirement(
    ead: float,
    pd: float,
    lgd: float,
    maturity: float,
    is_financial: bool = False,
) -> dict:
    """
    Full Basel III IRB capital requirement for a single counterparty.

    Formula (BCBS §272):
        R   = asset_correlation(PD, is_financial)
        MA  = maturity_adjustment(PD, M)
        K   = LGD × [N(√(R/(1-R)) × G(PD) + √(1/(1-R)) × G(0.999)) - PD] × MA
        RWA = 12.5 × EAD × K

    Returns
    -------
    dict
        Contains R, MA, K, RWA_CCR, EL, and UL.
    """
    pd = max(pd, 1e-8)
    R  = asset_correlation(pd, is_financial)
    MA = maturity_adjustment(pd, maturity)

    G_pd    = norm.ppf(pd)
    G_999   = norm.ppf(0.999)
    N_inner = norm.cdf(
        np.sqrt(R / (1.0 - R)) * G_pd
        + np.sqrt(1.0 / (1.0 - R)) * G_999
    )

    K   = lgd * max(N_inner - pd, 0.0) * MA
    RWA = RWA_SCALAR * ead * K
    EL  = pd * lgd * ead

    return {
        "R"      : R,
        "MA"     : MA,
        "K"      : K,
        "RWA_CCR": RWA,
        "EL"     : EL,
        "UL"     : K * ead,
    }


def sa_capital_requirement(
    ead: float,
    counterparty_type: str,
) -> dict:
    """
    Standardised approach CCR capital for non-IRB banks.

        K_SA  = RW × EAD × 8%
        RWA   = RW × EAD
    """
    rw = SA_RISK_WEIGHT.get(counterparty_type, 1.00)
    rwa = rw * ead
    return {
        "RW"     : rw,
        "RWA_CCR": rwa,
        "K_SA"   : TOTAL_CAPITAL_MIN * rwa,
    }


# =============================================================================
# CVA CAPITAL — BA-CVA
# =============================================================================

def ba_cva_capital(
    counterparties: List[CounterpartyCapitalInput],
    include_hedges: bool = False,
    hedge_notionals: Optional[Dict[str, float]] = None,
) -> dict:
    """
    Basic Approach CVA capital (BA-CVA) per BCBS d424 §20.22–20.30.

    The capital formula aggregates "CVA sensitivities" SC_c across
    counterparties using a one-factor correlation model:

        SC_c = RW_c × M_c_eff × EAD_c_disc
        K_BA_CVA = 0.25 × sqrt(ρ² × (Σ_c SC_c)² + (1-ρ²) × Σ_c SC_c²)
        RWA_CVA  = 12.5 × K_BA_CVA
    """
    hedge_notionals = hedge_notionals or {}
    SC: Dict[str, float] = {}

    for cp in counterparties:
        rw      = BA_CVA_RISK_WEIGHT.get(cp.credit_quality.upper(), 0.020)
        m_eff   = min(cp.maturity, 5.0)

        # Discounted EAD: EAD × (1 - e^{-0.05M}) / (0.05M)
        if m_eff > 1e-6:
            disc_factor = (1.0 - np.exp(-0.05 * m_eff)) / (0.05 * m_eff)
        else:
            disc_factor = 1.0
            
        ead_disc = cp.ead * disc_factor
        sc_c = rw * m_eff * ead_disc

        # Hedge offset (reduced BA-CVA)
        if include_hedges and cp.name in hedge_notionals:
            hedge_sc = rw * m_eff * hedge_notionals[cp.name]
            sc_c    -= hedge_sc   # can drive SC_c negative (over-hedged)

        SC[cp.name] = sc_c

    sc_values     = np.array(list(SC.values()))
    rho           = BA_CVA_RHO
    systematic    = (rho * np.sum(sc_values)) ** 2
    idiosyncratic = (1.0 - rho ** 2) * np.sum(sc_values ** 2)

    K_ba_cva    = 0.25 * np.sqrt(max(systematic + idiosyncratic, 0.0))
    rwa_cva     = RWA_SCALAR * K_ba_cva

    return {
        "K_BA_CVA"      : K_ba_cva,
        "RWA_CVA"       : rwa_cva,
        "SC"            : SC,
        "systematic"    : systematic,
        "idiosyncratic" : idiosyncratic,
    }


# =============================================================================
# CVA CAPITAL — SA-CVA (DELTA-ONLY, SIMPLIFIED)
# =============================================================================

def sa_cva_capital(
    cvA_sensitivities: Dict[str, Dict[str, float]],
    correlation_ir: float = 0.50,
    correlation_cs: float = 0.50,
) -> dict:
    """
    Simplified SA-CVA capital based on CVA sensitivities (delta only).

    Aggregation formula (within one risk class):
        K_delta = sqrt( (Σ_b WS_b)² × ρ² + Σ_b WS_b² × (1-ρ²) )
    """
    # Simplified IR supervisory risk weights by tenor bucket
    IR_RW: Dict[str, float] = {
        "0.25Y": 0.0160,
        "0.5Y" : 0.0160,
        "1Y"   : 0.0114,
        "2Y"   : 0.0098,
        "3Y"   : 0.0098,
        "5Y"   : 0.0098,
        "10Y"  : 0.0098,
        "15Y"  : 0.0098,
        "20Y"  : 0.0098,
        "30Y"  : 0.0098,
    }

    WS: Dict[str, Dict[str, float]] = {}
    capital_components: Dict[str, float] = {}

    for risk_class, buckets in cvA_sensitivities.items():
        ws_class: Dict[str, float] = {}

        for bucket, sensitivity in buckets.items():
            if risk_class == "IR":
                rw = IR_RW.get(bucket, 0.0114)   # default to 1.14% if bucket unknown
            elif risk_class == "CS":
                rw = BA_CVA_RISK_WEIGHT.get(bucket.upper(), 0.020)
            else:
                rw = 0.0
                
            ws_class[bucket] = rw * sensitivity

        WS[risk_class] = ws_class
        ws_vals = np.array(list(ws_class.values()))

        rho = correlation_ir if risk_class == "IR" else correlation_cs

        systematic    = (rho * np.sum(ws_vals)) ** 2
        idiosyncratic = (1.0 - rho ** 2) * np.sum(ws_vals ** 2)
        capital_components[risk_class] = np.sqrt(max(systematic + idiosyncratic, 0.0))

    K_sa_cva = np.sqrt(sum(v ** 2 for v in capital_components.values()))
    rwa_cva  = RWA_SCALAR * K_sa_cva

    return {
        "K_IR"      : capital_components.get("IR", 0.0),
        "K_CS"      : capital_components.get("CS", 0.0),
        "K_SA_CVA"  : K_sa_cva,
        "RWA_CVA"   : rwa_cva,
        "WS"        : WS,
    }


# =============================================================================
# FULL CAPITAL AGGREGATION
# =============================================================================

class CapitalEngine:
    """
    Aggregate regulatory capital across all counterparties.

    Combines CCR capital (IRB or SA) and CVA capital (BA-CVA or SA-CVA)
    into total RWA and minimum capital requirements.
    """

    def __init__(
        self,
        counterparties     : List[CounterpartyCapitalInput],
        use_irb            : bool = True,
        use_sa_cva         : bool = False,
        cva_sensitivities  : Optional[Dict] = None,
        include_cva_hedges : bool = False,
        hedge_notionals    : Optional[Dict[str, float]] = None,
    ) -> None:
        self.counterparties     = counterparties
        self.use_irb            = use_irb
        self.use_sa_cva         = use_sa_cva
        self.cva_sensitivities  = cva_sensitivities or {}
        self.include_cva_hedges = include_cva_hedges
        self.hedge_notionals    = hedge_notionals or {}

    def compute_ccr(self) -> Dict[str, dict]:
        results = {}
        for cp in self.counterparties:
            if self.use_irb:
                results[cp.name] = irb_capital_requirement(
                    ead          = cp.ead,
                    pd           = cp.pd,
                    lgd          = cp.lgd,
                    maturity     = cp.maturity,
                    is_financial = cp.is_financial,
                )
            else:
                results[cp.name] = sa_capital_requirement(
                    ead               = cp.ead,
                    counterparty_type = cp.counterparty_type,
                )
        return results

    def compute_cva(self) -> dict:
        if self.use_sa_cva:
            return sa_cva_capital(self.cva_sensitivities)
        else:
            return ba_cva_capital(
                counterparties  = self.counterparties,
                include_hedges  = self.include_cva_hedges,
                hedge_notionals = self.hedge_notionals,
            )

    def compute(self) -> dict:
        ccr_results = self.compute_ccr()
        cva_result  = self.compute_cva()

        rwa_ccr_total = sum(r.get("RWA_CCR", 0.0) for r in ccr_results.values())
        el_total      = sum(r.get("EL", 0.0) for r in ccr_results.values())
        rwa_cva       = cva_result.get("RWA_CVA", 0.0)
        rwa_total     = rwa_ccr_total + rwa_cva

        return {
            "CCR"               : ccr_results,
            "CVA"               : cva_result,
            "RWA_CCR_total"     : rwa_ccr_total,
            "RWA_CVA"           : rwa_cva,
            "RWA_total"         : rwa_total,
            "EL_total"          : el_total,
            "capital_min"       : {
                "CET1_min"        : CET1_RATIO_MIN         * rwa_total,
                "CET1_with_buffer": CET1_RATIO_WITH_BUFFER * rwa_total,
                "Tier1_min"       : TIER1_RATIO_MIN        * rwa_total,
                "Total_min"       : TOTAL_CAPITAL_MIN      * rwa_total,
            },
            "counterparty_type": "IRB" if self.use_irb else "SA",
            "cva_approach"     : "SA-CVA" if self.use_sa_cva else "BA-CVA",
        }


# =============================================================================
# CAPITAL ADEQUACY CHECK
# =============================================================================

def capital_adequacy_check(
    cet1_capital   : float,
    tier1_capital  : float,
    total_capital  : float,
    rwa_total      : float,
) -> dict:
    """
    Check whether actual capital levels meet Basel III minimum requirements.
    """
    if rwa_total <= 0:
        raise ValueError("RWA must be positive.")

    cet1_ratio  = cet1_capital  / rwa_total
    t1_ratio    = tier1_capital / rwa_total
    tc_ratio    = total_capital / rwa_total

    cet1_min     = CET1_RATIO_MIN         <= cet1_ratio
    cet1_buf     = CET1_RATIO_WITH_BUFFER <= cet1_ratio
    t1_pass      = TIER1_RATIO_MIN        <= t1_ratio
    tc_pass      = TOTAL_CAPITAL_MIN      <= tc_ratio

    return {
        "CET1_ratio"             : cet1_ratio,
        "Tier1_ratio"            : t1_ratio,
        "Total_capital_ratio"    : tc_ratio,
        "CET1_passes_minimum"    : cet1_min,
        "CET1_passes_with_buffer": cet1_buf,
        "Tier1_passes"           : t1_pass,
        "Total_passes"           : tc_pass,
        "all_pass"               : all([cet1_min, t1_pass, tc_pass]),
        "headroom"               : {
            "CET1_vs_minimum"    : (cet1_ratio  - CET1_RATIO_MIN)         * rwa_total,
            "CET1_vs_buffer"     : (cet1_ratio  - CET1_RATIO_WITH_BUFFER) * rwa_total,
            "Tier1_vs_minimum"   : (t1_ratio    - TIER1_RATIO_MIN)        * rwa_total,
            "Total_vs_minimum"   : (tc_ratio    - TOTAL_CAPITAL_MIN)      * rwa_total,
        },
    }