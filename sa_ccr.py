
"""
sa_ccr.py
---------
Basel III SA-CCR (Standardised Approach for Counterparty Credit Risk)
implementation for the Quant Desk Toolkit.

SA-CCR replaces the Current Exposure Method (CEM) and the Standardised Method
(SM) as the regulatory capital framework for OTC derivatives and SFTs under
BCBS 279 (April 2014), implemented in the US via the Federal Reserve's final
rule (November 2019).

Structure
---------
SA-CCR decomposes Exposure at Default (EAD) into two additive components:

    EAD = alpha * (RC + PFE_addon)

    where alpha = 1.4 (regulatory multiplier)

    RC (Replacement Cost): the current loss if the counterparty were to
    default today. For unmargined netting sets: max(V_net, 0). For margined
    netting sets: accounts for the collateral already posted/received.

    PFE_addon (Potential Future Exposure): forward-looking measure capturing
    the risk that MTM deteriorates before close-out. Computed as the product
    of an aggregated AddOn and a multiplier that accounts for
    over-collateralisation.

        PFE_addon = multiplier * AddOn_aggregate

Asset Class AddOns
------------------
The aggregate AddOn is the sum of asset-class-specific AddOns. Each asset
class uses a different aggregation formula:

    Interest Rate:  AddOn_IR   = sum over hedging sets of sqrt(sum_k D_k^2 + cross terms)
    Credit:         AddOn_Cr   = sum over single-name entities of sqrt(sum_k)^2 structure
    Equity:         AddOn_Eq   = sum over entities
    Commodity:      AddOn_Comm = sum over commodity types
    FX:             AddOn_FX   = sum over currency pairs

This module implements:
  - RC computation for margined and unmargined netting sets
  - PFE multiplier
  - IR AddOn (full hedging set aggregation)
  - FX AddOn
  - Equity AddOn
  - EAD computation

References
----------
BCBS (2014). "The Standardised Approach for Measuring Counterparty Credit
    Risk Exposures." Basel Committee on Banking Supervision, April 2014.
Federal Reserve (2019). "Standardized Approach for Calculating the Exposure
    Amount of Derivative Contracts." Final Rule, November 2019.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from scipy.stats import norm


# =============================================================================
# REGULATORY CONSTANTS
# =============================================================================

ALPHA = 1.4          # Regulatory multiplier (BCBS 279, §128)

# Supervisory factors by asset class (BCBS 279, Annex A, Table 1)
# Units: fraction (e.g. 0.005 = 0.5%)
SUPERVISORY_FACTOR: Dict[str, float] = {
    # Interest rate
    "IR"              : 0.005,
    # FX
    "FX"              : 0.04,
    # Credit: single-name
    "Credit_AAA"      : 0.0038,
    "Credit_AA"       : 0.0038,
    "Credit_A"        : 0.0042,
    "Credit_BBB"      : 0.0054,
    "Credit_BB"       : 0.0106,
    "Credit_B"        : 0.0106,
    "Credit_CCC"      : 0.0106,
    "Credit_NR"       : 0.0106,
    # Credit: index
    "Credit_IG_index" : 0.0038,
    "Credit_SG_index" : 0.0106,
    # Equity: single-name
    "Equity_single"   : 0.32,
    # Equity: index
    "Equity_index"    : 0.20,
    # Commodity
    "Comm_electricity": 0.40,
    "Comm_oil_gas"    : 0.18,
    "Comm_metals"     : 0.18,
    "Comm_agri"       : 0.18,
    "Comm_other"      : 0.18,
}

# Supervisory correlation rho by asset class (BCBS 279, Annex A, Table 2)
# Used in the aggregation of entity-level AddOns within an asset class.
SUPERVISORY_CORRELATION: Dict[str, float] = {
    "IR"              : 0.50,   # across currency pairs within same hedging set
    "FX"              : 1.00,   # each currency pair is its own hedging set
    "Credit_single"   : 0.50,
    "Credit_index"    : 0.80,
    "Equity_single"   : 0.50,
    "Equity_index"    : 0.80,
    "Comm_electricity": 0.40,
    "Comm_oil_gas"    : 0.40,
    "Comm_metals"     : 0.40,
    "Comm_agri"       : 0.40,
    "Comm_other"      : 0.40,
}

# IR supervisory duration boundaries (years)
IR_MATURITY_BUCKETS = {1: (0.0, 1.0), 2: (1.0, 5.0), 3: (5.0, np.inf)}
IR_BUCKET_CORRELATION = np.array([
    [1.00, 0.70, 0.30],
    [0.70, 1.00, 0.70],
    [0.30, 0.70, 1.00],
])  # correlation between buckets 1-2-3


# =============================================================================
# TRADE-LEVEL DATA STRUCTURES
# =============================================================================

@dataclass
class IRTrade:
    """
    Interest rate derivative trade for SA-CCR.

    Parameters
    ----------
    notional : float
        Notional in domestic currency.
    maturity : float
        Remaining maturity in years.
    start_date : float
        Start date in years from today. 0 for spot-starting swaps.
    end_date : float
        End date in years from today (= maturity for vanilla swaps).
    reference_currency : str
        ISO currency code (e.g. 'USD', 'EUR'). Defines hedging set.
    payer : bool
        True if payer (pay fixed). Sign of adjusted notional.
    current_mtm : float
        Current mark-to-market (positive = asset, negative = liability).
    """
    notional           : float
    maturity           : float
    start_date         : float
    end_date           : float
    reference_currency : str
    payer              : bool
    current_mtm        : float = 0.0


@dataclass
class FXTrade:
    """
    FX derivative trade for SA-CCR.

    Parameters
    ----------
    notional : float
        Notional in domestic currency.
    maturity : float
        Remaining maturity in years.
    currency_pair : str
        Currency pair (e.g. 'EURUSD'). Defines hedging set.
    long_foreign : bool
        True if long the foreign currency (call on foreign CCY).
    current_mtm : float
    """
    notional      : float
    maturity      : float
    currency_pair : str
    long_foreign  : bool
    current_mtm   : float = 0.0


@dataclass
class EquityTrade:
    """
    Equity derivative trade for SA-CCR.

    Parameters
    ----------
    notional : float
    maturity : float
    underlying : str
        Ticker / entity name. Defines the entity-level grouping.
    is_index : bool
        True for index products (lower supervisory factor + higher rho).
    long : bool
        True if long the underlying.
    current_mtm : float
    """
    notional    : float
    maturity    : float
    underlying  : str
    is_index    : bool
    long        : bool
    current_mtm : float = 0.0


# =============================================================================
# MATURITY FACTOR
# =============================================================================

def maturity_factor(maturity: float, margined: bool = False) -> float:
    """
    Compute the SA-CCR maturity factor MF.

    For unmargined netting sets (BCBS 279, §164):
        MF = sqrt(min(M, 1) / 1)  — square root of years, capped at 1 year

    For margined netting sets (BCBS 279, §165):
        MF = 3/2 * sqrt(MPOR / 1)  — scaled by 3/2 to account for settlement risk

    Parameters
    ----------
    maturity : float
        For unmargined: remaining maturity of the trade in years.
        For margined: Margin Period of Risk (MPoR) in years.
    margined : bool
        True for margined netting sets.

    Returns
    -------
    float
        Maturity factor MF.
    """
    if margined:
        # MPOR-based maturity factor (Equation 6, BCBS 279)
        return 1.5 * np.sqrt(maturity)
    else:
        # Trade-level maturity factor (Equation 5)
        return np.sqrt(min(maturity, 1.0))


# =============================================================================
# SUPERVISORY DURATION (IR ONLY)
# =============================================================================

def supervisory_duration(start: float, end: float) -> float:
    """
    Supervisory duration for interest rate trades (BCBS 279, §167):

        SD = (exp(-0.05 * S) - exp(-0.05 * E)) / 0.05

    where S = start date and E = end date in years, and 0.05 is the
    supervisory discount rate.

    Parameters
    ----------
    start : float
        Start date of the IR trade in years (0 for spot-starting).
    end : float
        End date / maturity in years.

    Returns
    -------
    float
        Supervisory duration SD, always ≥ 0.

    Notes
    -----
    For standard vanilla swaps, start ≈ 0 and SD ≈ annuity factor.
    For forward-starting swaps, SD is smaller for the same total tenor.
    """
    return max((np.exp(-0.05 * start) - np.exp(-0.05 * end)) / 0.05, 0.0)


# =============================================================================
# ADJUSTED NOTIONAL
# =============================================================================

def adjusted_notional_ir(trade: IRTrade) -> float:
    """
    Adjusted notional for an IR trade:

        d_i = N_i * SD_i

    where SD_i = supervisory duration for trade i.

    Parameters
    ----------
    trade : IRTrade

    Returns
    -------
    float
        Adjusted notional (signed: positive for payer, negative for receiver).
    """
    sd   = supervisory_duration(trade.start_date, trade.end_date)
    sign = 1.0 if trade.payer else -1.0
    return sign * trade.notional * sd


def adjusted_notional_fx(trade: FXTrade) -> float:
    """
    Adjusted notional for an FX trade: N_i (no duration scaling).

    Parameters
    ----------
    trade : FXTrade

    Returns
    -------
    float
        Signed adjusted notional.
    """
    sign = 1.0 if trade.long_foreign else -1.0
    return sign * trade.notional


def adjusted_notional_equity(trade: EquityTrade) -> float:
    """
    Adjusted notional for an equity trade: N_i (current price * quantity).

    Parameters
    ----------
    trade : EquityTrade

    Returns
    -------
    float
        Signed adjusted notional.
    """
    sign = 1.0 if trade.long else -1.0
    return sign * trade.notional


# =============================================================================
# EFFECTIVE NOTIONAL (TRADE-LEVEL)
# =============================================================================

def effective_notional_ir(trade: IRTrade, margined: bool = False) -> float:
    """
    Effective notional for an IR trade at the trade level:

        EN_i = d_i * MF_i * SF

    Parameters
    ----------
    trade : IRTrade
    margined : bool

    Returns
    -------
    float
        Effective notional (signed).
    """
    d_i  = adjusted_notional_ir(trade)
    mf_i = maturity_factor(trade.maturity, margined=margined)
    sf   = SUPERVISORY_FACTOR["IR"]
    return d_i * mf_i * sf


# =============================================================================
# REPLACEMENT COST (RC)
# =============================================================================

def replacement_cost_unmargined(
    netting_set_mtm: float,
    collateral_posted_by_cpty: float = 0.0,
) -> float:
    """
    Replacement Cost for an unmargined netting set (BCBS 279, §136):

        RC = max(V_net - C, 0)

    where C is the collateral posted by the counterparty (net of what we
    have posted to them — negative if we have posted more).

    Parameters
    ----------
    netting_set_mtm : float
        Current net MTM of the netting set. Positive = in our favour.
    collateral_posted_by_cpty : float
        Net collateral received from counterparty. Positive = we hold their
        collateral. Negative = we have posted more than we received.

    Returns
    -------
    float
        Replacement cost (non-negative).
    """
    return max(netting_set_mtm - collateral_posted_by_cpty, 0.0)


def replacement_cost_margined(
    netting_set_mtm: float,
    vm_received: float = 0.0,
    im_received: float = 0.0,
    im_posted: float = 0.0,
    threshold: float = 0.0,
    mta: float = 0.0,
) -> float:
    """
    Replacement Cost for a margined netting set (BCBS 279, §145–148):

        RC = max(V_net - VM_received - IM_net, TH + MTA - IM_net, 0)

    where:
        VM_received  = variation margin received from counterparty
        IM_net       = IM_received - IM_posted (net initial margin)
        TH           = VM threshold
        MTA          = minimum transfer amount

    The max with (TH + MTA - IM_net) captures the scenario where the
    netting set is in-the-money by less than the threshold, so no VM
    has been called yet. In that case RC = TH + MTA - IM_net.

    Parameters
    ----------
    netting_set_mtm : float
    vm_received : float
        Current VM received from counterparty.
    im_received : float
        Initial margin received from counterparty.
    im_posted : float
        Initial margin posted to counterparty.
    threshold : float
        VM threshold (CSA threshold).
    mta : float
        Minimum transfer amount.

    Returns
    -------
    float
        Replacement cost (non-negative).
    """
    im_net = im_received - im_posted
    rc1    = netting_set_mtm - vm_received - im_net
    rc2    = threshold + mta - im_net
    return max(rc1, rc2, 0.0)


# =============================================================================
# PFE MULTIPLIER
# =============================================================================

def pfe_multiplier(
    netting_set_mtm: float,
    collateral_net: float,
    aggregate_addon: float,
    floor: float = 0.05,
) -> float:
    """
    PFE multiplier for over-collateralised netting sets (BCBS 279, §148):

        multiplier = min(1, floor + (1 - floor) * exp(V_net - C) / (2 * (1-floor) * AddOn))

    When the netting set is deeply over-collateralised (V_net - C << 0),
    the multiplier approaches the floor (default 5%), reflecting the
    reduced but non-zero risk of future positive exposure.

    When V_net - C ≥ 0, the multiplier = 1 (standard PFE).

    Parameters
    ----------
    netting_set_mtm : float
        Current net MTM of the netting set.
    collateral_net : float
        Net collateral held (received - posted). Positive = over-collateralised.
    aggregate_addon : float
        Sum of all asset-class AddOns (before multiplier).
    floor : float
        Multiplier floor. Default 0.05 (5% per BCBS).

    Returns
    -------
    float
        Multiplier ∈ [floor, 1].
    """
    if aggregate_addon <= 0.0:
        return floor

    v_minus_c = netting_set_mtm - collateral_net
    exponent  = v_minus_c / (2.0 * (1.0 - floor) * aggregate_addon)
    mult      = floor + (1.0 - floor) * np.exp(exponent)
    return min(mult, 1.0)


# =============================================================================
# IR ADDON
# =============================================================================

def ir_addon(
    trades: List[IRTrade],
    margined: bool = False,
) -> float:
    """
    Compute the SA-CCR Interest Rate AddOn across all IR trades.

    Aggregation hierarchy (BCBS 279, §166–171):
      1. Trade → hedging set (currency). Within each currency:
      2. Each trade maps to one of 3 maturity buckets (< 1yr, 1–5yr, > 5yr).
      3. Within a bucket, effective notionals sum algebraically (netting).
      4. Across buckets, aggregation uses partial correlation:

             AddOn_ccy = sqrt(D1² + D2² + D3²
                              + 1.4 * D1*D2 + 1.4 * D2*D3 + 0.6 * D1*D3)

         where D_k = bucket-level effective notional and correlations are
         (ρ₁₂, ρ₂₃, ρ₁₃) = (0.70, 0.70, 0.30) from Table 2.

      5. Across currency hedging sets, AddOns sum (no cross-currency netting).

    Parameters
    ----------
    trades : list of IRTrade
    margined : bool

    Returns
    -------
    float
        IR AddOn (non-negative).
    """
    # Group by currency
    currencies: Dict[str, List[IRTrade]] = {}
    for t in trades:
        currencies.setdefault(t.reference_currency, []).append(t)

    total_addon = 0.0

    for ccy, ccy_trades in currencies.items():
        # Bucket effective notionals: sum within each bucket
        D = np.zeros(3)  # D[0] = bucket 1 (<1yr), D[1] = bucket 2 (1-5yr), D[2] = bucket 3 (>5yr)

        for t in ccy_trades:
            en = effective_notional_ir(t, margined=margined)
            if t.maturity < 1.0:
                D[0] += en
            elif t.maturity <= 5.0:
                D[1] += en
            else:
                D[2] += en

        # Intra-currency aggregation using bucket correlation matrix
        # AddOn_ccy = sqrt(D^T * Rho * D)
        rho = IR_BUCKET_CORRELATION
        addon_ccy = np.sqrt(max(D @ rho @ D, 0.0))  # numerical guard against -epsilon
        total_addon += addon_ccy

    return total_addon


# =============================================================================
# FX ADDON
# =============================================================================

def fx_addon(
    trades: List[FXTrade],
    margined: bool = False,
) -> float:
    """
    Compute the SA-CCR FX AddOn.

    Each currency pair forms its own hedging set. Within a currency pair,
    trades net algebraically. Across pairs, AddOns sum (no cross-pair netting).

        AddOn_FX = sum over currency pairs of |sum_k EN_k|
        where EN_k = N_k * MF_k * SF_FX (signed by direction)

    Parameters
    ----------
    trades : list of FXTrade
    margined : bool

    Returns
    -------
    float
        FX AddOn (non-negative).
    """
    # Group by currency pair
    pairs: Dict[str, float] = {}
    for t in trades:
        mf = maturity_factor(t.maturity, margined=margined)
        en = adjusted_notional_fx(t) * mf * SUPERVISORY_FACTOR["FX"]
        pairs[t.currency_pair] = pairs.get(t.currency_pair, 0.0) + en

    return sum(abs(v) for v in pairs.values())


# =============================================================================
# EQUITY ADDON
# =============================================================================

def equity_addon(
    trades: List[EquityTrade],
    margined: bool = False,
) -> float:
    """
    Compute the SA-CCR Equity AddOn.

    Aggregation (BCBS 279, §183–184):
      1. Within each entity (single-name or index), effective notionals sum.
      2. Across entities:

         AddOn_Eq = sqrt((rho * sum_k EN_k)² + (1 - rho²) * sum_k EN_k²)

         where rho = 0.50 (single-name) or 0.80 (index).

    When trades are a mix of single-name and index, they are grouped
    separately and their AddOns summed.

    Parameters
    ----------
    trades : list of EquityTrade
    margined : bool

    Returns
    -------
    float
        Equity AddOn (non-negative).
    """
    # Separate single-name vs index
    single_trades = [t for t in trades if not t.is_index]
    index_trades  = [t for t in trades if t.is_index]

    def _entity_addon(trade_list: List[EquityTrade], is_index: bool) -> float:
        sf  = SUPERVISORY_FACTOR["Equity_index" if is_index else "Equity_single"]
        rho = SUPERVISORY_CORRELATION["Equity_index" if is_index else "Equity_single"]

        # Group by underlying entity and compute entity-level effective notional
        entities: Dict[str, float] = {}
        for t in trade_list:
            mf = maturity_factor(t.maturity, margined=margined)
            en = adjusted_notional_equity(t) * mf * sf
            entities[t.underlying] = entities.get(t.underlying, 0.0) + en

        if not entities:
            return 0.0

        en_values = np.array(list(entities.values()))

        # Correlation aggregation formula
        systematic    = (rho * np.sum(en_values)) ** 2
        idiosyncratic = (1.0 - rho ** 2) * np.sum(en_values ** 2)
        return float(np.sqrt(max(systematic + idiosyncratic, 0.0)))

    return _entity_addon(single_trades, False) + _entity_addon(index_trades, True)


# =============================================================================
# EAD COMPUTATION
# =============================================================================

class SACCREngine:
    """
    SA-CCR EAD calculator for a single netting set.

    Computes:
        EAD = alpha * (RC + PFE_addon)
        PFE_addon = multiplier * AddOn_aggregate
        AddOn_aggregate = AddOn_IR + AddOn_FX + AddOn_EQ + ...

    Parameters
    ----------
    ir_trades : list of IRTrade, optional
    fx_trades : list of FXTrade, optional
    equity_trades : list of EquityTrade, optional
    margined : bool
        True if the netting set is subject to daily margining.
    netting_set_mtm : float
        Current net MTM of the netting set.
    collateral_net : float
        Net collateral held by us (received - posted). Used in RC and multiplier.
    vm_received : float
        VM received (margined sets only).
    im_received : float
        IM received (margined sets only).
    im_posted : float
        IM posted (margined sets only).
    threshold : float
        CSA threshold.
    mta : float
        Minimum transfer amount.
    mpor : float
        Margin period of risk in years (margined sets only). Default 10/252.
    """

    def __init__(
        self,
        ir_trades       : Optional[List[IRTrade]]     = None,
        fx_trades       : Optional[List[FXTrade]]     = None,
        equity_trades   : Optional[List[EquityTrade]] = None,
        margined        : bool    = False,
        netting_set_mtm : float   = 0.0,
        collateral_net  : float   = 0.0,
        vm_received     : float   = 0.0,
        im_received     : float   = 0.0,
        im_posted       : float   = 0.0,
        threshold       : float   = 0.0,
        mta             : float   = 0.0,
        mpor            : float   = 10 / 252,
    ) -> None:
        self.ir_trades       = ir_trades     or []
        self.fx_trades       = fx_trades     or []
        self.equity_trades   = equity_trades or []
        self.margined        = margined
        self.netting_set_mtm = float(netting_set_mtm)
        self.collateral_net  = float(collateral_net)
        self.vm_received     = float(vm_received)
        self.im_received     = float(im_received)
        self.im_posted       = float(im_posted)
        self.threshold       = float(threshold)
        self.mta             = float(mta)
        self.mpor            = float(mpor)

    def compute_rc(self) -> float:
        """Compute Replacement Cost."""
        if self.margined:
            return replacement_cost_margined(
                netting_set_mtm = self.netting_set_mtm,
                vm_received     = self.vm_received,
                im_received     = self.im_received,
                im_posted       = self.im_posted,
                threshold       = self.threshold,
                mta             = self.mta,
            )
        else:
            return replacement_cost_unmargined(
                netting_set_mtm          = self.netting_set_mtm,
                collateral_posted_by_cpty= self.collateral_net,
            )

    def compute_aggregate_addon(self) -> Dict[str, float]:
        """
        Compute asset-class AddOns and their aggregate.

        Returns
        -------
        dict with keys:
            IR, FX, Equity, aggregate
        """
        # For margined sets, the maturity factor uses the MPoR instead of trade maturity.
        # For unmargined, each trade uses its own remaining maturity.
        # The margined flag is passed through to each asset class function.
        addons = {
            "IR"    : ir_addon(self.ir_trades, margined=self.margined),
            "FX"    : fx_addon(self.fx_trades, margined=self.margined),
            "Equity": equity_addon(self.equity_trades, margined=self.margined),
        }
        addons["aggregate"] = sum(v for k, v in addons.items() if k != "aggregate")
        return addons

    def compute(self) -> dict:
        """
        Compute full SA-CCR EAD and return a summary dictionary.

        Returns
        -------
        dict with keys:
            RC              : float — Replacement Cost
            aggregate_addon : float — Sum of asset-class AddOns
            IR_addon        : float
            FX_addon        : float
            Equity_addon    : float
            multiplier      : float — PFE multiplier ∈ [0.05, 1.0]
            PFE_addon       : float — multiplier * aggregate_addon
            EAD             : float — alpha * (RC + PFE_addon)
            alpha           : float — 1.4 (regulatory constant)
        """
        rc        = self.compute_rc()
        addons    = self.compute_aggregate_addon()
        agg_addon = addons["aggregate"]

        mult      = pfe_multiplier(
            netting_set_mtm = self.netting_set_mtm,
            collateral_net  = self.collateral_net,
            aggregate_addon = agg_addon,
        )

        pfe_addon = mult * agg_addon
        ead       = ALPHA * (rc + pfe_addon)

        return {
            "RC"             : rc,
            "aggregate_addon": agg_addon,
            "IR_addon"       : addons["IR"],
            "FX_addon"       : addons["FX"],
            "Equity_addon"   : addons["Equity"],
            "multiplier"     : mult,
            "PFE_addon"      : pfe_addon,
            "EAD"            : ead,
            "alpha"          : ALPHA,
        }


# =============================================================================
# CAPITAL REQUIREMENT (RWA)
# =============================================================================

def counterparty_rwa(
    ead: float,
    pd: float,
    lgd: float = 0.45,
    maturity: float = 1.0,
    asset_correlation: Optional[float] = None,
) -> dict:
    """
    Compute counterparty credit risk capital requirement and RWA
    using the Basel III IRB formula.

    For corporate/financial institution counterparties:

        R = 0.12 * (1 - exp(-50*PD)) / (1 - exp(-50))
            + 0.24 * (1 - (1-exp(-50*PD))/(1-exp(-50)))

        b = (0.11852 - 0.05478 * ln(PD))²

        K = LGD * [N(sqrt(R/(1-R))*G(PD) + sqrt(1/(1-R))*G(0.999))
                   - PD * LGD] * (1 + (M-2.5)*b)/(1 - 1.5*b)

        RWA = 12.5 * EAD * K

    The maturity adjustment accounts for rollover/refinancing risk.

    Parameters
    ----------
    ead : float
        Exposure at Default from SA-CCR.
    pd : float
        Probability of default (1-year). E.g. 0.01 for 1% PD.
    lgd : float
        Loss Given Default (supervisory LGD). Default 0.45 for senior unsecured.
    maturity : float
        Effective maturity in years. Default 1.0.
    asset_correlation : float, optional
        Override the supervisory asset correlation R. If None, the Basel
        formula for corporate/FI counterparties is used.

    Returns
    -------
    dict with keys:
        R           : float — Asset correlation
        K           : float — Capital requirement (fraction of EAD)
        RWA         : float — Risk-weighted assets
        capital     : float — Minimum capital = 8% * RWA
    """
    pd = max(pd, 1e-8)  # floor to avoid log(0)

    if asset_correlation is None:
        # Basel III corporate/FI correlation formula
        exp_term = np.exp(-50.0 * pd)
        R = 0.12 * (1.0 - exp_term) / (1.0 - np.exp(-50.0)) \
          + 0.24 * (1.0 - (1.0 - exp_term) / (1.0 - np.exp(-50.0)))
    else:
        R = float(asset_correlation)

    # Maturity adjustment
    b = (0.11852 - 0.05478 * np.log(pd)) ** 2
    ma = (1.0 + (maturity - 2.5) * b) / (1.0 - 1.5 * b)

    # Capital formula
    G_pd    = norm.ppf(pd)
    G_999   = norm.ppf(0.999)
    N_inner = norm.cdf(np.sqrt(R / (1.0 - R)) * G_pd + np.sqrt(1.0 / (1.0 - R)) * G_999)

    K    = lgd * (N_inner - pd) * ma
    K    = max(K, 0.0)
    RWA  = 12.5 * ead * K
    cap  = 0.08 * RWA

    return {
        "R"      : R,
        "K"      : K,
        "RWA"    : RWA,
        "capital": cap,
    }