"""
greeks.py
---------
Sensitivities and risk measures for the Quant Desk Toolkit.

Implements bump-and-reprice Greeks for interest rate instruments,
with bucketed decomposition for hedging. All Greeks are computed
via finite difference against the curve objects from curve_factory.py
and instrument objects from instruments.py — no closed-form shortcuts
are taken, so the same framework extends cleanly to exotic payoffs.

Glossary
--------
DV01  (Dollar Value of a Basis Point)
    Change in instrument PV for a 1 bp parallel upward shift in rates.
    Also called PV01. Sign convention: positive DV01 = PV rises when
    rates rise (e.g. payer IRS, floating rate note). Negative DV01 =
    PV falls when rates rise (e.g. fixed-rate bond, receiver IRS).

Bucketed DV01
    DV01 decomposed by tenor bucket. The sum of all buckets equals
    the parallel DV01 (up to interpolation rounding). Used to construct
    a hedging portfolio of benchmark instruments at each tenor.

Modified Duration
    Duration = -DV01_parallel * 10_000 / PV
    Expressed in years. Measures percentage PV change per 100 bp shift.

Convexity
    Second-order rate sensitivity. C = (PV_up + PV_down - 2*PV) / (bump^2 * PV)
    Positive convexity means the instrument benefits from large rate moves
    (in either direction) relative to the linear duration approximation.

IR DV01 on CVA
    CVA's sensitivity to a parallel shift in the OIS discount curve.
    Requires a full Monte Carlo re-run at the bumped curve — this is
    the most expensive Greek in the XVA toolkit. The wrapper here
    accepts a factory callable that rebuilds the simulation from scratch.

Design
------
All bump functions return new Curve objects (immutable — the originals
are never modified). This makes bucketed loops safe and keeps the
call signatures stateless.
"""

import numpy as np
from typing import Callable, Dict, Optional

from curve_factory import Curve
from instruments import InterestRateSwap, Bond


# =============================================================================
# CURVE BUMPING UTILITIES
# =============================================================================

def bump_curve_parallel(curve: Curve, bump_bps: float = 1.0) -> Curve:
    """
    Apply a parallel shift to a Curve object and return a new Curve.

    All zero rates are shifted by bump_bps basis points simultaneously.
    Discount factors are recomputed from the shifted zero rates:

        P_bumped(0, T) = exp(-(z(T) + Δz) * T)

    Parameters
    ----------
    curve : Curve
        Original curve (unmodified).
    bump_bps : float
        Shift in basis points. Default 1.0 (1 bp = 0.0001).

    Returns
    -------
    Curve
        New Curve with parallel-shifted discount factors.

    Notes
    -----
    We shift zero rates (not par rates or forward rates) for consistency
    with the zero-curve representation in curve_factory.Curve. The
    resulting discount factors are exactly consistent with the shifted zeros.
    """
    bump    = bump_bps * 1e-4
    tenors  = curve.tenors
    
    # Derive current zero rates, apply bump, recompute DFs
    # Using the curve's built-in zero_rate method to handle the T=0 limit gracefully
    z_orig   = curve.zero_rate(tenors)
    z_bumped = z_orig + bump
    
    dfs_bumped = np.exp(-z_bumped * tenors)
    
    # Ensure T=0 remains exactly 1.0
    dfs_bumped[0] = 1.0
    
    return Curve(tenors=tenors, discount_factors=dfs_bumped, label=f"{curve.label}_bumped")


def bump_curve_tenor(
    curve: Curve,
    tenor: float,
    bump_bps: float = 1.0,
    width: float = 0.0,
) -> Curve:
    """
    Apply a localised bump to a single tenor point of a Curve.

    The bump is applied as a triangular perturbation centred at `tenor`
    (or as a point bump if width=0, which falls back to bumping the
    nearest grid point only). This produces bucketed DV01 when iterated
    across all tenor points.

    Parameters
    ----------
    curve : Curve
        Original curve.
    tenor : float
        Centre of the bump in years.
    bump_bps : float
        Bump size in basis points.
    width : float
        Half-width of the triangular bump in years. If 0 (default),
        only the nearest tenor grid point is bumped.

    Returns
    -------
    Curve
        New Curve with localised bump applied.
    """
    bump    = bump_bps * 1e-4
    tenors  = curve.tenors
    z_orig  = curve.zero_rate(tenors)

    if width == 0.0:
        # Point bump: find nearest tenor
        idx = int(np.argmin(np.abs(tenors - tenor)))
        z_bumped = z_orig.copy()
        z_bumped[idx] += bump
    else:
        # Triangular bump: linear taper from peak at tenor to zero at ±width
        weights  = np.maximum(1.0 - np.abs(tenors - tenor) / width, 0.0)
        z_bumped = z_orig + bump * weights

    dfs_bumped = np.exp(-z_bumped * tenors)
    dfs_bumped[0] = 1.0
    
    return Curve(tenors=tenors, discount_factors=dfs_bumped, label=f"{curve.label}_{tenor}Y_bumped")


# =============================================================================
# INTEREST RATE SWAP GREEKS
# =============================================================================

def irs_dv01_parallel(
    swap: InterestRateSwap,
    discount_curve: Curve,
    projection_curve: Curve,
    bump_bps: float = 1.0,
) -> float:
    """
    Parallel DV01 of an Interest Rate Swap.

    Bumps both the OIS discount curve and the SOFR projection curve
    simultaneously by bump_bps basis points and computes the PV change:

        DV01 = PV(z + Δz) - PV(z)

    Both curves are bumped because a parallel rate shift affects both
    the discount factors applied to cashflows and the implied forward
    SOFR rates used to compute the floating leg.
    """
    pv_base = swap.pv(discount_curve, projection_curve)["pv_net"]

    dc_up   = bump_curve_parallel(discount_curve, bump_bps)
    pc_up   = bump_curve_parallel(projection_curve, bump_bps)
    pv_up   = swap.pv(dc_up, pc_up)["pv_net"]

    return pv_up - pv_base


def irs_dv01_bucketed(
    swap: InterestRateSwap,
    discount_curve: Curve,
    projection_curve: Curve,
    bump_tenors: Optional[np.ndarray] = None,
    bump_bps: float = 1.0,
) -> Dict[float, float]:
    """
    Bucketed DV01 of an IRS: sensitivity at each tenor independently.

    Each tenor in bump_tenors is bumped by bump_bps in isolation on
    both curves simultaneously. The result is a dictionary mapping
    tenor → DV01, whose values sum to approximately the parallel DV01.
    """
    if bump_tenors is None:
        bump_tenors = discount_curve.tenors

    pv_base = swap.pv(discount_curve, projection_curve)["pv_net"]
    result  = {}

    for t in bump_tenors:
        dc_up = bump_curve_tenor(discount_curve,  t, bump_bps)
        pc_up = bump_curve_tenor(projection_curve, t, bump_bps)
        pv_up = swap.pv(dc_up, pc_up)["pv_net"]
        result[float(t)] = pv_up - pv_base

    return result


def irs_duration(
    swap: InterestRateSwap,
    discount_curve: Curve,
    projection_curve: Curve,
    bump_bps: float = 1.0,
) -> float:
    """
    Modified duration of an IRS (annualised, in years).

        Duration = -DV01_parallel * 10_000 / PV

    Raises ValueError if the PV of the swap is near zero (at-par swap).
    """
    pv = swap.pv(discount_curve, projection_curve)["pv_net"]
    if abs(pv) < 1e-8:
        raise ValueError(
            "Swap PV is near zero — duration is ill-defined for at-par swaps. "
            "Use irs_dv01_parallel() directly."
        )
    dv01 = irs_dv01_parallel(swap, discount_curve, projection_curve, bump_bps)
    return -dv01 * 1e4 / pv


def irs_convexity(
    swap: InterestRateSwap,
    discount_curve: Curve,
    projection_curve: Curve,
    bump_bps: float = 1.0,
) -> float:
    """
    Convexity of an IRS via central finite difference.

        Convexity = (PV_up + PV_down - 2*PV) / (bump^2 * PV)
    """
    pv      = swap.pv(discount_curve, projection_curve)["pv_net"]
    if abs(pv) < 1e-8:
        raise ValueError("Swap PV is near zero — convexity is ill-defined.")

    dc_up   = bump_curve_parallel(discount_curve,  +bump_bps)
    pc_up   = bump_curve_parallel(projection_curve, +bump_bps)
    dc_dn   = bump_curve_parallel(discount_curve,  -bump_bps)
    pc_dn   = bump_curve_parallel(projection_curve, -bump_bps)

    pv_up   = swap.pv(dc_up, pc_up)["pv_net"]
    pv_dn   = swap.pv(dc_dn, pc_dn)["pv_net"]

    bump    = bump_bps * 1e-4
    return (pv_up + pv_dn - 2.0 * pv) / (bump ** 2 * pv)


# =============================================================================
# BOND GREEKS
# =============================================================================

def bond_dv01_parallel(
    bond: Bond,
    discount_curve: Curve,
    bump_bps: float = 1.0,
) -> float:
    """
    Parallel DV01 of a fixed-rate bond.
    Only the OIS discount curve is bumped.
    """
    pv_base = bond.dirty_price(discount_curve)
    dc_up   = bump_curve_parallel(discount_curve, bump_bps)
    pv_up   = bond.dirty_price(dc_up)
    return pv_up - pv_base


def bond_dv01_bucketed(
    bond: Bond,
    discount_curve: Curve,
    bump_tenors: Optional[np.ndarray] = None,
    bump_bps: float = 1.0,
) -> Dict[float, float]:
    """Bucketed DV01 of a fixed-rate bond."""
    if bump_tenors is None:
        bump_tenors = discount_curve.tenors

    pv_base = bond.dirty_price(discount_curve)
    result  = {}

    for t in bump_tenors:
        dc_up         = bump_curve_tenor(discount_curve, t, bump_bps)
        result[float(t)] = bond.dirty_price(dc_up) - pv_base

    return result


def bond_modified_duration(
    bond: Bond,
    discount_curve: Curve,
    bump_bps: float = 1.0,
) -> float:
    """Modified duration of a bond in years."""
    dirty = bond.dirty_price(discount_curve)
    dv01  = bond_dv01_parallel(bond, discount_curve, bump_bps)
    return -dv01 * 1e4 / dirty


def bond_convexity(
    bond: Bond,
    discount_curve: Curve,
    bump_bps: float = 1.0,
) -> float:
    """Convexity of a fixed-rate bond via central finite difference."""
    pv    = bond.dirty_price(discount_curve)
    dc_up = bump_curve_parallel(discount_curve, +bump_bps)
    dc_dn = bump_curve_parallel(discount_curve, -bump_bps)
    pv_up = bond.dirty_price(dc_up)
    pv_dn = bond.dirty_price(dc_dn)

    bump  = bump_bps * 1e-4
    return (pv_up + pv_dn - 2.0 * pv) / (bump ** 2 * pv)


# =============================================================================
# GREEK SUMMARY TABLE
# =============================================================================

def irs_greek_summary(
    swap: InterestRateSwap,
    discount_curve: Curve,
    projection_curve: Curve,
    bump_tenors: Optional[np.ndarray] = None,
    bump_bps: float = 1.0,
) -> dict:
    """Compute a full Greek summary for an IRS in one call."""
    pv       = swap.pv(discount_curve, projection_curve)["pv_net"]
    par      = swap.par_rate(discount_curve, projection_curve)
    dv01_par = irs_dv01_parallel(swap, discount_curve, projection_curve, bump_bps)
    dv01_bkt = irs_dv01_bucketed(swap, discount_curve, projection_curve, bump_tenors, bump_bps)

    try:
        dur = irs_duration(swap, discount_curve, projection_curve, bump_bps)
    except ValueError:
        dur = None

    try:
        cvx = irs_convexity(swap, discount_curve, projection_curve, bump_bps)
    except ValueError:
        cvx = None

    return {
        "PV"            : pv,
        "par_rate"      : par,
        "DV01_parallel" : dv01_par,
        "DV01_bucketed" : dv01_bkt,
        "duration"      : dur,
        "convexity"     : cvx,
    }


def bond_greek_summary(
    bond: Bond,
    discount_curve: Curve,
    bump_tenors: Optional[np.ndarray] = None,
    bump_bps: float = 1.0,
) -> dict:
    """Compute a full Greek summary for a Bond in one call."""
    dirty  = bond.dirty_price(discount_curve)
    clean  = bond.clean_price(discount_curve)
    ai     = bond.accrued_interest(0.0) # Assuming 0.0 time since last coupon for summary
    ytm    = bond.yield_to_maturity(discount_curve)
    dv01   = bond_dv01_parallel(bond, discount_curve, bump_bps)
    dv01_b = bond_dv01_bucketed(bond, discount_curve, bump_tenors, bump_bps)
    dur    = bond_modified_duration(bond, discount_curve, bump_bps)
    cvx    = bond_convexity(bond, discount_curve, bump_bps)

    return {
        "dirty_price"       : dirty,
        "clean_price"       : clean,
        "accrued_interest"  : ai,
        "ytm"               : ytm,
        "DV01_parallel"     : dv01,
        "DV01_bucketed"     : dv01_b,
        "modified_duration" : dur,
        "convexity"         : cvx,
    }


# =============================================================================
# CVA IR DV01 (BUMP-AND-REPRICE, FULL MC)
# =============================================================================

def cva_ir_dv01(
    mc_factory: Callable[[Curve, Curve], dict],
    xva_factory: Callable[[dict], float],
    discount_curve: Curve,
    projection_curve: Curve,
    bump_bps: float = 1.0,
) -> float:
    """
    IR DV01 of CVA via bump-and-reprice with a full Monte Carlo re-run.

    Parameters
    ----------
    mc_factory : Callable[[Curve, Curve], dict]
        A callable f(discount_curve, projection_curve) → mc_results dict.
    xva_factory : Callable[[dict], float]
        A callable f(mc_results) → CVA (float).
    discount_curve : Curve
    projection_curve : Curve
    bump_bps : float

    Returns
    -------
    float
        CVA IR DV01.
    """
    # Base CVA
    mc_base  = mc_factory(discount_curve, projection_curve)
    cva_base = xva_factory(mc_base)

    # Bumped CVA — full MC re-run at shifted curves
    dc_up    = bump_curve_parallel(discount_curve, bump_bps)
    pc_up    = bump_curve_parallel(projection_curve, bump_bps)
    mc_up    = mc_factory(dc_up, pc_up)
    cva_up   = xva_factory(mc_up)

    return cva_up - cva_base


def cva_ir_dv01_bucketed(
    mc_factory: Callable[[Curve, Curve], dict],
    xva_factory: Callable[[dict], float],
    discount_curve: Curve,
    projection_curve: Curve,
    bump_tenors: Optional[np.ndarray] = None,
    bump_bps: float = 1.0,
) -> Dict[float, float]:
    """
    Tenor-bucketed CVA IR DV01 via full MC re-run at each tenor bump.

    Iterates over each tenor in bump_tenors, applies a localised bump
    to both curves, re-runs MC, and computes the CVA change.
    """
    if bump_tenors is None:
        bump_tenors = discount_curve.tenors

    mc_base  = mc_factory(discount_curve, projection_curve)
    cva_base = xva_factory(mc_base)
    result   = {}

    for t in bump_tenors:
        dc_up         = bump_curve_tenor(discount_curve,   t, bump_bps)
        pc_up         = bump_curve_tenor(projection_curve, t, bump_bps)
        mc_up         = mc_factory(dc_up, pc_up)
        cva_up        = xva_factory(mc_up)
        result[float(t)] = cva_up - cva_base

    return result
