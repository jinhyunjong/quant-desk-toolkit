
"""
xva.py
------
XVA pricing engine for the Quant Desk Toolkit.

Computes the three primary valuation adjustments on a netting set:

    CVA  — Credit Valuation Adjustment
    DVA  — Debt Valuation Adjustment
    FVA  — Funding Valuation Adjustment
    MVA  — Margin Valuation Adjustment

Theory
------

CVA (unilateral, continuous-time):
    CVA = (1 - R_c) * integral_0^T EE(t) * lambda_c(t) * P_surv_c(t) dt

    where:
        EE(t)        = E[max(V_net(t), 0)] discounted to t=0
        lambda_c(t)  = counterparty hazard rate (instantaneous default intensity)
        P_surv_c(t)  = exp(-integral_0^t lambda_c(s) ds)  — counterparty survival prob
        R_c          = counterparty recovery rate
        (1 - R_c)    = Loss Given Default (LGD)

DVA (unilateral):
    DVA = (1 - R_b) * integral_0^T ENE(t) * lambda_b(t) * P_surv_b(t) dt
    where ENE(t) = E[min(V_net(t), 0)] (negative expected exposure, our liability).
    DVA is always non-positive (a benefit to us).

Bilateral CVA (BCVA):
    BCVA = CVA + DVA       (DVA offsets CVA when we also bear default risk)

FVA (symmetric funding adjustment):
    FVA = integral_0^T FCA(t) dt + integral_0^T FBA(t) dt

    FCA (Funding Cost Adjustment — cost of funding uncollateralised asset):
        FCA = -s_f * integral_0^T EE(t) dt

    FBA (Funding Benefit Adjustment — benefit from investing collateral received):
        FBA = +s_f * integral_0^T ENE(t) dt    [sign: ENE < 0, so FBA < 0 in cost terms]

    where s_f = funding spread = unsecured borrowing rate - OIS rate.
    The FVA formula here follows Hull & White (2012) symmetric convention.
    Industry practice varies: some desks compute FCA only (conservative).

MVA (Margin Valuation Adjustment):
    MVA = s_f * integral_0^T IM(t) * P_surv(t) dt

    where:
        IM(t)        = projected Initial Margin at future time t
        P_surv(t)    = our own survival probability (we stop paying IM if we default)
        s_f          = funding spread (unsecured borrowing rate - OIS)

    Regulations (UMR / BCBS-IOSCO) require banks to post ISDA SIMM IM into
    segregated accounts. This cash earns OIS but is funded at the bank's
    unsecured borrowing rate. The gap is the funding spread s_f, and
    MVA is the present cost of that spread over the projected IM profile.

    MVA is always positive (a cost). It is the primary reason XVA desks care
    about SIMM IM — a large rate derivatives portfolio can carry hundreds of
    millions in IM, generating significant ongoing MVA.

Discretisation
--------------
All integrals are evaluated numerically on the simulation time grid using
Simpson's rule (falls back to trapezoidal for even-length grids). The hazard
rate λ(t) is assumed piecewise constant between tenor points and interpolated
linearly on the simulation grid.

The survival probability is computed from the cumulative default probability:
    P_surv(t) = exp(-integral_0^t lambda(s) ds)
evaluated by trapezoidal integration of λ on the time grid.

References
----------
Gregory, J. (2015). The xVA Challenge. 3rd ed. Wiley.
Hull, J. & White, A. (2012). "The FVA Debate." Risk, July 2012.
BCBS (2015). "Review of the Credit Valuation Adjustment Risk Framework."
"""

import numpy as np
from typing import Optional, Tuple
from common_utils.math_helpers import trapezoidal_integration, simpsons_integration, linear_interp


# =============================================================================
# HAZARD RATE AND SURVIVAL PROBABILITY
# =============================================================================

def build_hazard_rate_curve(
    tenors: np.ndarray,
    hazard_rates: np.ndarray,
    time_grid: np.ndarray,
) -> np.ndarray:
    """
    Interpolate a piecewise-constant hazard rate onto the simulation time grid.

    Parameters
    ----------
    tenors : np.ndarray
        Tenor points in years at which hazard rates are specified.
        Must include 0.0 as the first element (lambda(0) = lambda(tenors[1])).
    hazard_rates : np.ndarray
        Instantaneous hazard rates lambda(t) at each tenor point.
        Units: per year (e.g. 0.01 = 1% p.a. default intensity).
    time_grid : np.ndarray
        Simulation time grid onto which to interpolate.

    Returns
    -------
    np.ndarray, shape (len(time_grid),)
        Hazard rate lambda(t) evaluated at each simulation time step.

    Notes
    -----
    Linear interpolation is used between tenors. For most credit curves,
    the hazard rate is slowly varying and linear interpolation is adequate.
    For bootstrapped CDS curves a piecewise-constant (flat-forward) scheme
    is more common — this can be achieved by repeating each hazard_rate
    value at the next tenor boundary.
    """
    return np.array([float(linear_interp(t, tenors, hazard_rates)) for t in time_grid])


def survival_probability(
    lambda_grid: np.ndarray,
    time_grid: np.ndarray,
) -> np.ndarray:
    """
    Compute the survival probability curve from a hazard rate grid.

        P_surv(t) = exp(-integral_0^t lambda(s) ds)

    The integral is evaluated cumulatively by trapezoidal rule.

    Parameters
    ----------
    lambda_grid : np.ndarray, shape (n_steps+1,)
        Hazard rate values at each time grid point.
    time_grid : np.ndarray, shape (n_steps+1,)

    Returns
    -------
    np.ndarray, shape (n_steps+1,)
        Survival probability P_surv(t) at each time grid point.
        P_surv(0) = 1.0 by construction.
    """
    n = len(time_grid)
    cum_integral = np.zeros(n)
    for i in range(1, n):
        # Trapezoidal increment: integral from t_{i-1} to t_i
        dt       = time_grid[i] - time_grid[i - 1]
        cum_integral[i] = cum_integral[i - 1] + 0.5 * (lambda_grid[i - 1] + lambda_grid[i]) * dt
    return np.exp(-cum_integral)


def default_probability_density(
    lambda_grid: np.ndarray,
    surv_prob: np.ndarray,
) -> np.ndarray:
    """
    Compute the risk-neutral default probability density (the integrand weight
    for CVA/DVA integrals).

        q(t) = lambda(t) * P_surv(t)

    This is the probability that default occurs in [t, t+dt]:
        P(tau in [t, t+dt]) ≈ lambda(t) * P_surv(t) * dt

    Parameters
    ----------
    lambda_grid : np.ndarray, shape (n_steps+1,)
    surv_prob   : np.ndarray, shape (n_steps+1,)

    Returns
    -------
    np.ndarray, shape (n_steps+1,)
        Default probability density q(t) = lambda(t) * P_surv(t).
    """
    return lambda_grid * surv_prob


# =============================================================================
# CVA
# =============================================================================

class CVAEngine:
    """
    Credit Valuation Adjustment (CVA) calculator.

    Computes unilateral CVA using the standard integral formula:

        CVA = LGD_c * integral_0^T EE(t) * q_c(t) dt

    where q_c(t) = lambda_c(t) * P_surv_c(t) is the counterparty's
    default probability density.

    Parameters
    ----------
    ee_profile : np.ndarray, shape (n_steps+1,)
        Expected Exposure profile (discounted to t=0). Output of
        ExposureEngine.expected_exposure() with discount_to_t0=True.
    time_grid : np.ndarray, shape (n_steps+1,)
        Simulation time grid in years.
    counterparty_tenors : np.ndarray
        Tenor points for the counterparty credit curve.
    counterparty_hazard_rates : np.ndarray
        Hazard rates at each tenor point (per year).
    recovery_rate : float
        Counterparty recovery rate. Default 0.40 (Basel convention).

    Notes
    -----
    EE must already be discounted to t=0 before being passed in.
    The CVA formula integrates the discounted EE against the default
    probability density — double-discounting would be an error.
    """

    def __init__(
        self,
        ee_profile: np.ndarray,
        time_grid: np.ndarray,
        counterparty_tenors: np.ndarray,
        counterparty_hazard_rates: np.ndarray,
        recovery_rate: float = 0.40,
    ) -> None:
        self.ee_profile   = np.asarray(ee_profile, dtype=float)
        self.time_grid    = np.asarray(time_grid, dtype=float)
        self.recovery_rate = float(recovery_rate)

        # Build credit curves on simulation grid
        self.lambda_c = build_hazard_rate_curve(
            counterparty_tenors, counterparty_hazard_rates, time_grid
        )
        self.surv_c   = survival_probability(self.lambda_c, time_grid)
        self.q_c      = default_probability_density(self.lambda_c, self.surv_c)

    @property
    def lgd(self) -> float:
        """Loss Given Default = 1 - Recovery Rate."""
        return 1.0 - self.recovery_rate

    def compute(self) -> float:
        """
        Compute CVA via numerical integration.

        Returns
        -------
        float
            CVA (positive number — a cost, reduces the trade value).

        Formula
        -------
            CVA = LGD * integral_0^T EE(t) * q_c(t) dt
        """
        integrand = self.ee_profile * self.q_c
        return float(self.lgd * simpsons_integration(self.time_grid, integrand))

    def term_structure(self) -> np.ndarray:
        """
        CVA term structure: cumulative CVA from 0 to each time step.

        Useful for understanding how CVA accretes over time and for
        computing time-bucketed CVA sensitivities (CS01).

        Returns
        -------
        np.ndarray, shape (n_steps+1,)
            Cumulative CVA up to each time grid point.
        """
        integrand    = self.ee_profile * self.q_c
        n            = len(self.time_grid)
        cumulative   = np.zeros(n)
        for i in range(1, n):
            dt            = self.time_grid[i] - self.time_grid[i - 1]
            cumulative[i] = cumulative[i - 1] + 0.5 * (integrand[i - 1] + integrand[i]) * dt
        return self.lgd * cumulative

    def cs01(self) -> float:
        """
        Credit Spread 01 (CS01): sensitivity of CVA to a 1 bp parallel
        shift in the CDS par spread curve.

        Traders quote credit sensitivity relative to the CDS par spread s,
        not the hazard rate λ. Via the credit triangle approximation:

            s ≈ λ × LGD  →  λ = s / LGD  →  Δλ = Δs / LGD = Δs / (1 - R)

        A 1 bp shift in spread therefore maps to a 1/(1-R) bp shift in
        the hazard rate. With R = 0.40, Δλ ≈ 1.667 bp per 1 bp in spread.

        Bumping λ directly by 1 bp gives HR01 (hazard rate sensitivity),
        not CS01. Use hr01() if you want the hazard rate sensitivity.

        Approximated by one-sided finite difference:
            CS01 ≈ CVA(λ + Δs/(1-R)) - CVA(λ)

        Returns
        -------
        float
            CVA change per 1 bp parallel shift in the CDS par spread curve.
        """
        bump_spread   = 1e-4                                     # 1 bp in CDS spread
        bump_lambda   = bump_spread / (1.0 - self.recovery_rate) # Δλ = Δs / LGD
        lambda_bumped = self.lambda_c + bump_lambda
        surv_bumped   = survival_probability(lambda_bumped, self.time_grid)
        q_bumped      = default_probability_density(lambda_bumped, surv_bumped)
        integrand_up  = self.ee_profile * q_bumped
        cva_up        = float(self.lgd * simpsons_integration(self.time_grid, integrand_up))
        return cva_up - self.compute()

    def hr01(self) -> float:
        """
        Hazard Rate 01 (HR01): sensitivity of CVA to a 1 bp parallel
        shift in the hazard rate curve λ(t).

        This is the raw model sensitivity used by quants and model risk.
        Traders typically report CS01 instead — see cs01().

        The relationship between the two:
            CS01 = HR01 × (1 - R)     [credit triangle approximation]

        Returns
        -------
        float
            CVA change per 1 bp parallel shift in the hazard rate curve.
        """
        bump          = 1e-4  # 1 bp in hazard rate
        lambda_bumped = self.lambda_c + bump
        surv_bumped   = survival_probability(lambda_bumped, self.time_grid)
        q_bumped      = default_probability_density(lambda_bumped, surv_bumped)
        integrand_up  = self.ee_profile * q_bumped
        cva_up        = float(self.lgd * simpsons_integration(self.time_grid, integrand_up))
        return cva_up - self.compute()


# =============================================================================
# DVA
# =============================================================================

class DVAEngine:
    """
    Debt Valuation Adjustment (DVA) calculator.

    DVA represents the value of our own default to us — when we default,
    we avoid paying our liabilities to the counterparty. It offsets CVA
    in the bilateral close-out convention (BCVA = CVA + DVA, DVA < 0).

        DVA = LGD_b * integral_0^T ENE(t) * q_b(t) dt

    Since ENE(t) ≤ 0, DVA ≤ 0 (a benefit).

    Parameters
    ----------
    ene_profile : np.ndarray, shape (n_steps+1,)
        Expected Negative Exposure profile (discounted to t=0). Output of
        ExposureEngine.negative_exposure() with discount_to_t0=True.
        Values should be ≤ 0.
    time_grid : np.ndarray, shape (n_steps+1,)
    own_tenors : np.ndarray
        Tenor points for our own credit curve (bank's own CDS curve).
    own_hazard_rates : np.ndarray
        Our own hazard rates at each tenor.
    recovery_rate : float
        Our own recovery rate. Default 0.40.
    """

    def __init__(
        self,
        ene_profile: np.ndarray,
        time_grid: np.ndarray,
        own_tenors: np.ndarray,
        own_hazard_rates: np.ndarray,
        recovery_rate: float = 0.40,
    ) -> None:
        self.ene_profile  = np.asarray(ene_profile, dtype=float)
        self.time_grid    = np.asarray(time_grid, dtype=float)
        self.recovery_rate = float(recovery_rate)

        self.lambda_b = build_hazard_rate_curve(own_tenors, own_hazard_rates, time_grid)
        self.surv_b   = survival_probability(self.lambda_b, time_grid)
        self.q_b      = default_probability_density(self.lambda_b, self.surv_b)

    @property
    def lgd(self) -> float:
        return 1.0 - self.recovery_rate

    def compute(self) -> float:
        """
        Compute DVA.

        Returns
        -------
        float
            DVA (non-positive — a benefit that reduces the net XVA charge).

        Formula
        -------
            DVA = LGD_b * integral_0^T ENE(t) * q_b(t) dt    (ENE ≤ 0 → DVA ≤ 0)
        """
        integrand = self.ene_profile * self.q_b
        return float(self.lgd * simpsons_integration(self.time_grid, integrand))


# =============================================================================
# FVA
# =============================================================================

class FVAEngine:
    """
    Funding Valuation Adjustment (FVA) calculator.

    FVA captures the cost (or benefit) of funding uncollateralised derivative
    positions. Under the Hull-White (2012) symmetric framework:

        FCA = -s_f * integral_0^T EE(t) dt      (funding cost: positive exposure
                                                 requires borrowed funding)
        FBA = +s_f * integral_0^T ENE(t) dt     (funding benefit: when we owe,
                                                 counterparty funds us; ENE ≤ 0,
                                                 so FBA ≤ 0 — reduces FCA)
        FVA = FCA + FBA

    where s_f = funding_spread = cost of unsecured funding above OIS.

    Note on sign convention
    -----------------------
    FCA ≥ 0 (a cost to us). FBA ≤ 0 (a benefit).
    FVA = FCA + FBA may be positive (net cost) or negative (net benefit).

    In practice, many institutions compute FCA only (conservative / asymmetric).
    Toggle use_symmetric=False to exclude FBA.

    Parameters
    ----------
    ee_profile : np.ndarray, shape (n_steps+1,)
        Expected Exposure profile discounted to t=0.
    ene_profile : np.ndarray, shape (n_steps+1,)
        Expected Negative Exposure profile discounted to t=0.
    time_grid : np.ndarray, shape (n_steps+1,)
    funding_spread : float
        Funding spread s_f in decimal (e.g. 0.0050 for 50 bps).
    use_symmetric : bool
        If True (default), include FBA. If False, FVA = FCA only.
    """

    def __init__(
        self,
        ee_profile: np.ndarray,
        ene_profile: np.ndarray,
        time_grid: np.ndarray,
        funding_spread: float,
        use_symmetric: bool = True,
    ) -> None:
        self.ee_profile    = np.asarray(ee_profile, dtype=float)
        self.ene_profile   = np.asarray(ene_profile, dtype=float)
        self.time_grid     = np.asarray(time_grid, dtype=float)
        self.funding_spread = float(funding_spread)
        self.use_symmetric = use_symmetric

    def fca(self) -> float:
        """
        Funding Cost Adjustment.

            FCA = s_f * integral_0^T EE(t) dt

        Returns
        -------
        float
            FCA (non-negative cost).
        """
        return float(
            self.funding_spread
            * simpsons_integration(self.time_grid, self.ee_profile)
        )

    def fba(self) -> float:
        """
        Funding Benefit Adjustment.

            FBA = s_f * integral_0^T ENE(t) dt    (ENE ≤ 0 → FBA ≤ 0)

        Returns
        -------
        float
            FBA (non-positive benefit).
        """
        return float(
            self.funding_spread
            * simpsons_integration(self.time_grid, self.ene_profile)
        )

    def compute(self) -> float:
        """
        Compute total FVA = FCA + FBA (or FCA only if use_symmetric=False).

        Returns
        -------
        float
            FVA (positive = net funding cost, negative = net funding benefit).
        """
        fca = self.fca()
        if self.use_symmetric:
            return fca + self.fba()
        return fca


# =============================================================================
# MVA
# =============================================================================

class MVAEngine:
    """
    Margin Valuation Adjustment (MVA) calculator.

    MVA is the cost of funding Initial Margin (IM) over the life of
    the portfolio. Under UMR (Uncleared Margin Rules / BCBS-IOSCO),
    banks must post ISDA SIMM IM into segregated third-party accounts.
    The cash earns OIS but is funded at the bank's unsecured borrowing
    rate — the gap is the funding spread s_f.

        MVA = s_f * integral_0^T IM(t) * P_surv_b(t) dt

    The own survival probability P_surv_b(t) appears because if the bank
    defaults, it stops posting IM. Omitting it (setting P_surv_b = 1)
    gives a slightly conservative upper bound and is common for internal
    pricing.

    MVA is always positive (a cost to us). For a large rates book posting
    $500mm SIMM IM at a 50bp funding spread, MVA ≈ $2.5mm per year —
    material enough that desks explicitly charge it to trades at inception.

    Parameters
    ----------
    im_profile : np.ndarray, shape (n_steps+1,)
        Projected IM at each simulation time step. Use
        MarginEngine.im_profile() from margin.py to generate this.
    time_grid : np.ndarray, shape (n_steps+1,)
    funding_spread : float
        s_f in decimal (e.g. 0.005 for 50 bps).
    own_tenors : np.ndarray, optional
        Tenor points for the bank's own survival probability curve.
        If None, P_surv_b(t) = 1 (no own-default adjustment).
    own_hazard_rates : np.ndarray, optional
        Bank's own hazard rates at each tenor.
    """

    def __init__(
        self,
        im_profile      : np.ndarray,
        time_grid       : np.ndarray,
        funding_spread  : float,
        own_tenors      : Optional[np.ndarray] = None,
        own_hazard_rates: Optional[np.ndarray] = None,
    ) -> None:
        self.im_profile     = np.asarray(im_profile,  dtype=float)
        self.time_grid      = np.asarray(time_grid,   dtype=float)
        self.funding_spread = float(funding_spread)

        if own_tenors is not None and own_hazard_rates is not None:
            lambda_b      = build_hazard_rate_curve(own_tenors, own_hazard_rates, time_grid)
            self.surv_b   = survival_probability(lambda_b, time_grid)
        else:
            # No own-default adjustment: P_surv_b(t) = 1 everywhere
            self.surv_b   = np.ones_like(time_grid)

    def compute(self) -> float:
        """
        Compute MVA.

        Returns
        -------
        float
            MVA (positive — a cost that reduces trade value at inception).

        Formula
        -------
            MVA = s_f * integral_0^T IM(t) * P_surv_b(t) dt
        """
        integrand = self.im_profile * self.surv_b
        return float(self.funding_spread * simpsons_integration(self.time_grid, integrand))

    def term_structure(self) -> np.ndarray:
        """
        Cumulative MVA from 0 to each time step.

        Useful for understanding how MVA accretes and for computing
        the incremental MVA charge of a new trade (marginal MVA).

        Returns
        -------
        np.ndarray, shape (n_steps+1,)
        """
        integrand  = self.im_profile * self.surv_b
        n          = len(self.time_grid)
        cumulative = np.zeros(n)
        for i in range(1, n):
            dt             = self.time_grid[i] - self.time_grid[i - 1]
            cumulative[i]  = cumulative[i - 1] + 0.5 * (integrand[i - 1] + integrand[i]) * dt
        return self.funding_spread * cumulative


# =============================================================================
# XVA AGGREGATOR
# =============================================================================

class XVAEngine:
    """
    Unified XVA engine: computes CVA, DVA, FVA, and MVA for a netting set
    and returns an XVA summary.

    This is the top-level interface consumed by notebooks and counterparty
    risk reports. It wraps CVAEngine, DVAEngine, FVAEngine, and MVAEngine and 
    provides a single compute() call returning all adjustments and their sum.

    Parameters
    ----------
    ee_profile : np.ndarray
        Expected Exposure (discounted to t=0). From ExposureEngine.
    ene_profile : np.ndarray
        Expected Negative Exposure (discounted to t=0). From ExposureEngine.
    time_grid : np.ndarray
    counterparty_tenors : np.ndarray
    counterparty_hazard_rates : np.ndarray
    counterparty_recovery : float
        Counterparty LGD = 1 - recovery. Default 0.40.
    own_tenors : np.ndarray, optional
        Required for DVA. If None, DVA is set to 0.
    own_hazard_rates : np.ndarray, optional
        Required for DVA.
    own_recovery : float
        Our own recovery rate. Default 0.40.
    funding_spread : float
        Funding spread for FVA and MVA. Set to 0 to exclude both.
    use_symmetric_fva : bool
        Whether to include FBA in FVA. Default True.
    im_profile : np.ndarray, optional
        Projected IM through time for MVA. Use MarginEngine.im_profile()
        from margin.py. If None, MVA = 0.
    """

    def __init__(
        self,
        ee_profile: np.ndarray,
        ene_profile: np.ndarray,
        time_grid: np.ndarray,
        counterparty_tenors: np.ndarray,
        counterparty_hazard_rates: np.ndarray,
        counterparty_recovery: float = 0.40,
        own_tenors: Optional[np.ndarray] = None,
        own_hazard_rates: Optional[np.ndarray] = None,
        own_recovery: float = 0.40,
        funding_spread: float = 0.0,
        use_symmetric_fva: bool = True,
        im_profile: Optional[np.ndarray] = None,
    ) -> None:
        self.ee_profile  = np.asarray(ee_profile, dtype=float)
        self.ene_profile = np.asarray(ene_profile, dtype=float)
        self.time_grid   = np.asarray(time_grid, dtype=float)

        # CVA engine (always constructed)
        self.cva_engine = CVAEngine(
            ee_profile               = self.ee_profile,
            time_grid                = self.time_grid,
            counterparty_tenors      = counterparty_tenors,
            counterparty_hazard_rates= counterparty_hazard_rates,
            recovery_rate            = counterparty_recovery,
        )

        # DVA engine (optional)
        if own_tenors is not None and own_hazard_rates is not None:
            self.dva_engine = DVAEngine(
                ene_profile      = self.ene_profile,
                time_grid        = self.time_grid,
                own_tenors       = own_tenors,
                own_hazard_rates = own_hazard_rates,
                recovery_rate    = own_recovery,
            )
        else:
            self.dva_engine = None

        # FVA engine (optional — only if funding_spread > 0)
        if funding_spread > 0.0:
            self.fva_engine = FVAEngine(
                ee_profile     = self.ee_profile,
                ene_profile    = self.ene_profile,
                time_grid      = self.time_grid,
                funding_spread = funding_spread,
                use_symmetric  = use_symmetric_fva,
            )
        else:
            self.fva_engine = None

        # MVA engine (optional — requires im_profile and funding_spread > 0)
        if im_profile is not None and funding_spread > 0.0:
            self.mva_engine = MVAEngine(
                im_profile       = im_profile,
                time_grid        = self.time_grid,
                funding_spread   = funding_spread,
                own_tenors       = own_tenors,
                own_hazard_rates = own_hazard_rates,
            )
        else:
            self.mva_engine = None

    def compute(self) -> dict:
        """
        Compute all XVA components and return a summary dictionary.

        Returns
        -------
        dict with keys:
            CVA         : float  — Credit valuation adjustment (cost, > 0)
            DVA         : float  — Debt valuation adjustment (benefit, ≤ 0); 0 if not computed
            FVA         : float  — Funding valuation adjustment; 0 if funding_spread=0
            FCA         : float  — Funding cost component of FVA
            FBA         : float  — Funding benefit component of FVA; 0 if asymmetric
            MVA         : float  — Margin valuation adjustment; 0 if im_profile not provided
            BCVA        : float  — Bilateral CVA = CVA + DVA
            total_XVA   : float  — CVA + DVA + FVA + MVA (net XVA charge to trade value)
            CS01        : float  — CVA sensitivity to 1 bp CDS par spread shift
            HR01        : float  — CVA sensitivity to 1 bp hazard rate shift
            time_grid   : np.ndarray
            CVA_term_structure : np.ndarray — Cumulative CVA profile
            MVA_term_structure : np.ndarray — Cumulative MVA profile; zeros if MVA=0
        """
        cva    = self.cva_engine.compute()
        dva    = self.dva_engine.compute() if self.dva_engine is not None else 0.0
        fca    = self.fva_engine.fca()     if self.fva_engine is not None else 0.0
        fba    = self.fva_engine.fba()     if self.fva_engine is not None else 0.0
        fva    = self.fva_engine.compute() if self.fva_engine is not None else 0.0
        mva    = self.mva_engine.compute() if self.mva_engine is not None else 0.0
        cs01   = self.cva_engine.cs01()
        hr01   = self.cva_engine.hr01()
        cva_ts = self.cva_engine.term_structure()
        mva_ts = (self.mva_engine.term_structure()
                  if self.mva_engine is not None
                  else np.zeros_like(self.time_grid))

        return {
            "CVA"               : cva,
            "DVA"               : dva,
            "FVA"               : fva,
            "FCA"               : fca,
            "FBA"               : fba,
            "MVA"               : mva,
            "BCVA"              : cva + dva,
            "total_XVA"         : cva + dva + fva + mva,
            "CS01"              : cs01,
            "HR01"              : hr01,
            "time_grid"         : self.time_grid,
            "CVA_term_structure": cva_ts,
            "MVA_term_structure": mva_ts,
        }

    def sensitivity_table(self) -> dict:
        """
        Compute a standard sensitivity table for risk reporting.

        Returns
        -------
        dict with keys:
            CS01    : float — CVA sensitivity to counterparty spread (+1 bp)
            HR01    : float — CVA sensitivity to counterparty hazard rate (+1 bp)
            CS01_HR01_ratio : float — Ratio mapping (sanity check)
            IR_DV01 : str   — Placeholder note (IR DV01 requires bumping EE,
                              which requires a full MC re-run; see demo notebook)
        """
        return {
            "CS01"           : self.cva_engine.cs01(),
            "HR01"           : self.cva_engine.hr01(),
            "CS01_HR01_ratio": 1.0 - self.cva_engine.recovery_rate,  # = LGD; sanity check
            "IR_DV01"        : "Requires full MC re-run with bumped rate curve — see demo notebook.",
        }