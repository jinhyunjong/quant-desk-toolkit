"""
test_xva.py
-----------
Unit tests for xva.py: CVA, DVA, FVA, MVA engines and XVA aggregator.

Test strategy
-------------
Where analytical solutions exist, we derive them from first principles
and compare against the numerical result. This validates both the
integration machinery and the formula implementation simultaneously.

Analytical benchmarks used
---------------------------

1. Constant-EE, constant-hazard-rate CVA:
   EE(t) = E (constant), lambda(t) = λ (constant), recovery R.

       P_surv(t) = exp(-λt)
       q(t) = λ * exp(-λt)
       CVA = (1-R) * E * λ * ∫_0^T exp(-λt) dt
           = (1-R) * E * [1 - exp(-λT)]

2. CS01 / HR01 credit triangle relationship:
       CS01 ≈ HR01 × LGD = HR01 × (1 - R)

   Holds exactly in the limit of infinitesimally small bumps; passes
   to a relative tolerance of 0.1% for the 1 bp bump used here.

3. Constant-IM, unit-survival MVA:
       MVA = s_f * IM * T    (when P_surv = 1 everywhere)

4. FVA sign conventions:
       FCA ≥ 0 (EE ≥ 0, spread > 0)
       FBA ≤ 0 (ENE ≤ 0, spread > 0)
       Symmetric FVA = FCA + FBA

5. Survival probability boundary conditions:
       P_surv(0) = 1
       dP_surv/dt = -λ(t) * P_surv(t) < 0  (strictly decreasing for λ > 0)

Run with:
    python -m pytest test_xva.py -v
    python test_xva.py          (unittest runner)
"""

import unittest
import numpy as np
import numpy.testing as npt

# Module under test
from xva import (
    build_hazard_rate_curve,
    survival_probability,
    default_probability_density,
    CVAEngine,
    DVAEngine,
    FVAEngine,
    MVAEngine,
    XVAEngine,
)


# =============================================================================
# SHARED FIXTURES
# =============================================================================

def make_time_grid(T: float = 5.0, n: int = 100) -> np.ndarray:
    """Uniform time grid from 0 to T with n+1 points."""
    return np.linspace(0.0, T, n + 1)


def constant_ee(time_grid: np.ndarray, level: float = 1_000_000.0) -> np.ndarray:
    """Flat EE profile — simplest analytical benchmark."""
    return np.full_like(time_grid, level)


def constant_ene(time_grid: np.ndarray, level: float = -500_000.0) -> np.ndarray:
    """Flat ENE profile (must be ≤ 0)."""
    return np.full_like(time_grid, level)


def flat_hazard_curve(
    level: float = 0.01,
    tenors: np.ndarray = None,
) -> tuple:
    """Flat (constant) hazard rate curve at given level."""
    if tenors is None:
        tenors = np.array([0.0, 1.0, 2.0, 5.0, 10.0])
    rates = np.full_like(tenors, level)
    return tenors, rates


# =============================================================================
# HAZARD RATE AND SURVIVAL PROBABILITY
# =============================================================================

class TestHazardRateCurve(unittest.TestCase):

    def test_flat_hazard_interpolated_correctly(self):
        """Flat hazard rate must reproduce the same value at all tenors."""
        tenors = np.array([0.0, 1.0, 5.0, 10.0])
        rates  = np.array([0.01, 0.01, 0.01, 0.01])
        tg     = make_time_grid(T=10.0, n=200)
        lam    = build_hazard_rate_curve(tenors, rates, tg)
        npt.assert_allclose(lam, 0.01, atol=1e-10,
                            err_msg="Flat hazard rate should be constant on grid.")

    def test_linear_interpolation_midpoint(self):
        """Hazard rate at midpoint should be linearly interpolated."""
        tenors = np.array([0.0, 2.0])
        rates  = np.array([0.00, 0.02])
        tg     = np.array([0.0, 1.0, 2.0])
        lam    = build_hazard_rate_curve(tenors, rates, tg)
        self.assertAlmostEqual(lam[1], 0.01, places=8,
                               msg="Midpoint should be linearly interpolated to 0.01.")

    def test_hazard_rate_nonnegative(self):
        """Hazard rates must be non-negative for valid credit curves."""
        tenors, rates = flat_hazard_curve(level=0.005)
        tg  = make_time_grid(T=5.0, n=50)
        lam = build_hazard_rate_curve(tenors, rates, tg)
        self.assertTrue(np.all(lam >= 0.0),
                        "All hazard rates must be non-negative.")


class TestSurvivalProbability(unittest.TestCase):

    def test_survival_at_zero_is_one(self):
        """P_surv(0) must equal 1 by definition."""
        lam_grid = np.full(101, 0.05)
        tg       = make_time_grid(T=5.0, n=100)
        surv     = survival_probability(lam_grid, tg)
        self.assertAlmostEqual(surv[0], 1.0, places=12,
                               msg="P_surv(0) must be exactly 1.")

    def test_survival_is_monotonically_decreasing(self):
        """Survival probability must be strictly decreasing for λ > 0."""
        lam_grid = np.full(101, 0.02)
        tg       = make_time_grid(T=10.0, n=100)
        surv     = survival_probability(lam_grid, tg)
        self.assertTrue(np.all(np.diff(surv) <= 0.0),
                        "P_surv must be non-increasing.")
        self.assertTrue(np.all(np.diff(surv) < 0.0),
                        "P_surv must be strictly decreasing for λ > 0.")

    def test_constant_hazard_rate_analytical(self):
        """For constant λ, P_surv(t) = exp(-λ*t). Check at horizon T."""
        lam = 0.03
        T   = 5.0
        n   = 500
        tg  = make_time_grid(T=T, n=n)
        lam_grid = np.full(n + 1, lam)
        surv     = survival_probability(lam_grid, tg)
        expected = np.exp(-lam * tg)
        npt.assert_allclose(surv, expected, rtol=1e-4,
                            err_msg="P_surv must match exp(-λt) for constant λ.")

    def test_zero_hazard_rate_gives_unit_survival(self):
        """λ = 0 everywhere → P_surv(t) = 1 for all t."""
        tg       = make_time_grid(T=5.0, n=100)
        lam_grid = np.zeros(len(tg))
        surv     = survival_probability(lam_grid, tg)
        npt.assert_allclose(surv, 1.0, atol=1e-12,
                            err_msg="Zero hazard rate must give unit survival.")

    def test_survival_bounded_in_unit_interval(self):
        """P_surv(t) ∈ [0, 1] for all t."""
        lam_grid = np.linspace(0.01, 0.10, 101)
        tg       = make_time_grid(T=10.0, n=100)
        surv     = survival_probability(lam_grid, tg)
        self.assertTrue(np.all(surv >= 0.0) and np.all(surv <= 1.0),
                        "Survival probability must be in [0, 1].")


class TestDefaultProbabilityDensity(unittest.TestCase):

    def test_density_equals_lambda_times_surv(self):
        """q(t) = λ(t) * P_surv(t) element-wise."""
        lam_grid = np.array([0.01, 0.02, 0.03, 0.02, 0.01])
        tg       = np.linspace(0.0, 4.0, 5)
        surv     = survival_probability(lam_grid, tg)
        q        = default_probability_density(lam_grid, surv)
        npt.assert_allclose(q, lam_grid * surv, atol=1e-15)

    def test_density_nonnegative(self):
        """Default density must be non-negative."""
        tenors, rates = flat_hazard_curve(level=0.02)
        tg       = make_time_grid(T=5.0, n=50)
        lam_grid = build_hazard_rate_curve(tenors, rates, tg)
        surv     = survival_probability(lam_grid, tg)
        q        = default_probability_density(lam_grid, surv)
        self.assertTrue(np.all(q >= 0.0))


# =============================================================================
# CVA ENGINE
# =============================================================================

class TestCVAEngine(unittest.TestCase):

    def _make_engine(
        self,
        ee_level: float = 1_000_000.0,
        lam: float = 0.01,
        recovery: float = 0.40,
        T: float = 5.0,
        n: int = 500,
    ) -> CVAEngine:
        tg           = make_time_grid(T=T, n=n)
        ee           = constant_ee(tg, ee_level)
        tenors, rates = flat_hazard_curve(level=lam)
        return CVAEngine(
            ee_profile                = ee,
            time_grid                 = tg,
            counterparty_tenors       = tenors,
            counterparty_hazard_rates = rates,
            recovery_rate             = recovery,
        )

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------

    def test_zero_ee_gives_zero_cva(self):
        """No exposure → no CVA."""
        engine = self._make_engine(ee_level=0.0)
        self.assertAlmostEqual(engine.compute(), 0.0, places=6)

    def test_zero_hazard_rate_gives_zero_cva(self):
        """λ = 0 → no default probability → CVA = 0."""
        tg    = make_time_grid(T=5.0, n=200)
        ee    = constant_ee(tg, 1_000_000.0)
        engine = CVAEngine(
            ee_profile                = ee,
            time_grid                 = tg,
            counterparty_tenors       = np.array([0.0, 5.0]),
            counterparty_hazard_rates = np.array([0.0, 0.0]),
            recovery_rate             = 0.40,
        )
        self.assertAlmostEqual(engine.compute(), 0.0, places=2,
                               msg="Zero hazard rate must give zero CVA.")

    def test_cva_positive(self):
        """CVA must always be positive (it is a cost)."""
        engine = self._make_engine()
        self.assertGreater(engine.compute(), 0.0)

    # ------------------------------------------------------------------
    # Analytical benchmark
    # ------------------------------------------------------------------

    def test_constant_ee_constant_hazard_analytical(self):
        """
        For constant EE = E and constant λ:
            CVA = LGD * E * (1 - exp(-λT))
        """
        E   = 1_000_000.0
        lam = 0.02
        R   = 0.40
        T   = 5.0
        lgd = 1.0 - R

        engine = self._make_engine(ee_level=E, lam=lam, recovery=R, T=T, n=1000)
        cva_numerical  = engine.compute()
        cva_analytical = lgd * E * (1.0 - np.exp(-lam * T))

        npt.assert_allclose(
            cva_numerical, cva_analytical, rtol=1e-3,
            err_msg=(
                f"CVA numerical={cva_numerical:.2f} vs "
                f"analytical={cva_analytical:.2f}"
            ),
        )

    # ------------------------------------------------------------------
    # Monotonicity
    # ------------------------------------------------------------------

    def test_cva_increases_with_hazard_rate(self):
        """Higher counterparty hazard rate → higher CVA."""
        cva_low  = self._make_engine(lam=0.005).compute()
        cva_high = self._make_engine(lam=0.050).compute()
        self.assertGreater(cva_high, cva_low)

    def test_cva_increases_with_ee(self):
        """Higher exposure → higher CVA."""
        cva_small = self._make_engine(ee_level=500_000).compute()
        cva_large = self._make_engine(ee_level=2_000_000).compute()
        self.assertGreater(cva_large, cva_small)

    def test_cva_decreases_with_recovery(self):
        """Higher recovery → lower LGD → lower CVA."""
        cva_low_R  = self._make_engine(recovery=0.20).compute()
        cva_high_R = self._make_engine(recovery=0.80).compute()
        self.assertGreater(cva_low_R, cva_high_R)

    # ------------------------------------------------------------------
    # Term structure
    # ------------------------------------------------------------------

    def test_term_structure_starts_at_zero(self):
        """Cumulative CVA at t=0 must be zero."""
        engine = self._make_engine()
        ts     = engine.term_structure()
        self.assertAlmostEqual(ts[0], 0.0, places=10)

    def test_term_structure_monotonically_increasing(self):
        """Cumulative CVA must be non-decreasing."""
        engine = self._make_engine()
        ts     = engine.term_structure()
        self.assertTrue(np.all(np.diff(ts) >= 0.0),
                        "CVA term structure must be non-decreasing.")

    def test_term_structure_terminal_equals_compute(self):
        """Final cumulative CVA must match compute() to high precision."""
        engine = self._make_engine(n=1000)
        ts     = engine.term_structure()
        cva    = engine.compute()
        npt.assert_allclose(ts[-1], cva, rtol=1e-6)

    # ------------------------------------------------------------------
    # CS01 / HR01
    # ------------------------------------------------------------------

    def test_cs01_positive(self):
        """Rising credit spread → rising CVA → CS01 > 0."""
        self.assertGreater(self._make_engine().cs01(), 0.0)

    def test_hr01_positive(self):
        """Rising hazard rate → rising CVA → HR01 > 0."""
        self.assertGreater(self._make_engine().hr01(), 0.0)

    def test_cs01_greater_than_hr01(self):
        """CS01 = HR01 / LGD > HR01 for LGD < 1."""
        engine = self._make_engine(recovery=0.40)   # LGD = 0.60
        self.assertGreater(engine.cs01(), engine.hr01())

    def test_cs01_hr01_credit_triangle(self):
        """
        Credit triangle: CS01 ≈ HR01 / LGD = HR01 / (1 - R).
        Equivalently: CS01 * LGD ≈ HR01.
        Holds to within 0.1% relative tolerance for 1bp bumps.
        """
        engine = self._make_engine(recovery=0.40, n=1000)
        lgd    = 1.0 - 0.40
        npt.assert_allclose(
            engine.cs01() * lgd, engine.hr01(), rtol=1e-3,
            err_msg="CS01 * LGD must approximate HR01 (credit triangle).",
        )

    def test_cs01_scales_with_ee(self):
        """Doubling EE should approximately double CS01."""
        cs01_base   = self._make_engine(ee_level=1_000_000).cs01()
        cs01_double = self._make_engine(ee_level=2_000_000).cs01()
        npt.assert_allclose(cs01_double, 2.0 * cs01_base, rtol=1e-6)


# =============================================================================
# DVA ENGINE
# =============================================================================

class TestDVAEngine(unittest.TestCase):

    def _make_engine(
        self,
        ene_level: float = -500_000.0,
        lam: float = 0.01,
        recovery: float = 0.40,
        T: float = 5.0,
        n: int = 500,
    ) -> DVAEngine:
        tg            = make_time_grid(T=T, n=n)
        ene           = constant_ene(tg, ene_level)
        tenors, rates = flat_hazard_curve(level=lam)
        return DVAEngine(
            ene_profile      = ene,
            time_grid        = tg,
            own_tenors       = tenors,
            own_hazard_rates = rates,
            recovery_rate    = recovery,
        )

    def test_dva_is_nonpositive(self):
        """DVA is a benefit — must be ≤ 0."""
        self.assertLessEqual(self._make_engine().compute(), 0.0)

    def test_zero_ene_gives_zero_dva(self):
        """No liability → no DVA."""
        engine = self._make_engine(ene_level=0.0)
        self.assertAlmostEqual(engine.compute(), 0.0, places=6)

    def test_dva_magnitude_increases_with_hazard(self):
        """Higher own hazard rate → larger DVA benefit."""
        dva_low  = self._make_engine(lam=0.005).compute()
        dva_high = self._make_engine(lam=0.050).compute()
        self.assertLess(dva_high, dva_low,   # more negative = larger benefit
                        msg="Higher own hazard must increase |DVA|.")

    def test_dva_analytical_constant_ene(self):
        """
        For constant ENE = -N and constant own λ:
            DVA = -LGD * N * (1 - exp(-λT))   [DVA ≤ 0]
        """
        N   = 500_000.0
        lam = 0.015
        R   = 0.40
        T   = 5.0
        lgd = 1.0 - R

        engine = self._make_engine(ene_level=-N, lam=lam, recovery=R, T=T, n=1000)
        dva_numerical  = engine.compute()
        dva_analytical = -(lgd * N * (1.0 - np.exp(-lam * T)))

        npt.assert_allclose(dva_numerical, dva_analytical, rtol=1e-3)


# =============================================================================
# FVA ENGINE
# =============================================================================

class TestFVAEngine(unittest.TestCase):

    def _make_engine(
        self,
        ee_level   : float = 1_000_000.0,
        ene_level  : float = -500_000.0,
        spread     : float = 0.005,
        symmetric  : bool  = True,
        T          : float = 5.0,
        n          : int   = 500,
    ) -> FVAEngine:
        tg  = make_time_grid(T=T, n=n)
        return FVAEngine(
            ee_profile     = constant_ee(tg, ee_level),
            ene_profile    = constant_ene(tg, ene_level),
            time_grid      = tg,
            funding_spread = spread,
            use_symmetric  = symmetric,
        )

    def test_fca_nonnegative(self):
        """Funding cost must be non-negative."""
        self.assertGreaterEqual(self._make_engine().fca(), 0.0)

    def test_fba_nonpositive(self):
        """Funding benefit must be non-positive (ENE ≤ 0)."""
        self.assertLessEqual(self._make_engine().fba(), 0.0)

    def test_symmetric_fva_equals_fca_plus_fba(self):
        """Symmetric FVA = FCA + FBA."""
        engine = self._make_engine(symmetric=True)
        npt.assert_allclose(engine.compute(), engine.fca() + engine.fba(), atol=1e-10)

    def test_asymmetric_fva_equals_fca_only(self):
        """Asymmetric FVA = FCA only (no FBA)."""
        engine = self._make_engine(symmetric=False)
        npt.assert_allclose(engine.compute(), engine.fca(), atol=1e-10)

    def test_zero_spread_gives_zero_fva(self):
        """Zero funding spread → zero FVA."""
        engine = self._make_engine(spread=0.0)
        self.assertAlmostEqual(engine.compute(), 0.0, places=10)

    def test_fca_analytical_constant_ee(self):
        """
        For constant EE = E and constant spread s_f over horizon T:
            FCA = s_f * E * T
        """
        E  = 1_000_000.0
        sf = 0.005
        T  = 5.0
        engine = self._make_engine(ee_level=E, spread=sf, T=T, n=2000)
        npt.assert_allclose(engine.fca(), sf * E * T, rtol=1e-3)

    def test_fva_scales_linearly_with_spread(self):
        """Doubling the funding spread doubles FCA."""
        fca_base   = self._make_engine(spread=0.005).fca()
        fca_double = self._make_engine(spread=0.010).fca()
        npt.assert_allclose(fca_double, 2.0 * fca_base, rtol=1e-10)

    def test_fva_increases_with_ee(self):
        """Higher EE → higher FCA."""
        fca_small = self._make_engine(ee_level=500_000).fca()
        fca_large = self._make_engine(ee_level=2_000_000).fca()
        self.assertGreater(fca_large, fca_small)


# =============================================================================
# MVA ENGINE
# =============================================================================

class TestMVAEngine(unittest.TestCase):

    def _make_engine(
        self,
        im_level   : float = 2_000_000.0,
        spread     : float = 0.005,
        T          : float = 5.0,
        n          : int   = 500,
        with_surv  : bool  = False,
        lam        : float = 0.01,
    ) -> MVAEngine:
        tg = make_time_grid(T=T, n=n)
        im = np.full_like(tg, im_level)
        if with_surv:
            tenors, rates = flat_hazard_curve(level=lam)
            return MVAEngine(
                im_profile       = im,
                time_grid        = tg,
                funding_spread   = spread,
                own_tenors       = tenors,
                own_hazard_rates = rates,
            )
        return MVAEngine(im_profile=im, time_grid=tg, funding_spread=spread)

    def test_mva_nonnegative(self):
        """MVA must be non-negative (it is a cost)."""
        self.assertGreaterEqual(self._make_engine().compute(), 0.0)

    def test_zero_spread_gives_zero_mva(self):
        """Zero funding spread → zero MVA."""
        engine = self._make_engine(spread=0.0)
        self.assertAlmostEqual(engine.compute(), 0.0, places=10)

    def test_zero_im_gives_zero_mva(self):
        """Zero IM → zero MVA."""
        engine = self._make_engine(im_level=0.0)
        self.assertAlmostEqual(engine.compute(), 0.0, places=10)

    def test_analytical_constant_im_unit_survival(self):
        """
        For constant IM = M, P_surv = 1, spread s_f, horizon T:
            MVA = s_f * M * T
        """
        M  = 2_000_000.0
        sf = 0.005
        T  = 5.0
        engine = self._make_engine(im_level=M, spread=sf, T=T, n=2000, with_surv=False)
        npt.assert_allclose(engine.compute(), sf * M * T, rtol=1e-3)

    def test_own_default_reduces_mva(self):
        """
        Including own survival probability (λ > 0) must reduce MVA
        relative to ignoring it (P_surv = 1 everywhere).
        """
        mva_no_surv = self._make_engine(with_surv=False).compute()
        mva_surv    = self._make_engine(with_surv=True, lam=0.02).compute()
        self.assertLess(mva_surv, mva_no_surv,
                        msg="Own default should reduce MVA (P_surv < 1).")

    def test_mva_scales_with_im(self):
        """Doubling IM should double MVA."""
        mva_base   = self._make_engine(im_level=1_000_000).compute()
        mva_double = self._make_engine(im_level=2_000_000).compute()
        npt.assert_allclose(mva_double, 2.0 * mva_base, rtol=1e-10)

    def test_term_structure_starts_at_zero(self):
        """Cumulative MVA at t=0 must be zero."""
        ts = self._make_engine().term_structure()
        self.assertAlmostEqual(ts[0], 0.0, places=10)

    def test_term_structure_monotonically_increasing(self):
        """Cumulative MVA must be non-decreasing."""
        ts = self._make_engine().term_structure()
        self.assertTrue(np.all(np.diff(ts) >= -1e-10))

    def test_term_structure_terminal_equals_compute(self):
        """Final cumulative MVA must match compute()."""
        engine = self._make_engine(n=1000)
        npt.assert_allclose(engine.term_structure()[-1], engine.compute(), rtol=1e-6)


# =============================================================================
# XVA ENGINE (AGGREGATOR)
# =============================================================================

class TestXVAEngine(unittest.TestCase):

    def _make_engine(
        self,
        T              : float = 5.0,
        n              : int   = 500,
        ee_level       : float = 1_000_000.0,
        ene_level      : float = -500_000.0,
        cpty_lam       : float = 0.02,
        own_lam        : float = 0.01,
        recovery       : float = 0.40,
        funding_spread : float = 0.005,
        im_level       : float = 2_000_000.0,
        include_mva    : bool  = True,
    ) -> XVAEngine:
        tg     = make_time_grid(T=T, n=n)
        ee     = constant_ee(tg, ee_level)
        ene    = constant_ene(tg, ene_level)
        c_ten, c_rates = flat_hazard_curve(level=cpty_lam)
        o_ten, o_rates = flat_hazard_curve(level=own_lam)
        im     = np.full_like(tg, im_level) if include_mva else None

        return XVAEngine(
            ee_profile                = ee,
            ene_profile               = ene,
            time_grid                 = tg,
            counterparty_tenors       = c_ten,
            counterparty_hazard_rates = c_rates,
            counterparty_recovery     = recovery,
            own_tenors                = o_ten,
            own_hazard_rates          = o_rates,
            own_recovery              = recovery,
            funding_spread            = funding_spread,
            use_symmetric_fva         = True,
            im_profile                = im,
        )

    def test_total_xva_equals_sum_of_components(self):
        """total_XVA must equal CVA + DVA + FVA + MVA."""
        result = self._make_engine().compute()
        expected = (
            result["CVA"] + result["DVA"]
            + result["FVA"] + result["MVA"]
        )
        npt.assert_allclose(result["total_XVA"], expected, rtol=1e-10)

    def test_bcva_equals_cva_plus_dva(self):
        """BCVA = CVA + DVA."""
        result = self._make_engine().compute()
        npt.assert_allclose(result["BCVA"], result["CVA"] + result["DVA"], rtol=1e-10)

    def test_cva_positive(self):
        """CVA must be positive."""
        self.assertGreater(self._make_engine().compute()["CVA"], 0.0)

    def test_dva_nonpositive(self):
        """DVA must be non-positive."""
        self.assertLessEqual(self._make_engine().compute()["DVA"], 0.0)

    def test_fva_is_finite(self):
        """FVA must be finite."""
        result = self._make_engine().compute()
        self.assertTrue(np.isfinite(result["FVA"]))

    def test_mva_positive_when_im_provided(self):
        """MVA must be positive when IM profile is provided."""
        result = self._make_engine(include_mva=True).compute()
        self.assertGreater(result["MVA"], 0.0)

    def test_mva_zero_when_im_not_provided(self):
        """MVA must be zero when no IM profile is passed."""
        result = self._make_engine(include_mva=False).compute()
        self.assertAlmostEqual(result["MVA"], 0.0, places=10)

    def test_no_dva_when_own_hazard_omitted(self):
        """Omitting own hazard curve must give DVA = 0."""
        tg  = make_time_grid(T=5.0, n=200)
        ee  = constant_ee(tg)
        ene = constant_ene(tg)
        c_ten, c_rates = flat_hazard_curve(level=0.02)
        engine = XVAEngine(
            ee_profile                = ee,
            ene_profile               = ene,
            time_grid                 = tg,
            counterparty_tenors       = c_ten,
            counterparty_hazard_rates = c_rates,
        )
        result = engine.compute()
        self.assertAlmostEqual(result["DVA"], 0.0, places=10)

    def test_cva_term_structure_terminal_matches_cva(self):
        """Last entry of CVA term structure must match CVA scalar."""
        result = self._make_engine(n=1000).compute()
        npt.assert_allclose(result["CVA_term_structure"][-1], result["CVA"], rtol=1e-6)

    def test_mva_term_structure_terminal_matches_mva(self):
        """Last entry of MVA term structure must match MVA scalar."""
        result = self._make_engine(n=1000, include_mva=True).compute()
        npt.assert_allclose(result["MVA_term_structure"][-1], result["MVA"], rtol=1e-6)

    def test_mva_term_structure_zeros_when_no_im(self):
        """MVA term structure must be all zeros when IM not provided."""
        result = self._make_engine(include_mva=False).compute()
        npt.assert_allclose(result["MVA_term_structure"], 0.0, atol=1e-12)

    def test_cs01_positive(self):
        """CS01 must be positive (rising counterparty spread → rising CVA)."""
        result = self._make_engine().compute()
        self.assertGreater(result["CS01"], 0.0)

    def test_hr01_positive(self):
        """HR01 must be positive."""
        result = self._make_engine().compute()
        self.assertGreater(result["HR01"], 0.0)

    def test_cs01_greater_than_hr01(self):
        """CS01 > HR01 for recovery < 1 (CS01 = HR01 / LGD)."""
        result = self._make_engine(recovery=0.40).compute()
        self.assertGreater(result["CS01"], result["HR01"])

    def test_result_keys_complete(self):
        """compute() must return all expected keys."""
        required = {
            "CVA", "DVA", "FVA", "FCA", "FBA", "MVA",
            "BCVA", "total_XVA", "CS01", "HR01",
            "time_grid", "CVA_term_structure", "MVA_term_structure",
        }
        result = self._make_engine().compute()
        self.assertTrue(required.issubset(result.keys()),
                        f"Missing keys: {required - result.keys()}")


# =============================================================================
# NUMERICAL PRECISION BENCHMARKS
# =============================================================================

class TestNumericalPrecision(unittest.TestCase):
    """
    Verify that Simpson integration converges to known analytical values
    at the resolution used by the toolkit (n=500 steps).
    """

    def test_cva_convergence_with_grid_refinement(self):
        """
        CVA error must decrease as grid becomes finer.
        Verifies that numerical integration is doing useful work.
        """
        E, lam, R, T = 1_000_000.0, 0.02, 0.40, 5.0
        analytical   = (1 - R) * E * (1 - np.exp(-lam * T))
        tenors, rates = flat_hazard_curve(level=lam)

        errors = []
        for n in [50, 200, 1000]:
            tg     = make_time_grid(T=T, n=n)
            engine = CVAEngine(
                ee_profile                = constant_ee(tg, E),
                time_grid                 = tg,
                counterparty_tenors       = tenors,
                counterparty_hazard_rates = rates,
                recovery_rate             = R,
            )
            errors.append(abs(engine.compute() - analytical))

        # Each grid refinement must reduce error
        self.assertLess(errors[1], errors[0],
                        "Finer grid must reduce CVA integration error.")
        self.assertLess(errors[2], errors[1],
                        "Finest grid must give smallest CVA error.")

    def test_mva_convergence(self):
        """MVA = s_f * M * T for constant IM; verify sub-0.1% error at n=500."""
        M, sf, T = 2_000_000.0, 0.005, 5.0
        analytical = sf * M * T
        tg = make_time_grid(T=T, n=500)
        engine = MVAEngine(
            im_profile     = np.full_like(tg, M),
            time_grid      = tg,
            funding_spread = sf,
        )
        npt.assert_allclose(engine.compute(), analytical, rtol=1e-3)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
