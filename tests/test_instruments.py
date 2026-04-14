"""
test_instruments.py
-------------------
Unit tests for instruments.py: InterestRateSwap, Bond, SFT.

All benchmarks are derived analytically so no external market data
or simulation is required.

Key analytical results used
---------------------------
Floating leg telescope (single-curve, spot-start):
    PV_float = N * sum_i (df(t_{i-1})/df(t_i) - 1) * df(t_i)
             = N * sum_i (df(t_{i-1}) - df(t_i))
             = N * (df(0) - df(T))
             = N * (1 - exp(-r*T))    for flat rate r

Fixed leg (flat curve r, fixed rate K, step tau):
    PV_fixed = K * N * sum_i tau * exp(-r * T_i)

Par rate condition:
    K_par = PV_float / (N * annuity)

Zero-coupon bond on flat curve:
    P_dirty = Face * exp(-r * T)  (coupon = 0)

Par bond:
    Face = C * sum_i exp(-r*T_i) + Face * exp(-r*T)
    => C/Face = (1 - exp(-r*T)) / annuity_bond

YTM roundtrip:
    dirty_price(curve) -> y -> reprice with flat y curve -> same dirty_price

SFT repurchase price:
    P_repo = notional * exp(r * T)
"""

import sys
import os
import unittest
import numpy as np

# ---------------------------------------------------------------------------
# Path wiring: tests run from the outputs directory
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from curve_factory import Curve
from instruments import InterestRateSwap, Bond, SFT


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def flat_curve(rate: float, label: str = "flat") -> Curve:
    """
    Construct a Curve object with a flat continuously compounded zero rate.

    Grid is dense enough to bracket all payment dates up to 10Y.
    P(0, 0) = 1.0 exactly; P(0, T) = exp(-rate * T) for T > 0.
    """
    tenors = np.array(
        [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
         2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        dtype=float,
    )
    dfs = np.where(tenors == 0.0, 1.0, np.exp(-rate * tenors))
    return Curve(tenors, dfs, label=label)


def bond_annuity(rate: float, maturity: float, freq: int) -> float:
    """Sum of exp(-r*T_i) over semi/annual coupon dates (analytical)."""
    step = 1.0 / freq
    dates = np.arange(step, maturity + 1e-9, step)
    return float(np.sum(np.exp(-rate * dates)))


def par_bond_coupon_rate(rate: float, maturity: float, freq: int) -> float:
    """
    Annual coupon rate that prices a bond at par on a flat curve.

    Face = (face * c / freq) * annuity + face * exp(-r*T)
    => c = freq * (1 - exp(-r*T)) / annuity
    """
    ann = bond_annuity(rate, maturity, freq)
    return freq * (1.0 - np.exp(-rate * maturity)) / ann


# ===========================================================================
# InterestRateSwap Tests
# ===========================================================================

class TestIRSSchedule(unittest.TestCase):
    """Payment date generation."""

    def setUp(self):
        self.swap = InterestRateSwap(
            notional=1_000_000,
            fixed_rate=0.05,
            tenor=5.0,
            payment_frequency=2,
            float_frequency=4,
        )

    def test_fixed_leg_date_count(self):
        """5Y semi-annual fixed leg has exactly 10 payment dates."""
        dates = self.swap._fixed_payment_dates()
        self.assertEqual(len(dates), 10)

    def test_float_leg_date_count(self):
        """5Y quarterly floating leg has exactly 20 payment dates."""
        dates = self.swap._float_payment_dates()
        self.assertEqual(len(dates), 20)

    def test_fixed_first_date(self):
        """First fixed payment is 0.5Y from start."""
        dates = self.swap._fixed_payment_dates()
        self.assertAlmostEqual(dates[0], 0.5, places=7)

    def test_fixed_last_date(self):
        """Last fixed payment equals tenor."""
        dates = self.swap._fixed_payment_dates()
        self.assertAlmostEqual(dates[-1], 5.0, places=7)

    def test_float_last_date(self):
        """Last floating payment equals tenor."""
        dates = self.swap._float_payment_dates()
        self.assertAlmostEqual(dates[-1], 5.0, places=7)

    def test_forward_starting_schedule(self):
        """Forward-starting swap: dates begin at start + step."""
        swap = InterestRateSwap(
            notional=1e6, fixed_rate=0.04, tenor=2.0,
            payment_frequency=2, start=1.0,
        )
        dates = swap._fixed_payment_dates()
        self.assertAlmostEqual(dates[0], 1.5, places=7)
        self.assertAlmostEqual(dates[-1], 3.0, places=7)

    def test_annual_fixed_leg_count(self):
        """3Y annual fixed leg has exactly 3 payment dates."""
        swap = InterestRateSwap(
            notional=1e6, fixed_rate=0.04, tenor=3.0, payment_frequency=1,
        )
        dates = swap._fixed_payment_dates()
        self.assertEqual(len(dates), 3)


class TestIRSFixedLeg(unittest.TestCase):
    """Fixed leg PV: analytical benchmark vs implementation."""

    def setUp(self):
        self.r = 0.04
        self.K = 0.05
        self.N = 1_000_000.0
        self.T = 5.0
        self.freq = 2
        self.curve = flat_curve(self.r)
        self.swap = InterestRateSwap(
            notional=self.N,
            fixed_rate=self.K,
            tenor=self.T,
            payment_frequency=self.freq,
        )

    def test_pv_fixed_leg_analytical(self):
        """
        PV_fixed = K * N * sum_i (tau_i * exp(-r * T_i))

        Tau = 0.5 (semi-annual), dates = 0.5, 1.0, ..., 5.0.
        """
        step = 1.0 / self.freq
        dates = np.arange(step, self.T + 1e-9, step)
        expected = self.K * self.N * step * np.sum(np.exp(-self.r * dates))

        actual = self.swap.pv_fixed_leg(self.curve)
        self.assertAlmostEqual(actual, expected, delta=0.01)

    def test_fixed_leg_positive(self):
        """PV of fixed leg is always positive (asset to receiver)."""
        self.assertGreater(self.swap.pv_fixed_leg(self.curve), 0.0)

    def test_fixed_leg_increases_with_notional(self):
        """Doubling notional doubles fixed leg PV."""
        swap2 = InterestRateSwap(
            notional=2 * self.N, fixed_rate=self.K, tenor=self.T,
            payment_frequency=self.freq,
        )
        pv1 = self.swap.pv_fixed_leg(self.curve)
        pv2 = swap2.pv_fixed_leg(self.curve)
        self.assertAlmostEqual(pv2 / pv1, 2.0, places=10)

    def test_fixed_leg_decreases_with_rate(self):
        """Higher discount rate lowers the fixed leg PV."""
        curve_hi = flat_curve(self.r + 0.02)
        pv_lo = self.swap.pv_fixed_leg(self.curve)
        pv_hi = self.swap.pv_fixed_leg(curve_hi)
        self.assertGreater(pv_lo, pv_hi)


class TestIRSFloatingLeg(unittest.TestCase):
    """Floating leg PV: telescope benchmark."""

    def test_pv_floating_leg_telescopes(self):
        """
        Single-curve (discount == projection), spot start:

            PV_float = N * (1 - exp(-r*T))

        The term-by-term sum telescopes exactly to df(0) - df(T).
        """
        r = 0.04
        N = 1_000_000.0
        T = 5.0
        curve = flat_curve(r)
        swap = InterestRateSwap(
            notional=N, fixed_rate=0.04, tenor=T,
            payment_frequency=2, float_frequency=4,
        )
        expected = N * (1.0 - np.exp(-r * T))
        actual = swap.pv_floating_leg(curve, curve)
        self.assertAlmostEqual(actual, expected, delta=0.01)

    def test_float_leg_increases_with_discount_rate(self):
        """
        Counterintuitive: PV_float = N*(1 - df(T)) rises with r
        because df(T) = exp(-r*T) falls.
        """
        N = 1_000_000.0
        T = 3.0
        curve_lo = flat_curve(0.02)
        curve_hi = flat_curve(0.06)
        swap = InterestRateSwap(
            notional=N, fixed_rate=0.04, tenor=T,
            payment_frequency=2, float_frequency=4,
        )
        pv_lo = swap.pv_floating_leg(curve_lo, curve_lo)
        pv_hi = swap.pv_floating_leg(curve_hi, curve_hi)
        self.assertGreater(pv_hi, pv_lo)

    def test_float_leg_multi_curve_vs_single_curve(self):
        """
        Projecting at a higher rate than discounting raises the float PV
        vs a single-curve setup at the discount rate.
        """
        r_ois = 0.04
        r_sofr = 0.045  # SOFR basis above OIS
        N = 1_000_000.0
        T = 5.0
        ois_curve = flat_curve(r_ois, label="OIS")
        sofr_curve = flat_curve(r_sofr, label="SOFR")
        swap = InterestRateSwap(
            notional=N, fixed_rate=0.04, tenor=T,
            payment_frequency=2, float_frequency=4,
        )
        pv_single = swap.pv_floating_leg(ois_curve, ois_curve)
        pv_multi = swap.pv_floating_leg(ois_curve, sofr_curve)
        self.assertGreater(pv_multi, pv_single)


class TestIRSParRateAndPV(unittest.TestCase):
    """Par rate computation and net PV tests."""

    def setUp(self):
        self.r = 0.04
        self.N = 1_000_000.0
        self.T = 5.0
        self.curve = flat_curve(self.r)

    def test_par_rate_formula(self):
        """
        K_par = PV_float / (N * annuity)

        For single-curve flat r, T=5Y semi-annual:
            PV_float = N * (1 - exp(-r*T))
            annuity  = 0.5 * sum_i exp(-r*T_i)
            K_par    = (1 - exp(-r*T)) / (0.5 * sum_i exp(-r*T_i))
        """
        dates = np.arange(0.5, self.T + 1e-9, 0.5)
        annuity = 0.5 * np.sum(np.exp(-self.r * dates))
        expected_par = (1.0 - np.exp(-self.r * self.T)) / annuity

        swap = InterestRateSwap(
            notional=self.N, fixed_rate=0.04, tenor=self.T,
            payment_frequency=2, float_frequency=4,
        )
        actual_par = swap.par_rate(self.curve, self.curve)
        self.assertAlmostEqual(actual_par, expected_par, places=8)

    def test_par_swap_zero_pv(self):
        """Swap set to par rate has near-zero net PV."""
        swap = InterestRateSwap(
            notional=self.N, fixed_rate=0.04, tenor=self.T,
            payment_frequency=2, float_frequency=4,
        )
        K_par = swap.par_rate(self.curve, self.curve)
        par_swap = InterestRateSwap(
            notional=self.N, fixed_rate=K_par, tenor=self.T,
            payment_frequency=2, float_frequency=4, payer=True,
        )
        result = par_swap.pv(self.curve, self.curve)
        self.assertAlmostEqual(result["pv_net"], 0.0, delta=1.0)

    def test_payer_receiver_symmetry(self):
        """Payer PV == -Receiver PV for identical fixed rate."""
        K = 0.045
        payer = InterestRateSwap(
            notional=self.N, fixed_rate=K, tenor=self.T,
            payment_frequency=2, float_frequency=4, payer=True,
        )
        receiver = InterestRateSwap(
            notional=self.N, fixed_rate=K, tenor=self.T,
            payment_frequency=2, float_frequency=4, payer=False,
        )
        pv_p = payer.pv(self.curve, self.curve)["pv_net"]
        pv_r = receiver.pv(self.curve, self.curve)["pv_net"]
        self.assertAlmostEqual(pv_p, -pv_r, places=8)

    def test_payer_positive_when_fixed_below_par(self):
        """
        Payer IRS PV > 0 when fixed rate < par rate.
        (Paying below-market fixed; receiving above-market float.)
        """
        swap = InterestRateSwap(
            notional=self.N, fixed_rate=0.04, tenor=self.T,
            payment_frequency=2, float_frequency=4, payer=True,
        )
        K_par = swap.par_rate(self.curve, self.curve)
        below_par_swap = InterestRateSwap(
            notional=self.N, fixed_rate=K_par - 0.01, tenor=self.T,
            payment_frequency=2, float_frequency=4, payer=True,
        )
        pv_net = below_par_swap.pv(self.curve, self.curve)["pv_net"]
        self.assertGreater(pv_net, 0.0)

    def test_pv_dict_keys(self):
        """pv() output has all required keys."""
        swap = InterestRateSwap(
            notional=self.N, fixed_rate=0.04, tenor=self.T,
            payment_frequency=2, float_frequency=4,
        )
        result = swap.pv(self.curve, self.curve)
        for key in ("pv_fixed", "pv_float", "pv_net", "par_rate"):
            self.assertIn(key, result)

    def test_par_rate_in_pv_dict(self):
        """par_rate in pv() dict matches standalone par_rate() call."""
        swap = InterestRateSwap(
            notional=self.N, fixed_rate=0.04, tenor=self.T,
            payment_frequency=2, float_frequency=4,
        )
        pv_dict = swap.pv(self.curve, self.curve)
        standalone = swap.par_rate(self.curve, self.curve)
        self.assertAlmostEqual(pv_dict["par_rate"], standalone, places=10)

    def test_repr(self):
        """__repr__ includes direction and key fields."""
        swap = InterestRateSwap(
            notional=self.N, fixed_rate=0.04, tenor=self.T, payer=True,
        )
        r = repr(swap)
        self.assertIn("Payer", r)
        self.assertIn("5.0Y", r)


# ===========================================================================
# Bond Tests
# ===========================================================================

class TestBondCashFlows(unittest.TestCase):
    """Coupon amounts and accrued interest."""

    def setUp(self):
        self.bond = Bond(
            face_value=1000.0,
            coupon_rate=0.06,
            maturity=5.0,
            coupon_frequency=2,
        )

    def test_coupon_amount(self):
        """Semi-annual coupon = face * rate / 2."""
        expected = 1000.0 * 0.06 / 2  # = 30.0
        self.assertAlmostEqual(self.bond.coupon_amount(), expected, places=10)

    def test_coupon_amount_annual(self):
        """Annual coupon = face * rate."""
        bond = Bond(face_value=1000.0, coupon_rate=0.05, maturity=3.0, coupon_frequency=1)
        self.assertAlmostEqual(bond.coupon_amount(), 50.0, places=10)

    def test_accrued_interest_zero(self):
        """AI at t=0 (settlement) is zero."""
        self.assertAlmostEqual(self.bond.accrued_interest(0.0), 0.0, places=10)

    def test_accrued_interest_analytical(self):
        """AI = face * rate * t_since_last_coupon."""
        t = 0.25  # 3 months since last coupon
        expected = 1000.0 * 0.06 * 0.25
        self.assertAlmostEqual(self.bond.accrued_interest(t), expected, places=10)

    def test_accrued_interest_at_one_period(self):
        """AI at t = 1/freq equals one full period's coupon."""
        t = 0.5
        expected = 1000.0 * 0.06 * 0.5  # = 30.0
        self.assertAlmostEqual(self.bond.accrued_interest(t), expected, places=10)

    def test_coupon_date_count(self):
        """5Y semi-annual bond has 10 coupon dates."""
        dates = self.bond._coupon_dates()
        self.assertEqual(len(dates), 10)

    def test_coupon_dates_last(self):
        """Last coupon date equals maturity."""
        dates = self.bond._coupon_dates()
        self.assertAlmostEqual(dates[-1], 5.0, places=7)


class TestBondPricing(unittest.TestCase):
    """Dirty price, clean price, and analytical benchmarks."""

    def test_zero_coupon_bond_dirty_price(self):
        """
        Zero coupon bond: P_dirty = Face * exp(-r * T).
        """
        r = 0.05
        T = 5.0
        face = 1000.0
        curve = flat_curve(r)
        bond = Bond(face_value=face, coupon_rate=0.0, maturity=T, coupon_frequency=2)

        expected = face * np.exp(-r * T)
        actual = bond.dirty_price(curve)
        self.assertAlmostEqual(actual, expected, places=6)

    def test_dirty_price_analytical(self):
        """
        P_dirty = C * sum_i exp(-r*T_i) + Face * exp(-r*T)

        Verified numerically for face=1000, c=6%, T=5Y, semi-annual, r=4%.
        """
        r = 0.04
        c = 0.06
        T = 5.0
        face = 1000.0
        freq = 2
        curve = flat_curve(r)
        bond = Bond(face_value=face, coupon_rate=c, maturity=T, coupon_frequency=freq)

        step = 1.0 / freq
        dates = np.arange(step, T + 1e-9, step)
        C = face * c / freq
        expected = C * np.sum(np.exp(-r * dates)) + face * np.exp(-r * T)

        actual = bond.dirty_price(curve)
        self.assertAlmostEqual(actual, expected, delta=0.001)

    def test_par_bond_prices_at_face(self):
        """
        Bond with coupon = par rate prices at face value.

        par coupon = freq * (1 - exp(-r*T)) / sum_i exp(-r*T_i)
        """
        r = 0.05
        T = 5.0
        face = 1000.0
        freq = 2
        c_par = par_bond_coupon_rate(r, T, freq)

        curve = flat_curve(r)
        bond = Bond(face_value=face, coupon_rate=c_par, maturity=T, coupon_frequency=freq)
        dirty = bond.dirty_price(curve)
        self.assertAlmostEqual(dirty, face, delta=0.01)

    def test_above_par_premium_bond(self):
        """
        When discount rate < coupon rate, bond trades above par.
        """
        r = 0.03   # low discount rate
        c = 0.06   # high coupon
        curve = flat_curve(r)
        bond = Bond(face_value=1000.0, coupon_rate=c, maturity=5.0, coupon_frequency=2)
        self.assertGreater(bond.dirty_price(curve), 1000.0)

    def test_below_par_discount_bond(self):
        """
        When discount rate > coupon rate, bond trades below par.
        """
        r = 0.08   # high discount rate
        c = 0.04   # low coupon
        curve = flat_curve(r)
        bond = Bond(face_value=1000.0, coupon_rate=c, maturity=5.0, coupon_frequency=2)
        self.assertLess(bond.dirty_price(curve), 1000.0)

    def test_price_decreases_with_yield(self):
        """Higher discount rate → strictly lower dirty price."""
        c = 0.05
        T = 5.0
        face = 1000.0
        bond = Bond(face_value=face, coupon_rate=c, maturity=T, coupon_frequency=2)

        prices = [bond.dirty_price(flat_curve(r)) for r in [0.03, 0.05, 0.07, 0.10]]
        for i in range(len(prices) - 1):
            self.assertGreater(prices[i], prices[i + 1])

    def test_clean_price_equals_dirty_minus_ai(self):
        """clean_price(curve, t) == dirty_price(curve) - accrued_interest(t)."""
        r = 0.04
        t_ai = 0.25
        curve = flat_curve(r)
        bond = Bond(face_value=1000.0, coupon_rate=0.06, maturity=5.0, coupon_frequency=2)

        dirty = bond.dirty_price(curve)
        ai = bond.accrued_interest(t_ai)
        clean = bond.clean_price(curve, t_since_last_coupon=t_ai)

        self.assertAlmostEqual(clean, dirty - ai, places=10)

    def test_clean_price_zero_at_coupon_date(self):
        """Clean price equals dirty price when AI is zero (coupon date)."""
        r = 0.04
        curve = flat_curve(r)
        bond = Bond(face_value=1000.0, coupon_rate=0.06, maturity=5.0, coupon_frequency=2)

        self.assertAlmostEqual(
            bond.clean_price(curve, t_since_last_coupon=0.0),
            bond.dirty_price(curve),
            places=10,
        )


class TestBondYTM(unittest.TestCase):
    """YTM solver: roundtrip and closed-form cases."""

    def test_ytm_roundtrip(self):
        """
        YTM roundtrip: compute dirty price → solve YTM → reprice with flat
        YTM curve → recover original dirty price.

        This tests the Brent solver and the bond_pv function in isolation.
        """
        r = 0.04
        c = 0.06
        T = 5.0
        face = 1000.0
        freq = 2
        curve = flat_curve(r)
        bond = Bond(face_value=face, coupon_rate=c, maturity=T, coupon_frequency=freq)

        original_dirty = bond.dirty_price(curve)
        ytm = bond.yield_to_maturity(curve)

        # Reprice: P = C * sum exp(-y*T_i) + Face * exp(-y*T)
        step = 1.0 / freq
        dates = np.arange(step, T + 1e-9, step)
        C = face * c / freq
        repriced = C * np.sum(np.exp(-ytm * dates)) + face * np.exp(-ytm * T)

        self.assertAlmostEqual(repriced, original_dirty, places=6)

    def test_zero_coupon_ytm_equals_curve_rate(self):
        """
        For a zero coupon bond on a flat curve at rate r,
        the YTM equals r exactly (continuously compounded convention is shared).
        """
        r = 0.05
        T = 5.0
        face = 1000.0
        curve = flat_curve(r)
        bond = Bond(face_value=face, coupon_rate=0.0, maturity=T, coupon_frequency=1)

        ytm = bond.yield_to_maturity(curve)
        self.assertAlmostEqual(ytm, r, places=8)

    def test_par_bond_ytm_near_par_coupon_rate(self):
        """
        Par bond: YTM (cc) is close to the cc par coupon rate.
        Exact equality holds only for zero coupon bonds; for coupon bonds
        there is a small convexity difference between cc and simple discounting.
        We check that YTM is positive and finite.
        """
        r = 0.04
        T = 5.0
        freq = 2
        c_par = par_bond_coupon_rate(r, T, freq)

        curve = flat_curve(r)
        bond = Bond(face_value=1000.0, coupon_rate=c_par, maturity=T, coupon_frequency=freq)
        ytm = bond.yield_to_maturity(curve)
        self.assertGreater(ytm, 0.0)
        self.assertLess(ytm, 0.5)

    def test_ytm_increases_with_discount_rate(self):
        """Higher discount rate → lower price → higher YTM."""
        T = 5.0
        c = 0.05
        face = 1000.0
        bond = Bond(face_value=face, coupon_rate=c, maturity=T, coupon_frequency=2)

        ytms = [bond.yield_to_maturity(flat_curve(r)) for r in [0.03, 0.05, 0.07]]
        self.assertLess(ytms[0], ytms[1])
        self.assertLess(ytms[1], ytms[2])

    def test_pv_dict_keys(self):
        """Bond.pv() output has all required keys."""
        curve = flat_curve(0.04)
        bond = Bond(face_value=1000.0, coupon_rate=0.05, maturity=5.0)
        result = bond.pv(curve, t_since_last_coupon=0.0)
        for key in ("dirty_price", "clean_price", "accrued_interest", "ytm"):
            self.assertIn(key, result)

    def test_pv_dict_internal_consistency(self):
        """pv() dict: dirty_price - accrued_interest == clean_price."""
        curve = flat_curve(0.04)
        bond = Bond(face_value=1000.0, coupon_rate=0.05, maturity=5.0)
        t_ai = 0.3
        result = bond.pv(curve, t_since_last_coupon=t_ai)
        self.assertAlmostEqual(
            result["dirty_price"] - result["accrued_interest"],
            result["clean_price"],
            places=10,
        )

    def test_repr(self):
        """__repr__ includes Face, Coupon, Maturity fields."""
        bond = Bond(face_value=1000.0, coupon_rate=0.05, maturity=5.0)
        r = repr(bond)
        self.assertIn("Bond", r)
        self.assertIn("5.0Y", r)


# ===========================================================================
# SFT (Repo / Securities Lending) Tests
# ===========================================================================

class TestSFTConstruction(unittest.TestCase):
    """Construction, defaults, and initial quantities."""

    def setUp(self):
        # notional=980, haircut=2% => collateral = 980/0.98 = 1000
        self.sft = SFT(
            notional=980.0,
            repo_rate=0.03,
            tenor=0.5,
            haircut=0.02,
            is_repo=True,
        )

    def test_default_collateral_value(self):
        """Default collateral = notional / (1 - haircut)."""
        expected = 980.0 / 0.98  # = 1000.0
        self.assertAlmostEqual(self.sft.collateral_value, expected, places=8)

    def test_initial_margin(self):
        """IM = collateral_value * haircut."""
        expected = (980.0 / 0.98) * 0.02  # = 20.0
        self.assertAlmostEqual(self.sft.initial_margin(), expected, places=8)

    def test_explicit_collateral_value(self):
        """Explicit collateral value is stored as-is."""
        sft = SFT(
            notional=1000.0, repo_rate=0.03, tenor=0.5, haircut=0.02,
            collateral_value=1050.0,
        )
        self.assertAlmostEqual(sft.collateral_value, 1050.0, places=10)

    def test_zero_haircut(self):
        """Zero haircut: collateral == notional, IM == 0."""
        sft = SFT(notional=1000.0, repo_rate=0.03, tenor=0.5, haircut=0.0)
        self.assertAlmostEqual(sft.collateral_value, 1000.0, places=8)
        self.assertAlmostEqual(sft.initial_margin(), 0.0, places=10)


class TestSFTAccruedAndRepurchase(unittest.TestCase):
    """Accrued interest and repurchase price."""

    def setUp(self):
        self.notional = 1_000_000.0
        self.r = 0.04
        self.T = 0.5
        self.sft = SFT(
            notional=self.notional,
            repo_rate=self.r,
            tenor=self.T,
            haircut=0.02,
        )

    def test_repurchase_price_formula(self):
        """P_repo = notional * exp(repo_rate * tenor)."""
        expected = self.notional * np.exp(self.r * self.T)
        self.assertAlmostEqual(self.sft.repurchase_price(), expected, places=6)

    def test_repurchase_price_greater_than_notional(self):
        """Repurchase price > notional for positive repo rate."""
        self.assertGreater(self.sft.repurchase_price(), self.notional)

    def test_accrued_interest_at_zero(self):
        """AI(0) = notional * (exp(0) - 1) = 0."""
        self.assertAlmostEqual(self.sft.accrued_interest(0.0), 0.0, places=10)

    def test_accrued_interest_at_maturity(self):
        """AI(T) = repurchase_price - notional."""
        ai_T = self.sft.accrued_interest(self.T)
        expected = self.sft.repurchase_price() - self.notional
        self.assertAlmostEqual(ai_T, expected, places=6)

    def test_accrued_interest_midpoint(self):
        """AI(t) = notional * (exp(r*t) - 1) at arbitrary t."""
        t = 0.25
        expected = self.notional * (np.exp(self.r * t) - 1.0)
        self.assertAlmostEqual(self.sft.accrued_interest(t), expected, places=6)

    def test_accrued_interest_invalid_t(self):
        """t > tenor raises ValueError."""
        with self.assertRaises(ValueError):
            self.sft.accrued_interest(self.T + 0.01)

    def test_accrued_interest_negative_t(self):
        """Negative t raises ValueError."""
        with self.assertRaises(ValueError):
            self.sft.accrued_interest(-0.01)

    def test_repurchase_increases_with_rate(self):
        """Higher repo rate → higher repurchase price."""
        sft_lo = SFT(notional=1e6, repo_rate=0.02, tenor=1.0, haircut=0.02)
        sft_hi = SFT(notional=1e6, repo_rate=0.06, tenor=1.0, haircut=0.02)
        self.assertGreater(sft_hi.repurchase_price(), sft_lo.repurchase_price())


class TestSFTMtMExposure(unittest.TestCase):
    """MtM exposure for repo and reverse repo."""

    def setUp(self):
        self.notional = 1_000_000.0
        self.r = 0.03
        self.T = 1.0
        self.t = 0.0  # test at inception

    def test_repo_seller_exposure_when_collateral_appreciates(self):
        """
        Repo seller (cash borrower) is exposed when collateral > cash owed.
        At t=0: cash_owed = notional; if collateral > notional, exposure > 0.
        """
        sft = SFT(notional=self.notional, repo_rate=self.r, tenor=self.T, haircut=0.02, is_repo=True)
        # Collateral has appreciated above notional
        high_coll = self.notional * 1.10
        exposure = sft.mtm_exposure(high_coll, t=0.0)
        self.assertGreater(exposure, 0.0)
        self.assertAlmostEqual(exposure, high_coll - self.notional, places=6)

    def test_repo_seller_exposure_zero_when_collateral_below_cash(self):
        """
        Repo seller has zero exposure if collateral < cash owed
        (the lender bears the loss in this case, not the borrower).
        """
        sft = SFT(notional=self.notional, repo_rate=self.r, tenor=self.T, haircut=0.02, is_repo=True)
        low_coll = self.notional * 0.90  # collateral has fallen
        exposure = sft.mtm_exposure(low_coll, t=0.0)
        self.assertEqual(exposure, 0.0)

    def test_reverse_repo_exposure_when_collateral_falls(self):
        """
        Reverse repo buyer (cash lender) exposed when cash owed > collateral.
        At t=0 with fallen collateral.
        """
        sft = SFT(notional=self.notional, repo_rate=self.r, tenor=self.T,
                  haircut=0.02, is_repo=False)
        low_coll = self.notional * 0.90
        exposure = sft.mtm_exposure(low_coll, t=0.0)
        expected = self.notional - low_coll  # cash_owed at t=0 = notional
        self.assertAlmostEqual(exposure, expected, places=6)

    def test_reverse_repo_exposure_zero_when_collateral_appreciates(self):
        """Reverse repo buyer has zero exposure when collateral > cash owed."""
        sft = SFT(notional=self.notional, repo_rate=self.r, tenor=self.T,
                  haircut=0.02, is_repo=False)
        high_coll = self.notional * 1.10
        self.assertEqual(sft.mtm_exposure(high_coll, t=0.0), 0.0)

    def test_exposure_nonneg(self):
        """MtM exposure is always >= 0 regardless of inputs."""
        sft = SFT(notional=1e6, repo_rate=0.04, tenor=1.0, haircut=0.02, is_repo=True)
        for coll in [5e5, 9e5, 1e6, 1.1e6, 1.5e6]:
            self.assertGreaterEqual(sft.mtm_exposure(coll, t=0.0), 0.0)


class TestSFTMarginCall(unittest.TestCase):
    """Margin call triggering logic."""

    def setUp(self):
        self.notional = 1_000_000.0
        self.r = 0.03
        self.T = 1.0
        self.h = 0.02
        # At t=0, cash_owed = notional; required_coll = notional / (1-h)
        self.sft = SFT(
            notional=self.notional, repo_rate=self.r, tenor=self.T, haircut=self.h,
        )
        self.required_coll_t0 = self.notional / (1.0 - self.h)  # ~1_020_408

    def test_margin_call_triggered_when_shortfall_exceeds_threshold(self):
        """
        If collateral = face (shortfall = required - face ≈ 20408) and
        threshold = 10000, call is triggered.
        """
        coll = self.notional  # 1_000_000 < required 1_020_408
        shortfall = self.required_coll_t0 - coll  # ≈ 20_408
        mc = self.sft.margin_call(coll, t=0.0, threshold=10_000.0, minimum_transfer_amount=0.0)
        self.assertAlmostEqual(mc, shortfall, places=4)

    def test_no_margin_call_when_shortfall_below_threshold(self):
        """Shortfall within threshold: call is zero."""
        # Collateral is just slightly below required
        coll = self.required_coll_t0 - 5_000.0  # shortfall = 5000 < threshold
        mc = self.sft.margin_call(coll, t=0.0, threshold=10_000.0, minimum_transfer_amount=0.0)
        self.assertEqual(mc, 0.0)

    def test_no_margin_call_below_mta(self):
        """Shortfall exceeds threshold but is below MTA: call is zero."""
        coll = self.notional  # shortfall ≈ 20408
        mc = self.sft.margin_call(
            coll, t=0.0, threshold=0.0, minimum_transfer_amount=50_000.0,
        )
        self.assertEqual(mc, 0.0)

    def test_negative_margin_call_when_overcollateralised(self):
        """
        If collateral > required (overcollateralised), shortfall is negative.
        Counterparty may recall excess — returned as negative call.
        """
        coll = self.required_coll_t0 + 50_000.0  # 50k excess
        mc = self.sft.margin_call(coll, t=0.0, threshold=0.0, minimum_transfer_amount=0.0)
        self.assertLess(mc, 0.0)

    def test_margin_call_at_exact_threshold(self):
        """At exactly the threshold, no call is triggered (|shortfall| <= threshold)."""
        shortfall_target = 10_000.0
        coll = self.required_coll_t0 - shortfall_target
        mc = self.sft.margin_call(coll, t=0.0, threshold=shortfall_target, minimum_transfer_amount=0.0)
        self.assertEqual(mc, 0.0)

    def test_margin_call_with_both_threshold_and_mta(self):
        """
        Large shortfall passes both threshold and MTA filters: call is triggered.
        """
        coll = self.notional * 0.90  # significant shortfall
        mc = self.sft.margin_call(
            coll, t=0.0, threshold=5_000.0, minimum_transfer_amount=5_000.0,
        )
        self.assertGreater(mc, 0.0)

    def test_zero_threshold_zero_mta_returns_shortfall(self):
        """With no threshold or MTA, call equals exact shortfall."""
        coll = self.notional  # shortfall = required_coll - notional ≈ 20408
        mc = self.sft.margin_call(coll, t=0.0, threshold=0.0, minimum_transfer_amount=0.0)
        expected = self.required_coll_t0 - coll
        self.assertAlmostEqual(mc, expected, places=4)


class TestSFTPV(unittest.TestCase):
    """SFT full valuation (pv dict)."""

    def setUp(self):
        self.r_repo = 0.04
        self.r_disc = 0.03
        self.T = 1.0
        self.notional = 1_000_000.0
        self.h = 0.02

    def test_pv_dict_keys(self):
        """pv() output contains all required keys."""
        curve = flat_curve(self.r_disc)
        sft = SFT(notional=self.notional, repo_rate=self.r_repo,
                  tenor=self.T, haircut=self.h, is_repo=True)
        result = sft.pv(curve)
        for key in ("pv_cash_leg", "collateral_value", "initial_margin",
                    "repurchase_price", "net_pv"):
            self.assertIn(key, result)

    def test_pv_cash_leg_discounted_repurchase(self):
        """pv_cash_leg = repurchase_price * df(T)."""
        curve = flat_curve(self.r_disc)
        sft = SFT(notional=self.notional, repo_rate=self.r_repo,
                  tenor=self.T, haircut=self.h, is_repo=True)
        result = sft.pv(curve)
        expected_pv_cash = sft.repurchase_price() * np.exp(-self.r_disc * self.T)
        self.assertAlmostEqual(result["pv_cash_leg"], expected_pv_cash, places=4)

    def test_repo_vs_reverse_repo_sign(self):
        """
        Repo seller and reverse repo buyer have opposite net_pv signs
        (assuming equal terms).
        """
        curve = flat_curve(self.r_disc)
        repo = SFT(notional=self.notional, repo_rate=self.r_repo,
                   tenor=self.T, haircut=self.h, is_repo=True)
        rrepo = SFT(notional=self.notional, repo_rate=self.r_repo,
                    tenor=self.T, haircut=self.h, is_repo=False)
        pv_repo = repo.pv(curve)["net_pv"]
        pv_rrepo = rrepo.pv(curve)["net_pv"]
        self.assertAlmostEqual(pv_repo, -pv_rrepo, places=4)

    def test_pv_initial_margin_in_dict(self):
        """pv()['initial_margin'] matches standalone initial_margin()."""
        curve = flat_curve(self.r_disc)
        sft = SFT(notional=self.notional, repo_rate=self.r_repo,
                  tenor=self.T, haircut=self.h)
        result = sft.pv(curve)
        self.assertAlmostEqual(result["initial_margin"], sft.initial_margin(), places=8)

    def test_repr(self):
        """__repr__ contains side, rate, and tenor."""
        sft = SFT(notional=1e6, repo_rate=0.03, tenor=1.0, haircut=0.02, is_repo=True)
        r = repr(sft)
        self.assertIn("Repo", r)
        self.assertIn("1.0Y", r)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
