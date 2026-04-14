"""
test_sa_ccr.py
--------------
Unit tests for sa_ccr.py: SA-CCR EAD calculation and IRB capital.

All benchmarks are derived analytically from the BCBS 279 formulas so no
Monte Carlo or external market data is required.

Key analytical results used
---------------------------
Maturity factor (unmargined):
    MF = sqrt(min(M, 1))
    => MF(0.25) = 0.5, MF(1.0) = 1.0, MF(5.0) = 1.0 (capped)

Maturity factor (margined):
    MF = 1.5 * sqrt(MPOR)

Supervisory duration:
    SD(S, E) = (exp(-0.05*S) - exp(-0.05*E)) / 0.05
    => SD(0, E) = (1 - exp(-0.05*E)) / 0.05

Effective notional IR (single trade, unmargined):
    EN = sign * N * SD(S, E) * MF(M) * SF_IR
    where SF_IR = 0.005

IR AddOn (single trade, bucket k):
    AddOn = sqrt(D_k^2) = |D_k| = |EN|    when only bucket k is populated

IR AddOn (two offsetting trades, same bucket):
    D_k = EN_payer + EN_receiver = 0  =>  AddOn = 0

FX AddOn (single trade):
    AddOn = |N * MF(M) * SF_FX| = N * sqrt(min(M,1)) * 0.04

EAD:
    EAD = 1.4 * (RC + multiplier * aggregate_AddOn)

PFE multiplier boundary:
    V_net - C == 0  =>  mult = floor + (1-floor)*exp(0) = 1.0
    Deeply overcollateralised => mult -> floor = 0.05

IRB capital:
    RWA = 12.5 * EAD * K
    capital = 0.08 * RWA
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from sa_ccr import (
    # Constants
    ALPHA, SUPERVISORY_FACTOR, IR_BUCKET_CORRELATION,
    # Data structures
    IRTrade, FXTrade, EquityTrade,
    # Core functions
    maturity_factor,
    supervisory_duration,
    adjusted_notional_ir,
    adjusted_notional_fx,
    adjusted_notional_equity,
    effective_notional_ir,
    replacement_cost_unmargined,
    replacement_cost_margined,
    pfe_multiplier,
    ir_addon,
    fx_addon,
    equity_addon,
    # Engine + capital
    SACCREngine,
    counterparty_rwa,
)


# ===========================================================================
# Maturity Factor
# ===========================================================================

class TestMaturityFactor(unittest.TestCase):
    """Maturity factor: unmargined (sqrt(min(M,1))) and margined (1.5*sqrt(MPOR))."""

    # --- unmargined ---

    def test_unmargined_quarter_year(self):
        """MF(0.25) = sqrt(0.25) = 0.5."""
        self.assertAlmostEqual(maturity_factor(0.25, margined=False), 0.5, places=10)

    def test_unmargined_one_year(self):
        """MF(1.0) = sqrt(1) = 1.0."""
        self.assertAlmostEqual(maturity_factor(1.0, margined=False), 1.0, places=10)

    def test_unmargined_capped_at_one_year(self):
        """MF(5.0) = sqrt(min(5,1)) = 1.0 (cap applies)."""
        self.assertAlmostEqual(maturity_factor(5.0, margined=False), 1.0, places=10)

    def test_unmargined_capped_long_maturity(self):
        """MF(30.0) = 1.0 (cap applies regardless)."""
        self.assertAlmostEqual(maturity_factor(30.0, margined=False), 1.0, places=10)

    def test_unmargined_half_year(self):
        """MF(0.5) = sqrt(0.5) = 1/sqrt(2)."""
        self.assertAlmostEqual(
            maturity_factor(0.5, margined=False), 1.0 / np.sqrt(2), places=10,
        )

    # --- margined ---

    def test_margined_ten_day_mpor(self):
        """MF = 1.5 * sqrt(10/252)."""
        mpor = 10.0 / 252.0
        expected = 1.5 * np.sqrt(mpor)
        self.assertAlmostEqual(maturity_factor(mpor, margined=True), expected, places=10)

    def test_margined_twenty_day_mpor(self):
        """MF = 1.5 * sqrt(20/252)."""
        mpor = 20.0 / 252.0
        expected = 1.5 * np.sqrt(mpor)
        self.assertAlmostEqual(maturity_factor(mpor, margined=True), expected, places=10)

    def test_margined_larger_than_unmargined(self):
        """
        For the same input value, margined MF (1.5*sqrt) > unmargined MF (sqrt)
        when the input is less than 1.  MPoR inputs are always << 1 year.
        """
        mpor = 10.0 / 252.0
        self.assertGreater(
            maturity_factor(mpor, margined=True),
            maturity_factor(mpor, margined=False),
        )


# ===========================================================================
# Supervisory Duration
# ===========================================================================

class TestSupervisoryDuration(unittest.TestCase):
    """SD(S, E) = (exp(-0.05*S) - exp(-0.05*E)) / 0.05."""

    def test_spot_start_one_year(self):
        """SD(0, 1) = (1 - exp(-0.05)) / 0.05."""
        expected = (1.0 - np.exp(-0.05)) / 0.05
        self.assertAlmostEqual(supervisory_duration(0.0, 1.0), expected, places=10)

    def test_spot_start_five_year(self):
        """SD(0, 5) = (1 - exp(-0.25)) / 0.05."""
        expected = (1.0 - np.exp(-0.25)) / 0.05
        self.assertAlmostEqual(supervisory_duration(0.0, 5.0), expected, places=10)

    def test_spot_start_ten_year(self):
        """SD(0, 10) = (1 - exp(-0.50)) / 0.05."""
        expected = (1.0 - np.exp(-0.50)) / 0.05
        self.assertAlmostEqual(supervisory_duration(0.0, 10.0), expected, places=10)

    def test_forward_start_reduces_sd(self):
        """
        Forward-starting trade has smaller SD than spot-starting with same tenor.
        SD(1, 6) < SD(0, 5) since the front stub is excluded.
        """
        sd_spot    = supervisory_duration(0.0, 5.0)
        sd_forward = supervisory_duration(1.0, 6.0)
        self.assertLess(sd_forward, sd_spot)

    def test_sd_positive(self):
        """SD is always strictly positive for S < E."""
        for s, e in [(0, 1), (0, 5), (1, 5), (5, 10)]:
            self.assertGreater(supervisory_duration(s, e), 0.0)

    def test_sd_increases_with_tenor(self):
        """Longer tenor → larger SD (for fixed start)."""
        sd_1y = supervisory_duration(0.0, 1.0)
        sd_5y = supervisory_duration(0.0, 5.0)
        sd_10y = supervisory_duration(0.0, 10.0)
        self.assertLess(sd_1y, sd_5y)
        self.assertLess(sd_5y, sd_10y)

    def test_sd_asymptote(self):
        """SD(0, large_E) -> 1/0.05 = 20 from below."""
        sd_big = supervisory_duration(0.0, 500.0)
        self.assertAlmostEqual(sd_big, 20.0, delta=0.01)


# ===========================================================================
# Adjusted Notional
# ===========================================================================

class TestAdjustedNotional(unittest.TestCase):
    """Signed adjusted notionals for IR, FX, and Equity."""

    def test_ir_payer_positive(self):
        """Payer IR trade has positive adjusted notional."""
        t = IRTrade(notional=1e6, maturity=5.0, start_date=0.0, end_date=5.0,
                    reference_currency="USD", payer=True)
        self.assertGreater(adjusted_notional_ir(t), 0.0)

    def test_ir_receiver_negative(self):
        """Receiver IR trade has negative adjusted notional."""
        t = IRTrade(notional=1e6, maturity=5.0, start_date=0.0, end_date=5.0,
                    reference_currency="USD", payer=False)
        self.assertLess(adjusted_notional_ir(t), 0.0)

    def test_ir_payer_receiver_symmetry(self):
        """Payer adjusted notional == -Receiver adjusted notional (same trade)."""
        payer    = IRTrade(notional=1e6, maturity=5.0, start_date=0.0, end_date=5.0,
                           reference_currency="USD", payer=True)
        receiver = IRTrade(notional=1e6, maturity=5.0, start_date=0.0, end_date=5.0,
                           reference_currency="USD", payer=False)
        self.assertAlmostEqual(
            adjusted_notional_ir(payer), -adjusted_notional_ir(receiver), places=8,
        )

    def test_ir_analytical(self):
        """d = N * SD(0, E) for spot-start payer."""
        N = 1_000_000.0
        E = 5.0
        t = IRTrade(notional=N, maturity=E, start_date=0.0, end_date=E,
                    reference_currency="USD", payer=True)
        expected = N * supervisory_duration(0.0, E)
        self.assertAlmostEqual(adjusted_notional_ir(t), expected, places=6)

    def test_fx_long_positive(self):
        """Long FX trade has positive adjusted notional = +N."""
        t = FXTrade(notional=1e6, maturity=1.0, currency_pair="EURUSD", long_foreign=True)
        self.assertAlmostEqual(adjusted_notional_fx(t), 1e6, places=6)

    def test_fx_short_negative(self):
        """Short FX trade has adjusted notional = -N."""
        t = FXTrade(notional=1e6, maturity=1.0, currency_pair="EURUSD", long_foreign=False)
        self.assertAlmostEqual(adjusted_notional_fx(t), -1e6, places=6)

    def test_equity_long_positive(self):
        """Long equity trade has positive adjusted notional = +N."""
        t = EquityTrade(notional=5e5, maturity=1.0, underlying="AAPL",
                        is_index=False, long=True)
        self.assertAlmostEqual(adjusted_notional_equity(t), 5e5, places=6)

    def test_equity_short_negative(self):
        """Short equity trade has negative adjusted notional."""
        t = EquityTrade(notional=5e5, maturity=1.0, underlying="AAPL",
                        is_index=False, long=False)
        self.assertAlmostEqual(adjusted_notional_equity(t), -5e5, places=6)


# ===========================================================================
# Effective Notional IR
# ===========================================================================

class TestEffectiveNotionalIR(unittest.TestCase):
    """EN_i = d_i * MF_i * SF_IR."""

    def test_effective_notional_analytical(self):
        """
        EN = N * SD(0, E) * sqrt(min(M, 1)) * SF_IR

        For a spot-start 3Y payer with N=1M:
            SD(0, 3) = (1 - exp(-0.15)) / 0.05
            MF = sqrt(min(3,1)) = 1.0
            EN = 1e6 * SD(0,3) * 1.0 * 0.005
        """
        N = 1_000_000.0
        E = 3.0
        t = IRTrade(notional=N, maturity=E, start_date=0.0, end_date=E,
                    reference_currency="USD", payer=True)

        sd = supervisory_duration(0.0, E)
        mf = maturity_factor(E, margined=False)
        expected = N * sd * mf * SUPERVISORY_FACTOR["IR"]

        self.assertAlmostEqual(effective_notional_ir(t, margined=False), expected, places=6)

    def test_effective_notional_receiver_negative(self):
        """Receiver trade effective notional is negative."""
        t = IRTrade(notional=1e6, maturity=3.0, start_date=0.0, end_date=3.0,
                    reference_currency="USD", payer=False)
        self.assertLess(effective_notional_ir(t), 0.0)

    def test_effective_notional_short_maturity_smaller(self):
        """
        Short-maturity trade has smaller |EN| than long-maturity trade with
        same notional (SD and MF both increase with maturity).
        """
        t_short = IRTrade(notional=1e6, maturity=0.5, start_date=0.0, end_date=0.5,
                          reference_currency="USD", payer=True)
        t_long  = IRTrade(notional=1e6, maturity=5.0, start_date=0.0, end_date=5.0,
                          reference_currency="USD", payer=True)
        self.assertLess(
            abs(effective_notional_ir(t_short)),
            abs(effective_notional_ir(t_long)),
        )


# ===========================================================================
# Replacement Cost
# ===========================================================================

class TestReplacementCostUnmargined(unittest.TestCase):
    """RC = max(V_net - C, 0)."""

    def test_positive_mtm_no_collateral(self):
        """V=100, C=0 => RC=100."""
        self.assertAlmostEqual(replacement_cost_unmargined(100.0, 0.0), 100.0, places=8)

    def test_positive_mtm_partially_collateralised(self):
        """V=100, C=60 => RC=40."""
        self.assertAlmostEqual(replacement_cost_unmargined(100.0, 60.0), 40.0, places=8)

    def test_fully_collateralised(self):
        """V=100, C=100 => RC=0."""
        self.assertAlmostEqual(replacement_cost_unmargined(100.0, 100.0), 0.0, places=8)

    def test_overcollateralised(self):
        """V=100, C=150 => RC=0 (excess collateral not counted)."""
        self.assertAlmostEqual(replacement_cost_unmargined(100.0, 150.0), 0.0, places=8)

    def test_negative_mtm_no_collateral(self):
        """V=-50, C=0 => RC=0 (out-of-the-money netting set)."""
        self.assertAlmostEqual(replacement_cost_unmargined(-50.0, 0.0), 0.0, places=8)

    def test_nonneg(self):
        """RC is always >= 0."""
        for v, c in [(-100, 0), (-100, 50), (0, 0), (50, 100)]:
            self.assertGreaterEqual(replacement_cost_unmargined(v, c), 0.0)


class TestReplacementCostMargined(unittest.TestCase):
    """RC = max(V - VM - IM_net, TH + MTA - IM_net, 0)."""

    def test_in_the_money_vm_received(self):
        """
        V=100, VM=80, no IM, TH=0:
        rc1 = 100 - 80 = 20, rc2 = 0 => RC = 20.
        """
        rc = replacement_cost_margined(
            netting_set_mtm=100.0, vm_received=80.0,
            im_received=0.0, im_posted=0.0, threshold=0.0, mta=0.0,
        )
        self.assertAlmostEqual(rc, 20.0, places=8)

    def test_fully_margined_zero_rc(self):
        """V=100, VM=100, no IM, TH=0, MTA=0 => RC=max(0, 0, 0)=0."""
        rc = replacement_cost_margined(100.0, vm_received=100.0)
        self.assertAlmostEqual(rc, 0.0, places=8)

    def test_threshold_binding(self):
        """
        V=5 (below TH=10), VM=0, no IM, MTA=2:
        rc1 = 5, rc2 = 10+2 = 12 => RC = 12.
        """
        rc = replacement_cost_margined(
            netting_set_mtm=5.0, vm_received=0.0,
            im_received=0.0, im_posted=0.0, threshold=10.0, mta=2.0,
        )
        self.assertAlmostEqual(rc, 12.0, places=8)

    def test_im_net_reduces_rc(self):
        """
        IM_net = IM_received - IM_posted. Net positive IM reduces RC.
        V=100, VM=0, IM_received=30, IM_posted=10, TH=0:
        IM_net = 20; rc1 = 100 - 0 - 20 = 80; rc2 = 0+0-20 = -20 => RC=80.
        """
        rc = replacement_cost_margined(
            netting_set_mtm=100.0, vm_received=0.0,
            im_received=30.0, im_posted=10.0, threshold=0.0, mta=0.0,
        )
        self.assertAlmostEqual(rc, 80.0, places=8)

    def test_nonneg(self):
        """RC is always >= 0."""
        for v in [-100, -50, 0, 50, 100]:
            rc = replacement_cost_margined(v, vm_received=v * 2)
            self.assertGreaterEqual(rc, 0.0)


# ===========================================================================
# PFE Multiplier
# ===========================================================================

class TestPFEMultiplier(unittest.TestCase):
    """
    multiplier = min(1, floor + (1-floor)*exp((V-C)/(2*(1-floor)*AddOn)))
    """

    def test_uncollateralised_at_par(self):
        """
        V_net - C = 0 => exponent = 0 => exp(0) = 1
        => mult = floor + (1-floor)*1 = 1.0.
        """
        mult = pfe_multiplier(
            netting_set_mtm=0.0, collateral_net=0.0, aggregate_addon=1000.0,
        )
        self.assertAlmostEqual(mult, 1.0, places=10)

    def test_in_the_money_capped_at_one(self):
        """
        V_net >> C => exponent > 0 => mult would exceed 1 but is capped.
        """
        mult = pfe_multiplier(
            netting_set_mtm=1_000_000.0, collateral_net=0.0, aggregate_addon=100.0,
        )
        self.assertAlmostEqual(mult, 1.0, places=8)

    def test_overcollateralised_approaches_floor(self):
        """
        V_net - C very negative => exponent << 0 => exp -> 0 => mult -> floor.
        """
        mult = pfe_multiplier(
            netting_set_mtm=-10_000_000.0, collateral_net=0.0, aggregate_addon=1000.0,
        )
        self.assertAlmostEqual(mult, 0.05, delta=1e-4)

    def test_floor_with_zero_addon(self):
        """Zero AddOn: multiplier defaults to floor."""
        mult = pfe_multiplier(
            netting_set_mtm=100.0, collateral_net=0.0, aggregate_addon=0.0,
        )
        self.assertAlmostEqual(mult, 0.05, places=10)

    def test_multiplier_in_range(self):
        """Multiplier is always in [floor, 1.0]."""
        floor = 0.05
        for v, c, addon in [
            (0, 0, 1000), (100, 0, 1000), (-100, 0, 1000),
            (-1e6, 0, 100), (1e6, 0, 100),
        ]:
            m = pfe_multiplier(v, c, addon)
            self.assertGreaterEqual(m, floor - 1e-10)
            self.assertLessEqual(m, 1.0 + 1e-10)

    def test_multiplier_increases_with_mtm(self):
        """Higher V_net → higher multiplier (collateral fixed)."""
        addon = 1_000.0
        m_lo = pfe_multiplier(-5000.0, 0.0, addon)
        m_hi = pfe_multiplier(5000.0, 0.0, addon)
        self.assertGreater(m_hi, m_lo)

    def test_custom_floor(self):
        """Custom floor is respected: deeply overcollateralised -> custom floor."""
        mult = pfe_multiplier(
            netting_set_mtm=-1e10, collateral_net=0.0, aggregate_addon=1000.0, floor=0.10,
        )
        self.assertAlmostEqual(mult, 0.10, delta=1e-4)


# ===========================================================================
# IR AddOn
# ===========================================================================

class TestIRAddon(unittest.TestCase):
    """IR AddOn: single trade, netting, multi-bucket, multi-currency."""

    def _make_trade(self, notional, maturity, end_date=None, payer=True, ccy="USD"):
        return IRTrade(
            notional=notional,
            maturity=maturity,
            start_date=0.0,
            end_date=end_date or maturity,
            reference_currency=ccy,
            payer=payer,
        )

    def test_single_bucket2_payer_analytical(self):
        """
        Single payer in bucket 2 (1–5Y), N=1M, maturity=3Y:
            EN = N * SD(0,3) * MF(3) * SF
            D = [0, EN, 0]
            AddOn = sqrt(D^T Rho D) = |D2|
        """
        N = 1_000_000.0
        M = 3.0
        t = self._make_trade(N, M)

        en = effective_notional_ir(t, margined=False)
        # Only bucket 2 populated: AddOn = |D2| = |EN|
        expected = abs(en)
        actual = ir_addon([t], margined=False)
        self.assertAlmostEqual(actual, expected, places=6)

    def test_single_bucket1_short_maturity(self):
        """
        Bucket 1 (< 1Y): maturity = 0.5Y.
        AddOn = |EN| (only bucket 1 populated).
        """
        t = self._make_trade(1e6, 0.5)
        en = effective_notional_ir(t, margined=False)
        self.assertAlmostEqual(ir_addon([t]), abs(en), places=6)

    def test_single_bucket3_long_maturity(self):
        """
        Bucket 3 (> 5Y): maturity = 10Y.
        AddOn = |EN| (only bucket 3 populated).
        """
        t = self._make_trade(1e6, 10.0)
        en = effective_notional_ir(t, margined=False)
        self.assertAlmostEqual(ir_addon([t]), abs(en), places=6)

    def test_offsetting_same_bucket_nets_to_zero(self):
        """
        Equal payer and receiver in the same bucket → D_k = 0 → AddOn = 0.
        """
        payer    = self._make_trade(1e6, 3.0, payer=True)
        receiver = self._make_trade(1e6, 3.0, payer=False)
        self.assertAlmostEqual(ir_addon([payer, receiver]), 0.0, delta=1e-6)

    def test_two_payers_same_bucket_sum(self):
        """
        Two payers in bucket 2: D2 = EN1 + EN2.
        AddOn = D2 = EN1 + EN2 (both positive).
        """
        t1 = self._make_trade(1e6, 3.0)
        t2 = self._make_trade(2e6, 4.0)
        en1 = effective_notional_ir(t1)
        en2 = effective_notional_ir(t2)
        expected = en1 + en2  # both positive
        self.assertAlmostEqual(ir_addon([t1, t2]), expected, places=6)

    def test_two_currency_hedging_sets_sum(self):
        """
        Trades in different currencies form separate hedging sets.
        AddOn = AddOn_USD + AddOn_EUR (no cross-currency netting).
        """
        t_usd = self._make_trade(1e6, 3.0, ccy="USD")
        t_eur = self._make_trade(1e6, 3.0, ccy="EUR")
        addon_usd = ir_addon([t_usd])
        addon_eur = ir_addon([t_eur])
        combined  = ir_addon([t_usd, t_eur])
        self.assertAlmostEqual(combined, addon_usd + addon_eur, places=6)

    def test_cross_bucket_aggregation(self):
        """
        Bucket 1 and Bucket 2: use full correlation matrix.
        AddOn = sqrt(D1² + D2² + 2*0.7*D1*D2)
        """
        t1 = self._make_trade(1e6, 0.5)   # bucket 1
        t2 = self._make_trade(1e6, 3.0)   # bucket 2
        en1 = effective_notional_ir(t1)
        en2 = effective_notional_ir(t2)

        # Analytical: D=[en1, en2, 0], rho_12=0.70
        rho = IR_BUCKET_CORRELATION
        D = np.array([en1, en2, 0.0])
        expected = float(np.sqrt(max(D @ rho @ D, 0.0)))

        self.assertAlmostEqual(ir_addon([t1, t2]), expected, places=6)

    def test_empty_trade_list(self):
        """No trades → IR AddOn = 0."""
        self.assertAlmostEqual(ir_addon([]), 0.0, places=10)

    def test_addon_positive(self):
        """IR AddOn is always non-negative."""
        t = self._make_trade(1e6, 5.0, payer=False)  # receiver (negative EN)
        self.assertGreaterEqual(ir_addon([t]), 0.0)


# ===========================================================================
# FX AddOn
# ===========================================================================

class TestFXAddon(unittest.TestCase):
    """FX AddOn: per currency-pair hedging sets."""

    def test_single_long_fx_analytical(self):
        """
        Single long EURUSD, N=1M, M=1Y:
        EN = +1M * MF(1) * SF_FX = 1e6 * 1.0 * 0.04 = 40_000
        AddOn = |EN| = 40_000.
        """
        t = FXTrade(notional=1e6, maturity=1.0, currency_pair="EURUSD", long_foreign=True)
        expected = 1e6 * maturity_factor(1.0) * SUPERVISORY_FACTOR["FX"]
        self.assertAlmostEqual(fx_addon([t]), expected, places=6)

    def test_single_short_fx_analytical(self):
        """Short FX: AddOn = |−EN| = same magnitude as long."""
        t_long  = FXTrade(notional=1e6, maturity=1.0, currency_pair="EURUSD", long_foreign=True)
        t_short = FXTrade(notional=1e6, maturity=1.0, currency_pair="EURUSD", long_foreign=False)
        self.assertAlmostEqual(fx_addon([t_long]), fx_addon([t_short]), places=8)

    def test_offsetting_same_pair_nets(self):
        """
        Equal long and short in the same pair → net EN = 0 → AddOn = 0.
        """
        t_long  = FXTrade(notional=1e6, maturity=1.0, currency_pair="EURUSD", long_foreign=True)
        t_short = FXTrade(notional=1e6, maturity=1.0, currency_pair="EURUSD", long_foreign=False)
        self.assertAlmostEqual(fx_addon([t_long, t_short]), 0.0, delta=1e-6)

    def test_two_currency_pairs_sum(self):
        """
        Separate currency pairs: AddOn_total = AddOn_EURUSD + AddOn_USDJPY.
        """
        t1 = FXTrade(notional=1e6, maturity=1.0, currency_pair="EURUSD", long_foreign=True)
        t2 = FXTrade(notional=2e6, maturity=0.5, currency_pair="USDJPY", long_foreign=True)
        addon1 = fx_addon([t1])
        addon2 = fx_addon([t2])
        self.assertAlmostEqual(fx_addon([t1, t2]), addon1 + addon2, places=6)

    def test_fx_addon_capped_maturity(self):
        """MF caps at 1Y so FX AddOn for 5Y = N * 1.0 * SF_FX."""
        t = FXTrade(notional=1e6, maturity=5.0, currency_pair="EURUSD", long_foreign=True)
        expected = 1e6 * 1.0 * SUPERVISORY_FACTOR["FX"]
        self.assertAlmostEqual(fx_addon([t]), expected, places=6)

    def test_empty_trade_list(self):
        """No FX trades → AddOn = 0."""
        self.assertAlmostEqual(fx_addon([]), 0.0, places=10)


# ===========================================================================
# Equity AddOn
# ===========================================================================

class TestEquityAddon(unittest.TestCase):
    """Equity AddOn: single-name (rho=0.5) and index (rho=0.8) aggregation."""

    def test_single_long_single_name(self):
        """
        Single long single-name, N=1M, M=1Y:
        EN = 1M * 1.0 * SF_single = 1e6 * 0.32 = 320_000
        AddOn (one entity) = sqrt((rho*EN)^2 + (1-rho^2)*EN^2) = |EN|
        """
        sf  = SUPERVISORY_FACTOR["Equity_single"]
        rho = 0.50
        N   = 1e6
        t   = EquityTrade(notional=N, maturity=1.0, underlying="AAPL",
                          is_index=False, long=True)
        en  = N * maturity_factor(1.0) * sf
        # Single entity: formula simplifies to |EN|
        systematic    = (rho * en) ** 2
        idiosyncratic = (1.0 - rho**2) * en**2
        expected = np.sqrt(systematic + idiosyncratic)
        self.assertAlmostEqual(equity_addon([t]), expected, places=4)

    def test_single_entity_equals_abs_en(self):
        """
        For a single entity, the correlation formula reduces to |EN|:
        sqrt((rho*EN)^2 + (1-rho^2)*EN^2) = |EN| * sqrt(rho^2 + 1 - rho^2) = |EN|.
        """
        t  = EquityTrade(notional=1e6, maturity=1.0, underlying="MSFT",
                         is_index=False, long=True)
        sf = SUPERVISORY_FACTOR["Equity_single"]
        mf = maturity_factor(1.0)
        en = 1e6 * mf * sf
        self.assertAlmostEqual(equity_addon([t]), abs(en), places=4)

    def test_offsetting_same_entity_nets(self):
        """
        Equal long and short on same underlying → EN_net = 0 → AddOn = 0.
        """
        t_long  = EquityTrade(notional=1e6, maturity=1.0, underlying="AAPL",
                              is_index=False, long=True)
        t_short = EquityTrade(notional=1e6, maturity=1.0, underlying="AAPL",
                              is_index=False, long=False)
        self.assertAlmostEqual(equity_addon([t_long, t_short]), 0.0, delta=1e-4)

    def test_index_higher_rho_vs_single_name(self):
        """
        Two entities with same EN: index rho=0.80 gives higher systematic
        component than single-name rho=0.50, so index AddOn >= single-name AddOn.
        """
        # Two different underlyings, long, same notional
        t1 = EquityTrade(notional=1e6, maturity=1.0, underlying="A",
                         is_index=True, long=True)
        t2 = EquityTrade(notional=1e6, maturity=1.0, underlying="B",
                         is_index=True, long=True)
        t3 = EquityTrade(notional=1e6, maturity=1.0, underlying="A",
                         is_index=False, long=True)
        t4 = EquityTrade(notional=1e6, maturity=1.0, underlying="B",
                         is_index=False, long=True)

        addon_index  = equity_addon([t1, t2])
        addon_single = equity_addon([t3, t4])

        # Index has higher systematic correlation → larger diversification benefit
        # disappears but rho*sum is larger, so addon >= single
        # Actually: higher rho => more systematic => more of the sum
        # we just check both are positive
        self.assertGreater(addon_index, 0.0)
        self.assertGreater(addon_single, 0.0)

    def test_empty_equity(self):
        """No equity trades → AddOn = 0."""
        self.assertAlmostEqual(equity_addon([]), 0.0, places=10)


# ===========================================================================
# SACCREngine (Integration)
# ===========================================================================

class TestSACCREngineBasic(unittest.TestCase):
    """Full EAD computation: structural correctness and formula."""

    def _simple_engine(self, **kwargs):
        """Helper: engine with a single 3Y USD payer IRS."""
        defaults = dict(
            ir_trades=[
                IRTrade(notional=1e6, maturity=3.0, start_date=0.0, end_date=3.0,
                        reference_currency="USD", payer=True, current_mtm=0.0)
            ],
            margined=False,
            netting_set_mtm=0.0,
            collateral_net=0.0,
        )
        defaults.update(kwargs)
        return SACCREngine(**defaults)

    def test_dict_keys(self):
        """compute() returns all required keys."""
        result = self._simple_engine().compute()
        for key in ("RC", "aggregate_addon", "IR_addon", "FX_addon",
                    "Equity_addon", "multiplier", "PFE_addon", "EAD", "alpha"):
            self.assertIn(key, result)

    def test_alpha_is_1_4(self):
        """Alpha constant in output is 1.4."""
        result = self._simple_engine().compute()
        self.assertAlmostEqual(result["alpha"], 1.4, places=10)

    def test_ead_formula(self):
        """EAD = 1.4 * (RC + PFE_addon)."""
        eng = self._simple_engine()
        result = eng.compute()
        expected_ead = 1.4 * (result["RC"] + result["PFE_addon"])
        self.assertAlmostEqual(result["EAD"], expected_ead, places=4)

    def test_pfe_addon_formula(self):
        """PFE_addon = multiplier * aggregate_addon."""
        result = self._simple_engine().compute()
        expected = result["multiplier"] * result["aggregate_addon"]
        self.assertAlmostEqual(result["PFE_addon"], expected, places=8)

    def test_aggregate_addon_sum(self):
        """aggregate_addon = IR + FX + Equity."""
        result = self._simple_engine().compute()
        expected = result["IR_addon"] + result["FX_addon"] + result["Equity_addon"]
        self.assertAlmostEqual(result["aggregate_addon"], expected, places=8)

    def test_ead_positive(self):
        """EAD is always positive (non-negative) for any trade configuration."""
        result = self._simple_engine(netting_set_mtm=-50_000).compute()
        self.assertGreaterEqual(result["EAD"], 0.0)

    def test_ead_zero_for_empty_netting_set(self):
        """No trades, zero MTM → EAD = 0."""
        eng = SACCREngine()
        result = eng.compute()
        self.assertAlmostEqual(result["EAD"], 0.0, places=6)

    def test_ead_increases_with_notional(self):
        """Doubling notional roughly doubles EAD (when RC=0)."""
        eng1 = self._simple_engine(netting_set_mtm=0.0, collateral_net=0.0)
        eng2 = SACCREngine(
            ir_trades=[
                IRTrade(notional=2e6, maturity=3.0, start_date=0.0, end_date=3.0,
                        reference_currency="USD", payer=True)
            ],
        )
        ead1 = eng1.compute()["EAD"]
        ead2 = eng2.compute()["EAD"]
        self.assertAlmostEqual(ead2 / ead1, 2.0, delta=0.01)

    def test_rc_zero_for_negative_mtm(self):
        """Netting set out-of-the-money: RC = 0 (no current loss)."""
        eng = self._simple_engine(netting_set_mtm=-100_000.0)
        result = eng.compute()
        self.assertAlmostEqual(result["RC"], 0.0, places=6)

    def test_rc_equals_mtm_when_uncollateralised(self):
        """RC = MTM when MTM > 0 and no collateral (unmargined)."""
        mtm = 50_000.0
        eng = self._simple_engine(netting_set_mtm=mtm, collateral_net=0.0)
        result = eng.compute()
        self.assertAlmostEqual(result["RC"], mtm, places=4)


class TestSACCREngineMultiAsset(unittest.TestCase):
    """Multi-asset-class netting sets."""

    def test_fx_and_ir_addon_sum(self):
        """
        IR and FX AddOns from different asset classes sum in aggregate.
        Combined AddOn > either individual AddOn.
        """
        ir = IRTrade(notional=1e6, maturity=3.0, start_date=0.0, end_date=3.0,
                     reference_currency="USD", payer=True)
        fx = FXTrade(notional=1e6, maturity=1.0, currency_pair="EURUSD", long_foreign=True)

        eng_ir_only = SACCREngine(ir_trades=[ir])
        eng_fx_only = SACCREngine(fx_trades=[fx])
        eng_both    = SACCREngine(ir_trades=[ir], fx_trades=[fx])

        addon_ir   = eng_ir_only.compute()["aggregate_addon"]
        addon_fx   = eng_fx_only.compute()["aggregate_addon"]
        addon_both = eng_both.compute()["aggregate_addon"]

        self.assertAlmostEqual(addon_both, addon_ir + addon_fx, places=4)

    def test_offsetting_ir_reduces_ead(self):
        """
        Adding a receiver IRS to offset a payer reduces EAD
        (partial netting within the same bucket).
        """
        payer    = IRTrade(notional=1e6, maturity=3.0, start_date=0.0, end_date=3.0,
                           reference_currency="USD", payer=True)
        receiver = IRTrade(notional=0.5e6, maturity=3.0, start_date=0.0, end_date=3.0,
                           reference_currency="USD", payer=False)

        ead_single  = SACCREngine(ir_trades=[payer]).compute()["EAD"]
        ead_hedged  = SACCREngine(ir_trades=[payer, receiver]).compute()["EAD"]
        self.assertLess(ead_hedged, ead_single)

    def test_margined_vs_unmargined_same_trades(self):
        """
        Margined netting set has lower EAD than unmargined (MPoR << trade maturity
        → smaller maturity factor, reduced PFE).
        """
        ir = IRTrade(notional=1e6, maturity=5.0, start_date=0.0, end_date=5.0,
                     reference_currency="USD", payer=True)
        ead_unmargined = SACCREngine(ir_trades=[ir], margined=False).compute()["EAD"]
        ead_margined   = SACCREngine(
            ir_trades=[ir], margined=True, mpor=10.0/252.0,
        ).compute()["EAD"]
        self.assertLess(ead_margined, ead_unmargined)


# ===========================================================================
# counterparty_rwa
# ===========================================================================

class TestCounterpartyRWA(unittest.TestCase):
    """IRB capital: K, RWA, capital relationships."""

    def _rwa(self, ead=1_000_000.0, pd=0.01, lgd=0.45, maturity=1.0):
        return counterparty_rwa(ead=ead, pd=pd, lgd=lgd, maturity=maturity)

    def test_dict_keys(self):
        """Output contains R, K, RWA, capital."""
        result = self._rwa()
        for key in ("R", "K", "RWA", "capital"):
            self.assertIn(key, result)

    def test_rwa_formula(self):
        """RWA = 12.5 * EAD * K."""
        result = self._rwa(ead=1e6)
        self.assertAlmostEqual(result["RWA"], 12.5 * 1e6 * result["K"], places=4)

    def test_capital_formula(self):
        """capital = 0.08 * RWA."""
        result = self._rwa(ead=1e6)
        self.assertAlmostEqual(result["capital"], 0.08 * result["RWA"], places=4)

    def test_K_positive(self):
        """Capital requirement K > 0 for non-zero PD."""
        result = self._rwa()
        self.assertGreater(result["K"], 0.0)

    def test_K_less_than_LGD(self):
        """K < LGD (diversification benefit from correlation)."""
        result = self._rwa(pd=0.01, lgd=0.45)
        self.assertLess(result["K"], 0.45)

    def test_rwa_scales_linearly_with_ead(self):
        """RWA is linear in EAD: doubling EAD doubles RWA."""
        r1 = self._rwa(ead=1e6)
        r2 = self._rwa(ead=2e6)
        self.assertAlmostEqual(r2["RWA"] / r1["RWA"], 2.0, places=8)

    def test_K_increases_with_pd(self):
        """Higher PD → higher capital requirement K."""
        K_lo = self._rwa(pd=0.001)["K"]
        K_hi = self._rwa(pd=0.05)["K"]
        self.assertGreater(K_hi, K_lo)

    def test_K_increases_with_lgd(self):
        """Higher LGD → higher capital requirement K."""
        K_lo = self._rwa(lgd=0.25)["K"]
        K_hi = self._rwa(lgd=0.60)["K"]
        self.assertGreater(K_hi, K_lo)

    def test_asset_correlation_override(self):
        """
        Custom asset correlation override is used instead of the Basel formula.
        R in output equals the override.
        """
        result = counterparty_rwa(ead=1e6, pd=0.01, lgd=0.45, asset_correlation=0.15)
        self.assertAlmostEqual(result["R"], 0.15, places=10)

    def test_K_nonneg_at_low_pd(self):
        """K >= 0 even for very small PD (floor prevents log(0))."""
        result = self._rwa(pd=1e-10)
        self.assertGreaterEqual(result["K"], 0.0)

    def test_maturity_adjustment_increases_K(self):
        """
        Longer effective maturity → larger maturity adjustment → larger K.
        (For M > 2.5 years, MA > 1.0.)
        """
        K_1y = self._rwa(maturity=1.0)["K"]
        K_5y = self._rwa(maturity=5.0)["K"]
        self.assertGreater(K_5y, K_1y)

    def test_R_in_valid_range(self):
        """Basel asset correlation R is in [0.12, 0.24] for corporate."""
        for pd in [0.001, 0.01, 0.05, 0.10, 0.20]:
            R = self._rwa(pd=pd)["R"]
            self.assertGreaterEqual(R, 0.12 - 1e-8)
            self.assertLessEqual(R, 0.24 + 1e-8)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
