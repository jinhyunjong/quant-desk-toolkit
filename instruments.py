
"""
instruments.py
--------------
Financial instrument definitions and pricing for the Quant Desk Toolkit.

Implements:
  - InterestRateSwap : IRS valuation under the multi-curve framework
  - Bond             : Fixed coupon bond pricing with YTM and accrued interest
  - SFT              : Securities financing transaction (repo / sec lending)

All instruments consume Curve objects from curve_factory.py.
Pricing follows market conventions:
  - Discounting  : OIS curve (CSA-consistent)
  - Projection   : SOFR curve (floating leg forward rates)
  - Conventions  : Act/360 or Act/365 passed via day_count parameter
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from curve_factory import Curve
from common_utils.math_helpers import brent_solver


# =============================================================================
# INTEREST RATE SWAP
# =============================================================================

@dataclass
class InterestRateSwap:
    """
    Vanilla interest rate swap under the multi-curve OIS/SOFR framework.

    Conventions
    -----------
    - Payer IRS  : pay fixed, receive floating  → PV = PV_float - PV_fixed
    - Receiver IRS: receive fixed, pay floating → PV = PV_fixed - PV_float
    - Fixed leg  : semi-annual or annual, Act/360
    - Floating leg: quarterly or semi-annual, Act/360, indexed to SOFR

    Parameters
    ----------
    notional : float
        Notional amount in currency units.
    fixed_rate : float
        Fixed coupon rate (annualised, e.g. 0.05 for 5%).
    tenor : float
        Swap maturity in years.
    payment_frequency : int
        Number of fixed leg payments per year (e.g. 2 = semi-annual).
    float_frequency : int
        Number of floating leg payments per year (e.g. 4 = quarterly).
    payer : bool
        If True, this is a payer IRS (pay fixed, receive float).
        If False, receiver IRS (receive fixed, pay float).
    start : float
        Trade start in years from today. Default 0 (spot starting).
    day_count : float
        Day count multiplier. Default 1.0 (adjust externally if needed).
    """

    notional        : float
    fixed_rate      : float
    tenor           : float
    payment_frequency : int   = 2
    float_frequency : int     = 4
    payer           : bool    = True
    start           : float   = 0.0
    day_count       : float   = 1.0

    def _fixed_payment_dates(self) -> np.ndarray:
        """Generate fixed leg payment schedule."""
        step = 1.0 / self.payment_frequency
        dates = np.arange(
            self.start + step,
            self.start + self.tenor + 1e-9,
            step,
        )
        return np.round(dates, 8)

    def _float_payment_dates(self) -> np.ndarray:
        """Generate floating leg payment schedule."""
        step = 1.0 / self.float_frequency
        dates = np.arange(
            self.start + step,
            self.start + self.tenor + 1e-9,
            step,
        )
        return np.round(dates, 8)

    def pv_fixed_leg(self, discount_curve: Curve) -> float:
        """
        Present value of the fixed leg.

        PV_fixed = K * N * sum_i [ tau_i * P(0, T_i) ]

        Parameters
        ----------
        discount_curve : Curve
            OIS discount curve.

        Returns
        -------
        float
            PV of fixed leg (positive = asset).
        """
        dates      = self._fixed_payment_dates()
        prev_dates = np.insert(dates[:-1], 0, self.start)
        tau        = (dates - prev_dates) * self.day_count
        dfs        = discount_curve.df(dates)
        return float(self.fixed_rate * self.notional * np.sum(tau * dfs))

    def pv_floating_leg(
        self,
        discount_curve: Curve,
        projection_curve: Curve,
    ) -> float:
        """
        Present value of the floating leg using exact discrete discount factor ratios.

        PV_float = N * sum_i [ (P_SOFR(0, T_{i-1}) / P_SOFR(0, T_i) - 1) * P_OIS(0, T_i) ]

        This avoids compounding convention assumptions and numerical approximation 
        errors associated with extracting and multiplying forward rates.

        Parameters
        ----------
        discount_curve : Curve
            OIS curve for discounting.
        projection_curve : Curve
            SOFR curve for projecting forward rates.

        Returns
        -------
        float
            PV of floating leg (positive = asset).
        """
        dates      = self._float_payment_dates()
        prev_dates = np.insert(dates[:-1], 0, self.start)

        pv = 0.0
        for t_prev, t_i in zip(prev_dates, dates):
            df_sofr_prev = float(projection_curve.df(t_prev))
            df_sofr_curr = float(projection_curve.df(t_i))
            df_ois       = float(discount_curve.df(t_i))
            
            # Exact discrete floating payment PV
            pv += (df_sofr_prev / df_sofr_curr - 1.0) * df_ois

        return float(self.notional * pv)

    def pv(
        self,
        discount_curve: Curve,
        projection_curve: Curve,
    ) -> dict:
        """
        Full mark-to-market valuation of the IRS.

        Parameters
        ----------
        discount_curve : Curve
            OIS discount curve.
        projection_curve : Curve
            SOFR projection curve.

        Returns
        -------
        dict with keys:
            pv_fixed   : float — PV of fixed leg
            pv_float   : float — PV of floating leg
            pv_net     : float — Net PV from this counterparty's perspective
            par_rate   : float — Current par swap rate
        """
        pv_fixed = self.pv_fixed_leg(discount_curve)
        pv_float = self.pv_floating_leg(discount_curve, projection_curve)

        # Net PV: payer pays fixed (negative), receives float (positive)
        sign    = 1.0 if self.payer else -1.0
        pv_net  = sign * (pv_float - pv_fixed)

        par = self.par_rate(discount_curve, projection_curve)

        return {
            "pv_fixed"  : pv_fixed,
            "pv_float"  : pv_float,
            "pv_net"    : pv_net,
            "par_rate"  : par,
        }

    def par_rate(
        self,
        discount_curve: Curve,
        projection_curve: Curve,
    ) -> float:
        """
        Compute the par fixed rate that makes the swap PV zero.

        K_par = PV_float / (N * annuity)

        Parameters
        ----------
        discount_curve : Curve
        projection_curve : Curve

        Returns
        -------
        float
            Par swap rate.
        """
        dates      = self._fixed_payment_dates()
        prev_dates = np.insert(dates[:-1], 0, self.start)
        tau        = (dates - prev_dates) * self.day_count
        dfs        = discount_curve.df(dates)
        annuity    = float(np.sum(tau * dfs))

        pv_float   = self.pv_floating_leg(discount_curve, projection_curve)
        return float(pv_float / (self.notional * annuity))

    def __repr__(self) -> str:
        direction = "Payer" if self.payer else "Receiver"
        return (
            f"InterestRateSwap({direction}, N={self.notional:,.0f}, "
            f"K={self.fixed_rate:.4%}, T={self.tenor}Y)"
        )


# =============================================================================
# BOND
# =============================================================================

@dataclass
class Bond:
    """
    Fixed coupon bond pricing with clean/dirty price and YTM.

    Parameters
    ----------
    face_value : float
        Par / face value of the bond.
    coupon_rate : float
        Annual coupon rate (e.g. 0.05 for 5%).
    maturity : float
        Time to maturity in years.
    coupon_frequency : int
        Coupon payments per year. Default 2 (semi-annual).
    settlement : float
        Settlement date in years from today. Default 0.
    day_count : float
        Day count multiplier. Default 1.0.
    """

    face_value       : float
    coupon_rate      : float
    maturity         : float
    coupon_frequency : int   = 2
    settlement       : float = 0.0
    day_count        : float = 1.0

    def _coupon_dates(self) -> np.ndarray:
        """Generate coupon payment schedule from settlement to maturity."""
        step  = 1.0 / self.coupon_frequency
        dates = np.arange(
            self.settlement + step,
            self.maturity + 1e-9,
            step,
        )
        return np.round(dates, 8)

    def coupon_amount(self) -> float:
        """
        Periodic coupon cash flow.

        Returns
        -------
        float
            Coupon per period = face_value * coupon_rate / coupon_frequency.
        """
        return self.face_value * self.coupon_rate / self.coupon_frequency

    def dirty_price(self, discount_curve: Curve) -> float:
        """
        Full (dirty) price of the bond: PV of all future cash flows.

        P_dirty = sum_i [ C * P(0, T_i) ] + Face * P(0, T_n)

        Parameters
        ----------
        discount_curve : Curve
            OIS discount curve.

        Returns
        -------
        float
            Dirty price.
        """
        dates  = self._coupon_dates()
        dfs    = discount_curve.df(dates)
        coupon = self.coupon_amount()

        pv_coupons   = coupon * float(np.sum(dfs))
        pv_principal = self.face_value * float(discount_curve.df(self.maturity))

        return pv_coupons + pv_principal

    def accrued_interest(self, t_since_last_coupon: float) -> float:
        """
        Accrued interest since last coupon date.

        AI = face_value * coupon_rate * t_since_last_coupon

        Parameters
        ----------
        t_since_last_coupon : float
            Time elapsed since last coupon in years.

        Returns
        -------
        float
            Accrued interest.
        """
        return self.face_value * self.coupon_rate * t_since_last_coupon * self.day_count

    def clean_price(
        self,
        discount_curve: Curve,
        t_since_last_coupon: float = 0.0,
    ) -> float:
        """
        Clean price = dirty price minus accrued interest.

        Parameters
        ----------
        discount_curve : Curve
        t_since_last_coupon : float
            Time since last coupon in years.

        Returns
        -------
        float
            Clean price.
        """
        return (
            self.dirty_price(discount_curve)
            - self.accrued_interest(t_since_last_coupon)
        )

    def yield_to_maturity(self, discount_curve: Curve) -> float:
        """
        Solve for the yield to maturity (YTM) consistent with the dirty price.

        YTM is the flat continuously compounded rate y such that:

            P_dirty = sum_i C * exp(-y * T_i) + Face * exp(-y * T_n)

        Uses Brent's method from math_helpers.

        Parameters
        ----------
        discount_curve : Curve

        Returns
        -------
        float
            YTM as a continuously compounded rate.
        """
        target = self.dirty_price(discount_curve)
        dates  = self._coupon_dates()
        coupon = self.coupon_amount()

        def bond_pv(y: float) -> float:
            dfs  = np.exp(-y * dates)
            pv_c = coupon * float(np.sum(dfs))
            pv_p = self.face_value * np.exp(-y * self.maturity)
            return pv_c + pv_p - target

        return brent_solver(bond_pv, a=-0.20, b=1.0, tol=1e-10)

    def pv(self, discount_curve: Curve, t_since_last_coupon: float = 0.0) -> dict:
        """
        Full bond valuation output.

        Returns
        -------
        dict with keys:
            dirty_price       : float
            clean_price       : float
            accrued_interest  : float
            ytm               : float
        """
        dirty  = self.dirty_price(discount_curve)
        ai     = self.accrued_interest(t_since_last_coupon)
        ytm    = self.yield_to_maturity(discount_curve)

        return {
            "dirty_price"      : dirty,
            "clean_price"      : dirty - ai,
            "accrued_interest" : ai,
            "ytm"              : ytm,
        }

    def __repr__(self) -> str:
        return (
            f"Bond(Face={self.face_value:,.0f}, "
            f"Coupon={self.coupon_rate:.4%}, "
            f"Maturity={self.maturity}Y)"
        )


# =============================================================================
# SECURITIES FINANCING TRANSACTION (REPO / SEC LENDING)
# =============================================================================

@dataclass
class SFT:
    """
    Securities Financing Transaction — repo and securities lending.

    Models the cash flow structure and mark-to-market exposure of:
      - Classic repo (sell and buy-back)
      - Reverse repo (buy and sell-back)
      - Securities lending (lend securities, receive cash collateral)

    Haircut reduces the initial cash leg to create overcollateralisation,
    providing a buffer against collateral value deterioration.

    Parameters
    ----------
    notional : float
        Initial cash leg of the transaction (dirty price × quantity).
    repo_rate : float
        Agreed repo rate (annualised, continuously compounded).
    tenor : float
        Transaction maturity in years.
    haircut : float
        Collateral haircut as a decimal (e.g. 0.02 for 2%).
        Cash lent = collateral_value × (1 - haircut).
    is_repo : bool
        If True, this side is the cash borrower (repo seller).
        If False, this side is the cash lender (reverse repo buyer).
    collateral_value : float, optional
        Current mark-to-market value of the collateral securities.
        If None, defaults to notional / (1 - haircut).
    """

    notional          : float
    repo_rate         : float
    tenor             : float
    haircut           : float
    is_repo           : bool  = True
    collateral_value  : Optional[float] = None

    def __post_init__(self) -> None:
        if self.collateral_value is None:
            # Infer initial collateral value from cash and haircut
            self.collateral_value = self.notional / (1.0 - self.haircut)

    def initial_margin(self) -> float:
        """
        Initial margin (overcollateralisation buffer).

        IM = collateral_value × haircut

        Returns
        -------
        float
            Initial margin amount.
        """
        return self.collateral_value * self.haircut

    def repurchase_price(self) -> float:
        """
        Agreed repurchase price at maturity.

        P_repo = notional × exp(repo_rate × tenor)

        For continuously compounded repo rate convention.

        Returns
        -------
        float
            Repurchase price at tenor.
        """
        return self.notional * np.exp(self.repo_rate * self.tenor)

    def accrued_interest(self, t: float) -> float:
        """
        Accrued repo interest at time t.

        AI(t) = notional × (exp(repo_rate × t) - 1)

        Parameters
        ----------
        t : float
            Elapsed time in years (0 <= t <= tenor).

        Returns
        -------
        float
            Accrued repo interest.
        """
        if t < 0 or t > self.tenor + 1e-9:
            raise ValueError(f"t={t} outside [0, {self.tenor}].")
        return float(self.notional * (np.exp(self.repo_rate * t) - 1.0))

    def mtm_exposure(
        self,
        current_collateral_value: float,
        t: float,
    ) -> float:
        """
        Mark-to-market exposure at time t.

        Exposure is the net economic value at risk if the counterparty
        defaults at time t. Measured as the difference between what is
        owed and the current collateral value.

        For a repo seller (cash borrower):
            Exposure = max(cash_owed(t) - collateral_value, 0)

        For a reverse repo buyer (cash lender):
            Exposure = max(collateral_value - cash_owed(t), 0)

        Parameters
        ----------
        current_collateral_value : float
            Current market value of the collateral securities.
        t : float
            Current time in years.

        Returns
        -------
        float
            Positive exposure (loss given default of the counterparty).
        """
        cash_owed = self.notional * np.exp(self.repo_rate * t)

        if self.is_repo:
            # Cash borrower: exposed if collateral worth less than cash owed
            exposure = current_collateral_value - cash_owed
        else:
            # Cash lender: exposed if cash owed less than collateral delivered
            exposure = cash_owed - current_collateral_value

        return float(max(exposure, 0.0))

    def margin_call(
        self,
        current_collateral_value: float,
        t: float,
        threshold: float = 0.0,
        minimum_transfer_amount: float = 0.0,
    ) -> float:
        """
        Compute margin call amount triggered by collateral value changes.

        A margin call is triggered when the overcollateralisation falls
        below the required haircut buffer.

        Required collateral = cash_owed(t) / (1 - haircut)
        Shortfall = required_collateral - current_collateral_value

        Parameters
        ----------
        current_collateral_value : float
            Current MtM value of collateral.
        t : float
            Current time in years.
        threshold : float
            Minimum shortfall before a margin call is triggered.
        minimum_transfer_amount : float
            Minimum transfer size (MTA) — calls below this are waived.

        Returns
        -------
        float
            Margin call amount (positive = counterparty must post collateral,
            negative = counterparty may recall excess).
        """
        cash_owed            = self.notional * np.exp(self.repo_rate * t)
        required_collateral  = cash_owed / (1.0 - self.haircut)
        shortfall            = required_collateral - current_collateral_value

        if abs(shortfall) <= threshold:
            return 0.0
        if abs(shortfall) < minimum_transfer_amount:
            return 0.0

        return float(shortfall)

    def pv(
        self,
        discount_curve: Curve,
        current_collateral_value: Optional[float] = None,
    ) -> dict:
        """
        Full SFT valuation at current market.

        Parameters
        ----------
        discount_curve : Curve
            OIS discount curve for discounting the terminal cash flow.
        current_collateral_value : float, optional
            Current MtM of collateral. Defaults to initial collateral value.

        Returns
        -------
        dict with keys:
            pv_cash_leg        : float — PV of cash repayment
            collateral_value   : float — Current collateral MtM
            initial_margin     : float — Haircut buffer
            repurchase_price   : float — Agreed repurchase amount at maturity
            net_pv             : float — Net economic value of the position
        """
        coll_value    = current_collateral_value or self.collateral_value
        repo_price    = self.repurchase_price()
        pv_cash_leg   = repo_price * float(discount_curve.df(self.tenor))
        im            = self.initial_margin()

        # Net PV: collateral received minus PV of cash owed (for repo seller)
        sign   = 1.0 if self.is_repo else -1.0
        net_pv = sign * (coll_value - pv_cash_leg)

        return {
            "pv_cash_leg"      : pv_cash_leg,
            "collateral_value" : coll_value,
            "initial_margin"   : im,
            "repurchase_price" : repo_price,
            "net_pv"           : net_pv,
        }

    def __repr__(self) -> str:
        side = "Repo" if self.is_repo else "Reverse Repo"
        return (
            f"SFT({side}, Notional={self.notional:,.0f}, "
            f"Rate={self.repo_rate:.4%}, "
            f"Haircut={self.haircut:.2%}, "
            f"Tenor={self.tenor}Y)"
        )