from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_left
from typing import Sequence

import math


def discount_factor(rate: float, t: float) -> float:
    """
    Compute a discount factor under continuous compounding.

    Parameters
    ----------
    rate : float
        Continuously compounded annual rate.
    t : float
        Time to maturity in years.

    Returns
    -------
    float
        Discount factor exp(-r * t).
    """
    if t < 0:
        raise ValueError("Time t must be non-negative.")
    return math.exp(-rate * t)


def zero_rate(df: float, t: float) -> float:
    """
    Convert a discount factor into a continuously compounded zero rate.

    Parameters
    ----------
    df : float
        Discount factor.
    t : float
        Time to maturity in years.

    Returns
    -------
    float
        Continuously compounded zero rate.
    """
    if t <= 0:
        raise ValueError("Time t must be positive.")
    if df <= 0:
        raise ValueError("Discount factor must be positive.")
    return -math.log(df) / t


@dataclass(frozen=True)
class FlatYieldCurve:
    """
    Flat continuously compounded yield curve.

    Attributes
    ----------
    rate : float
        Flat annual continuously compounded rate.
    """

    rate: float

    def get_zero_rate(self, t: float) -> float:
        """
        Return the zero rate at time t.
        """
        if t < 0:
            raise ValueError("Time t must be non-negative.")
        return self.rate

    def get_discount_factor(self, t: float) -> float:
        """
        Return the discount factor at time t.
        """
        return discount_factor(self.rate, t)


@dataclass(frozen=True)
class ZeroCurve:
    """
    Piecewise-linear zero curve under continuous compounding.

    Attributes
    ----------
    times : Sequence[float]
        Curve pillar times in years, sorted ascending.
    zero_rates : Sequence[float]
        Continuously compounded zero rates corresponding to times.
    """

    times: Sequence[float]
    zero_rates: Sequence[float]

    def __post_init__(self) -> None:
        if len(self.times) == 0:
            raise ValueError("times must not be empty.")
        if len(self.times) != len(self.zero_rates):
            raise ValueError("times and zero_rates must have the same length.")

        times = list(self.times)
        rates = list(self.zero_rates)

        if any(t <= 0 for t in times):
            raise ValueError("All curve times must be positive.")
        if any(times[i] >= times[i + 1] for i in range(len(times) - 1)):
            raise ValueError("Curve times must be strictly increasing.")

        object.__setattr__(self, "times", tuple(times))
        object.__setattr__(self, "zero_rates", tuple(rates))

    def get_zero_rate(self, t: float) -> float:
        """
        Interpolate the continuously compounded zero rate at time t
        using piecewise-linear interpolation.

        For times below the first pillar, return the first rate.
        For times above the last pillar, return the last rate.
        """
        if t < 0:
            raise ValueError("Time t must be non-negative.")
        if t == 0:
            return self.zero_rates[0]

        times = self.times
        rates = self.zero_rates

        if t <= times[0]:
            return rates[0]
        if t >= times[-1]:
            return rates[-1]

        idx = bisect_left(times, t)
        t1, t2 = times[idx - 1], times[idx]
        r1, r2 = rates[idx - 1], rates[idx]

        weight = (t - t1) / (t2 - t1)
        return r1 + weight * (r2 - r1)

    def get_discount_factor(self, t: float) -> float:
        """
        Return the discount factor at time t using interpolated zero rate.
        """
        if t == 0:
            return 1.0
        r = self.get_zero_rate(t)
        return discount_factor(r, t)


def build_flat_curve(rate: float) -> FlatYieldCurve:
    """
    Convenience function to build a flat yield curve.
    """
    return FlatYieldCurve(rate=rate)


def build_zero_curve(times: Sequence[float], zero_rates: Sequence[float]) -> ZeroCurve:
    """
    Convenience function to build a piecewise-linear zero curve.
    """
    return ZeroCurve(times=times, zero_rates=zero_rates)
