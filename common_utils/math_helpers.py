
"""
math_helpers.py
---------------
Shared numerical utilities for the Quant Desk Toolkit.

Covers:
  - Numerical integration (trapezoidal, Simpson's rule)
  - Root-finding (Brent's method)
  - Interpolation (linear, log-linear, cubic spline)
  - Monte Carlo variance reduction and diagnostics
"""

import numpy as np
from scipy.interpolate import CubicSpline
from typing import Callable, Tuple


# =============================================================================
# NUMERICAL INTEGRATION
# =============================================================================

def trapezoidal_integration(x: np.ndarray, y: np.ndarray) -> float:
    """
    Numerical integration using the trapezoidal rule.

    Parameters
    ----------
    x : np.ndarray
        Strictly increasing array of abscissae (e.g. time grid in years).
    y : np.ndarray
        Function values evaluated at x.

    Returns
    -------
    float
        Approximation of integral of y over x.

    Example
    -------
    >>> t = np.linspace(0, 5, 100)
    >>> ee = np.exp(-0.05 * t)          # toy expected exposure
    >>> trapezoidal_integration(t, ee)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    return float(np.trapz(y, x))


def simpsons_integration(x: np.ndarray, y: np.ndarray) -> float:
    """
    Numerical integration using Simpson's rule (higher-order accuracy).

    Parameters
    ----------
    x : np.ndarray
        Strictly increasing array of abscissae. Length must be odd
        (even number of intervals) for pure Simpson's; falls back to
        trapezoidal on the last interval if even.
    y : np.ndarray
        Function values evaluated at x.

    Returns
    -------
    float
        Approximation of integral of y over x.
    """
    from scipy.integrate import simpson
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    return float(simpson(y, x=x))


# =============================================================================
# ROOT FINDING
# =============================================================================

def brent_solver(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-8,
    max_iter: int = 500,
) -> float:
    """
    Brent's method for finding a root of f in the bracket [a, b].

    Used for implied rate/spread/volatility solving where Newton-Raphson
    may not converge reliably.

    Parameters
    ----------
    f : Callable
        Scalar function whose root is sought. Must satisfy f(a)*f(b) < 0.
    a, b : float
        Bracket endpoints. f(a) and f(b) must have opposite signs.
    tol : float
        Convergence tolerance on the root value.
    max_iter : int
        Maximum number of iterations before raising RuntimeError.

    Returns
    -------
    float
        Approximate root x* such that |f(x*)| < tol.

    Raises
    ------
    ValueError
        If f(a) and f(b) have the same sign (no bracket).
    RuntimeError
        If the method does not converge within max_iter iterations.

    Example
    -------
    >>> # Solve for par swap rate
    >>> brent_solver(lambda r: pv_fixed(r) - pv_float(), 0.0, 0.2)
    """
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError(
            f"f(a)={fa:.6f} and f(b)={fb:.6f} have the same sign. "
            "No root bracketed in [a, b]."
        )

    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c, fc = a, fa
    mflag = True
    s = d = 0.0

    for _ in range(max_iter):
        if abs(b - a) < tol:
            return b

        if fa != fc and fb != fc:
            # Inverse quadratic interpolation
            s = (
                a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb))
            )
        else:
            # Secant method
            s = b - fb * (b - a) / (fb - fa)

        cond1 = not ((3 * a + b) / 4 < s < b or b < s < (3 * a + b) / 4)
        cond2 = mflag and abs(s - b) >= abs(b - c) / 2
        cond3 = (not mflag) and abs(s - b) >= abs(c - d) / 2
        cond4 = mflag and abs(b - c) < tol
        cond5 = (not mflag) and abs(c - d) < tol

        if cond1 or cond2 or cond3 or cond4 or cond5:
            s = (a + b) / 2
            mflag = True
        else:
            mflag = False

        fs = f(s)
        d, c, fc = c, b, fb

        if fa * fs < 0:
            b, fb = s, fs
        else:
            a, fa = s, fs

        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

    raise RuntimeError(
        f"Brent's method did not converge after {max_iter} iterations. "
        f"Last bracket: [{a:.8f}, {b:.8f}]"
    )


# =============================================================================
# INTERPOLATION
# =============================================================================

def linear_interp(
    x: float | np.ndarray,
    xp: np.ndarray,
    yp: np.ndarray,
) -> float | np.ndarray:
    """
    Piecewise linear interpolation.

    Parameters
    ----------
    x : float or np.ndarray
        Query point(s).
    xp : np.ndarray
        Known x-coordinates (strictly increasing).
    yp : np.ndarray
        Known y-values at xp.

    Returns
    -------
    float or np.ndarray
        Interpolated value(s) at x.
    """
    return np.interp(x, xp, yp)


def log_linear_interp(
    x: float | np.ndarray,
    xp: np.ndarray,
    yp: np.ndarray,
) -> float | np.ndarray:
    """
    Log-linear interpolation — standard for discount factor curves.

    Interpolates in log space: ln(y) is linearly interpolated,
    then exponentiated. Ensures positivity and smooth discount factors.

    Parameters
    ----------
    x : float or np.ndarray
        Query point(s).
    xp : np.ndarray
        Known x-coordinates (strictly increasing).
    yp : np.ndarray
        Known y-values at xp. Must be strictly positive (discount factors).

    Returns
    -------
    float or np.ndarray
        Interpolated value(s) at x.

    Raises
    ------
    ValueError
        If any yp value is non-positive.
    """
    yp = np.asarray(yp, dtype=float)
    if np.any(yp <= 0):
        raise ValueError("Log-linear interpolation requires strictly positive yp values.")
    log_yp = np.log(yp)
    log_y = np.interp(x, xp, log_yp)
    return np.exp(log_y)


def cubic_spline_interp(
    x: float | np.ndarray,
    xp: np.ndarray,
    yp: np.ndarray,
    bc_type: str = "not-a-knot",
) -> float | np.ndarray:
    """
    Cubic spline interpolation — standard for zero rate curves.

    Parameters
    ----------
    x : float or np.ndarray
        Query point(s).
    xp : np.ndarray
        Known x-coordinates (strictly increasing).
    yp : np.ndarray
        Known y-values at xp.
    bc_type : str
        Boundary condition type passed to scipy CubicSpline.
        Default is 'not-a-knot'.

    Returns
    -------
    float or np.ndarray
        Interpolated value(s) at x.
    """
    cs = CubicSpline(xp, yp, bc_type=bc_type)
    return cs(x)


# =============================================================================
# MONTE CARLO UTILITIES
# =============================================================================

def antithetic_variates(z: np.ndarray) -> np.ndarray:
    """
    Apply antithetic variates variance reduction to a standard normal array.

    Doubles the effective sample size by pairing each draw z with -z.
    Reduces variance of the estimator when the integrand is monotone.

    Parameters
    ----------
    z : np.ndarray
        Array of standard normal draws, shape (n_paths, n_steps).

    Returns
    -------
    np.ndarray
        Array of shape (2 * n_paths, n_steps) with original and
        antithetic draws concatenated.

    Example
    -------
    >>> z = np.random.standard_normal((10_000, 50))
    >>> z_av = antithetic_variates(z)
    >>> z_av.shape
    (20000, 50)
    """
    return np.concatenate([z, -z], axis=0)


def standard_error(values: np.ndarray) -> float:
    """
    Compute the Monte Carlo standard error of an estimator.

    Parameters
    ----------
    values : np.ndarray
        Array of simulated payoffs or valuations, shape (n_paths,).

    Returns
    -------
    float
        Standard error = std(values) / sqrt(n_paths).
    """
    values = np.asarray(values, dtype=float)
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def confidence_interval(
    values: np.ndarray,
    alpha: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute a symmetric confidence interval for a Monte Carlo estimator.

    Parameters
    ----------
    values : np.ndarray
        Array of simulated values, shape (n_paths,).
    alpha : float
        Confidence level. Default is 0.95 (95% CI).

    Returns
    -------
    Tuple[float, float]
        (lower_bound, upper_bound) of the confidence interval.

    Example
    -------
    >>> ci = confidence_interval(simulated_cva, alpha=0.95)
    >>> print(f"CVA 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
    """
    from scipy.stats import norm
    values = np.asarray(values, dtype=float)
    mean = np.mean(values)
    se = standard_error(values)
    z = norm.ppf((1 + alpha) / 2)
    return (float(mean - z * se), float(mean + z * se))


def control_variate_adjustment(
    simulated: np.ndarray,
    control: np.ndarray,
    control_mean: float,
) -> np.ndarray:
    """
    Apply control variate variance reduction.

    Adjusts simulated values using a correlated control variable
    whose true mean is known analytically.

    Parameters
    ----------
    simulated : np.ndarray
        Raw Monte Carlo output, shape (n_paths,).
    control : np.ndarray
        Simulated values of the control variable, shape (n_paths,).
    control_mean : float
        Known analytical mean of the control variable.

    Returns
    -------
    np.ndarray
        Variance-reduced estimator, shape (n_paths,).

    Notes
    -----
    The optimal coefficient beta* is estimated via OLS regression
    of simulated on control. The adjusted estimator is:

        Y_adj = Y - beta* * (C - E[C])

    where E[C] = control_mean.
    """
    simulated = np.asarray(simulated, dtype=float)
    control = np.asarray(control, dtype=float)

    # Estimate optimal beta via covariance
    cov_matrix = np.cov(simulated, control)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]

    return simulated - beta * (control - control_mean)