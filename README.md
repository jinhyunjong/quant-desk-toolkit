![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen)

# Quant Desk Toolkit

A professional-grade Python library for derivatives pricing, counterparty
exposure modeling, XVA analytics, and regulatory capital computation —
built to reflect the analytical frameworks used on derivatives trading desks.

---

## Overview

This toolkit implements the core quantitative models underlying modern
counterparty risk management and valuation adjustment pricing. It covers
four interconnected domains:

- **Derivatives Pricing** — Multi-curve OIS/SOFR curve construction,
  interest rate swap and fixed income instrument valuation, and
  interest rate Greeks via bump-and-reprice finite difference
- **XVA Pricing** — CVA, DVA, and FVA computed from simulated exposure
  profiles and market-implied credit/funding curves
- **Counterparty Exposure** — Monte Carlo simulation of Expected Exposure
  (EE), Expected Positive Exposure (EPE), and Potential Future Exposure
  (PFE) across netting sets, with and without collateral
- **Regulatory Capital** — SA-CCR exposure-at-default (EAD), SIMM/ISDA
  initial margin, and risk-weighted asset (RWA) calculations under Basel III

The implementation prioritizes numerical accuracy, modularity, and
transparency of methodology — consistent with how these models are
built and validated in production trading environments.

---

## Architecture
quant-desk-toolkit/
│
├── quant_desk/                  # Installable package
│   ├── init.py
│   ├── curve_factory.py         # Multi-curve OIS/SOFR bootstrapping
│   ├── instruments.py           # IRS, bond, SFT pricing
│   ├── simulator.py             # Hull-White Monte Carlo engine
│   ├── exposure.py              # EPE/PFE/EE profiles
│   ├── xva.py                   # CVA, DVA, FVA, MVA
│   ├── margin.py                # SIMM/ISDA IM and variation margin
│   ├── sa_ccr.py                # SA-CCR RC, PFE add-on, EAD
│   ├── capital_rwa.py           # Basel III RWA and K-value
│   ├── counterparty.py          # CDS bootstrapping, survival curves
│   ├── greeks.py                # DV01, duration, convexity, CVA IR DV01
│   └── common_utils/
│       ├── init.py
│       └── math_helpers.py      # Numerical integration, root-finding, MC variance reduction
│
├── notebooks/
│   ├── 01_curves_and_instruments.ipynb
│   ├── 02_monte_carlo.ipynb
│   ├── 03_exposure_xva.ipynb
│   ├── 04_sa_ccr_capital.ipynb
│   ├── 05_margin_simm.ipynb
│   └── 06_greeks.ipynb
│
├── scripts/
│   └── run_pipeline.py          # End-to-end integration script
│
├── tests/
│   └── test_*.py
│
├── requirements.txt
├── setup.py
└── README.md

---

## Core Formulas

**Credit Valuation Adjustment (CVA)**

$$CVA \approx (1 - R) \int_0^T EE^*(t) \, dPD(0, t)$$

Where $EE^*(t)$ is the discounted expected exposure at time $t$,
$R$ is the recovery rate, and $PD(0,t)$ is the risk-neutral
default probability.

**Expected Positive Exposure (EPE)**

$$EPE = \frac{1}{T} \int_0^T EE(t) \, dt$$

Where $EE(t) = \mathbb{E}[\max(V(t), 0)]$ is the expected exposure
at time $t$ under the risk-neutral measure.

**SA-CCR Exposure at Default (EAD)**

$$EAD = \alpha \cdot (RC + PFE_{aggregate})$$

Where $\alpha = 1.4$ is the supervisory scaling factor, $RC$ is the
replacement cost, and $PFE_{aggregate}$ is the aggregated
potential future exposure add-on across asset classes.

---

## Modules

### `curve_factory`
Multi-curve bootstrapping under an OIS/SOFR framework. Separates discount
and projection curves with log-linear interpolation on discount factors and
cubic spline interpolation on zero rates. Supports arbitrary tenor grids and
market data inputs for SOFR swaps and OIS instruments.

### `instruments`
Fixed income instrument valuation under the multi-curve framework — interest
rate swaps (IRS), fixed-rate bonds, FRAs, and securities financing transactions
(SFTs). Computes fair value, accrued interest, par rate, and yield to maturity.

### `simulator`
Hull-White one-factor Monte Carlo simulation engine. Generates short rate paths
under the risk-neutral measure for use in counterparty exposure and XVA pricing
workflows. Supports antithetic variates and control variates for variance
reduction.

### `exposure`
Counterparty exposure analytics — Expected Positive Exposure (EPE), Potential
Future Exposure (PFE), and Expected Exposure (EE) profiles across simulation
paths and time horizons. Supports netting set aggregation with and without
collateral.

### `xva`
XVA pricing engine covering CVA, DVA, FVA, and MVA. Integrates with the
exposure engine and curve environment for full valuation adjustment computation
under a multi-curve OIS/SOFR framework. CVA and DVA computed as discounted
integrals of EPE/ENE against credit hazard rates. FVA computed from asymmetric
funding spread framework.

### `margin`
Initial margin and variation margin computation. Implements the ISDA SIMM
delta sensitivity-based IM methodology across IR, FX, Credit, and Equity risk
classes. Variation margin mechanics include margin period of risk (MPoR) and
collateral threshold modeling.

### `sa_ccr`
SA-CCR (Standardised Approach for Counterparty Credit Risk) implementation
per Basel III final rules. Computes Replacement Cost (RC) for margined and
unmargined netting sets, PFE add-ons across IR, FX, Credit, and Equity asset
classes, and Exposure at Default (EAD) with supervisory delta and maturity
factor adjustments.

### `capital_rwa`
Basel III regulatory capital and RWA calculation. Integrates SA-CCR EAD
outputs with counterparty risk weights to produce capital requirements and
K-value capital charges across derivatives portfolios.

### `counterparty`
Counterparty-level analytics — CDS spread bootstrapping to survival probability
curves, default probability term structure under flat and term hazard rate
assumptions, and LGD parameterization for use in CVA/DVA computation.

### `greeks`
Interest rate sensitivities via bump-and-reprice finite difference — no
closed-form shortcuts, so the framework extends cleanly to exotic payoffs.
Covers parallel and bucketed DV01, modified duration, and convexity for
interest rate swaps and fixed-rate bonds. Includes CVA IR DV01 via full Monte
Carlo re-run at bumped curves, with both parallel and tenor-bucketed variants.

### `common_utils/math_helpers`
Shared numerical utilities — trapezoidal and Simpson's rule integration,
Brent's method root-finding for implied rate and volatility solving, and Monte
Carlo variance reduction methods (antithetic variates, control variates).

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_curves_and_instruments` | OIS/SOFR curve bootstrapping, IRS and bond pricing, par rate validation |
| `02_monte_carlo` | Hull-White simulation engine — rate path generation, calibration, and variance reduction |
| `03_exposure_xva` | EPE/PFE profiles and full XVA pricing (CVA, DVA, FVA, MVA) with collateral impact |
| `04_sa_ccr_capital` | SA-CCR EAD calculation and Basel III regulatory capital across a mixed derivatives portfolio |
| `05_margin_simm` | ISDA SIMM initial margin and variation margin computation across risk classes |
| `06_greeks` | DV01, bucketed sensitivities, modified duration, convexity, and CVA IR DV01 |

---

## Mathematical Framework

| Component | Model / Method |
|-----------|----------------|
| IR simulation | Hull-White one-factor |
| Curve construction | Multi-curve OIS/SOFR bootstrapping |
| Exposure aggregation | Netting set with collateral threshold |
| CVA / DVA | Hazard rate model, CDS-implied survival curve |
| FVA | Asymmetric funding spread framework |
| MVA | SIMM-based IM projection over simulation paths |
| IM | ISDA SIMM delta sensitivity approach |
| Regulatory EAD | Basel III SA-CCR |
| Greeks | Bump-and-reprice finite difference |

---

## Usage

```python
from quant_desk.curve_factory import CurveFactory
from quant_desk.instruments import InterestRateSwap
from quant_desk.simulator import HullWhiteSimulator
from quant_desk.exposure import ExposureEngine
from quant_desk.xva import XVAEngine
from quant_desk.greeks import irs_greek_summary

# Bootstrap OIS/SOFR curves
curves = CurveFactory.build(market_data)

# Price an interest rate swap
swap = InterestRateSwap(notional=10_000_000, fixed_rate=0.035, tenor=5)
npv = swap.pv(curves.discount, curves.projection)["pv_net"]

# Compute Greeks
greeks = irs_greek_summary(swap, curves.discount, curves.projection)

# Simulate exposure and compute XVA
simulator = HullWhiteSimulator(curves, n_paths=5000)
engine = ExposureEngine(simulator)
xva = XVAEngine(engine, curves)
print(f"CVA: {xva.cva():.0f}  FVA: {xva.fva():.0f}")
```

---

## Tech Stack

- **Python 3.10+** — NumPy, SciPy, Pandas, Matplotlib
- **Jupyter** — Interactive notebooks with inline analytics
- **pytest** — Unit testing for core pricing and regulatory logic

---

## Setup

```bash
git clone https://github.com/hyun-quant/quant-desk-toolkit.git
cd quant-desk-toolkit
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

---

## Tests

```bash
pytest tests/
```

Core test coverage includes CVA calculation on known analytical solutions,
SA-CCR EAD against published Basel Committee example portfolios, and curve
bootstrapping consistency checks.

---

## Background

Built independently as a quantitative finance research and implementation
project, with public release timed to support a return to institutional quant
practice. The toolkit reflects hands-on implementation experience across
derivatives pricing, counterparty risk, and regulatory capital — areas covered
professionally across roles in sell-side risk management and financial risk
advisory.

---

## License

MIT
