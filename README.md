![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen)

# Quant Desk Toolkit

A professional-grade Python library for XVA pricing, counterparty credit risk
modeling, and regulatory capital computation — built to reflect the analytical
frameworks used on derivatives trading desks.

---

## Overview

This toolkit implements the core quantitative models underlying modern
counterparty risk management and valuation adjustment pricing. It covers
three interconnected domains:

- **XVA Pricing** — CVA, DVA, and FVA computed from simulated exposure
  profiles and market-implied credit/funding curves
- **Counterparty Exposure** — Monte Carlo simulation of Expected Exposure
  (EE), Expected Positive Exposure (EPE), and Potential Future Exposure
  (PFE) across netting sets, with and without collateral
- **Regulatory Capital** — SA-CCR exposure-at-default (EAD) and
  risk-weighted asset (RWA) calculations under Basel III

The implementation prioritizes numerical accuracy, modularity, and
transparency of methodology — consistent with how these models are
built and validated in production trading environments.

---

## Architecture

```
quant-desk-toolkit/
├── xva-engine/          # Core: XVA pricing and exposure simulation
├── rates-analytics/     # Supporting: Curve construction and instrument pricing
├── regulatory-capital/  # Regulatory: SA-CCR and RWA
├── common-utils/        # Shared: Numerical methods and market data I/O
└── tests/               # Unit tests for core pricing logic
```

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

### XVA Engine
*The primary module — implements the full bilateral XVA pricing framework.*

**Exposure Simulation** (`simulator.py`, `exposure.py`)
- Hull-White one-factor model for interest rate path generation
- GBM for equity/FX underlying simulation
- Netting set aggregation with bilateral close-out
- Expected Exposure (EE), EPE, and PFE term structures

**XVA Pricing** (`xva.py`)
- CVA: Unilateral and bilateral, computed as discounted integral
  of EPE against credit hazard rates
- DVA: Own-default benefit using firm's CDS-implied survival probability
- FVA: Funding cost/benefit from unsecured funding of variation margin
  shortfalls, using OIS/internal funding spread

**Margin and Collateral** (`margin.py`)
- ISDA SIMM delta sensitivity-based IM calculation
- Variation margin (VM) mechanics with margin period of risk (MPoR)
- Impact of collateral on exposure reduction

**Counterparty Module** (`counterparty.py`)
- CDS spread bootstrapping to survival probability curves
- Default probability term structure under flat and term hazard rates
- Loss given default (LGD) parameterization

**Key Notebooks**
- `demo_xva_pricing.ipynb` — Incremental CVA/FVA pricing on a new
  IRS trade added to an existing portfolio
- `collateral_impact.ipynb` — Sensitivity of EPE/PFE to margin
  frequency and MPoR assumptions

---

### Rates Analytics
*Underlying curve infrastructure and instrument pricing.*

**Curve Construction** (`curve_factory.py`)
- Multi-curve bootstrapping: OIS discounting with SOFR-linked
  projection curves
- Interpolation methods: log-linear on discount factors,
  cubic spline on zero rates

**Instrument Pricing** (`instruments.py`)
- Interest rate swap (IRS) valuation under multi-curve framework
- Fixed income bond pricing with accrued interest
- Securities financing transaction (SFT) cash flow modeling

**Sensitivities** (`greeks.py`)
- Analytic DV01 and PV01 for swaps and bonds
- Bucketed sensitivity across curve tenors

**Key Notebooks**
- `curve_construction.ipynb` — OIS/SOFR curve build with
  swap pricing validation

---

### Regulatory Capital
*SA-CCR implementation for CCR capital computation.*

The SA-CCR module provides transparency into the RWA impact of trades,
enabling capital-efficient structuring of derivatives portfolios.

**SA-CCR** (`sa_ccr.py`)
- Replacement cost (RC) for margined and unmargined netting sets
- PFE add-on calculation across asset classes:
  IR, FX, Credit, Equity
- EAD aggregation with supervisory delta and maturity factor

**Capital and RWA** (`capital_rwa.py`)
- RWA computation from SA-CCR EAD
- K-value capital requirement

**Key Notebooks**
- `sa_ccr_walkthrough.ipynb` — Step-by-step SA-CCR calculation
  on a mixed derivatives portfolio, validated against
  Basel Committee published example portfolios

---

### Common Utilities

**Math Helpers** (`math_helpers.py`)
- Numerical integration (trapezoidal, Simpson's rule)
- Root-finding (Brent's method for implied vol/rate solving)
- Monte Carlo variance reduction (antithetic variates,
  control variates)

**Data I/O** (`data_io.py`)
- Market data ingestion from JSON, CSV, and SQL sources
- Discount factor and hazard rate curve construction
  from raw market inputs

---

## Mathematical Framework

| Component | Model / Method |
|-----------|---------------|
| IR simulation | Hull-White one-factor |
| Equity/FX simulation | Geometric Brownian Motion |
| Exposure aggregation | Netting set with collateral threshold |
| CVA/DVA | Hazard rate model, CDS-implied |
| FVA | Asymmetric funding spread framework |
| IM | ISDA SIMM delta sensitivity approach |
| Regulatory EAD | Basel III SA-CCR |

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
pip install -r requirements.txt
```

---

## Tests

```bash
pytest tests/
```

Core test coverage includes CVA calculation on known analytical
solutions, SA-CCR EAD against published Basel Committee example
portfolios, and curve bootstrapping consistency checks.
