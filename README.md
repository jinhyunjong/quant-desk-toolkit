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
