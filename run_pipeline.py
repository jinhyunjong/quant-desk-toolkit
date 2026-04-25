
"""
run_pipeline.py
---------------
End-to-End Integration and Execution Script for the Quant Desk Toolkit.
"""

import numpy as np

# Import from the Quant Desk Toolkit
from curve_factory import Curve
from instruments import InterestRateSwap
from counterparty import Counterparty
from margin import CSATerms, MarginEngine
from simulator import HullWhiteSimulator, MonteCarloEngine
from exposure import ExposureEngine, NettingSet as ExposureNettingSet
from xva import XVAEngine
from sa_ccr import SACCREngine, IRTrade
from capital_rwa import CounterpartyCapitalInput, CapitalEngine

def main():
    print("=" * 70)
    print(" QUANT DESK TOOLKIT: END-TO-END PIPELINE EXECUTION")
    print("=" * 70)

    # 1. MARKET DATA & CURVES
    print("[1/7] Bootstrapping Market Curves...")
    time_grid = np.linspace(0, 5, 61)  # 5-year horizon, monthly simulation steps
    ois_curve = Curve(tenors=time_grid, discount_factors=np.exp(-0.0200 * time_grid), label="USD_OIS")
    sofr_curve = Curve(tenors=time_grid, discount_factors=np.exp(-0.0250 * time_grid), label="USD_SOFR")

    # Hazard rate arrays for XVA pricing
    cpty_tenors = np.array([0.0, 1.0, 3.0, 5.0])
    cpty_hazard_rates = np.array([0.010, 0.012, 0.016, 0.020])
    own_tenors = np.array([0.0, 1.0, 3.0, 5.0])
    own_hazard_rates = np.array([0.006, 0.008, 0.010, 0.012])

    # 2. ENTITIES & CSA
    print("[2/7] Structuring Counterparty and CSA Terms...")
    megacorp = Counterparty(
        name="MegaCorp Inc",
        credit_quality="BBB",
        pd=0.01,
        lgd=0.60,
        recovery_rate=0.40,
        maturity=5.0
    )
    csa_terms = CSATerms(threshold_them=500_000.0, mta_them=50_000.0)

    # 3. TRADE BOOKING
    print("[3/7] Booking Derivative Portfolio...")
    swap_1 = InterestRateSwap(notional=10_000_000.0, fixed_rate=0.0250, start=0.0, tenor=5.0, payer=True)
    spot_mtm = swap_1.pv(ois_curve, sofr_curve)["pv_net"]

    # 4. MONTE CARLO SIMULATION
    print("[4/7] Running Hull-White 1F Monte Carlo Engine (N=2000)...")
    hw_sim = HullWhiteSimulator(a=0.03, sigma=0.01, curve=ois_curve)
    mc_engine = MonteCarloEngine(hw_simulator=hw_sim)
    mc_results = mc_engine.run(time_grid, n_paths=2000, use_antithetic=True)

    # 5. EXPOSURE AGGREGATION
    print("[5/7] Simulating Collateral Mechanics and Aggregating Exposure...")
    exposure_engine = ExposureEngine(mc_results, hw_sim)
    val_1 = exposure_engine.irs_valuator(swap_1.fixed_rate, swap_1._fixed_payment_dates(), swap_1.notional, swap_1.payer)
    exp_netting_set = ExposureNettingSet(name="Net_MegaCorp", trade_valuators=[val_1], threshold=csa_terms.threshold_them, minimum_transfer_amount=csa_terms.mta_them)
    exp_summary = exposure_engine.exposure_summary(exp_netting_set)

    # 6. XVA PRICING
    print("[6/7] Computing XVA Integrals (CVA, DVA, FVA, MVA)...")
    xva_engine = XVAEngine(
        ee_profile=exp_summary["EE"],
        ene_profile=exp_summary["ENE"],
        time_grid=time_grid,
        counterparty_tenors=cpty_tenors,
        counterparty_hazard_rates=cpty_hazard_rates,
        counterparty_recovery=megacorp.recovery_rate,
        own_tenors=own_tenors,
        own_hazard_rates=own_hazard_rates,
        funding_spread=0.0050
    )
    xva_metrics = xva_engine.compute()

    # 7. REGULATORY CAPITAL
    print("[7/7] Executing Basel III Capital Framework...")
    saccr_engine = SACCREngine(
        ir_trades=[IRTrade(swap_1.notional, swap_1.tenor, swap_1.start, swap_1.tenor, "USD", swap_1.payer)],
        margined=True,
        netting_set_mtm=spot_mtm,
        threshold=csa_terms.threshold_them,
        mta=csa_terms.mta_them
    )
    saccr_res = saccr_engine.compute()

    cap_input = CounterpartyCapitalInput(
        name=megacorp.name,
        ead=saccr_res["EAD"],
        pd=megacorp.pd,
        lgd=megacorp.lgd,
        maturity=megacorp.maturity,
        credit_quality=megacorp.credit_quality,
        counterparty_type="Corporate_BBB"
    )
    cap_engine = CapitalEngine([cap_input], use_irb=True)
    cap_metrics = cap_engine.compute()

    # --- RESTORED DASHBOARD OUTPUT ---
    print("\n" + "=" * 70)
    print(f" FRONT OFFICE RISK DASHBOARD: {megacorp.name}")
    print("=" * 70)
    print(f"EXPOSURE METRICS:")
    print(f"  Current MTM:           ${spot_mtm:,.0f}")
    print(f"  Peak PFE (95%):        ${exp_summary['peak_PFE']:,.0f}")
    print(f"  Expected Pos Exp (EPE):${exp_summary['EPE']:,.0f}")
    print("-" * 70)
    print(f"VALUATION ADJUSTMENTS (XVA):")
    print(f"  CVA (Credit Cost):     ${xva_metrics['CVA']:,.0f}")
    print(f"  DVA (Debt Benefit):   -${abs(xva_metrics['DVA']):,.0f}")
    print(f"  FVA (Funding Cost):    ${xva_metrics['FVA']:,.0f}")
    print(f"  Net XVA Charge:        ${xva_metrics['total_XVA']:,.0f}")
    print("-" * 70)
    print(f"REGULATORY CAPITAL (BASEL III/IV):")
    print(f"  SA-CCR EAD:            ${saccr_res['EAD']:,.0f}")
    print(f"  IRB RWA (CCR):         ${cap_metrics['RWA_CCR_total']:,.0f}")
    print(f"  Min Capital Required:  ${cap_metrics['capital_min']['Total_min']:,.0f}")
    print("=" * 70)
    print("PIPELINE EXECUTED SUCCESSFULLY. NO INTEGRATION ERRORS.")

if __name__ == "__main__":
    main()