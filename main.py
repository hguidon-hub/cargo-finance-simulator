"""Hedged Cargo Financing Simulator — command-line entry point.

Examples
--------
    python main.py                                   # Brent cargo, live Yahoo data
    python main.py --spot 65 --vol 0.30              # offline, no internet needed
    python main.py --contango 0.04 --horizon 126     # flatter curve, 6-month carry
    python main.py --hedge-mode cross                # Brent cargo hedged with WTI futures
    python main.py --save cargo.png --no-plot
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

import pandas as pd

from cargofinance import __version__
from cargofinance.carry import carry_sensitivity, cash_and_carry
from cargofinance.config import CROSS, DIRECT, Config
from cargofinance.market import fetch_market_data, offline_inputs
from cargofinance.report import interpret
from cargofinance.simulation import BASE, build_scenarios, run_scenario


def parse_args(argv=None) -> argparse.Namespace:
    d = Config()
    p = argparse.ArgumentParser(description="Hedged, bank-financed physical cargo: carry, hedge, liquidity and LTV risk.")
    p.add_argument("--spot", type=float, help="Spot $/bbl (with --vol: run offline, skip Yahoo)")
    p.add_argument("--vol", type=float, help="Annualised volatility, e.g. 0.30")
    p.add_argument("--quantity", type=float, default=d.quantity_bbl, help="Cargo size in barrels")
    p.add_argument("--horizon", type=int, default=d.horizon_days, help="Storage horizon in trading days")
    p.add_argument("--contango", type=float, default=d.contango_annual, help="Annualised curve slope (assumption)")
    p.add_argument("--storage", type=float, default=d.storage_cost_bbl_month, help="Storage $/bbl/month")
    p.add_argument("--funding-rate", type=float, default=d.funding_rate, help="Bank loan rate (ACT/360)")
    p.add_argument("--equity-rate", type=float, help="Cost of the trader's own equity (default: funding rate)")
    p.add_argument("--ltv", type=float, default=d.ltv_initial, help="Initial loan-to-value")
    p.add_argument("--ltv-max", type=float, default=d.ltv_max, help="LTV breach threshold")
    p.add_argument("--buffer", type=float, default=d.trader_cash_buffer, help="Trader cash for margin ($)")
    p.add_argument("--im", type=float, default=d.initial_margin_per_contract, help="Initial margin per contract ($)")
    p.add_argument("--hedge-mode", choices=(DIRECT, CROSS), default=d.hedge_mode)
    p.add_argument("--hedge-ratio", type=float, default=d.hedge_ratio)
    p.add_argument("--basis-vol", type=float, default=d.basis_vol, help="Basis volatility (direct mode)")
    p.add_argument("--paths", type=int, default=d.n_paths)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--save", help="Save the base-case dashboard to this PNG path")
    p.add_argument("--no-plot", action="store_true", help="Do not open the chart window")
    p.add_argument("--version", action="version", version=f"cargo-finance-simulator {__version__}")
    return p.parse_args(argv)


def build_config(a: argparse.Namespace) -> Config:
    cfg = replace(
        Config(),
        quantity_bbl=a.quantity, horizon_days=a.horizon, contango_annual=a.contango,
        storage_cost_bbl_month=a.storage, funding_rate=a.funding_rate, equity_cost_rate=a.equity_rate,
        ltv_initial=a.ltv, ltv_max=a.ltv_max, trader_cash_buffer=a.buffer,
        initial_margin_per_contract=a.im, hedge_mode=a.hedge_mode, hedge_ratio=a.hedge_ratio,
        basis_vol=a.basis_vol, n_paths=a.paths, seed=a.seed,
    )
    cfg.validate()
    return cfg


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    pd.set_option("display.width", 160)

    if (args.spot is None) != (args.vol is None):
        raise ValueError("Give both --spot and --vol to run offline, or neither to use Yahoo Finance.")
    if args.spot is not None:
        if cfg.hedge_mode == CROSS:
            raise ValueError("Cross-hedge mode needs live data for both contracts; drop --spot/--vol.")
        mkt = offline_inputs(args.spot, args.vol)
    else:
        mkt = fetch_market_data(cfg)

    when = f"close of {mkt.as_of}" if mkt.as_of else "no market date"
    print(f"\n=== Market ({mkt.source}, {when}) ===")
    print(f"{cfg.commodity} spot: {mkt.spot:.2f} $/bbl | annualised vol: {mkt.vol:.1%}")
    if mkt.h_cross is not None:
        print(f"Cross-hedge {cfg.ticker_physical} via {cfg.ticker_hedge}: h* = {mkt.h_cross:.2f}, "
              f"correlation = {mkt.corr:.2f}, residual vol = {mkt.resid_vol:.1%}")

    carry = cash_and_carry(cfg, mkt.spot)
    print("\n=== Cash-and-carry (contango is an assumption: see sensitivity) ===")
    print(carry.to_string())
    print("\n=== Carry margin vs curve shape ===")
    print(carry_sensitivity(cfg, mkt.spot).to_string())

    results = {name: run_scenario(c, mkt) for name, c in build_scenarios(cfg).items()}
    table = pd.DataFrame({name: r.metrics for name, r in results.items()})
    print("\n=== Monte Carlo results ===")
    print(table.to_string())

    print("\n=== Interpretation ===\n")
    print(interpret(cfg, mkt, carry, table))

    if args.save or not args.no_plot:
        from cargofinance.plots import plot_base_case

        plot_base_case(cfg, results[BASE], save_path=args.save, show=not args.no_plot,
                       locked_carry=carry["Total carry margin ($M)"])
        if args.save:
            print(f"\nChart saved to {args.save}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(130)
    except (ValueError, RuntimeError) as err:
        print(f"\nError: {err}", file=sys.stderr)
        sys.exit(1)
