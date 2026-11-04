"""Turn the numbers into a plain-English credit and trading reading."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CROSS, Config
from .market import MarketInputs
from .simulation import BASE, SHOCK_DOWN, SHOCK_UP


def _get(table: pd.DataFrame, row: str, col: str):
    return table.loc[row, col] if col in table.columns else None


def interpret(cfg: Config, mkt: MarketInputs, carry: pd.Series, table: pd.DataFrame) -> str:
    b = table[BASE]
    out = []

    move = mkt.vol * np.sqrt(cfg.T)
    out.append(
        f"[Market] {cfg.commodity} at {mkt.spot:.2f} $/bbl, {mkt.vol:.0%} annualised volatility "
        f"({mkt.source}): a one-sigma move over {cfg.horizon_days} trading days is about "
        f"±{move:.0%} of the cargo, i.e. ±{move * mkt.spot * cfg.quantity_bbl / 1e6:.0f} $M.")

    if mkt.h_cross is not None:
        out.append(
            f"[Cross-hedge] Hedging {cfg.ticker_physical} with {cfg.ticker_hedge} needs h* = "
            f"{mkt.h_cross:.2f} (correlation {mkt.corr:.2f}). The {1 - mkt.corr**2:.0%} of variance "
            f"left unexplained ({mkt.resid_vol:.0%} annualised residual vol) is basis risk no "
            f"futures position can remove.")

    margin = carry["Carry margin ($/bbl)"]
    costs = carry["Storage ($/bbl)"] + carry["Bank interest ($/bbl)"] + carry["Equity cost ($/bbl)"]
    verdict = "pays for" if margin > 0 else "does NOT pay for"
    out.append(
        f"[Carry] At the ASSUMED {cfg.contango_annual:.1%} annualised contango, the spread "
        f"({carry['Contango F0-S0 ($/bbl)']:.2f} $/bbl) {verdict} storage + financing "
        f"({costs:.2f} $/bbl): margin {margin:+.2f} $/bbl, {carry['Total carry margin ($M)']:+.2f} $M. "
        f"Break-even is {carry['Break-even contango (annualised)']:.1%}: check the live curve "
        f"against that number before trusting the verdict.")

    hedge_desc = "1:1 in the same grade" if cfg.hedge_mode != CROSS else f"via {cfg.ticker_hedge} at h*"
    locked = carry["Total carry margin ($M)"]
    funding = b["Margin funding cost, mean ($M)"]
    basis = b["Mean P&L ($M)"] - locked + funding
    hedge_txt = (
        f"[Hedge] Hedging {hedge_desc} cuts P&L volatility from {b['Unhedged std dev ($M)']:.1f} $M "
        f"to {b['P&L std dev ($M)']:.2f} $M ({b['Variance reduction']:.0%} of variance removed). "
        f"P&L bridge: locked-in carry {locked:+.2f} $M, margin funding {-funding:+.2f} $M, "
        f"basis {basis:+.2f} $M = mean hedged P&L {b['Mean P&L ($M)']:+.2f} $M."
    )
    if locked > 0 and funding > 0:
        hedge_txt += f" Funding the margin account alone eats {funding / locked:.0%} of the carry."
    out.append(hedge_txt)
    if b["Variance reduction"] > 0.99 and cfg.hedge_mode != CROSS:
        out.append(
            f"[Caveat] A >99% variance reduction comes from the {cfg.basis_vol:.0%} basis-vol "
            f"assumption, not from data. Run --hedge-mode cross to see a basis estimated from "
            f"market prices, or raise --basis-vol to stress it.")

    up = _get(table, "P(cash buffer exhausted)", SHOCK_UP)
    up_txt = f" ({up:.0%} after a +30% shock)" if up is not None else ""
    out.append(
        f"[Liquidity] Price risk is hedged, cash risk is not. Initial margin locks "
        f"{b['Initial margin ($M)']:.0f} $M of the {cfg.trader_cash_buffer / 1e6:.0f} $M buffer on day one; "
        f"a rally then forces variation-margin payments before the cargo is sold. The p99 total "
        f"cash need is {b['Cash need p99, IM + VM ($M)']:.0f} $M and the buffer runs out in "
        f"{b['P(cash buffer exhausted)']:.0%} of base paths{up_txt}. A perfectly hedged trader "
        f"can still default on liquidity.")

    down = _get(table, "P(LTV breach) no assignment", SHOCK_DOWN)
    down_txt = f" ({down:.0%} after a -30% shock)" if down is not None else ""
    out.append(
        f"[Bank] With only the inventory as collateral, LTV breaches {cfg.ltv_max:.0%} in "
        f"{b['P(LTV breach) no assignment']:.0%} of base paths{down_txt}. With the hedge assigned, "
        f"futures gains offset the inventory loss and the breach probability falls to "
        f"{b['P(LTV breach) with assignment']:.0%}. This is why lenders take an assignment of the "
        f"hedging contracts, and why they also size the trader's liquidity line.")

    worst = table.loc["ES 95% ($M)"].idxmax()
    out.append(
        f"[Stress] Worst scenario: '{worst}' (ES 95% = {table.loc['ES 95% ($M)', worst]:.2f} $M, "
        f"P(loss) = {table.loc['P(loss)', worst]:.0%}). With the hedge in place, losses come from "
        f"funding margin calls and basis, not from the outright price.")

    out.append("[Reading tip] A negative VaR / ES means even the worst 5% of outcomes is a profit.")
    return "\n\n".join(out)
