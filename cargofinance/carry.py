"""Cash-and-carry economics: does the contango pay for storage and financing?

The same cost function is used here and in the Monte Carlo, so the two analyses can
never disagree on what the cargo costs to carry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config


@dataclass(frozen=True)
class CarryCosts:
    """Carrying costs per barrel over the horizon."""
    storage: float
    debt_interest: float   # bank loan: LTV x S0 at the funding rate
    equity_cost: float     # trader's own (1 - LTV) x S0 at the equity cost rate

    @property
    def financing(self) -> float:
        return self.debt_interest + self.equity_cost

    @property
    def total(self) -> float:
        return self.storage + self.financing


def carrying_costs(cfg: Config, s0: float) -> CarryCosts:
    return CarryCosts(
        storage=cfg.storage_cost_bbl_month * cfg.T * 12.0,
        debt_interest=cfg.ltv_initial * s0 * cfg.funding_rate * cfg.act360,
        equity_cost=(1.0 - cfg.ltv_initial) * s0 * cfg.equity_rate * cfg.act360,
    )


def forward_price(s0: float, contango: float, T: float) -> float:
    return s0 * math.exp(contango * T)


def breakeven_contango(cfg: Config, s0: float) -> float:
    """Annualised contango at which the carry margin is exactly zero."""
    return math.log((s0 + carrying_costs(cfg, s0).total) / s0) / cfg.T


def cash_and_carry(cfg: Config, s0: float) -> pd.Series:
    costs = carrying_costs(cfg, s0)
    f0 = forward_price(s0, cfg.contango_annual, cfg.T)
    margin = (f0 - s0) - costs.total
    return pd.Series({
        "Spot S0 ($/bbl)": s0,
        "Forward F0 ($/bbl)": f0,
        "Contango F0-S0 ($/bbl)": f0 - s0,
        "Storage ($/bbl)": costs.storage,
        "Bank interest ($/bbl)": costs.debt_interest,
        "Equity cost ($/bbl)": costs.equity_cost,
        "Carry margin ($/bbl)": margin,
        "Total carry margin ($M)": margin * cfg.quantity_bbl / 1e6,
        "Break-even contango (annualised)": breakeven_contango(cfg, s0),
    })


def carry_sensitivity(cfg: Config, s0: float, contangos=None) -> pd.DataFrame:
    """Carry margin across curve shapes: the verdict depends on the contango assumption,
    so show it as a range rather than a single number."""
    if contangos is None:
        contangos = np.array([-0.10, -0.05, 0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20])
        contangos = np.unique(np.append(contangos, [cfg.contango_annual, breakeven_contango(cfg, s0)]))
    costs = carrying_costs(cfg, s0).total
    rows = []
    for c in contangos:
        spread = forward_price(s0, c, cfg.T) - s0
        rows.append({
            "Contango (annualised)": c,
            "F0-S0 ($/bbl)": spread,
            "Carry margin ($/bbl)": spread - costs,
            "Carry margin ($M)": (spread - costs) * cfg.quantity_bbl / 1e6,
        })
    return pd.DataFrame(rows).set_index("Contango (annualised)")
