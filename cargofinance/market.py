"""Market inputs: spot, volatility, and cross-hedge statistics.

``inputs_from_prices`` is pure (testable offline); ``fetch_market_data`` only adds the
Yahoo Finance download around it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .config import TRADING_DAYS, Config


@dataclass(frozen=True)
class MarketInputs:
    as_of: date | None
    spot: float
    vol: float
    source: str
    hedge_spot: float | None = None
    hedge_vol: float | None = None
    corr: float | None = None
    h_cross: float | None = None     # minimum-variance hedge ratio (return terms)
    resid_vol: float | None = None   # annualised vol of the unhedgeable residual


def min_variance_hedge_ratio(spot_ret: pd.Series, fut_ret: pd.Series) -> float:
    """h* = Cov(dS, dF) / Var(dF): minimises the variance of the hedged position."""
    return float(np.cov(spot_ret, fut_ret, ddof=1)[0, 1] / np.var(fut_ret, ddof=1))


def inputs_from_prices(px: pd.DataFrame, physical: str, hedge: str, source: str = "prices") -> MarketInputs:
    """Estimate spot, vols, correlation, h* and residual vol from daily closes."""
    px = px[[physical, hedge]].dropna()
    if len(px) < 30:
        raise RuntimeError("Fewer than 30 common daily prices: cannot estimate volatility.")
    rets = np.log(px.where(px > 0)).diff().dropna()  # drop non-positive prices (WTI, April 2020)
    s, h = rets[physical], rets[hedge]
    h_star = min_variance_hedge_ratio(s, h)
    resid = s - h_star * h
    ann = math.sqrt(TRADING_DAYS)
    return MarketInputs(
        as_of=pd.Timestamp(px.index[-1]).date(),
        spot=float(px[physical].iloc[-1]),
        vol=float(s.std(ddof=1) * ann),
        source=source,
        hedge_spot=float(px[hedge].iloc[-1]),
        hedge_vol=float(h.std(ddof=1) * ann),
        corr=float(s.corr(h)),
        h_cross=h_star,
        resid_vol=float(resid.std(ddof=1) * ann),
    )


def fetch_market_data(cfg: Config, period: str = "2y") -> MarketInputs:
    import yfinance as yf  # lazy: tests and offline runs do not need it

    raw = yf.download([cfg.ticker_physical, cfg.ticker_hedge], period=period,
                      progress=False, auto_adjust=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"No data for {cfg.ticker_physical} / {cfg.ticker_hedge}: "
                           "check the connection, or run offline with --spot and --vol.")
    return inputs_from_prices(raw["Close"], cfg.ticker_physical, cfg.ticker_hedge,
                              source="Yahoo Finance")


def offline_inputs(spot: float, vol: float) -> MarketInputs:
    if spot <= 0 or vol <= 0:
        raise ValueError("--spot and --vol must be positive.")
    return MarketInputs(as_of=None, spot=spot, vol=vol, source="user input (offline)")
