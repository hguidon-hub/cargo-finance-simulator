"""Monte Carlo of the hedged, bank-financed cargo."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .carry import carrying_costs
from .config import CROSS, DIRECT, TRADING_DAYS, Config
from .market import MarketInputs


def _gbm(s0: float, sigma: float, z: np.ndarray, dt: float) -> np.ndarray:
    """Driftless (martingale) geometric Brownian motion, path 0 = s0."""
    inc = -0.5 * sigma**2 * dt + sigma * np.sqrt(dt) * z
    log_path = np.concatenate([np.zeros((z.shape[0], 1)), np.cumsum(inc, axis=1)], axis=1)
    return s0 * np.exp(log_path)


def _apply_jump(paths: np.ndarray, cfg: Config) -> np.ndarray:
    if cfg.jump_size != 0:
        paths = paths.copy()
        paths[:, cfg.jump_day:] *= 1.0 + cfg.jump_size
    return paths


def simulate_paths(cfg: Config, mkt: MarketInputs) -> tuple[np.ndarray, np.ndarray]:
    """Return (S, F): physical price paths and the hedge futures paths, shape (n_paths, N+1).

    direct mode: F_t = S_t * exp(c*tau + e_t*tau/T) with e an Ornstein-Uhlenbeck basis;
                 F converges to S at maturity, so the residual risk is the basis only.
    cross mode:  F is a correlated contract (WTI) H_t * exp(c*tau); the unhedgeable risk
                 is the 1 - rho^2 of variance the two grades do not share.
    """
    cfg.validate()
    rng = np.random.default_rng(cfg.seed)
    n, N, dt = cfg.n_paths, cfg.horizon_days, 1.0 / TRADING_DAYS
    tau = (N - np.arange(N + 1)) * dt
    T = N * dt

    z_s = rng.standard_normal((n, N))
    S = _apply_jump(_gbm(mkt.spot, mkt.vol * cfg.vol_multiplier, z_s, dt), cfg)

    if cfg.hedge_mode == DIRECT:
        zb = rng.standard_normal((n, N))
        e = np.zeros((n, N + 1))
        for k in range(N):
            e[:, k + 1] = e[:, k] * (1 - cfg.basis_kappa * dt) + cfg.basis_vol * np.sqrt(dt) * zb[:, k]
        F = S * np.exp(cfg.contango_annual * tau + e * tau / T)
    else:
        if mkt.h_cross is None or mkt.hedge_spot is None:
            raise ValueError("Cross-hedge mode needs market data for the hedge contract (not available offline).")
        rho = mkt.corr
        z_h = rho * z_s + np.sqrt(1.0 - rho**2) * rng.standard_normal((n, N))
        H = _apply_jump(_gbm(mkt.hedge_spot, mkt.hedge_vol * cfg.vol_multiplier, z_h, dt), cfg)
        F = H * np.exp(cfg.contango_annual * tau)
    return S, F


def hedge_size(cfg: Config, mkt: MarketInputs, f0: float) -> tuple[int, float]:
    """(number of contracts, hedged barrels of the hedge contract), rounded to whole contracts."""
    if cfg.hedge_mode == DIRECT:
        target_bbl = cfg.hedge_ratio * cfg.quantity_bbl
    else:
        # h* is in return terms: hedge notional = h* x cargo notional
        target_bbl = cfg.hedge_ratio * mkt.h_cross * cfg.quantity_bbl * mkt.spot / f0
    n_contracts = int(round(target_bbl / cfg.contract_size))
    return n_contracts, n_contracts * cfg.contract_size


@dataclass
class ScenarioResult:
    metrics: dict
    paths: dict


def run_scenario(cfg: Config, mkt: MarketInputs) -> ScenarioResult:
    S, F = simulate_paths(cfg, mkt)
    Q, s0, N = cfg.quantity_bbl, mkt.spot, cfg.horizon_days
    f0 = F[:, 0]
    n_contracts, hedge_bbl = hedge_size(cfg, mkt, float(f0[0]))

    # --- Margining: initial margin is posted on day 0 and stays locked ---
    vm_daily = -hedge_bbl * np.diff(F, axis=1)            # >0 = cash received (short futures)
    vm_cum = np.cumsum(vm_daily, axis=1)
    initial_margin = n_contracts * cfg.initial_margin_per_contract
    vm_need = np.maximum(0.0, -vm_cum).max(axis=1)        # peak cumulative variation margin paid
    cash_need = initial_margin + vm_need                  # total cash the trader must find
    buffer_breach = cash_need > cfg.trader_cash_buffer

    # --- Bank: loan accrues interest (ACT/360), collateral marked daily ---
    loan0 = cfg.ltv_initial * Q * s0
    cal_days = np.arange(N + 1) * 365.0 / TRADING_DAYS
    loan = loan0 * (1.0 + cfg.funding_rate * cal_days / 360.0)
    collat_phys = Q * S
    hedge_mtm = -hedge_bbl * (F - f0[:, None])
    collat_assigned = collat_phys + hedge_mtm
    ltv_no_assign = loan / collat_phys
    ltv_assign = np.where(collat_assigned > 0, loan / np.where(collat_assigned > 0, collat_assigned, 1.0), np.inf)
    breach_no_assign = (ltv_no_assign > cfg.ltv_max).any(axis=1)
    breach_assign = (ltv_assign > cfg.ltv_max).any(axis=1)

    # --- P&L at maturity: same cost definitions as the cash-and-carry analysis ---
    costs = carrying_costs(cfg, s0)
    fixed_costs = costs.total * Q
    # Margin funding, day by day: IM is funded for the whole horizon; cumulative VM is a cash
    # balance that costs interest when paid out and earns it when received (ACT/360).
    day_frac = (365.0 / TRADING_DAYS) / 360.0
    margin_funding = cfg.funding_rate * (initial_margin * cfg.act360 - vm_cum.sum(axis=1) * day_frac)
    margin_funding_mean = float(margin_funding.mean())
    phys_pnl = Q * (S[:, -1] - s0)
    hedge_pnl = hedge_bbl * (f0 - F[:, -1])
    pnl = phys_pnl + hedge_pnl - fixed_costs - margin_funding
    pnl_unhedged = phys_pnl - fixed_costs

    q05 = np.quantile(pnl, 0.05)
    metrics = {
        "Mean P&L ($M)": pnl.mean() / 1e6,
        "P&L std dev ($M)": pnl.std() / 1e6,
        "Unhedged std dev ($M)": pnl_unhedged.std() / 1e6,
        "Variance reduction": 1.0 - pnl.var() / pnl_unhedged.var(),
        "VaR 95% ($M)": -q05 / 1e6,
        "ES 95% ($M)": -pnl[pnl <= q05].mean() / 1e6,
        "P(loss)": (pnl < 0).mean(),
        "Margin funding cost, mean ($M)": margin_funding_mean / 1e6,
        "Contracts sold": float(n_contracts),
        "Initial margin ($M)": initial_margin / 1e6,
        "Cash need p99, IM + VM ($M)": np.quantile(cash_need, 0.99) / 1e6,
        "P(cash buffer exhausted)": buffer_breach.mean(),
        "P(LTV breach) no assignment": breach_no_assign.mean(),
        "P(LTV breach) with assignment": breach_assign.mean(),
    }
    paths = {"S": S, "F": F, "pnl": pnl, "pnl_unhedged": pnl_unhedged, "vm_cum": vm_cum,
             "initial_margin": initial_margin, "ltv_no_assign": ltv_no_assign, "ltv_assign": ltv_assign}
    return ScenarioResult(metrics, paths)


# Scenario names are constants so the report never depends on hand-typed strings.
BASE = "Base"
VOL_X2 = "Vol x2"
SHOCK_UP = "Shock +30%"
SHOCK_DOWN = "Shock -30%"
SQUEEZE = "Squeeze (+30%, vol x2)"


def build_scenarios(cfg: Config) -> dict[str, Config]:
    return {
        BASE: cfg,
        VOL_X2: replace(cfg, vol_multiplier=2.0),
        SHOCK_UP: replace(cfg, jump_size=0.30),
        SHOCK_DOWN: replace(cfg, jump_size=-0.30),
        SQUEEZE: replace(cfg, jump_size=0.30, vol_multiplier=2.0),
    }
