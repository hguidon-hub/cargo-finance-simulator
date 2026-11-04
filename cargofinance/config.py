"""Deal parameters. Every number here is an assumption the user can override from the CLI."""

from __future__ import annotations

from dataclasses import dataclass

TRADING_DAYS = 252
DIRECT = "direct"  # hedge the cargo with futures on the same grade (Brent vs ICE Brent)
CROSS = "cross"    # hedge with a correlated contract (Brent cargo vs NYMEX WTI)


@dataclass(frozen=True)
class Config:
    # --- Cargo ---
    commodity: str = "Brent"
    ticker_physical: str = "BZ=F"      # proxy for the physical price (front-month future)
    ticker_hedge: str = "CL=F"         # instrument used in cross-hedge mode
    quantity_bbl: float = 1_000_000    # 1 Mb ~ one Suezmax cargo
    contract_size: float = 1_000       # 1 Brent / WTI future = 1,000 bbl
    horizon_days: int = 63             # ~3 months of trading days

    # --- Forward curve (ASSUMPTION: not read from the market, see README) ---
    contango_annual: float = 0.16      # annualised slope, F = S * exp(c * tau)

    # --- Carrying costs ---
    storage_cost_bbl_month: float = 0.50   # $/bbl/month
    funding_rate: float = 0.063            # bank loan rate (benchmark + margin), ACT/360
    equity_cost_rate: float | None = None  # cost of the trader's own (1 - LTV) share; None = funding_rate

    # --- Hedge and margining ---
    hedge_mode: str = DIRECT
    hedge_ratio: float = 1.0               # multiplier on the 1:1 (direct) or h* (cross) hedge
    initial_margin_per_contract: float = 6_000
    trader_cash_buffer: float = 10_000_000  # must cover initial margin AND variation margin

    # --- Bank financing ---
    ltv_initial: float = 0.80
    ltv_max: float = 0.90

    # --- Basis between physical and paper (direct mode only) ---
    basis_vol: float = 0.02
    basis_kappa: float = 5.0

    # --- Stress tests ---
    vol_multiplier: float = 1.0
    jump_day: int = 10
    jump_size: float = 0.0

    # --- Monte Carlo ---
    n_paths: int = 10_000
    seed: int = 42

    @property
    def T(self) -> float:
        """Horizon in years (trading-day convention, used for diffusion and the curve)."""
        return self.horizon_days / TRADING_DAYS

    @property
    def act360(self) -> float:
        """Money-market accrual factor: calendar days / 360."""
        return self.T * 365.0 / 360.0

    @property
    def equity_rate(self) -> float:
        return self.funding_rate if self.equity_cost_rate is None else self.equity_cost_rate

    def validate(self) -> None:
        if self.hedge_mode not in (DIRECT, CROSS):
            raise ValueError(f"hedge_mode must be '{DIRECT}' or '{CROSS}'.")
        if not 0 < self.ltv_initial < self.ltv_max:
            raise ValueError("Need 0 < ltv_initial < ltv_max.")
        if self.horizon_days < 2 or not 0 <= self.jump_day <= self.horizon_days:
            raise ValueError("horizon_days must be >= 2 and jump_day within the horizon.")
        if self.quantity_bbl <= 0 or self.n_paths < 100:
            raise ValueError("quantity_bbl must be positive and n_paths >= 100.")
