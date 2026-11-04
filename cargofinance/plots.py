"""Base-case dashboard."""

from __future__ import annotations

import numpy as np

from .config import Config
from .simulation import ScenarioResult

BLUE, RED, GREY, GREEN, ORANGE = "#1f4e79", "#c0392b", "#7f8c8d", "#1e8449", "#d68910"


def plot_base_case(cfg: Config, result: ScenarioResult, save_path: str | None = None, show: bool = True,
                   locked_carry: float | None = None):
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = result.paths
    days = np.arange(cfg.horizon_days + 1)
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"{cfg.commodity} cargo {cfg.quantity_bbl / 1e6:.1f} Mb, {cfg.horizon_days} trading days "
                 f"— base case ({cfg.n_paths:,} paths)", fontsize=14, fontweight="bold")

    # 1. Spot fan chart
    q = np.percentile(p["S"], [1, 5, 50, 95, 99], axis=0)
    ax[0, 0].fill_between(days, q[0], q[4], color=BLUE, alpha=0.12, label="1–99%")
    ax[0, 0].fill_between(days, q[1], q[3], color=BLUE, alpha=0.25, label="5–95%")
    ax[0, 0].plot(days, q[2], color=BLUE, lw=2, label="median")
    ax[0, 0].set(title="Simulated physical price", xlabel="trading days", ylabel="\\$/bbl")
    ax[0, 0].legend(fontsize=9)

    # 2. Hedged P&L distribution (the unhedged one is ~100x wider: shown as a number, not bars)
    pnl, pnl_u = p["pnl"] / 1e6, p["pnl_unhedged"] / 1e6
    ax[0, 1].hist(pnl, bins=80, color=GREEN, alpha=0.8, label="hedged P&L")
    ax[0, 1].axvline(pnl.mean(), color=BLUE, lw=1.5, label=f"mean {pnl.mean():+.2f} M\\$")
    if locked_carry is not None:
        ax[0, 1].axvline(locked_carry, color=ORANGE, ls="--", lw=1.5, label=f"locked carry {locked_carry:+.2f} M\\$")
    ax[0, 1].axvline(0, color="black", lw=0.8)
    ax[0, 1].set(title=f"Hedged P&L at maturity (σ {pnl.std():.2f} M\\$ vs {pnl_u.std():.1f} M\\$ unhedged)",
                 xlabel="M\\$, after storage, financing and margin funding")
    ax[0, 1].legend(fontsize=9)

    # 3. Cash consumed: initial margin + cumulative variation margin paid
    vm_paid = np.maximum(0.0, -np.concatenate([np.zeros((cfg.n_paths, 1)), p["vm_cum"]], axis=1))
    need = (p["initial_margin"] + vm_paid) / 1e6
    for pct, color in ((50, GREEN), (95, ORANGE), (99, RED)):
        ax[1, 0].plot(days, np.percentile(need, pct, axis=0), color=color, lw=1.8, label=f"p{pct}")
    ax[1, 0].axhline(cfg.trader_cash_buffer / 1e6, color="black", ls="--", label="cash buffer")
    ax[1, 0].axhline(p["initial_margin"] / 1e6, color=GREY, ls=":", label="initial margin")
    ax[1, 0].set(title="Cash tied up in margin (IM + VM paid)", xlabel="trading days", ylabel="M\\$")
    ax[1, 0].legend(fontsize=9)

    # 4. Bank LTV
    for key, lab, color in (("ltv_no_assign", "inventory only", RED), ("ltv_assign", "hedge assigned", BLUE)):
        ax[1, 1].plot(days, np.percentile(p[key], 99, axis=0), color=color, lw=1.8, label=f"p99 {lab}")
    ax[1, 1].axhline(cfg.ltv_max, color="black", ls="--", label=f"max LTV {cfg.ltv_max:.0%}")
    ax[1, 1].axhline(cfg.ltv_initial, color=GREY, ls=":", label=f"initial LTV {cfg.ltv_initial:.0%}")
    ax[1, 1].set(title="Bank loan-to-value (99th percentile)", xlabel="trading days")
    ax[1, 1].legend(fontsize=9)

    for a in ax.flat:
        a.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    return fig
