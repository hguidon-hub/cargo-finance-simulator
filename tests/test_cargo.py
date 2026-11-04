"""Invariant tests: offline, deterministic, no network."""

import math
import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from cargofinance.carry import breakeven_contango, carry_sensitivity, carrying_costs, cash_and_carry
from cargofinance.config import CROSS, Config
from cargofinance.market import inputs_from_prices, min_variance_hedge_ratio, offline_inputs
from cargofinance.report import interpret
from cargofinance.simulation import BASE, build_scenarios, run_scenario, simulate_paths

MKT = offline_inputs(spot=70.0, vol=0.30)
SMALL = Config(n_paths=2_000)


def no_frictions(cfg: Config) -> Config:
    """No storage, no financing, no margin, no basis: only the curve is left."""
    return replace(cfg, storage_cost_bbl_month=0.0, funding_rate=0.0, equity_cost_rate=0.0,
                   initial_margin_per_contract=0.0, basis_vol=0.0)


class TestCarry(unittest.TestCase):
    def test_breakeven_contango_gives_zero_margin(self):
        be = breakeven_contango(SMALL, 70.0)
        margin = cash_and_carry(replace(SMALL, contango_annual=be), 70.0)["Carry margin ($/bbl)"]
        self.assertAlmostEqual(margin, 0.0, places=10)

    def test_financing_covers_100_percent_of_the_cargo(self):
        # bank share + equity share at the same rate = full cargo at that rate (v1 mixed 80% and 100%)
        c = carrying_costs(SMALL, 70.0)
        self.assertAlmostEqual(c.financing, 70.0 * SMALL.funding_rate * SMALL.act360, places=12)
        self.assertAlmostEqual(c.debt_interest / c.financing, SMALL.ltv_initial)

    def test_act360_accrual(self):
        self.assertAlmostEqual(SMALL.act360, 63 / 252 * 365 / 360)

    def test_sensitivity_is_monotonic_and_contains_breakeven(self):
        s = carry_sensitivity(SMALL, 70.0)
        self.assertTrue(s["Carry margin ($/bbl)"].is_monotonic_increasing)
        self.assertTrue(np.isclose(s["Carry margin ($/bbl)"], 0.0).any())


class TestSimulation(unittest.TestCase):
    def test_futures_converge_to_spot(self):
        S, F = simulate_paths(SMALL, MKT)
        np.testing.assert_allclose(F[:, -1], S[:, -1], rtol=1e-12)
        np.testing.assert_allclose(F[:, 0], 70.0 * math.exp(SMALL.contango_annual * SMALL.T))

    def test_spot_is_a_martingale(self):
        S, _ = simulate_paths(replace(SMALL, n_paths=40_000), MKT)
        self.assertAlmostEqual(S[:, -1].mean() / 70.0, 1.0, delta=0.01)

    def test_perfect_hedge_locks_the_carry_on_every_path(self):
        cfg = no_frictions(SMALL)
        r = run_scenario(cfg, MKT)
        locked = cash_and_carry(cfg, 70.0)["Total carry margin ($M)"] * 1e6
        np.testing.assert_allclose(r.paths["pnl"], locked, atol=1e-4)

    def test_variation_margin_equals_hedge_pnl(self):
        r = run_scenario(SMALL, MKT)
        S, F = r.paths["S"], r.paths["F"]
        hedge_pnl = SMALL.quantity_bbl * (F[:, 0] - F[:, -1])
        np.testing.assert_allclose(r.paths["vm_cum"][:, -1], hedge_pnl, rtol=1e-9)

    def test_initial_margin_counts_against_the_buffer(self):
        # buffer smaller than IM alone: the trader is short of cash on every path (v1 missed this)
        cfg = replace(SMALL, trader_cash_buffer=5_000_000, initial_margin_per_contract=6_000)
        self.assertEqual(run_scenario(cfg, MKT).metrics["P(cash buffer exhausted)"], 1.0)

    def test_assignment_protects_the_bank_in_a_crash(self):
        m = run_scenario(replace(SMALL, jump_size=-0.30), MKT).metrics
        self.assertEqual(m["P(LTV breach) no assignment"], 1.0)
        self.assertLess(m["P(LTV breach) with assignment"], 0.05)

    def test_loan_accrues_interest(self):
        cfg = replace(SMALL, vol_multiplier=1e-6, contango_annual=0.0, basis_vol=0.0)
        r = run_scenario(cfg, offline_inputs(70.0, 1e-4))
        self.assertAlmostEqual(r.paths["ltv_no_assign"][0, -1] / cfg.ltv_initial, 1 + cfg.funding_rate * cfg.act360, places=4)

    def test_reproducible(self):
        a = run_scenario(SMALL, MKT).metrics
        b = run_scenario(SMALL, MKT).metrics
        self.assertEqual(a, b)


def synthetic_prices(n=500, rho=0.9, seed=1):
    rng = np.random.default_rng(seed)
    z1, z2 = rng.standard_normal((2, n))
    r_h = 0.02 * z1
    r_s = 0.8 * r_h + 0.02 * np.sqrt(1 - rho**2) * z2
    idx = pd.bdate_range("2024-11-01", periods=n + 1)
    return pd.DataFrame({
        "BZ=F": 70 * np.exp(np.concatenate([[0], np.cumsum(r_s)])),
        "CL=F": 66 * np.exp(np.concatenate([[0], np.cumsum(r_h)])),
    }, index=idx)


class TestCrossHedge(unittest.TestCase):
    def test_min_variance_ratio(self):
        rng = np.random.default_rng(0)
        f = pd.Series(rng.normal(0, 0.02, 2000))
        s = 1.5 * f + pd.Series(rng.normal(0, 0.001, 2000))
        self.assertAlmostEqual(min_variance_hedge_ratio(s, f), 1.5, places=2)

    def test_inputs_from_prices(self):
        mkt = inputs_from_prices(synthetic_prices(), "BZ=F", "CL=F")
        self.assertAlmostEqual(mkt.h_cross, 0.8, delta=0.05)
        self.assertGreater(mkt.corr, 0.8)
        self.assertLess(mkt.resid_vol, mkt.vol)

    def test_cross_hedge_leaves_basis_risk(self):
        mkt = inputs_from_prices(synthetic_prices(), "BZ=F", "CL=F")
        cfg = replace(SMALL, hedge_mode=CROSS)
        m = run_scenario(cfg, mkt).metrics
        expected = mkt.corr**2  # share of variance a min-variance hedge can remove
        self.assertAlmostEqual(m["Variance reduction"], expected, delta=0.05)
        self.assertLess(m["Variance reduction"], run_scenario(SMALL, mkt).metrics["Variance reduction"])

    def test_cross_mode_requires_market_data(self):
        with self.assertRaises(ValueError):
            run_scenario(replace(SMALL, hedge_mode=CROSS), MKT)


class TestReport(unittest.TestCase):
    def test_interpretation_runs_for_all_scenarios(self):
        table = pd.DataFrame({k: run_scenario(c, MKT).metrics for k, c in build_scenarios(SMALL).items()})
        text = interpret(SMALL, MKT, cash_and_carry(SMALL, 70.0), table)
        for tag in ("[Carry]", "[Hedge]", "[Liquidity]", "[Bank]", "[Stress]"):
            self.assertIn(tag, text)

    def test_interpretation_survives_base_only(self):
        table = pd.DataFrame({BASE: run_scenario(SMALL, MKT).metrics})
        self.assertIn("[Bank]", interpret(SMALL, MKT, cash_and_carry(SMALL, 70.0), table))


if __name__ == "__main__":
    unittest.main()
