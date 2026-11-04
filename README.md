# Hedged Cargo Financing Simulator

Monte Carlo model of a **physical oil cargo bought, stored, hedged with futures and financed by a
bank** through an inventory-backed facility. It answers the four questions a commodity trade
finance desk asks before funding the deal:

1. **Carry**: does the contango pay for storage and financing?
2. **Hedge**: how much price risk does the futures hedge remove, and what basis risk is left?
3. **Liquidity**: how much cash does the trader need to meet margin calls on the hedge?
4. **Credit**: how likely is the bank's loan-to-value to breach its trigger, with and without an
   assignment of the hedging contracts?

```
$ python main.py --spot 65 --vol 0.30          # offline, no internet needed
```

## Key result (`--spot 65 --vol 0.30`: 1 Mb Brent, 3 months, 80% LTV, 16% contango assumed)

| | |
|---|---|
| Locked-in carry | +0.11 M$ (+0.11 $/bbl) |
| Margin funding cost | −0.07 M$, i.e. **64% of the carry** |
| Mean hedged P&L | +0.04 M$, σ 0.09 M$ vs 9.7 M$ unhedged |
| p99 cash tied up in margin | 33 M$ against a 10 M$ buffer (buffer exhausted in 54% of paths) |
| P(LTV breach), inventory only | 46% (100% after a −30% shock) |
| P(LTV breach), hedge assigned | 0% |

The trade is hedged against price and still fragile: a rally turns a profitable carry into a
liquidity crisis, because the short futures pay variation margin daily while the cargo is only
sold at the end. That is why lenders take an assignment of the hedge **and** size a separate
margin line. The LTV chart shows the mirror image: without the hedge assigned, a 30% fall breaches
the bank's 90% trigger on every path.

## How the model works

| Block | Method |
|---|---|
| Market inputs | Front-month Brent (`BZ=F`) and WTI (`CL=F`) from Yahoo: spot, annualised vol, correlation, minimum-variance hedge ratio h* = Cov(ΔS,ΔF)/Var(ΔF), residual vol. Or `--spot`/`--vol` offline. |
| Physical price | Driftless GBM (no price view), plus optional jump and vol-multiplier stresses. |
| Hedge, direct mode | Brent futures with F_t = S_t·exp(c·τ + e_t·τ/T); e is an Ornstein-Uhlenbeck basis, and F converges to S at expiry. |
| Hedge, cross mode | WTI futures sized at h*, simulated with the estimated correlation: the 1 − ρ² of variance the grades do not share is unhedgeable. |
| Carrying costs | Storage $/bbl/month; bank interest on the LTV share and an equity cost on the trader's share, ACT/360. The same function feeds the carry analysis and the Monte Carlo. |
| Margining | Initial margin locked on day one; variation margin funded day by day (cost when paid, interest when received). Buffer breach if IM + peak VM > cash buffer. |
| Bank | Loan accrues interest daily; LTV marked on the inventory alone, or on inventory + hedge MTM when the hedge is assigned. |
| Stress tests | Vol ×2, +30% and −30% price jumps, squeeze (+30% with vol ×2). |
| Output | VaR / ES 95%, P(loss), P&L bridge (carry − margin funding ± basis), plain-English interpretation, 2×2 dashboard. |

## Run it

```bash
git clone https://github.com/hguidon-hub/cargo-finance-simulator.git
cd cargo-finance-simulator
pip install -r requirements.txt

python main.py                                    # live Yahoo data, direct hedge
python main.py --hedge-mode cross                 # Brent cargo hedged with WTI
python main.py --spot 65 --vol 0.30 --contango 0.04 --horizon 126   # flat curve, 6-month carry
python main.py --buffer 30e6 --ltv 0.75 --save cargo.png --no-plot  # structure the facility
python main.py --help                             # every assumption is a flag
```

## Validation

18 offline tests run on every push (GitHub Actions):

- Futures converge to spot at expiry; the simulated spot is a martingale
- With no frictions, the hedged P&L equals the locked-in carry **on every path**
- Cumulative variation margin equals the hedge P&L
- Break-even contango gives exactly zero carry margin; financing covers 100% of the cargo
- Initial margin counts against the cash buffer
- A −30% crash breaches LTV on every path without assignment, almost never with it
- Cross-hedge: the simulated variance reduction matches ρ², and h* is recovered from synthetic prices

```bash
python -m unittest discover -s tests -t . -v
```

## Assumptions to challenge

- **Contango is an input, not market data.** The carry verdict depends on it entirely, so the report
  prints a sensitivity table and the break-even slope. Read the live Brent curve (front vs M+3)
  before trusting it.
- **Basis in direct mode is assumed** (2% vol, fast mean reversion), which makes the hedge almost
  perfect by construction. Cross mode estimates the basis from data; Dated Brent vs ICE futures
  basis data would be the proper calibration.
- **Front-month continuous futures** include roll jumps, which inflate the estimated vol and bias h*.
- Constant volatility and contango over the horizon; no curve-shape risk on rolled hedges, no
  quality, demurrage or counterparty risk on the physical buyer.

## Author

Hugo Guidon, MSc in Finance (Financial Markets), emlyon business school.
