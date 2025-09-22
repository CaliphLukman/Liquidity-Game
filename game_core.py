from dataclasses import dataclass, field
import random
import uuid
from typing import Dict, List


@dataclass
class SecuritySpec:
    ticker: str
    face_price: float = 100.0
    bid_ask_bps: float = 20.0
    liquidity_score: int = 1


@dataclass
class Portfolio:
    name: str
    current_account: float = 0.0
    securities: Dict[str, SecuritySpec] = field(default_factory=dict)
    pos_qty: Dict[str, float] = field(default_factory=dict)
    repo_liabilities: List[Dict] = field(default_factory=list)  # each repo: amount, qty, ticker, rate, maturity
    td_assets: List[Dict] = field(default_factory=list)          # each TD: amount, rate, maturity
    pnl_realized: float = 0.0

    def market_value(self, prices: Dict[str, float]) -> float:
        sec_val = sum(self.pos_qty[t] * prices.get(t, self.securities[t].face_price)
                      for t in self.pos_qty)
        td_val = sum(a["amount"] for a in self.td_assets)
        return self.current_account + sec_val + td_val

    def summary(self, prices: Dict[str, float]) -> dict:
        sec_val = sum(self.pos_qty[t] * prices.get(t, self.securities[t].face_price)
                      for t in self.pos_qty)
        return {
            "current_account": self.current_account,
            "securities_mv": sec_val,
            "repo_outstanding": sum(l["amount"] for l in self.repo_liabilities),
            "td_invested": sum(a["amount"] for a in self.td_assets),
            "pnl_realized": self.pnl_realized,
            "total_mv": self.market_value(prices),
        }


# ------------------------
# Init
# ------------------------
def init_portfolios(tickers: List[str], prices: Dict[str, float], total_reserve: float = 200000.0):
    portfolios = []
    for i in range(4):
        p = Portfolio(name=f"Group {i+1}")
        p.current_account = total_reserve * 0.25  # fixed % in cash
        remaining_val = total_reserve - p.current_account
        weights = [random.random() for _ in tickers]
        s = sum(weights)
        weights = [w/s for w in weights]
        for t, w in zip(tickers, weights):
            spec = SecuritySpec(ticker=t)
            p.securities[t] = spec
            qty = (remaining_val * w) / prices[t]
            p.pos_qty[t] = qty
        portfolios.append(p)
    return portfolios


# ------------------------
# Withdrawals
# ------------------------
def generate_withdrawal(round_idx: int, portfolio_reserve: float, rng: random.Random) -> float:
    base = rng.uniform(0.18, 0.28) + 0.06 * round_idx
    base = min(base, 0.65)
    return round(portfolio_reserve * base, 2)


# ------------------------
# Actions
# ------------------------
def execute_repo(portfolio: Portfolio, ticker: str, amount: float, price: float,
                 current_round: int, rate: float):
    cap_qty = portfolio.pos_qty.get(ticker, 0.0)
    max_amt = cap_qty * price
    got = min(amount, max_amt)
    if got <= 0:
        return 0.0, None
    qty_repoed = got / price
    portfolio.pos_qty[ticker] -= qty_repoed
    portfolio.current_account += got
    repo_id = str(uuid.uuid4())
    portfolio.repo_liabilities.append({
        "id": repo_id, "amount": got, "qty_repoed": qty_repoed,
        "ticker": ticker, "rate": rate, "maturity": current_round + 1
    })
    return got, repo_id


def restore_repo(portfolio: Portfolio, liab: Dict):
    ticker, qty = liab["ticker"], liab["qty_repoed"]
    portfolio.pos_qty[ticker] += qty


def execute_sale(portfolio: Portfolio, ticker: str, qty: float, price: float):
    if qty <= 0 or qty > portfolio.pos_qty.get(ticker, 0.0):
        return {"proceeds": 0.0, "qty": 0.0, "pnl_delta": 0.0, "eff_price": 0.0}
    spec = portfolio.securities[ticker]
    half_bps = (spec.bid_ask_bps / 2.0)
    eff_price = price * (1 - half_bps / 10000.0)
    proceeds = qty * eff_price
    pnl_delta = (eff_price - spec.face_price) * qty
    portfolio.pos_qty[ticker] -= qty
    portfolio.current_account += proceeds
    portfolio.pnl_realized += pnl_delta
    return {"proceeds": proceeds, "qty": qty, "pnl_delta": pnl_delta, "eff_price": eff_price}


def execute_buy(portfolio: Portfolio, ticker: str, qty: float, price: float):
    if qty <= 0:
        return {"cost": 0.0, "qty": 0.0, "ticker": ticker}
    spec = portfolio.securities[ticker]
    half_bps = (spec.bid_ask_bps / 2.0)
    eff_price = price * (1 + half_bps / 10000.0)
    cost = qty * eff_price
    if portfolio.current_account < cost:
        return {"cost": 0.0, "qty": 0.0, "ticker": ticker}
    portfolio.current_account -= cost
    portfolio.pos_qty[ticker] = portfolio.pos_qty.get(ticker, 0.0) + qty
    return {"cost": cost, "qty": qty, "ticker": ticker, "effective_price": eff_price}


def execute_invest_td(portfolio: Portfolio, amount: float, current_round: int, rate: float):
    if amount <= 0 or portfolio.current_account < amount:
        return []
    td_id = str(uuid.uuid4())
    portfolio.current_account -= amount
    portfolio.td_assets.append({
        "id": td_id, "amount": amount, "rate": rate,
        "maturity": current_round + 2  # updated: 2 rounds later
    })
    return [td_id]


def execute_redeem_td(portfolio: Portfolio, amount: float, current_round: int, penalty_rate=0.01):
    to_redeem = amount
    redeemed = []
    principal, penalty = 0.0, 0.0
    for asset in list(portfolio.td_assets):
        if to_redeem <= 0:
            break
        take = min(asset["amount"], to_redeem)
        principal += take
        if asset["maturity"] > current_round:
            penalty += take * penalty_rate
        asset["amount"] -= take
        if asset["amount"] <= 1e-9:
            portfolio.td_assets.remove(asset)
        redeemed.append({
            "id": asset["id"], "taken": take,
            "rate": asset["rate"], "maturity": asset["maturity"]
        })
        to_redeem -= take
    portfolio.current_account += principal - penalty
    portfolio.pnl_realized -= penalty
    return {"redeemed": redeemed, "principal": principal, "penalty": penalty}


# ------------------------
# Maturities
# ------------------------
def process_maturities(portfolio: Portfolio, current_round: int):
    # Repos
    for liab in list(portfolio.repo_liabilities):
        if liab["maturity"] == current_round:
            repay = liab["amount"] * (1 + liab["rate"])
            portfolio.current_account -= repay
            portfolio.pnl_realized -= (repay - liab["amount"])
            restore_repo(portfolio, liab)
            portfolio.repo_liabilities.remove(liab)
    # TDs
    for asset in list(portfolio.td_assets):
        if asset["maturity"] == current_round:
            repay = asset["amount"] * (1 + asset["rate"])
            portfolio.current_account += repay
            portfolio.pnl_realized += (repay - asset["amount"])
            portfolio.td_assets.remove(asset)
