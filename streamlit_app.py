# streamlit_app.py
import os
import json
import time
import random
from typing import List, Dict, Tuple, Any, Optional

import pandas as pd
import streamlit as st

from game_core import (
    Portfolio, init_portfolios, generate_withdrawal,
    execute_repo, execute_sale, execute_buy,
    execute_invest_td, execute_redeem_td,
    process_maturities
)

# ------------------------
# App + Theme (light)
# ------------------------
st.set_page_config(page_title="Liquidity Tranche Simulation", layout="wide")

st.markdown("""
<style>
:root{
  --navy:#0B1F3B;         /* sidebar bg */
  --navy-strong:#0B3D91;  /* headings & buttons */
  --white:#FFFFFF;
  --ink:#111827;          /* near-black text */
  --green:#006400;        /* big totals ($ figures) & captions */
  --light-navy:#9DB7FF;   /* subtle outlines on dark bg */
}

/* Main */
.stApp{ background:var(--white); color:var(--ink); }
h1,h2,h3{ color:var(--navy-strong) !important; }
.stMarkdown,.stText,label,p,span,div{ color:var(--ink); }

/* Captions in dark green */
.stCaption, .stCaption * { color: var(--green) !important; }

/* Big $ totals */
.stMetricValue{ color:var(--green) !important; font-weight:800 !important; }
.stMetricLabel{ color:var(--ink) !important; font-weight:600 !important; }

/* Ticker line (bold black) */
.ticker-line{ color:#000 !important; font-weight:700; }

/* Sidebar (dark navy, pure white text) */
section[data-testid="stSidebar"]{
  background:var(--navy);
  color:var(--white);
  border-right:4px solid var(--navy-strong);
  box-shadow:4px 0 12px rgba(0,0,0,.08);
}
section[data-testid="stSidebar"] *{
  color:var(--white) !important;
  opacity:1 !important;
}

/* Sidebar inputs: dark fields with white text */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] select{
  background:#102A53 !important;
  color:var(--white) !important;
  border:1.5px solid var(--light-navy) !important;
  border-radius:8px !important;
}
section[data-testid="stSidebar"] ::placeholder{
  color:#EAF2FF !important;
  opacity:1 !important;
}

/* File uploader — no borders/dots; black '×' on chip */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] label{
  border:none !important; outline:none !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"],
section[data-testid="stSidebar"] .stFileUploader div[role="button"]{
  background:#102A53 !important;
  border:none !important;
  box-shadow:none !important;
  border-radius:10px !important;
  color:var(--white) !important;
}
/* 'Browse files' button: white bg + navy text */
section[data-testid="stSidebar"] .stFileUploader button{
  background:var(--white) !important;
  color:var(--navy-strong) !important;
  border:1.5px solid var(--light-navy) !important;
  border-radius:8px !important;
  font-weight:700 !important;
}
/* Uploaded file chip: white with black '×' */
.stFileUploader [data-baseweb="tag"]{
  background:#FFFFFF !important;
  color:#111827 !important;
  border:1px solid #E6EAF2 !important;
  border-radius:10px !important;
}
.stFileUploader [data-baseweb="tag"] button svg,
.stFileUploader [data-baseweb="tag"] [role="button"] svg{
  display:none !important;
}
.stFileUploader [data-baseweb="tag"] button,
.stFileUploader [data-baseweb="tag"] [role="button"]{
  position:relative !important;
  background:transparent !important;
  border:none !important;
  box-shadow:none !important;
}
.stFileUploader [data-baseweb="tag"] button::after,
.stFileUploader [data-baseweb="tag"] [role="button"]::after{
  content:"\\00D7"; /* × */
  color:#000 !important;
  font-weight:900 !important;
  font-size:14px !important;
  line-height:1 !important;
  display:inline-block !important;
}

/* Number steppers: main = black icons; sidebar = white icons */
[data-testid="stNumberInput"] button{
  background:transparent !important;
  color:#000 !important;
  border:1px solid var(--navy-strong) !important;
  border-radius:6px !important;
}
[data-testid="stNumberInput"] button svg,
[data-testid="stNumberInput"] button span{
  color:#000 !important; fill:#000 !important;
}
section[data-testid="stSidebar"] [data-testid="stNumberInput"] button{
  background:transparent !important;
  color:var(--white) !important;
  border:1px solid var(--light-navy) !important;
}
section[data-testid="stSidebar"] [data-testid="stNumberInput"] button svg,
section[data-testid="stSidebar"] [data-testid="stNumberInput"] button span{
  color:var(--white) !important; fill:var(--white) !important;
}

/* Buttons: dark blue bg + white text everywhere */
div.stButton > button{
  background:var(--navy-strong) !important;
  color:var(--white) !important;
  border:0 !important;
  border-radius:8px !important;
  font-weight:700 !important;
  padding:.45rem .9rem !important;
}
div.stButton > button *{ color:var(--white) !important; fill:var(--white) !important; }
div.stButton > button:hover{ filter:brightness(.92); }
</style>
""", unsafe_allow_html=True)

# ------------------------
# Config
# ------------------------
BASE_REPO_RATE = 0.015
BASE_TD_RATE   = 0.0175
DAILY_SPREAD   = 0.005
TD_PENALTY     = 0.01
TD_MAT_GAP     = 2
MAX_GROUPS_UI  = 8

# Simple file-based sync (single game, no DB)
SHARED_STATE_PATH   = ".shared_state.json"   # host pushes session info + claims + current_round  
UPLOADED_CSV_PATH   = ".uploaded.csv"        # host saves CSV here
PLAYER_PORTFOLIOS_PATH = ".player_portfolios.json"  # real-time player portfolio states
SNAPSHOT_PATH       = ".snapshot.json"       # published round info for all
ACTIONS_QUEUE_PATH = ".actions_queue.json"  # player actions queue for host processing
# ------------------------
# Small helpers
# ------------------------
def _safe_rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

def _json_read(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _json_write_atomic(path: str, payload: Any):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp, path)

def _json_mutate(path: str, default, fn):
    data = _json_read(path, default)
    new_data = fn(data if isinstance(data, (dict, list)) else default)
    _json_write_atomic(path, new_data)
    return new_data

def _now_ts() -> float:
    return time.time()

def _fmt_money(x: float, nd: int = 2) -> str:
    """Format as dollar amount with proper comma separators."""
    try:
        return f"${x:,.{nd}f}"
    except Exception:
        return f"${x}"

# ------------------------
# Session State bootstrap
# ------------------------
def _init_state():
    ss = st.session_state
    ss.initialized = False
    ss.rng_seed = 1234
    ss.rounds = 3
    ss.current_round = 0
    ss.withdrawals: List[float] = []
    ss.portfolios: List[Portfolio] = []
    ss.logs: Dict[str, List[dict]] = {}
    ss.price_df = None
    ss.last_maturity_round = -1
    ss.inited_rounds = set()
    ss.num_groups = 4
    ss.role = "Host"
    ss.player_group_index = 0
    ss.player_name = ""
    # Player-side: keep a local staged "delta" to reflect actions immediately
    ss.staged_inputs: Dict[str, Dict[str, Any]] = {}  # key=f"r{r}_group_{name}"
    ss.player_portfolio_cache: Dict[str, Dict] = {}   # cached instant state adjustments

if "initialized" not in st.session_state:
    _init_state()

# ------------------------
# Game helpers
# ------------------------
def price_row_for_round(r: int) -> int:
    df = st.session_state.price_df
    return min(r, len(df) - 1)

def base_prices_for_round(r: int, df: pd.DataFrame, tickers: List[str]) -> Tuple[str, Dict[str, float]]:
    ix = price_row_for_round(r)
    return str(df.loc[ix, "date"]), {t: float(df.loc[ix, t]) for t in tickers}

def daily_rates_for_round(r: int) -> Tuple[float, float]:
    rng = random.Random(st.session_state.rng_seed * 991 + r * 7919)
    repo_delta = rng.uniform(-DAILY_SPREAD, DAILY_SPREAD)
    td_delta   = rng.uniform(-DAILY_SPREAD, DAILY_SPREAD)
    return max(0.0, BASE_REPO_RATE + repo_delta), max(0.0, BASE_TD_RATE + td_delta)

def ensure_round_initialized(r: int, prices_for_mv: Dict[str, float]):
    ss = st.session_state
    if r >= ss.rounds or r in ss.inited_rounds:
        return
    if len(ss.withdrawals) < ss.rounds:
        ss.withdrawals = [0.0 for _ in range(ss.rounds)]
    req = generate_withdrawal(
        r,
        ss.portfolios[0].market_value(prices_for_mv),
        random.Random(ss.rng_seed + 10007*r)
    )
    ss.withdrawals[r] = float(req)
    ss.inited_rounds.add(r)

def compute_remaining_for_group(group_name: str, r: int, req_for_round: float) -> float:
    logs = [L for L in st.session_state.logs.get(group_name, []) if L["round"] == r+1]
    used = 0.0
    for L in logs:
        for t, d in L["actions"]:
            if t == "cash":
                used += float(d.get("used", 0.0))
            elif t in ("repo","sell","redeem_td"):
                used += float(d.get("use", 0.0))
    return max(0.0, round(req_for_round - used, 2))

def calculate_effective_price_with_spread(base_price: float, is_buy: bool, liquidity_factor: float = 0.005) -> float:
    """Calculate effective price with bid-ask spread for trading."""
    if is_buy:
        return base_price * (1 + liquidity_factor)  # Pay slightly more when buying
    else:
        return base_price * (1 - liquidity_factor)  # Receive slightly less when selling

# ------------------------
# Player-side instant state calculation helpers
# ------------------------
def _apply_staged_deltas_to_summary(base_summary: dict, base_positions: dict, staged_inputs, prices: dict, r: int, total_withdrawal: float = 0.0, host_remaining: float = 0.0) -> Tuple[dict, dict]:
    """Apply staged inputs to create instant player-side view of their portfolio state."""
    # Copy base state
    new_summary = dict(base_summary)
    new_positions = dict(base_positions)
    
    # Initialize withdrawal tracking - CRITICAL: start fresh each time
    new_summary["_withdrawal_used"] = 0.0
    
    # If no staged inputs, return unchanged
    if not staged_inputs:
        return new_summary, new_positions
    
    # If staged_inputs is a list of actions (cumulative), process each one
    if isinstance(staged_inputs, list):
        remaining_for_actions = host_remaining
        for action_inputs in staged_inputs:
            # Make a copy to avoid modifying original
            action_copy = dict(action_inputs)
            action_copy["_total_withdrawal"] = total_withdrawal
            action_copy["_host_remaining"] = remaining_for_actions
            new_summary, new_positions = _apply_single_staged_action(new_summary, new_positions, action_copy, prices, r)
            # Update remaining for next action
            remaining_for_actions = max(0.0, remaining_for_actions - (new_summary.get("_withdrawal_used", 0.0) - action_copy.get("_prev_used", 0.0)))
    else:
        # Single action
        staged_copy = dict(staged_inputs)
        staged_copy["_total_withdrawal"] = total_withdrawal
        staged_copy["_host_remaining"] = host_remaining
        new_summary, new_positions = _apply_single_staged_action(new_summary, new_positions, staged_copy, prices, r)
    
    return new_summary, new_positions

def _apply_single_staged_action(summary: dict, positions: dict, inputs: dict, prices: dict, r: int) -> Tuple[dict, dict]:
    """Apply a single staged action to the current state."""
    # Extract inputs
    cash_use = max(0.0, float(inputs.get("cash", 0.0)))
    repo_amt = max(0.0, float(inputs.get("repo_amt", 0.0)))
    repo_tick = inputs.get("repo_tick", "(none)")
    redeem_amt = max(0.0, float(inputs.get("redeem", 0.0)))
    invest_amt = max(0.0, float(inputs.get("invest_td", 0.0)))
    sell_qty = max(0.0, float(inputs.get("sell_qty", 0.0)))
    sell_tick = inputs.get("sell_tick", "(none)")
    buy_qty = max(0.0, float(inputs.get("buy_qty", 0.0)))
    buy_tick = inputs.get("buy_tick", "(none)")
    
    # Apply changes to current account
    new_ca = float(summary["current_account"])
    
    # Track total used towards withdrawal requirement
    total_used_for_withdrawal = 0.0
    
    # 1. Cash use - reduces CA and counts towards withdrawal
    if cash_use > 0:
        actual_cash_use = min(cash_use, max(0.0, new_ca))
        new_ca -= actual_cash_use
        total_used_for_withdrawal += actual_cash_use
    
    # 2. Repo - get cash from repo, reduce securities, cash used for withdrawal
    if repo_tick != "(none)" and repo_amt > 0 and repo_tick in prices:
        price = float(prices[repo_tick])
        available_collateral = float(positions.get(repo_tick, 0.0)) * price
        actual_repo = min(repo_amt, available_collateral)
        if actual_repo > 0:
            securities_to_repo = actual_repo / price
            positions[repo_tick] = max(0.0, float(positions.get(repo_tick, 0.0)) - securities_to_repo)
            summary["repo_outstanding"] = float(summary["repo_outstanding"]) + actual_repo
            # Repo proceeds are used directly for withdrawal, not added to current account
            total_used_for_withdrawal += actual_repo
    
    # 3. Redeem TD - get cash, reduce TD investment, net proceeds used for withdrawal
    if redeem_amt > 0:
        available_td = float(summary["td_invested"])
        actual_redeem = min(redeem_amt, available_td)
        if actual_redeem > 0:
            penalty = actual_redeem * TD_PENALTY
            net_proceeds = actual_redeem - penalty
            summary["td_invested"] = available_td - actual_redeem
            summary["pnl_realized"] = float(summary["pnl_realized"]) - penalty
            # Net proceeds (after penalty) used for withdrawal, not added to current account
            total_used_for_withdrawal += net_proceeds
    
    # 4. Sell - get cash, reduce securities, proceeds used for withdrawal
    if sell_tick != "(none)" and sell_qty > 0 and sell_tick in prices:
        price = calculate_effective_price_with_spread(float(prices[sell_tick]), False)
        available_qty = float(positions.get(sell_tick, 0.0))
        actual_sell = min(sell_qty, available_qty)
        if actual_sell > 0:
            proceeds = actual_sell * price
            positions[sell_tick] = available_qty - actual_sell
            # Sell proceeds are used directly for withdrawal, not added to current account
            total_used_for_withdrawal += proceeds
    
    # 5. Invest TD - use cash, increase TD investment
    if invest_amt > 0:
        actual_invest = min(invest_amt, max(0.0, new_ca))
        if actual_invest > 0:
            new_ca -= actual_invest
            summary["td_invested"] = float(summary["td_invested"]) + actual_invest
    
    # 6. Buy - use cash, increase securities
    if buy_tick != "(none)" and buy_qty > 0 and buy_tick in prices:
        price = calculate_effective_price_with_spread(float(prices[buy_tick]), True)
        cost = buy_qty * price
        if cost <= new_ca:
            new_ca -= cost
            positions[buy_tick] = float(positions.get(buy_tick, 0.0)) + buy_qty
    
    # Update current account and recalculate securities MV
    summary["current_account"] = new_ca
    securities_mv = sum(float(positions.get(t, 0.0)) * float(prices.get(t, 0.0)) for t in prices.keys())
    summary["securities_mv"] = securities_mv
    summary["total_mv"] = new_ca + securities_mv + float(summary["td_invested"]) - float(summary["repo_outstanding"])
    
    # Store the amount used for withdrawal tracking
    summary["_withdrawal_used"] = summary.get("_withdrawal_used", 0.0) + total_used_for_withdrawal
    
    return summary, positions

# ------------------------
# Direct Portfolio Execution Functions
# ------------------------
def get_player_portfolio(group_name: str, r: int) -> Optional[Portfolio]:
    """Get a player's current portfolio state"""
    player_data = _json_read(PLAYER_PORTFOLIOS_PATH, {})
    if group_name not in player_data:
        return None
    
    portfolio_data = player_data[group_name]
    # Reconstruct portfolio object from saved data
    portfolio = Portfolio(
        name=portfolio_data["name"],
        current_account=portfolio_data["current_account"],
        pos_qty=portfolio_data["pos_qty"],
        pnl_realized=portfolio_data["pnl_realized"]
    )
    portfolio.repo_liabilities = portfolio_data.get("repo_liabilities", [])
    portfolio.td_assets = portfolio_data.get("td_assets", [])
    return portfolio

def save_player_portfolio(portfolio: Portfolio):
    """Save a player's portfolio state"""
    def update_portfolios(data):
        data = data or {}
        data[portfolio.name] = {
            "name": portfolio.name,
            "current_account": portfolio.current_account,
            "pos_qty": dict(portfolio.pos_qty),
            "pnl_realized": portfolio.pnl_realized,
            "repo_liabilities": portfolio.repo_liabilities,
            "td_assets": portfolio.td_assets,
            "last_updated": _now_ts()
        }
        return data
    
    _json_mutate(PLAYER_PORTFOLIOS_PATH, {}, update_portfolios)

def execute_player_action(group_name: str, inputs: dict, prices: dict, r: int, withdrawal_req: float) -> dict:
    """Execute a player action immediately and return result"""
    portfolio = get_player_portfolio(group_name, r)
    if not portfolio:
        return {"error": "Portfolio not found"}
    
    # Process maturities first
    process_maturities(portfolio, r)
    
    # Calculate current remaining withdrawal for this group
    logs = st.session_state.logs.get(group_name, [])
    current_round_logs = [L for L in logs if L["round"] == r+1]
    used_this_round = 0.0
    for L in current_round_logs:
        for t, d in L["actions"]:
            if t in ("cash", "repo", "sell", "redeem_td"):
                used_this_round += float(d.get("use", 0.0))
    
    remaining_withdrawal = max(0.0, withdrawal_req - used_this_round)
    
    # Execute the action
    result = _execute_single_action(portfolio, inputs, prices, r, remaining_withdrawal)
    
    # Save updated portfolio
    save_player_portfolio(portfolio)
    
    # Log the action
    if result.get("actions"):
        st.session_state.logs.setdefault(group_name, []).append({
            "round": r + 1,
            "request": withdrawal_req,
            "actions": result["actions"],
            "timestamp": _now_ts()
        })
    
    return result

def _execute_single_action(portfolio: Portfolio, inputs: dict, prices: dict, r: int, remaining_withdrawal: float) -> dict:
    """Execute a single action on a portfolio and return what happened"""
    def clamp_float(val): 
        try: return float(val or 0.0)
        except: return 0.0
    
    # Extract inputs
    cash_amt = clamp_float(inputs.get("cash"))
    repo_amt = clamp_float(inputs.get("repo_amt"))
    repo_tick = inputs.get("repo_tick") or "(none)"
    redeem_amt = clamp_float(inputs.get("redeem"))
    invest_amt = clamp_float(inputs.get("invest_td"))
    sell_qty = clamp_float(inputs.get("sell_qty"))
    sell_tick = inputs.get("sell_tick") or "(none)"
    buy_qty = clamp_float(inputs.get("buy_qty"))
    buy_tick = inputs.get("buy_tick") or "(none)"
    
    actions = []
    rem_left = remaining_withdrawal
    
    # 1) Cash use
    if cash_amt > 0:
        use = min(cash_amt, max(0.0, portfolio.current_account), rem_left)
        if use > 0:
            portfolio.current_account -= use
            rem_left -= use
            actions.append(("cash", {"used": round(use, 2)}))
    
    # 2) Repo
    if repo_tick != "(none)" and repo_amt > 0:
        price = float(prices.get(repo_tick, 0.0))
        max_amt = portfolio.pos_qty.get(repo_tick, 0.0) * price
        repo_amt = min(repo_amt, max_amt)
        if repo_amt > 0:
            repo_rate = daily_rates_for_round(r)[0]
            info = _safe_repo_call(portfolio, repo_tick, repo_amt, price, r, repo_rate)
            got = float(info["got"])
            
            # Split proceeds: withdrawal coverage + excess to current account
            use = min(got, rem_left)
            excess = got - use
            portfolio.current_account += excess
            rem_left -= use
            
            actions.append(("repo", {
                "ticker": repo_tick, "got": round(got, 2), "use": round(use, 2),
                "excess_to_ca": round(excess, 2), "repo_id": info["repo_id"], "rate": repo_rate
            }))
    
    # 3) Redeem TD
    if redeem_amt > 0:
        red = _safe_redeem_td(portfolio, redeem_amt, r)
        principal = float(red["principal"])
        
        # Split proceeds: withdrawal coverage + excess to current account
        use = min(principal, rem_left)
        excess = principal - use
        portfolio.current_account += excess
        rem_left -= use
        
        actions.append(("redeem_td", {
            "principal": round(principal, 2),
            "penalty": round(float(red["penalty"]), 2),
            "use": round(use, 2),
            "excess_to_ca": round(excess, 2),
            "chunks": red.get("redeemed", []),
        }))
    
    # 4) Sell
    if sell_tick != "(none)" and sell_qty > 0:
        sell_qty = min(sell_qty, portfolio.pos_qty.get(sell_tick, 0.0))
        if sell_qty > 0:
            sale = _safe_sale(portfolio, sell_tick, sell_qty, prices[sell_tick])
            proceeds = sale["proceeds"]
            
            # Split proceeds: withdrawal coverage + excess to current account
            use = min(proceeds, rem_left)
            excess = proceeds - use
            portfolio.current_account += excess
            rem_left -= use
            
            actions.append(("sell", {
                "ticker": sale["ticker"],
                "qty": round(sale["qty"], 2),
                "proceeds": round(proceeds, 2),
                "use": round(use, 2),
                "excess_to_ca": round(excess, 2),
                "pnl_delta": round(sale["pnl_delta"], 2),
                "effective_price": round(sale["effective_price"], 6),
            }))
    
    # 5) Invest TD
    if invest_amt > 0:
        invest_amt = min(invest_amt, max(0.0, portfolio.current_account))
        if invest_amt > 0:
            td_rate = daily_rates_for_round(r)[1]
            td_ids = _safe_invest_td(portfolio, invest_amt, r, td_rate)
            actions.append(("invest_td", {
                "amount": round(invest_amt, 2),
                "td_ids": td_ids,
                "rate": td_rate
            }))
    
    # 6) Buy
    if buy_tick != "(none)" and buy_qty > 0:
        buy = _safe_buy(portfolio, buy_tick, buy_qty, prices[buy_tick])
        actions.append(("buy", {
            "ticker": buy["ticker"],
            "qty": round(buy["qty"], 2),
            "cost": round(buy["cost"], 2),
            "effective_price": round(buy["effective_price"], 6),
        }))
    
    return {
        "actions": actions,
        "remaining_after": rem_left,
        "portfolio_summary": portfolio.summary(prices)
    }

# ---- safe adapters
def _safe_repo_call(portfolio: Portfolio, ticker: str, amount: float, price: float, rnow: int, rate: float) -> Dict[str, Any]:
    got, repo_id = 0.0, None
    try:
        res = execute_repo(portfolio, ticker, amount, price, rnow, rate=rate)
        if isinstance(res, tuple) and len(res) >= 2:
            got, repo_id = float(res[0]), res[1]
        elif isinstance(res, dict):
            got = float(res.get("got", amount)); repo_id = res.get("id")
        else:
            got = float(amount)
    except TypeError:
        res = execute_repo(portfolio, ticker, amount, rnow, rate=rate)  # type: ignore
        if isinstance(res, tuple) and len(res) >= 2:
            got, repo_id = float(res[0]), res[1]
        elif isinstance(res, dict):
            got = float(res.get("got", amount)); repo_id = res.get("id")
        else:
            got = float(amount)
    return {"got": got, "repo_id": repo_id}

def _safe_redeem_td(portfolio: Portfolio, amount: float, rnow: int) -> Dict[str, Any]:
    try:
        res = execute_redeem_td(portfolio, amount, rnow, penalty=TD_PENALTY)
        if isinstance(res, dict):
            return {"principal": float(res.get("principal", amount)),
                    "penalty": float(res.get("penalty", amount * TD_PENALTY)),
                    "redeemed": res.get("redeemed", [])}
    except TypeError:
        res = execute_redeem_td(portfolio, amount, rnow, penalty_rate=TD_PENALTY)  # type: ignore
        if isinstance(res, dict):
            return {"principal": float(res.get("principal", amount)),
                    "penalty": float(res.get("penalty", amount * TD_PENALTY)),
                    "redeemed": res.get("redeemed", [])}
    return {"principal": amount, "penalty": amount * TD_PENALTY, "redeemed": []}

def _safe_invest_td(portfolio: Portfolio, amount: float, rnow: int, rate: float):
    res = execute_invest_td(portfolio, amount, rnow, rate=rate)
    if isinstance(res, list): return res
    if isinstance(res, dict) and "ids" in res: return list(res["ids"])
    return []

def _safe_sale(portfolio: Portfolio, ticker: str, qty: float, px: float) -> Dict[str, Any]:
    effective_px = calculate_effective_price_with_spread(px, False)
    res = execute_sale(portfolio, ticker, qty, effective_px)
    if isinstance(res, dict):
        return {"ticker": res.get("ticker", ticker),
                "qty": float(res.get("qty", qty)),
                "proceeds": float(res.get("proceeds", qty * effective_px)),
                "pnl_delta": float(res.get("pnl_delta", 0.0)),
                "effective_price": float(res.get("effective_price", effective_px))}
    return {"ticker": ticker, "qty": qty, "proceeds": qty*effective_px, "pnl_delta": 0.0, "effective_price": effective_px}

def _safe_buy(portfolio: Portfolio, ticker: str, qty: float, px: float) -> Dict[str, Any]:
    effective_px = calculate_effective_price_with_spread(px, True)
    res = execute_buy(portfolio, ticker, qty, effective_px)
    if isinstance(res, dict):
        return {"ticker": res.get("ticker", ticker),
                "qty": float(res.get("qty", qty)),
                "cost": float(res.get("cost", qty * effective_px)),
                "effective_price": float(res.get("effective_price", effective_px))}
    return {"ticker": ticker, "qty": qty, "cost": qty*effective_px, "effective_price": effective_px}

# ------------------------
# Role selector (no auto-refresh)
# ------------------------
st.sidebar.markdown("### Session")
role = st.sidebar.radio("Role", ["Host", "Player"], index=(0 if st.session_state.role == "Host" else 1))
st.session_state.role = role

# ------------------------
# Sidebar — conditional by role
# ------------------------
if role == "Host":
    st.sidebar.header("Host Setup")
    uploaded = st.sidebar.file_uploader("Bond price CSV (date + ≥3 securities)", type=["csv"])
    seed = st.sidebar.number_input("RNG seed", value=st.session_state.rng_seed, step=1)
    rounds = st.sidebar.number_input("Rounds", value=st.session_state.rounds, min_value=1, max_value=10, step=1)
    groups = st.sidebar.number_input("Groups (up to 8)", value=st.session_state.num_groups, min_value=1, max_value=MAX_GROUPS_UI, step=1)
    c1, c2, c3 = st.sidebar.columns(3)
    start_clicked = c1.button("Start/Reset", type="primary")
    refresh_clicked = c2.button("Refresh status 🔄")
    end_clicked   = c3.button("End Game")
else:
    st.sidebar.header("Player Setup")
    st.session_state.player_name = st.sidebar.text_input("Your name", value=st.session_state.player_name or "")
    uploaded = None
    start_clicked = refresh_clicked = end_clicked = False
    
    # Check if snapshot.json exists (the key change)
    if not os.path.exists(SNAPSHOT_PATH):
        st.sidebar.info("Waiting for Host to start...")
    else:
        shared = _json_read(SHARED_STATE_PATH, {})
        if not shared.get("initialized"):
            st.sidebar.info("Host is setting up...")
        else:
            st.sidebar.caption(f"Rounds: {shared.get('rounds')} • Groups: {shared.get('num_groups')} • RNG seed: {shared.get('rng_seed')}")

with st.sidebar.expander("Game Instructions", expanded=False):
    st.markdown(f"""
- **Withdrawals:** One amount per round, **same for all groups**.
- **Term Deposits (TD):** Assets mature after **{TD_MAT_GAP} rounds**. Early redemption penalty: **{TD_PENALTY*100:.2f}%**.
- **Rates:** Repo & TD vary daily by **±50 bps**.
- **Initial TD allocation:** In **Round 1**, each group auto-invests **10–30%** CA into TD.
- **Players**: Claim a group → **Apply** or **Clear** your actions (only your group).
- **Host**: **Refresh status** to see latest player actions; only **Next Round** and **End Game**.
""")

# ------------------------
# Start / End (Host only)
# ------------------------
if role == "Host" and start_clicked:
    if uploaded is None:
        st.sidebar.error("Please upload a CSV with columns: date,BOND_A,BOND_B,BOND_C,...")
    else:
        df = pd.read_csv(uploaded)
        if "date" not in df.columns or len([c for c in df.columns if c != "date"]) < 3:
            st.sidebar.error("CSV must have 'date' and at least 3 security columns.")
        else:
            # Save CSV for all sessions
            with open(UPLOADED_CSV_PATH, "wb") as f:
                f.write(uploaded.getbuffer())

            # Reset local (authoritative) state
            seed_val = int(seed); rounds_val = int(rounds)
            desired_groups = int(groups)
            _init_state()
            st.session_state.initialized   = True
            st.session_state.rng_seed      = seed_val
            st.session_state.rounds        = rounds_val
            st.session_state.price_df      = df.copy()

            tickers_all = [c for c in df.columns if c != "date"]
            date0, prices_all0 = base_prices_for_round(0, df, tickers_all)

            base_portfolios = init_portfolios(tickers_all, prices_all0, total_reserve=200000.0)
            cap = min(MAX_GROUPS_UI, len(base_portfolios), desired_groups)
            if desired_groups > cap:
                st.sidebar.warning(f"Requested {desired_groups} groups, capped to {cap}.")
            st.session_state.portfolios = base_portfolios[:cap]
            st.session_state.num_groups = cap

            # Random initial TD allocation (deterministic by seed)
            random.seed(seed_val ^ 0xA5A5)
            for p in st.session_state.portfolios:
                frac = random.uniform(0.10, 0.30)
                amt = round(max(0.0, p.current_account) * frac, 2)
                if amt > 0:
                    execute_invest_td(p, amt, 0, rate=BASE_TD_RATE)

            st.session_state.logs        = {p.name: [] for p in st.session_state.portfolios}
            st.session_state.withdrawals = [0.0 for _ in range(rounds_val)]
            st.session_state.current_round = 0
            st.session_state.last_maturity_round = -1
            st.session_state.inited_rounds = set()
            st.session_state.player_group_index = 0

            # Clear old files and initialize new system
            for path in [PLAYER_PORTFOLIOS_PATH, SNAPSHOT_PATH]:
                if os.path.exists(path):
                    os.remove(path)

            # Initialize player portfolios
            player_portfolios = {}
            for p in st.session_state.portfolios:
                player_portfolios[p.name] = {
                    "name": p.name,
                    "current_account": p.current_account,
                    "pos_qty": dict(p.pos_qty),
                    "pnl_realized": p.pnl_realized,
                    "repo_liabilities": list(p.repo_liabilities),
                    "td_assets": list(p.td_assets),
                    "last_updated": _now_ts()
                }
            _json_write_atomic(PLAYER_PORTFOLIOS_PATH, player_portfolios)
            _json_write_atomic(SHARED_STATE_PATH, {
                "initialized": True,
                "rng_seed": seed_val,
                "rounds": rounds_val,
                "num_groups": cap,
                "current_round": 0,
                "csv_ready": True,
                "claims": {},   # group_name -> player_name
                "ts": _now_ts()
            })

            # IMPORTANT: Write the initial snapshot.json file immediately
            # This signals to players that the game has started
            _json_write_atomic(SNAPSHOT_PATH, {
                "published": True,
                "ts": _now_ts()
            })

            _safe_rerun()

if role == "Host" and end_clicked and st.session_state.initialized:
    _json_mutate(SHARED_STATE_PATH, {}, lambda s: {**s, "current_round": s.get("rounds", st.session_state.rounds), "ts": _now_ts()})
    st.session_state.current_round = st.session_state.rounds
    _safe_rerun()

# ------------------------
# Title
# ------------------------
st.title("Liquidity Tranche Simulation")

# ------------------------
# Player bootstrap
# ------------------------
if role == "Player" and not st.session_state.initialized:
    # Now we check for snapshot.json instead of just shared state
    if not os.path.exists(SNAPSHOT_PATH):
        st.info("Waiting for Host to start the session.")
        st.stop()
    
    shared = _json_read(SHARED_STATE_PATH, {})
    if not shared.get("initialized"):
        st.info("Host is still setting up...")
        st.stop()
        
    if not os.path.exists(UPLOADED_CSV_PATH):
        st.info("Waiting for Host CSV…")
        st.stop()

    df = pd.read_csv(UPLOADED_CSV_PATH)
    _init_state()
    st.session_state.initialized = True
    st.session_state.rng_seed = int(shared.get("rng_seed", 1234))
    st.session_state.rounds = int(shared.get("rounds", 3))
    st.session_state.num_groups = int(shared.get("num_groups", 4))
    st.session_state.price_df = df.copy()
    st.session_state.current_round = int(shared.get("current_round", 0))
    st.session_state.withdrawals = [0.0 for _ in range(st.session_state.rounds)]
    st.session_state.logs = {}
    st.session_state.inited_rounds = set()
    st.session_state.last_maturity_round = -1

# If still not initialized
if not st.session_state.initialized:
    if role == "Host":
        st.info("Upload a CSV and click **Start/Reset** to begin.")
    else:
        st.info("Waiting for the Host to start the session.")
    st.stop()

# Sync current round for players
if role == "Player":
    shared = _json_read(SHARED_STATE_PATH, {})
    if shared.get("initialized"):
        host_round = int(shared.get("current_round", st.session_state.current_round))
        if host_round != st.session_state.current_round:
            st.session_state.current_round = host_round
            # Clear staged inputs when round changes but preserve player selection
            st.session_state.staged_inputs = {}

df = st.session_state.price_df
all_tickers = [c for c in df.columns if c != "date"]
tickers_ui = all_tickers[:3]  # the 3 shown in UI
r = st.session_state.current_round
NG = max(1, int(st.session_state.get("num_groups", 1)))

# Compute prices (ALL for valuation, subset for UI lines)
date_str, prices_all = base_prices_for_round(r, df, all_tickers)
prices_ui = {t: prices_all[t] for t in tickers_ui}

# ------------------------
# Host: settle maturities, ensure withdrawal, consume queue, publish snapshot
# ------------------------
def _host_publish_snapshot():
    """Publish full round snapshot with per-group remaining to make Player UI richer & consistent."""
    groups_out = []
    req = float(st.session_state.withdrawals[r]) if r < len(st.session_state.withdrawals) else 0.0
    for p in st.session_state.portfolios:
        s = p.summary(prices_all)  # ALL tickers (for accurate totals)
        positions_ui = {t: float(p.pos_qty.get(t, 0.0)) for t in tickers_ui}
        remaining = compute_remaining_for_group(p.name, r, req)
        groups_out.append({
            "name": p.name,                  # stable id by name to map actions
            "summary": {
                "current_account": s["current_account"],
                "securities_mv": s["securities_mv"],
                "repo_outstanding": s["repo_outstanding"],
                "td_invested": s["td_invested"],
                "pnl_realized": s["pnl_realized"],
                "total_mv": s["total_mv"],
            },
            "positions": positions_ui,
            "remaining": remaining,          # NEW: remaining for this round (host truth)
        })
    _json_write_atomic(SNAPSHOT_PATH, {
        "published": True,
        "round": r,
        "tickers": tickers_ui,
        "groups": groups_out,
        "withdrawal": req,
        "rates": list(daily_rates_for_round(r)),
        "ts": _now_ts()
    })

def _reverse_actions_for_round(p: Portfolio, rr: int):
    logs_this_round = [L for L in st.session_state.logs.get(p.name, []) if L["round"] == rr + 1]
    for L in logs_this_round:
        for t, d in L["actions"]:
            if t == "cash":
                p.current_account += float(d.get("used", 0.0))
            elif t == "repo":
                got = float(d.get("got", 0.0)); use = float(d.get("use", 0.0))
                p.current_account -= got; p.current_account += use
                rid = d.get("repo_id")
                if rid is not None:
                    p.repo_liabilities = [l for l in p.repo_liabilities if l.get("id") != rid]
                else:
                    for i, l in enumerate(list(p.repo_liabilities)):
                        if abs(float(l.get("amount", 0.0)) - got) < 1e-6 and l.get("maturity", 10**9) > rr:
                            p.repo_liabilities.pop(i); break
            elif t == "redeem_td":
                principal = float(d.get("principal", 0.0))
                penalty   = float(d.get("penalty", 0.0))
                use       = float(d.get("use", 0.0))
                p.current_account -= principal
                if penalty > 0:
                    p.current_account += penalty
                    p.pnl_realized += penalty
                p.current_account += use
                for ch in d.get("chunks", []):
                    p.td_assets.append({
                        "id": ch.get("id"),
                        "amount": float(ch.get("taken", 0.0)),
                        "rate": float(ch.get("rate", BASE_TD_RATE)),
                        "maturity": int(ch.get("maturity", rr + TD_MAT_GAP)),
                    })
            elif t == "sell":
                proceeds  = float(d.get("proceeds", 0.0))
                use       = float(d.get("use", 0.0))
                ticker    = d.get("ticker")
                qty       = float(d.get("qty", 0.0))
                pnl_delta = float(d.get("pnl_delta", 0.0))
                p.current_account -= proceeds
                p.current_account += use
                if ticker is not None:
                    p.pos_qty[ticker] = p.pos_qty.get(ticker, 0.0) + qty
                p.pnl_realized -= pnl_delta
            elif t == "invest_td":
                amt = float(d.get("amount", 0.0))
                ids = set(d.get("td_ids", []))
                p.current_account += amt
                if ids:
                    p.td_assets = [a for a in p.td_assets if a.get("id") not in ids]
            elif t == "buy":
                cost   = float(d.get("cost", 0.0))
                ticker = d.get("ticker")
                qty    = float(d.get("qty", 0.0))
                p.current_account += cost
                if ticker is not None:
                    p.pos_qty[ticker] = p.pos_qty.get(ticker, 0.0) - qty
    st.session_state.logs[p.name] = [L for L in st.session_state.logs.get(p.name, []) if L["round"] != rr + 1]

def _host_apply_single_action(p: Portfolio, req_all: float, px_ui: Dict[str, float], action: Dict[str, Any]):
    rem_left = compute_remaining_for_group(p.name, r, req_all)
    history = []
    def clamp_float(val): 
        try: return float(val or 0.0)
        except: return 0.0
    cash_amt   = clamp_float(action.get("cash"))
    repo_amt   = clamp_float(action.get("repo_amt"))
    repo_tick  = action.get("repo_tick") or "(none)"
    redeem_amt = clamp_float(action.get("redeem"))
    invest_amt = clamp_float(action.get("invest_td"))
    sell_qty   = clamp_float(action.get("sell_qty"))
    sell_tick  = action.get("sell_tick") or "(none)"
    buy_qty    = clamp_float(action.get("buy_qty"))
    buy_tick   = action.get("buy_tick") or "(none)"

    # 1) cash
    if cash_amt > 0 and rem_left > 0:
        use = min(cash_amt, max(0.0, p.current_account), rem_left)
        if use > 0:
            p.current_account -= use
            rem_left -= use
            history.append(("cash", {"used": round(use, 2)}))
    
    # 2) repo - FIXED: handle excess properly
    if repo_tick != "(none)" and repo_amt > 0:
        price = float(px_ui.get(repo_tick, 0.0))
        max_amt = p.pos_qty.get(repo_tick, 0.0) * price
        repo_amt = min(repo_amt, max_amt)
        if repo_amt > 0:
            repo_rate = daily_rates_for_round(r)[0]
            info = _safe_repo_call(p, repo_tick, repo_amt, price, r, repo_rate)
            got = float(info["got"])
            
            # CRITICAL FIX: Split proceeds correctly
            if rem_left > 0:
                use = min(got, rem_left)
                excess = got - use
                rem_left -= use
            else:
                use = 0.0
                excess = got  # All goes to current account when no withdrawal needed
            
            p.current_account += excess  # Add excess to current account
            
            history.append(("repo", {
                "ticker": repo_tick, "got": round(got, 2), "use": round(use, 2),
                "repo_id": info["repo_id"], "rate": repo_rate
            }))
    
    # 3) redeem td - FIXED: handle excess properly
    if redeem_amt > 0:
        red = _safe_redeem_td(p, redeem_amt, r)
        principal = float(red["principal"])
        
        # CRITICAL FIX: Split proceeds correctly
        if rem_left > 0:
            use = min(principal, rem_left)
            excess = principal - use
            rem_left -= use
        else:
            use = 0.0
            excess = principal  # All goes to current account when no withdrawal needed
        
        p.current_account += excess  # Add excess to current account
        
        history.append(("redeem_td", {
            "principal": round(principal, 2),
            "penalty":   round(float(red["penalty"]), 2),
            "use":       round(use, 2),
            "chunks":    red.get("redeemed", []),
        }))
    
    # 4) sell - FIXED: handle excess properly
    if sell_tick != "(none)" and sell_qty > 0:
        sell_qty = min(sell_qty, p.pos_qty.get(sell_tick, 0.0))
        if sell_qty > 0:
            sale = _safe_sale(p, sell_tick, sell_qty, px_ui[sell_tick])
            proceeds = sale["proceeds"]
            
            # CRITICAL FIX: Split proceeds correctly
            if rem_left > 0:
                use = min(proceeds, rem_left)
                excess = proceeds - use
                rem_left -= use
            else:
                use = 0.0
                excess = proceeds  # All goes to current account when no withdrawal needed
            
            p.current_account += excess  # Add excess to current account
            
            history.append(("sell", {
                "ticker": sale["ticker"],
                "qty": round(sale["qty"], 2),
                "proceeds": round(proceeds, 2),
                "use": round(use, 2),
                "pnl_delta": round(sale["pnl_delta"], 2),
                "effective_price": round(sale["effective_price"], 6),
            }))
    
    # 5) invest td
    if invest_amt > 0:
        invest_amt = min(invest_amt, max(0.0, p.current_account))
        if invest_amt > 0:
            td_rate = daily_rates_for_round(r)[1]
            td_ids = _safe_invest_td(p, invest_amt, r, td_rate)
            history.append(("invest_td", {
                "amount": round(invest_amt, 2),
                "td_ids": td_ids,
                "rate": td_rate
            }))
    # 6) buy
    if buy_tick != "(none)" and buy_qty > 0:
        buy = _safe_buy(p, buy_tick, buy_qty, px_ui[buy_tick])
        history.append(("buy", {
            "ticker": buy["ticker"],
            "qty": round(buy["qty"], 2),
            "cost": round(buy["cost"], 2),
            "effective_price": round(buy["effective_price"], 6),
        }))
    if history:
        st.session_state.logs.setdefault(p.name, []).append({
            "round": r + 1,
            "request": float(st.session_state.withdrawals[r]),
            "actions": history
        })

def _find_portfolio_index_by_name(name: str) -> Optional[int]:
    for i, p in enumerate(st.session_state.portfolios):
        if p.name == name:
            return i
    return None

def _host_consume_actions_and_publish(req_all: float):
    actions = _json_read(ACTIONS_QUEUE_PATH, [])
    if not isinstance(actions, list):
        actions = []
    if actions:
        for a in actions:
            rr = int(a.get("round", -1))
            a_type = a.get("type", "apply")
            gname = a.get("group_name", "")
            gi = _find_portfolio_index_by_name(gname)  # <-- map by NAME (fixes 'always group 4' bug)
            if rr != r or gi is None:
                continue
            p = st.session_state.portfolios[gi]
            if a_type == "apply":
                _host_apply_single_action(p, req_all, prices_ui, a.get("inputs", {}))
            elif a_type == "clear":
                _reverse_actions_for_round(p, rr)
        _json_write_atomic(ACTIONS_QUEUE_PATH, [])
    _host_publish_snapshot()

# ------------------------
# Endgame helper
# ------------------------
def _render_endgame(final_px_all: Dict[str, float]):
    st.header("Scoreboard & Logs")
    rows = []
    for p in st.session_state.portfolios:
        s = p.summary(final_px_all)
        rows.append({
            "group": p.name,
            "current_account": s["current_account"],
            "securities_reserve": s["securities_mv"],
            "repo_outstanding": s["repo_outstanding"],
            "term_deposit": s["td_invested"],
            "pnl_realized": s["pnl_realized"],
            "total_reserve": s["total_mv"],
        })
    sb = pd.DataFrame(rows)
    
    # Format all monetary columns with dollar signs
    for col in ["current_account", "securities_reserve", "repo_outstanding", "term_deposit", "pnl_realized", "total_reserve"]:
        if col in sb.columns:
            sb[col] = sb[col].apply(lambda x: _fmt_money(x))
    
    st.dataframe(sb, use_container_width=True)
    st.download_button("Download scoreboard CSV", sb.to_csv(index=False).encode("utf-8"),
                       file_name="scoreboard.csv", mime="text/csv")
    st.download_button("Download logs JSON", json.dumps(st.session_state.logs, indent=2).encode("utf-8"),
                       file_name="logs.json", mime="application/json")
    st.stop()

# ------------------------
# Host flow: settle maturities, ensure withdrawal
# (consume queue + publish snapshot ONLY on Refresh or Next Round)
# ------------------------
if role == "Host":
    if st.session_state.last_maturity_round != r:
        for p in st.session_state.portfolios:
            process_maturities(p, r)
        st.session_state.last_maturity_round = r
    ensure_round_initialized(r, prices_all)
    req_all = float(st.session_state.withdrawals[r])

    # Manual refresh (host clicks)
    if 'refresh_clicked' in locals() and refresh_clicked:
        _host_consume_actions_and_publish(req_all)
        _safe_rerun()

# ------------------------
# Player flow: claim + Apply/Clear (send to queue), render from snapshot
# ------------------------
if role == "Player":
    shared = _json_read(SHARED_STATE_PATH, {})
    snapshot = _json_read(SNAPSHOT_PATH, {})
    if not snapshot or not snapshot.get("published") and not snapshot.get("groups"):
        st.info("Waiting for Host to publish the round snapshot…")
        st.stop()

    tickers_ui = snapshot.get("tickers", tickers_ui)
    req_all = float(snapshot.get("withdrawal", 0.0))
    repo_rate_today, td_rate_today = snapshot.get("rates", list(daily_rates_for_round(r)))[:2]

    group_names = [g["name"] for g in snapshot.get("groups", [])]
    if not group_names:
        st.info("Waiting for Host to initialize groups…")
        st.stop()

    claims: Dict[str, str] = shared.get("claims", {})
    # claims status
    claim_cols = st.columns(len(group_names))
    for i, c in enumerate(claim_cols):
        gname = group_names[i]
        with c:
            owner = claims.get(gname, "")
            st.caption(f"{gname}: {'(unclaimed)' if not owner else 'claimed by: ' + owner}")

    # choose + claim with persistent selection
    if group_names:
        # Ensure player_group_index is within valid range
        valid_index = min(max(st.session_state.player_group_index, 0), len(group_names)-1)
        if st.session_state.player_group_index != valid_index:
            st.session_state.player_group_index = valid_index
        
        chosen = st.sidebar.selectbox("Select your Group", group_names, 
                                    index=st.session_state.player_group_index,
                                    key="player_group_selector")
        # Update the stored index when selection changes
        new_index = group_names.index(chosen)
        if new_index != st.session_state.player_group_index:
            st.session_state.player_group_index = new_index

    can_claim = bool(st.session_state.player_name.strip())
    if st.sidebar.button("Claim Group", disabled=not can_claim):
        def _try_claim(s):
            s = dict(s or {})
            s.setdefault("claims", {})
            if chosen not in s["claims"]:
                s["claims"][chosen] = st.session_state.player_name.strip()
            return s
        before = _json_read(SHARED_STATE_PATH, {})
        after = _json_mutate(SHARED_STATE_PATH, {}, _try_claim)
        if chosen in after.get("claims", {}) and after["claims"][chosen] != before.get("claims", {}).get(chosen):
            st.success(f"Claimed {chosen} for {st.session_state.player_name} ✅")
        else:
            st.warning("Someone already claimed that group.")

    you_own = claims.get(chosen, "") == st.session_state.player_name.strip()

# ------------------------
# Shared UI: dashboard
# ------------------------
repo_rate_today, td_rate_today = daily_rates_for_round(r)

st.subheader(f"Round {r+1} — Date: {date_str}")
st.caption(f"Today's rates → Repo: {repo_rate_today*100:.2f}%  •  TD: {td_rate_today*100:.2f}%  •  Early TD penalty: {TD_PENALTY*100:.2f}%")

if role == "Host":
    cols = st.columns(NG)
    for g, c in enumerate(cols):
        if g >= len(st.session_state.portfolios): break
        with c:
            p = st.session_state.portfolios[g]
            rem = compute_remaining_for_group(p.name, r, float(st.session_state.withdrawals[r]))
            reserve = p.market_value(prices_all)
            st.markdown(f"### {p.name}")
            st.markdown(f"<div style='font-size:28px; font-weight:800; color:#006400;'>{_fmt_money(reserve,0)}</div>", unsafe_allow_html=True)
            for t in tickers_ui:
                st.markdown(f"<div class='ticker-line'>{t}: {p.pos_qty.get(t,0.0):,.0f} @ {_fmt_money(prices_ui[t])}</div>", unsafe_allow_html=True)
            prog = 0.0 if float(st.session_state.withdrawals[r]) <= 0 else max(0.0, 1 - rem/float(st.session_state.withdrawals[r]))
            st.progress(prog)
else:
    snapshot = _json_read(SNAPSHOT_PATH, {})
    groups = snapshot.get("groups", [])
    cols = st.columns(min(NG, len(groups)))
    for g, c in enumerate(cols):
        if g >= len(groups): break
        with c:
            G = groups[g]
            s = G["summary"]
            positions = G["positions"]
            
            # Apply instant local adjustments if this is the player's claimed group
            chosen_name = groups[st.session_state.player_group_index]["name"] if st.session_state.player_group_index < len(groups) else ""
            you_own_this = claims.get(G["name"], "") == st.session_state.player_name.strip()
            staged_key = f"r{r}_group_{G['name']}"
            staged_inputs = st.session_state.staged_inputs.get(staged_key, {})
            
            if you_own_this and staged_inputs:
                s, positions = _apply_staged_deltas_to_summary(s, positions, staged_inputs, prices_ui, r, req_all)
            
            st.markdown(f"### {G['name']}")
            st.markdown(f"<div style='font-size:28px; font-weight:800; color:#006400;'>{_fmt_money(s['total_mv'],0)}</div>", unsafe_allow_html=True)
            for t in tickers_ui:
                qty = float(positions.get(t, 0.0))
                st.markdown(f"<div class='ticker-line'>{t}: {qty:,.0f} @ {_fmt_money(prices_ui[t])}</div>", unsafe_allow_html=True)
            
            # Progress bar - use base remaining minus any staged consumption
            group_rem = float(G.get("remaining", 0.0))
            if you_own_this and staged_inputs:
                staged_used = _estimate_used_from_inputs(G, staged_inputs, prices_ui)
                group_rem = max(0.0, group_rem - staged_used)
            prog = 1.0 if req_all <= 0 else max(0.0, 1 - group_rem/req_all)
            st.progress(prog)

# ------------------------
# Detailed tabs
# ------------------------
if role == "Host":
    req_all = float(st.session_state.withdrawals[r])
    tab_labels = [p.name for p in st.session_state.portfolios[:NG]]
    tabs = st.tabs(tab_labels if tab_labels else ["Group 1"])
    for g, tab in enumerate(tabs):
        if g >= len(st.session_state.portfolios): break
        with tab:
            p = st.session_state.portfolios[g]
            rem = compute_remaining_for_group(p.name, r, req_all)
            st.markdown(f"### {p.name} (Host)")
            summary = p.summary(prices_all)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Current Account",        _fmt_money(summary['current_account']))
                st.metric("Repo Outstanding",       _fmt_money(summary['repo_outstanding']))
            with c2:
                st.metric("Securities Reserve",     _fmt_money(summary['securities_mv']))
                st.metric("Term Deposit (Asset)",   _fmt_money(summary['td_invested']))
            with c3:
                st.metric("PnL Realized",           _fmt_money(summary['pnl_realized']))
                st.metric("Total Reserve",          _fmt_money(summary['total_mv']))
            st.markdown(f"**Withdrawal (all groups):** :blue[{_fmt_money(req_all)}]")
            st.markdown(f"**Remaining for {p.name}:** :orange[{_fmt_money(rem)}]")
else:
    snapshot = _json_read(SNAPSHOT_PATH, {})
    groups = snapshot.get("groups", [])
    if groups:
        tabs = st.tabs([g["name"] for g in groups])
        chosen_idx = st.session_state.player_group_index if st.session_state.player_group_index < len(groups) else 0
        chosen_name = groups[chosen_idx]["name"] if groups else ""
        shared = _json_read(SHARED_STATE_PATH, {})
        claims = shared.get("claims", {})
        you_own = claims.get(chosen_name, "") == st.session_state.player_name.strip()

        for gi, tab in enumerate(tabs):
            with tab:
                G = groups[gi]
                base_s = G["summary"]
                base_pos = G["positions"]
                rem_host = float(G.get("remaining", 0.0))

                # Apply instant adjustments if this is the player's group
                you_own_this_group = claims.get(G["name"], "") == st.session_state.player_name.strip()
                staged_key = f"r{r}_group_{G['name']}"
                staged_inputs = st.session_state.staged_inputs.get(staged_key, [])
                
                # Calculate adjusted state with detailed debugging
                if you_own_this_group and staged_inputs:
                    total_withdrawal = float(snapshot.get('withdrawal', 0.0))
                    original_ca = float(base_s['current_account'])
                    
                    s, pos = _apply_staged_deltas_to_summary(base_s, base_pos, staged_inputs, prices_ui, r, total_withdrawal)
                    
                    # Debug: Show what changed with detailed breakdown
                    new_ca = float(s['current_account'])
                    ca_change = new_ca - original_ca
                    
                    # Calculate remaining based on how much we've used for withdrawals
                    withdrawal_used = s.get("_withdrawal_used", 0.0)
                    rem_effective = max(0.0, rem_host - withdrawal_used)
                    
                    # Debug display for troubleshooting
                    if ca_change != 0 or withdrawal_used > 0:
                        st.caption(f"DEBUG: CA change: {_fmt_money(ca_change)}, Withdrawal used: {_fmt_money(withdrawal_used)}")
                        # Show sell debug info if available
                        if "_debug_sell" in s:
                            sell_debug = s["_debug_sell"]
                            st.caption(f"SELL DEBUG: Proceeds: {_fmt_money(sell_debug['gross_proceeds'])}, To withdrawal: {_fmt_money(sell_debug['withdrawal_amount'])}, To CA: {_fmt_money(sell_debug['ca_amount'])}")

                else:
                    s, pos = base_s, base_pos
                    rem_effective = rem_host

                st.markdown(f"### {G['name']}{' (You)' if (gi==chosen_idx and you_own) else ''}")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Current Account",        _fmt_money(s['current_account']))
                    st.metric("Repo Outstanding",       _fmt_money(s['repo_outstanding']))
                with c2:
                    st.metric("Securities Reserve",     _fmt_money(s['securities_mv']))
                    st.metric("Term Deposit (Asset)",   _fmt_money(s['td_invested']))
                with c3:
                    st.metric("PnL Realized",           _fmt_money(s['pnl_realized']))
                    st.metric("Total Reserve",          _fmt_money(s['total_mv']))

                # Show withdrawal progress
                st.markdown(f"**Withdrawal (all groups):** :blue[{_fmt_money(snapshot.get('withdrawal', 0.0))}]")
                st.markdown(f"**Remaining:** :orange[{_fmt_money(rem_effective)}]")

                # Show updated positions with bid/ask spreads for trading decisions
                st.markdown("**Current Holdings & Market Prices:**")
                
                for t in tickers_ui:
                    qty = float(pos.get(t, 0.0))
                    market_px = float(prices_ui[t])
                    bid_px = calculate_effective_price_with_spread(market_px, False)  # What you get when selling
                    ask_px = calculate_effective_price_with_spread(market_px, True)   # What you pay when buying
                    
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.metric(f"**{t}**", f"{qty:,.0f} units", f"Market: {_fmt_money(market_px)}")
                    with col2:
                        st.caption(f"**Bid:** {_fmt_money(bid_px)}")
                        st.caption("*(You receive)*")
                    with col3:
                        st.caption(f"**Ask:** {_fmt_money(ask_px)}")
                        st.caption("*(You pay)*")

                # Inputs ONLY on your own claimed group; others read-only
                if gi == chosen_idx and you_own:
                    tickers_ui = snapshot.get("tickers", tickers_ui)

                    def get_float(key, default=0.0):
                        try: return float(st.session_state.get(key, default) or 0.0)
                        except: return 0.0

                    # Define input field keys
                    cash_key      = f"p_cash_{r}_{gi}"
                    repo_amt_key  = f"p_repo_{r}_{gi}"
                    repo_tick_key = f"p_repo_t_{r}_{gi}"
                    redeem_key    = f"p_redeemtd_{r}_{gi}"
                    invest_key    = f"p_investtd_{r}_{gi}"
                    sell_qty_key  = f"p_sell_{r}_{gi}"
                    sell_tick_key = f"p_sell_t_{r}_{gi}"
                    buy_qty_key   = f"p_buy_{r}_{gi}"
                    buy_tick_key  = f"p_buy_t_{r}_{gi}"

                    def _collect_current_inputs() -> Dict[str, Any]:
                        return {
                            "cash":      get_float(cash_key),
                            "repo_amt":  get_float(repo_amt_key),
                            "repo_tick": st.session_state.get(repo_tick_key, "(none)"),
                            "redeem":    get_float(redeem_key),
                            "invest_td": get_float(invest_key),
                            "sell_qty":  get_float(sell_qty_key),
                            "sell_tick": st.session_state.get(sell_tick_key, "(none)"),
                            "buy_qty":   get_float(buy_qty_key),
                            "buy_tick":  st.session_state.get(buy_tick_key, "(none)")
                        }

                    def get_float(key, default=0.0):
                        try: return float(st.session_state.get(key, default) or 0.0)
                        except: return 0.0

                    cash_key      = f"p_cash_{r}_{gi}"
                    repo_amt_key  = f"p_repo_{r}_{gi}"
                    repo_tick_key = f"p_repo_t_{r}_{gi}"
                    redeem_key    = f"p_redeemtd_{r}_{gi}"
                    invest_key    = f"p_investtd_{r}_{gi}"
                    sell_qty_key  = f"p_sell_{r}_{gi}"
                    sell_tick_key = f"p_sell_t_{r}_{gi}"
                    buy_qty_key   = f"p_buy_{r}_{gi}"
                    buy_tick_key  = f"p_buy_t_{r}_{gi}"

                    st.divider()
                    st.markdown("**Action Inputs:**")

                    # Cash input
                    st.number_input("Use cash", min_value=0.0, step=0.01, value=st.session_state.get(cash_key, 0.0), format="%.2f", key=cash_key)
                    
                    # Repo inputs with preview
                    st.selectbox("Repo ticker", ["(none)"] + tickers_ui, index=(["(none)"] + tickers_ui).index(st.session_state.get(repo_tick_key, "(none)")), key=repo_tick_key)
                    st.number_input("Repo amount", min_value=0.0, step=0.01, value=st.session_state.get(repo_amt_key, 0.0), format="%.2f", key=repo_amt_key)
                    
                    # Repo preview
                    repo_tick = st.session_state.get(repo_tick_key, "(none)")
                    repo_amt = get_float(repo_amt_key)
                    if repo_tick != "(none)" and repo_amt > 0 and repo_tick in prices_ui:
                        repo_units = repo_amt / float(prices_ui[repo_tick])
                        available_units = float(pos.get(repo_tick, 0.0))
                        max_units = min(repo_units, available_units)
                        st.caption(f"≈ {max_units:,.1f} units of {repo_tick} (max: {available_units:,.0f})")
                    
                    st.caption(f"Today's Repo rate: {daily_rates_for_round(r)[0]*100:.2f}%")

                    # TD Redemption
                    st.number_input("Redeem Term Deposit", min_value=0.0, step=0.01, value=st.session_state.get(redeem_key, 0.0), format="%.2f", key=redeem_key)
                    redeem_amt = get_float(redeem_key)
                    if redeem_amt > 0:
                        penalty = redeem_amt * TD_PENALTY
                        net = redeem_amt - penalty
                        st.caption(f"Penalty: {_fmt_money(penalty)} → Net: {_fmt_money(net)}")
                    else:
                        st.caption(f"Early redemption penalty: {TD_PENALTY*100:.2f}%")

                    # Sell inputs with preview
                    # Sell inputs with preview
                    st.selectbox("Sell ticker", ["(none)"] + tickers_ui, index=(["(none)"] + tickers_ui).index(st.session_state.get(sell_tick_key, "(none)")), key=sell_tick_key)
                    st.number_input("Sell qty", min_value=0.0, step=0.01, value=st.session_state.get(sell_qty_key, 0.0), format="%.2f", key=sell_qty_key)
                    
                    # Sell preview
                    sell_tick = st.session_state.get(sell_tick_key, "(none)")
                    sell_qty = get_float(sell_qty_key)
                    if sell_tick != "(none)" and sell_qty > 0 and sell_tick in prices_ui:
                        base_price = float(prices_ui[sell_tick])
                        effective_price = calculate_effective_price_with_spread(base_price, False)
                        available_qty = float(pos.get(sell_tick, 0.0))
                        max_qty = min(sell_qty, available_qty)
                        proceeds = max_qty * effective_price
                        st.caption(f"{max_qty:,.1f} × {_fmt_money(effective_price)} = {_fmt_money(proceeds)} (max: {available_qty:,.0f})")

                    # TD Investment
                    st.number_input("Invest in Term Deposit", min_value=0.0, step=0.01, value=st.session_state.get(invest_key, 0.0), format="%.2f", key=invest_key)
                    st.caption(f"Today's TD rate (if held to maturity): {daily_rates_for_round(r)[1]*100:.2f}% • Matures in {TD_MAT_GAP} rounds")

                    # Buy inputs with preview
                    st.selectbox("Buy ticker", ["(none)"] + tickers_ui, index=(["(none)"] + tickers_ui).index(st.session_state.get(buy_tick_key, "(none)")), key=buy_tick_key)
                    st.number_input("Buy qty", min_value=0.0, step=0.01, value=st.session_state.get(buy_qty_key, 0.0), format="%.2f", key=buy_qty_key)
                    
                    # Buy preview
                    buy_tick = st.session_state.get(buy_tick_key, "(none)")
                    buy_qty = get_float(buy_qty_key)
                    if buy_tick != "(none)" and buy_qty > 0 and buy_tick in prices_ui:
                        base_price = float(prices_ui[buy_tick])
                        effective_price = calculate_effective_price_with_spread(base_price, True)
                        cost = buy_qty * effective_price
                        current_cash = float(s['current_account'])
                        if cost <= current_cash:
                            st.caption(f"{buy_qty:,.1f} × {_fmt_money(effective_price)} = {_fmt_money(cost)}")
                        else:
                            st.caption(f"{buy_qty:,.1f} × {_fmt_money(effective_price)} = {_fmt_money(cost)} (insufficient cash: {_fmt_money(current_cash)})")

                    def _collect_current_inputs() -> Dict[str, Any]:
                        return {
                            "cash":      get_float(cash_key),
                            "repo_amt":  get_float(repo_amt_key),
                            "repo_tick": st.session_state.get(repo_tick_key, "(none)"),
                            "redeem":    get_float(redeem_key),
                            "invest_td": get_float(invest_key),
                            "sell_qty":  get_float(sell_qty_key),
                            "sell_tick": st.session_state.get(sell_tick_key, "(none)"),
                            "buy_qty":   get_float(buy_qty_key),
                            "buy_tick":  st.session_state.get(buy_tick_key, "(none)")
                        }

                    # Action buttons - Fixed to avoid session state modification in callbacks
                    b1, b2 = st.columns([2,1])
                    
                    with b1:
                        apply_clicked = st.button("Apply Action", type="primary", key=f"apply_{gi}_{r}")
                    with b2:
                        clear_clicked = st.button("Clear Actions", key=f"clear_{gi}_{r}")

                    # Handle button clicks outside of callbacks
                    if apply_clicked:
                        inputs = _collect_current_inputs()
                        
                        # Get existing staged inputs and make them cumulative
                        current_staged = st.session_state.staged_inputs.get(staged_key, [])
                        if not isinstance(current_staged, list):
                            current_staged = [current_staged] if current_staged else []
                        
                        # Only add non-zero actions to the staged list
                        has_action = any([
                            inputs.get("cash", 0) > 0,
                            inputs.get("repo_amt", 0) > 0 and inputs.get("repo_tick") != "(none)",
                            inputs.get("redeem", 0) > 0,
                            inputs.get("invest_td", 0) > 0,
                            inputs.get("sell_qty", 0) > 0 and inputs.get("sell_tick") != "(none)",
                            inputs.get("buy_qty", 0) > 0 and inputs.get("buy_tick") != "(none)"
                        ])
                        
                        if has_action:
                            current_staged.append(inputs)
                            st.session_state.staged_inputs[staged_key] = current_staged
                            
                            # Enqueue by NAME (not index) to avoid mismatches on Host
                            action = {
                                "ts": _now_ts(),
                                "type": "apply",
                                "by": st.session_state.player_name.strip(),
                                "group_name": G["name"],  # Use group name for correct binding
                                "round": r,
                                "inputs": inputs
                            }
                            _json_mutate(ACTIONS_QUEUE_PATH, [], lambda q: (q if isinstance(q, list) else []) + [action])
                            
                            # Clear input fields after successful apply
                            for k in [cash_key, repo_amt_key, redeem_key, invest_key, sell_qty_key, buy_qty_key]:
                                if k in st.session_state:
                                    del st.session_state[k]
                            for k in [repo_tick_key, sell_tick_key, buy_tick_key]:
                                if k in st.session_state:
                                    del st.session_state[k]
                            
                            st.success("Action applied! Enter new values and click Apply again to add more actions.")
                        else:
                            st.warning("Please enter some action values before applying.")
                        
                        _safe_rerun()

                    if clear_clicked:
                        # Clear staged inputs
                        st.session_state.staged_inputs[staged_key] = []
                        # Enqueue clear action
                        action = {
                            "ts": _now_ts(),
                            "type": "clear",
                            "by": st.session_state.player_name.strip(),
                            "group_name": G["name"],
                            "round": r,
                            "inputs": {}
                        }
                        _json_mutate(ACTIONS_QUEUE_PATH, [], lambda q: (q if isinstance(q, list) else []) + [action])
                        
                        # Clear input fields
                        for k in [cash_key, repo_amt_key, redeem_key, invest_key, sell_qty_key, buy_qty_key]:
                            if k in st.session_state:
                                del st.session_state[k]
                        for k in [repo_tick_key, sell_tick_key, buy_tick_key]:
                            if k in st.session_state:
                                del st.session_state[k]
                        
                        st.success("Actions cleared!")
                        _safe_rerun()
                        
                else:
                    st.caption("Read-only (not your claimed group).")

# ------------------------
# Controls (Host advances rounds for everyone)
# ------------------------
st.divider()
lft, rgt = st.columns([3,1])
with lft: st.subheader("Controls")

if st.session_state.role == "Host":
    with rgt:
        if st.button("Next Round ▶️"):
            req_all = float(st.session_state.withdrawals[r])
            all_covered = all(
                compute_remaining_for_group(st.session_state.portfolios[i].name, r, req_all) <= 0.01
                for i in range(min(NG, len(st.session_state.portfolios)))
            )
            if not all_covered:
                st.error("Cover all groups (remaining ≤ $0.01) before moving on.")
            else:
                # Clear all staged inputs when advancing round
                st.session_state.staged_inputs = {}
                
                if st.session_state.current_round + 1 < st.session_state.rounds:
                    st.session_state.current_round += 1
                else:
                    st.session_state.current_round = st.session_state.rounds
                _json_mutate(SHARED_STATE_PATH, {}, lambda s: {**s, "current_round": st.session_state.current_round, "ts": _now_ts()})
                _host_publish_snapshot()
                _safe_rerun()
else:
    with rgt:
        st.caption("Only the Host can advance rounds.")

# ------------------------
# End game
# ------------------------
if st.session_state.role == "Host" and r >= st.session_state.rounds:
    last_ix = min(st.session_state.rounds-1, len(df)-1)
    _, final_px_all = base_prices_for_round(last_ix, df, all_tickers)
    _render_endgame(final_px_all)
elif st.session_state.role == "Player":
    shared = _json_read(SHARED_STATE_PATH, {})
    if int(shared.get("current_round", 0)) >= int(shared.get("rounds", st.session_state.rounds)):
        st.header("Game finished — waiting for host to display final results.")

