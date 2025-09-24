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
SHARED_STATE_PATH       = ".shared_state.json"       # host pushes session info + claims + current_round  
UPLOADED_CSV_PATH       = ".uploaded.csv"            # host saves CSV here
PLAYER_PORTFOLIOS_PATH  = ".player_portfolios.json"  # real-time player portfolio states
SNAPSHOT_PATH           = ".snapshot.json"           # published round info for all

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

if "initialized" not in st.session_state:
    _init_state()

# ------------------------
# Game helpers
# ------------------------
def price_row_for_round(r: int) -> int:
    df = st.session_state.price_df
    if df is None or len(df) == 0:
        return 0
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
    
    # Safety check: ensure portfolios exist before trying to access them
    if not ss.portfolios or len(ss.portfolios) == 0:
        return  # Skip initialization if no portfolios exist yet
    
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
            # Remove the automatic counting of repo, sell, redeem_td toward withdrawals
            # Players must explicitly use cash to cover withdrawals
    return max(0.0, round(req_for_round - used, 2))

def calculate_effective_price_with_spread(base_price: float, is_buy: bool, liquidity_factor: float = 0.005) -> float:
    """Calculate effective price with bid-ask spread for trading."""
    if is_buy:
        return base_price * (1 + liquidity_factor)  # Pay slightly more when buying
    else:
        return base_price * (1 - liquidity_factor)  # Receive slightly less when selling

# ------------------------
# Player-side direct portfolio execution
# ------------------------
def get_player_portfolio(group_name: str) -> Optional[Portfolio]:
    """Get a player's current portfolio state directly from shared file"""
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
    
    # Reconstruct securities from saved data
    securities_data = portfolio_data.get("securities", {})
    from game_core import SecuritySpec
    for ticker, spec_data in securities_data.items():
        portfolio.securities[ticker] = SecuritySpec(
            ticker=spec_data["ticker"],
            face_price=spec_data.get("face_price", 100.0),
            bid_ask_bps=spec_data.get("bid_ask_bps", 20.0),
            liquidity_score=spec_data.get("liquidity_score", 1)
        )
    
    portfolio.repo_liabilities = portfolio_data.get("repo_liabilities", [])
    portfolio.td_assets = portfolio_data.get("td_assets", [])
    return portfolio

def save_player_portfolio(portfolio: Portfolio):
    """Save a player's portfolio state directly to shared file"""
    def update_portfolios(data):
        data = data or {}
        
        # Convert securities to serializable format
        securities_data = {}
        for ticker, spec in portfolio.securities.items():
            securities_data[ticker] = {
                "ticker": spec.ticker,
                "face_price": spec.face_price,
                "bid_ask_bps": spec.bid_ask_bps,
                "liquidity_score": spec.liquidity_score
            }
        
        data[portfolio.name] = {
            "name": portfolio.name,
            "current_account": portfolio.current_account,
            "securities": securities_data,
            "pos_qty": dict(portfolio.pos_qty),
            "pnl_realized": portfolio.pnl_realized,
            "repo_liabilities": portfolio.repo_liabilities,
            "td_assets": portfolio.td_assets,
            "last_updated": _now_ts()
        }
        return data
    
    _json_mutate(PLAYER_PORTFOLIOS_PATH, {}, update_portfolios)

def execute_player_action_direct(group_name: str, inputs: dict, prices: dict, r: int) -> dict:
    """Execute a player action directly and immediately save to shared state"""
    portfolio = get_player_portfolio(group_name)
    if not portfolio:
        return {"error": "Portfolio not found"}
    
    # Process maturities first
    process_maturities(portfolio, r)
    
    # Execute the action using your existing logic
    result = _execute_single_action_direct(portfolio, inputs, prices, r)
    
    # Save updated portfolio immediately
    save_player_portfolio(portfolio)
    
    # Log the action for host tracking
    if result.get("actions"):
        st.session_state.logs.setdefault(group_name, []).append({
            "round": r + 1,
            "request": float(st.session_state.withdrawals[r]) if r < len(st.session_state.withdrawals) else 0.0,
            "actions": result["actions"],
            "timestamp": _now_ts()
        })
    
    return result

def _execute_single_action_direct(portfolio: Portfolio, inputs: dict, prices: dict, r: int) -> dict:
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
    
    # 1) Cash use
    if cash_amt > 0:
        use = min(cash_amt, max(0.0, portfolio.current_account))
        if use > 0:
            portfolio.current_account -= use
            actions.append(("cash", {"used": round(use, 2)}))
    
    # 2) Repo
    if repo_tick != "(none)" and repo_amt > 0 and repo_tick in prices:
        price = float(prices.get(repo_tick, 0.0))
        max_amt = portfolio.pos_qty.get(repo_tick, 0.0) * price
        repo_amt = min(repo_amt, max_amt)
        if repo_amt > 0:
            repo_rate = daily_rates_for_round(r)[0]
            info = _safe_repo_call(portfolio, repo_tick, repo_amt, price, r, repo_rate)
            got = float(info["got"])
            
            actions.append(("repo", {
                "ticker": repo_tick, "got": round(got, 2), "use": round(got, 2),
                "repo_id": info["repo_id"], "rate": repo_rate
            }))
    
    # 3) Redeem TD
    if redeem_amt > 0:
        red = _safe_redeem_td(portfolio, redeem_amt, r)
        principal = float(red["principal"])
        
        actions.append(("redeem_td", {
            "principal": round(principal, 2),
            "penalty": round(float(red["penalty"]), 2),
            "use": round(principal, 2),
            "chunks": red.get("redeemed", []),
        }))
    
    # 4) Sell
    if sell_tick != "(none)" and sell_qty > 0 and sell_tick in prices:
        sell_qty = min(sell_qty, portfolio.pos_qty.get(sell_tick, 0.0))
        if sell_qty > 0:
            sale = _safe_sale(portfolio, sell_tick, sell_qty, prices[sell_tick])
            proceeds = sale["proceeds"]
            
            actions.append(("sell", {
                "ticker": sale["ticker"],
                "qty": round(sale["qty"], 2),
                "proceeds": round(proceeds, 2),
                "use": round(proceeds, 2),
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
    if buy_tick != "(none)" and buy_qty > 0 and buy_tick in prices:
        buy = _safe_buy(portfolio, buy_tick, buy_qty, prices[buy_tick])
        actions.append(("buy", {
            "ticker": buy["ticker"],
            "qty": round(buy["qty"], 2),
            "cost": round(buy["cost"], 2),
            "effective_price": round(buy["effective_price"], 6),
        }))
    
    return {
        "actions": actions,
        "portfolio_summary": portfolio.summary(prices)
    }

# ---- safe adapters (same as before)
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
- **Players**: Claim a group → make actions that update immediately.
- **Host**: **Refresh status** to see latest player actions (read-only).
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

            # Initialize player portfolios with securities data
            player_portfolios = {}
            for p in st.session_state.portfolios:
                # Convert securities to serializable format
                securities_data = {}
                for ticker, spec in p.securities.items():
                    securities_data[ticker] = {
                        "ticker": spec.ticker,
                        "face_price": spec.face_price,
                        "bid_ask_bps": spec.bid_ask_bps,
                        "liquidity_score": spec.liquidity_score
                    }
                
                player_portfolios[p.name] = {
                    "name": p.name,
                    "current_account": p.current_account,
                    "securities": securities_data,
                    "pos_qty": dict(p.pos_qty),
                    "pnl_realized": p.pnl_realized,
                    "repo_liabilities": list(p.repo_liabilities),
                    "td_assets": list(p.td_assets),
                    "last_updated": _now_ts()
                }
            _json_write_atomic(PLAYER_PORTFOLIOS_PATH, player_portfolios)
            # Generate withdrawal schedule for all rounds
            withdrawals = []
            for round_idx in range(rounds_val):
                req = generate_withdrawal(
                    round_idx,
                    base_portfolios[0].market_value(prices_all0),
                    random.Random(seed_val + 10007*round_idx)
                )
                withdrawals.append(float(req))
            
            st.session_state.withdrawals = withdrawals
            st.session_state.inited_rounds = set(range(rounds_val))
            
            _json_write_atomic(SHARED_STATE_PATH, {
                "initialized": True,
                "rng_seed": seed_val,
                "rounds": rounds_val,
                "num_groups": cap,
                "current_round": 0,
                "withdrawals": withdrawals,  # Share withdrawals with players
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
    st.session_state.withdrawals = shared.get("withdrawals", [0.0 for _ in range(st.session_state.rounds)])  # Get from shared state
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

df = st.session_state.price_df
all_tickers = [c for c in df.columns if c != "date"]
tickers_ui = all_tickers[:3]  # the 3 shown in UI
r = st.session_state.current_round
NG = max(1, int(st.session_state.get("num_groups", 1)))

# Compute prices (ALL for valuation, subset for UI lines)
date_str, prices_all = base_prices_for_round(r, df, all_tickers)
prices_ui = {t: prices_all[t] for t in tickers_ui}

# ------------------------
# Host flow: settle maturities, ensure withdrawal, read-only refresh
# ------------------------
if role == "Host":
    # Safety check: ensure portfolios exist before processing
    if not st.session_state.portfolios:
        st.info("Please upload a CSV and click Start/Reset to initialize portfolios.")
        st.stop()
        
    if st.session_state.last_maturity_round != r:
        for p in st.session_state.portfolios:
            process_maturities(p, r)
        st.session_state.last_maturity_round = r
    ensure_round_initialized(r, prices_all)
    # Safe withdrawal access - handle end game scenarios
    req_all = 0.0
    if r < len(st.session_state.withdrawals):
        req_all = float(st.session_state.withdrawals[r])

    # Manual refresh (host clicks) - Updates host view with latest player portfolio states
    if 'refresh_clicked' in locals() and refresh_clicked:
        # Sync host portfolios with latest player states from shared files
        all_portfolios = _json_read(PLAYER_PORTFOLIOS_PATH, {})
        for i, p in enumerate(st.session_state.portfolios):
            if p.name in all_portfolios:
                player_data = all_portfolios[p.name]
                # Update host portfolio with latest player data
                p.current_account = player_data["current_account"]
                p.pos_qty = dict(player_data["pos_qty"])
                p.pnl_realized = player_data["pnl_realized"]
                p.repo_liabilities = player_data.get("repo_liabilities", [])
                p.td_assets = player_data.get("td_assets", [])
                
                # Reconstruct securities if needed
                securities_data = player_data.get("securities", {})
                if securities_data:
                    from game_core import SecuritySpec
                    for ticker, spec_data in securities_data.items():
                        p.securities[ticker] = SecuritySpec(
                            ticker=spec_data["ticker"],
                            face_price=spec_data.get("face_price", 100.0),
                            bid_ask_bps=spec_data.get("bid_ask_bps", 20.0),
                            liquidity_score=spec_data.get("liquidity_score", 1)
                        )
        _safe_rerun()

# ------------------------
# Player flow: claim + direct action execution
# ------------------------
if role == "Player":
    shared = _json_read(SHARED_STATE_PATH, {})
    req_all = 0.0
    if r < len(st.session_state.withdrawals):
        req_all = float(st.session_state.withdrawals[r])

    # Read current portfolio states from shared file
    all_portfolios = _json_read(PLAYER_PORTFOLIOS_PATH, {})
    group_names = list(all_portfolios.keys())
    
    if not group_names:
        st.info("Waiting for Host to initialize groups…")
        st.stop()

    claims: Dict[str, str] = shared.get("claims", {})
    # Claims status
    claim_cols = st.columns(len(group_names))
    for i, c in enumerate(claim_cols):
        gname = group_names[i]
        with c:
            owner = claims.get(gname, "")
            st.caption(f"{gname}: {'(unclaimed)' if not owner else 'claimed by: ' + owner}")

    # Choose + claim with persistent selection
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
# Shared UI: dashboard (Host reads from session state, Player reads from files)
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
            # Safe withdrawal access for end game scenarios
            req_all_for_progress = 0.0
            if r < len(st.session_state.withdrawals):
                req_all_for_progress = float(st.session_state.withdrawals[r])
            rem = compute_remaining_for_group(p.name, r, req_all_for_progress)
            reserve = p.market_value(prices_all)
            st.markdown(f"### {p.name}")
            st.markdown(f"<div style='font-size:28px; font-weight:800; color:#006400;'>{_fmt_money(reserve,0)}</div>", unsafe_allow_html=True)
            for t in tickers_ui:
                st.markdown(f"<div class='ticker-line'>{t}: {p.pos_qty.get(t,0.0):,.0f} @ {_fmt_money(prices_ui[t])}</div>", unsafe_allow_html=True)
            prog = 0.0 if req_all_for_progress <= 0 else max(0.0, 1 - rem/req_all_for_progress)
            st.progress(prog)
else:
    # Player dashboard reads from shared files
    all_portfolios = _json_read(PLAYER_PORTFOLIOS_PATH, {})
    cols = st.columns(min(NG, len(all_portfolios)))
    for g, c in enumerate(cols):
        group_names_list = list(all_portfolios.keys())
        if g >= len(group_names_list): break
        with c:
            group_name = group_names_list[g]
            portfolio = get_player_portfolio(group_name)
            if not portfolio:
                st.error(f"Could not load {group_name}")
                continue
                
            process_maturities(portfolio, r)  # For display
            reserve = portfolio.market_value(prices_all)
            rem = compute_remaining_for_group(group_name, r, req_all)
            
            st.markdown(f"### {group_name}")
            st.markdown(f"<div style='font-size:28px; font-weight:800; color:#006400;'>{_fmt_money(reserve,0)}</div>", unsafe_allow_html=True)
            for t in tickers_ui:
                st.markdown(f"<div class='ticker-line'>{t}: {portfolio.pos_qty.get(t,0.0):,.0f} @ {_fmt_money(prices_ui[t])}</div>", unsafe_allow_html=True)
            prog = 0.0 if req_all <= 0 else max(0.0, 1 - rem/req_all)
            st.progress(prog)

# ------------------------
# Detailed tabs
# ------------------------
if role == "Host":
    # Host tabs show session state portfolios
    tab_labels = [p.name for p in st.session_state.portfolios[:NG]]
    tabs = st.tabs(tab_labels if tab_labels else ["Group 1"])
    for g, tab in enumerate(tabs):
        if g >= len(st.session_state.portfolios): break
        with tab:
            p = st.session_state.portfolios[g]
            # Safe withdrawal access
            req_for_tab = 0.0
            if r < len(st.session_state.withdrawals):
                req_for_tab = float(st.session_state.withdrawals[r])
            rem = compute_remaining_for_group(p.name, r, req_for_tab)
            st.markdown(f"### {p.name} (Host View)")
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
            st.markdown(f"**Withdrawal (all groups):** :blue[{_fmt_money(req_for_tab)}]")
            st.markdown(f"**Remaining for {p.name}:** :orange[{_fmt_money(rem)}]")
else:
    # Player tabs show portfolios from files
    all_portfolios = _json_read(PLAYER_PORTFOLIOS_PATH, {})
    group_names_list = list(all_portfolios.keys())
    
    if group_names_list:
        tabs = st.tabs(group_names_list)
        chosen_idx = st.session_state.player_group_index if st.session_state.player_group_index < len(group_names_list) else 0
        chosen_name = group_names_list[chosen_idx] if group_names_list else ""
        shared = _json_read(SHARED_STATE_PATH, {})
        claims = shared.get("claims", {})
        you_own = claims.get(chosen_name, "") == st.session_state.player_name.strip()

        for gi, tab in enumerate(tabs):
            with tab:
                group_name = group_names_list[gi]
                portfolio = get_player_portfolio(group_name)
                if not portfolio:
                    st.error(f"Could not load portfolio for {group_name}")
                    continue

                # Process maturities for display
                process_maturities(portfolio, r)
                summary = portfolio.summary(prices_all)
                rem = compute_remaining_for_group(group_name, r, req_all)

                you_own_this_group = claims.get(group_name, "") == st.session_state.player_name.strip()
                st.markdown(f"### {group_name}{' (You)' if (gi==chosen_idx and you_own) else ''}")
                
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

                # Show withdrawal progress
                st.markdown(f"**Withdrawal (all groups):** :blue[{_fmt_money(req_all)}]")
                st.markdown(f"**Remaining:** :orange[{_fmt_money(rem)}]")

                # Show updated positions with bid/ask spreads for trading decisions
                st.markdown("**Current Holdings & Market Prices:**")
                
                for t in tickers_ui:
                    qty = float(portfolio.pos_qty.get(t, 0.0))
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

                # Player action interface - ONLY for owned groups
                if gi == chosen_idx and you_own:
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
                        available_units = float(portfolio.pos_qty.get(repo_tick, 0.0))
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
                    st.selectbox("Sell ticker", ["(none)"] + tickers_ui, index=(["(none)"] + tickers_ui).index(st.session_state.get(sell_tick_key, "(none)")), key=sell_tick_key)
                    st.number_input("Sell qty", min_value=0.0, step=0.01, value=st.session_state.get(sell_qty_key, 0.0), format="%.2f", key=sell_qty_key)
                    
                    # Sell preview
                    sell_tick = st.session_state.get(sell_tick_key, "(none)")
                    sell_qty = get_float(sell_qty_key)
                    if sell_tick != "(none)" and sell_qty > 0 and sell_tick in prices_ui:
                        base_price = float(prices_ui[sell_tick])
                        effective_price = calculate_effective_price_with_spread(base_price, False)
                        available_qty = float(portfolio.pos_qty.get(sell_tick, 0.0))
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
                        current_cash = float(summary['current_account'])
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

                    # Action button - execute immediately
                    if st.button("Execute Action", type="primary", key=f"execute_{gi}_{r}"):
                        inputs = _collect_current_inputs()
                        
                        # Check if any action is specified
                        has_action = any([
                            inputs.get("cash", 0) > 0,
                            inputs.get("repo_amt", 0) > 0 and inputs.get("repo_tick") != "(none)",
                            inputs.get("redeem", 0) > 0,
                            inputs.get("invest_td", 0) > 0,
                            inputs.get("sell_qty", 0) > 0 and inputs.get("sell_tick") != "(none)",
                            inputs.get("buy_qty", 0) > 0 and inputs.get("buy_tick") != "(none)"
                        ])
                        
                        if has_action:
                            # Execute directly on shared portfolio state
                            result = execute_player_action_direct(group_name, inputs, prices_ui, r)
                            
                            if result.get("actions"):
                                st.success("Action executed successfully!")
                                
                                # Clear input fields after successful execution
                                for k in [cash_key, repo_amt_key, redeem_key, invest_key, sell_qty_key, buy_qty_key]:
                                    if k in st.session_state:
                                        del st.session_state[k]
                                for k in [repo_tick_key, sell_tick_key, buy_tick_key]:
                                    if k in st.session_state:
                                        del st.session_state[k]
                                        
                                _safe_rerun()
                            else:
                                st.warning("No valid action could be executed")
                        else:
                            st.warning("Please enter some action values before executing.")
                        
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
                if st.session_state.current_round + 1 < st.session_state.rounds:
                    st.session_state.current_round += 1
                else:
                    st.session_state.current_round = st.session_state.rounds
                _json_mutate(SHARED_STATE_PATH, {}, lambda s: {**s, "current_round": st.session_state.current_round, "ts": _now_ts()})
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
    
    st.header("Scoreboard & Logs")
    
    # Get latest player portfolio states for accurate final scores
    all_portfolios = _json_read(PLAYER_PORTFOLIOS_PATH, {})
    rows = []
    
    for portfolio_name in [p.name for p in st.session_state.portfolios]:
        if portfolio_name in all_portfolios:
            # Load latest player portfolio state
            portfolio = get_player_portfolio(portfolio_name)
            if portfolio:
                # Process final maturities
                process_maturities(portfolio, r)
                s = portfolio.summary(final_px_all)
                rows.append({
                    "group": portfolio_name,
                    "current_account": s["current_account"],
                    "securities_reserve": s["securities_mv"],
                    "repo_outstanding": s["repo_outstanding"],
                    "term_deposit": s["td_invested"],
                    "pnl_realized": s["pnl_realized"],
                    "total_reserve": s["total_mv"],
                })
        else:
            # Fallback to host session state if no player data found
            p = next((p for p in st.session_state.portfolios if p.name == portfolio_name), None)
            if p:
                process_maturities(p, r)
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
elif st.session_state.role == "Player":
    shared = _json_read(SHARED_STATE_PATH, {})
    if int(shared.get("current_round", 0)) >= int(shared.get("rounds", st.session_state.rounds)):
        st.header("Game finished — waiting for host to display final results.")
