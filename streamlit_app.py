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

# File paths for persistent state
GAME_CONFIG_PATH = ".game_config.json"       # Host configuration
PLAYER_PORTFOLIOS_PATH = ".player_portfolios.json"  # All portfolio states
GAME_STATE_PATH = ".game_state.json"         # Current round, withdrawals, etc.

# ------------------------
# Helper functions
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

def _fmt_money(x: float, nd: int = 2) -> str:
    try:
        return f"${x:,.{nd}f}"
    except Exception:
        return f"${x}"

def calculate_effective_price_with_spread(base_price: float, is_buy: bool, liquidity_factor: float = 0.005) -> float:
    """Calculate effective price with bid-ask spread for trading."""
    if is_buy:
        return base_price * (1 + liquidity_factor)  # Pay slightly more when buying
    else:
        return base_price * (1 - liquidity_factor)  # Receive slightly less when selling

# ------------------------
# Session State bootstrap
# ------------------------
def _init_state():
    ss = st.session_state
    ss.role = "Host"
    ss.player_name = ""
    ss.claimed_group = ""

if "role" not in st.session_state:
    _init_state()

# ------------------------
# Game state functions
# ------------------------
def get_game_config():
    return _json_read(GAME_CONFIG_PATH, {})

def get_game_state():
    return _json_read(GAME_STATE_PATH, {})

def get_all_portfolios():
    return _json_read(PLAYER_PORTFOLIOS_PATH, {})

def save_portfolio(portfolio: Portfolio):
    """Save a single portfolio"""
    def update_portfolios(data):
        data = data or {}
        data[portfolio.name] = {
            "name": portfolio.name,
            "current_account": portfolio.current_account,
            "pos_qty": dict(portfolio.pos_qty),
            "pnl_realized": portfolio.pnl_realized,
            "repo_liabilities": portfolio.repo_liabilities,
            "td_assets": portfolio.td_assets,
            "last_updated": time.time()
        }
        return data
    
    _json_mutate(PLAYER_PORTFOLIOS_PATH, {}, update_portfolios)

def load_portfolio(group_name: str) -> Optional[Portfolio]:
    """Load a single portfolio"""
    all_portfolios = get_all_portfolios()
    if group_name not in all_portfolios:
        return None
    
    data = all_portfolios[group_name]
    portfolio = Portfolio(
        name=data["name"],
        current_account=data["current_account"],
        pos_qty=data["pos_qty"],
        pnl_realized=data["pnl_realized"]
    )
    portfolio.repo_liabilities = data.get("repo_liabilities", [])
    portfolio.td_assets = data.get("td_assets", [])
    return portfolio

def get_current_prices():
    """Get current round prices"""
    config = get_game_config()
    state = get_game_state()
    
    if not config.get("initialized") or "price_data" not in config:
        return {}, []
    
    df_data = config["price_data"]
    df = pd.DataFrame(df_data)
    current_round = state.get("current_round", 0)
    
    all_tickers = [c for c in df.columns if c != "date"]
    tickers_ui = all_tickers[:3]
    
    row_index = min(current_round, len(df) - 1)
    date_str = str(df.loc[row_index, "date"])
    prices = {t: float(df.loc[row_index, t]) for t in all_tickers}
    
    return prices, tickers_ui

def daily_rates_for_round(r: int, seed: int) -> Tuple[float, float]:
    rng = random.Random(seed * 991 + r * 7919)
    repo_delta = rng.uniform(-DAILY_SPREAD, DAILY_SPREAD)
    td_delta = rng.uniform(-DAILY_SPREAD, DAILY_SPREAD)
    return max(0.0, BASE_REPO_RATE + repo_delta), max(0.0, BASE_TD_RATE + td_delta)

def get_withdrawal_requirement():
    """Get current withdrawal requirement"""
    config = get_game_config()
    state = get_game_state()
    
    if not config.get("initialized"):
        return 0.0
    
    current_round = state.get("current_round", 0)
    withdrawals = state.get("withdrawals", [])
    
    if current_round >= len(withdrawals):
        return 0.0
    
    return withdrawals[current_round]

# ------------------------
# Portfolio execution functions
# ------------------------
def execute_player_action(group_name: str, action_type: str, inputs: dict) -> dict:
    """Execute a player action and return results"""
    portfolio = load_portfolio(group_name)
    if not portfolio:
        return {"success": False, "error": "Portfolio not found"}
    
    config = get_game_config()
    state = get_game_state()
    current_round = state.get("current_round", 0)
    
    # Process maturities first
    process_maturities(portfolio, current_round)
    
    # Get current prices
    prices, _ = get_current_prices()
    if not prices:
        return {"success": False, "error": "Prices not available"}
    
    try:
        # Execute the action based on type
        if action_type == "cash":
            amount = float(inputs.get("amount", 0))
            if amount > 0:
                used = min(amount, max(0.0, portfolio.current_account))
                portfolio.current_account -= used
                result = {"used": used}
        
        elif action_type == "repo":
            ticker = inputs.get("ticker")
            amount = float(inputs.get("amount", 0))
            if ticker and ticker in prices and amount > 0:
                price = prices[ticker]
                max_collateral = portfolio.pos_qty.get(ticker, 0.0) * price
                actual_amount = min(amount, max_collateral)
                if actual_amount > 0:
                    repo_rate = daily_rates_for_round(current_round, config.get("rng_seed", 1234))[0]
                    got, repo_id = execute_repo(portfolio, ticker, actual_amount, price, current_round, rate=repo_rate)
                    portfolio.current_account += got
                    result = {"got": got, "repo_id": repo_id}
        
        elif action_type == "sell":
            ticker = inputs.get("ticker")
            qty = float(inputs.get("qty", 0))
            if ticker and ticker in prices and qty > 0:
                available = portfolio.pos_qty.get(ticker, 0.0)
                actual_qty = min(qty, available)
                if actual_qty > 0:
                    base_price = prices[ticker]
                    effective_price = calculate_effective_price_with_spread(base_price, False)
                    proceeds = actual_qty * effective_price
                    portfolio.pos_qty[ticker] -= actual_qty
                    portfolio.current_account += proceeds
                    result = {"qty": actual_qty, "proceeds": proceeds, "effective_price": effective_price}
        
        elif action_type == "buy":
            ticker = inputs.get("ticker")
            qty = float(inputs.get("qty", 0))
            if ticker and ticker in prices and qty > 0:
                base_price = prices[ticker]
                effective_price = calculate_effective_price_with_spread(base_price, True)
                cost = qty * effective_price
                if cost <= portfolio.current_account:
                    portfolio.current_account -= cost
                    portfolio.pos_qty[ticker] = portfolio.pos_qty.get(ticker, 0.0) + qty
                    result = {"qty": qty, "cost": cost, "effective_price": effective_price}
        
        elif action_type == "invest_td":
            amount = float(inputs.get("amount", 0))
            if amount > 0:
                actual_amount = min(amount, max(0.0, portfolio.current_account))
                if actual_amount > 0:
                    td_rate = daily_rates_for_round(current_round, config.get("rng_seed", 1234))[1]
                    td_ids = execute_invest_td(portfolio, actual_amount, current_round, rate=td_rate)
                    result = {"amount": actual_amount, "td_ids": td_ids, "rate": td_rate}
        
        elif action_type == "redeem_td":
            amount = float(inputs.get("amount", 0))
            if amount > 0:
                res = execute_redeem_td(portfolio, amount, current_round, penalty=TD_PENALTY)
                if isinstance(res, dict):
                    principal = float(res.get("principal", amount))
                    penalty = float(res.get("penalty", amount * TD_PENALTY))
                    portfolio.current_account += principal
                    result = {"principal": principal, "penalty": penalty}
        
        # Save the updated portfolio
        save_portfolio(portfolio)
        
        return {
            "success": True,
            "result": result if 'result' in locals() else {},
            "portfolio_summary": portfolio.summary(prices)
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

# ------------------------
# UI Components
# ------------------------
def render_host_setup():
    """Render host setup interface"""
    st.sidebar.header("Host Setup")
    
    config = get_game_config()
    
    uploaded = st.sidebar.file_uploader("Bond price CSV (date + ≥3 securities)", type=["csv"])
    seed = st.sidebar.number_input("RNG seed", value=config.get("rng_seed", 1234), step=1)
    rounds = st.sidebar.number_input("Rounds", value=config.get("rounds", 3), min_value=1, max_value=10, step=1)
    groups = st.sidebar.number_input("Groups", value=config.get("num_groups", 4), min_value=1, max_value=MAX_GROUPS_UI, step=1)
    
    if st.sidebar.button("Initialize Game", type="primary"):
        if uploaded is None:
            st.sidebar.error("Please upload a CSV file")
            return False
        
        df = pd.read_csv(uploaded)
        if "date" not in df.columns or len([c for c in df.columns if c != "date"]) < 3:
            st.sidebar.error("CSV must have 'date' and at least 3 security columns.")
            return False
        
        # Initialize game configuration
        all_tickers = [c for c in df.columns if c != "date"]
        prices_round_0 = {t: float(df.loc[0, t]) for t in all_tickers}
        
        # Create initial portfolios
        base_portfolios = init_portfolios(all_tickers, prices_round_0, total_reserve=200000.0)
        actual_groups = min(groups, len(base_portfolios))
        
        # Add random initial TD allocation
        random.seed(seed)
        for i in range(actual_groups):
            p = base_portfolios[i]
            frac = random.uniform(0.10, 0.30)
            amt = max(0.0, p.current_account) * frac
            if amt > 0:
                execute_invest_td(p, amt, 0, rate=BASE_TD_RATE)
        
        # Generate withdrawal schedule
        withdrawals = []
        for r in range(rounds):
            req = generate_withdrawal(r, base_portfolios[0].market_value(prices_round_0), 
                                    random.Random(seed + 10007*r))
            withdrawals.append(float(req))
        
        # Save configuration
        _json_write_atomic(GAME_CONFIG_PATH, {
            "initialized": True,
            "rng_seed": seed,
            "rounds": rounds,
            "num_groups": actual_groups,
            "price_data": df.to_dict(),
            "tickers": all_tickers
        })
        
        # Save initial game state
        _json_write_atomic(GAME_STATE_PATH, {
            "current_round": 0,
            "withdrawals": withdrawals,
            "last_maturity_round": -1
        })
        
        # Save initial portfolios
        portfolio_data = {}
        for p in base_portfolios[:actual_groups]:
            portfolio_data[p.name] = {
                "name": p.name,
                "current_account": p.current_account,
                "pos_qty": dict(p.pos_qty),
                "pnl_realized": p.pnl_realized,
                "repo_liabilities": list(p.repo_liabilities),
                "td_assets": list(p.td_assets),
                "last_updated": time.time()
            }
        _json_write_atomic(PLAYER_PORTFOLIOS_PATH, portfolio_data)
        
        st.sidebar.success("Game initialized!")
        _safe_rerun()
        return True
    
    return config.get("initialized", False)

def render_player_setup():
    """Render player setup interface"""
    st.sidebar.header("Player Setup")
    
    config = get_game_config()
    if not config.get("initialized"):
        st.sidebar.info("Waiting for Host to initialize the game...")
        return False, ""
    
    player_name = st.sidebar.text_input("Your name", value=st.session_state.player_name)
    st.session_state.player_name = player_name
    
    # Show available groups
    portfolios = get_all_portfolios()
    group_names = list(portfolios.keys())
    
    if not group_names:
        st.sidebar.info("No groups available yet")
        return False, ""
    
    selected_group = st.sidebar.selectbox("Select Group", group_names)
    
    if st.sidebar.button("Claim Group") and player_name.strip():
        st.session_state.claimed_group = selected_group
        st.sidebar.success(f"Playing as {selected_group}")
    
    return True, st.session_state.claimed_group

def render_host_dashboard():
    """Render host dashboard - read only"""
    config = get_game_config()
    state = get_game_state()
    current_round = state.get("current_round", 0)
    
    if st.button("Refresh Dashboard"):
        _safe_rerun()
    
    # Round controls
    if current_round < config.get("rounds", 3):
        if st.button("Next Round"):
            _json_mutate(GAME_STATE_PATH, {}, lambda s: {**s, "current_round": s.get("current_round", 0) + 1})
            _safe_rerun()
    
    prices, tickers_ui = get_current_prices()
    withdrawal_req = get_withdrawal_requirement()
    
    if not prices:
        st.info("Prices not available")
        return
    
    repo_rate, td_rate = daily_rates_for_round(current_round, config.get("rng_seed", 1234))
    
    st.subheader(f"Round {current_round + 1}")
    st.caption(f"Withdrawal Required: {_fmt_money(withdrawal_req)} | Repo: {repo_rate*100:.2f}% | TD: {td_rate*100:.2f}%")
    
    # Show all group summaries
    portfolios = get_all_portfolios()
    if portfolios:
        cols = st.columns(len(portfolios))
        for i, (group_name, portfolio_data) in enumerate(portfolios.items()):
            with cols[i]:
                # Reconstruct portfolio for calculations
                portfolio = Portfolio(
                    name=portfolio_data["name"],
                    current_account=portfolio_data["current_account"],
                    pos_qty=portfolio_data["pos_qty"],
                    pnl_realized=portfolio_data["pnl_realized"]
                )
                portfolio.repo_liabilities = portfolio_data.get("repo_liabilities", [])
                portfolio.td_assets = portfolio_data.get("td_assets", [])
                
                # Process maturities for display
                process_maturities(portfolio, current_round)
                
                summary = portfolio.summary(prices)
                
                st.markdown(f"### {group_name}")
                st.metric("Total Reserve", _fmt_money(summary["total_mv"]))
                st.metric("Current Account", _fmt_money(summary["current_account"]))
                st.metric("Securities", _fmt_money(summary["securities_mv"]))
                
                # Show holdings
                for ticker in tickers_ui:
                    qty = portfolio.pos_qty.get(ticker, 0.0)
                    st.caption(f"{ticker}: {qty:,.0f} @ {_fmt_money(prices[ticker])}")

def render_player_interface(group_name: str):
    """Render player interface for making actions"""
    prices, tickers_ui = get_current_prices()
    if not prices:
        st.info("Prices not available")
        return
    
    config = get_game_config()
    state = get_game_state()
    current_round = state.get("current_round", 0)
    withdrawal_req = get_withdrawal_requirement()
    
    repo_rate, td_rate = daily_rates_for_round(current_round, config.get("rng_seed", 1234))
    
    st.subheader(f"Round {current_round + 1} - Playing as {group_name}")
    st.caption(f"Withdrawal Required: {_fmt_money(withdrawal_req)} | Repo: {repo_rate*100:.2f}% | TD: {td_rate*100:.2f}%")
    
    # Load current portfolio
    portfolio = load_portfolio(group_name)
    if not portfolio:
        st.error("Portfolio not found")
        return
    
    # Process maturities for display
    process_maturities(portfolio, current_round)
    summary = portfolio.summary(prices)
    
    # Display current status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Reserve", _fmt_money(summary["total_mv"]))
        st.metric("Current Account", _fmt_money(summary["current_account"]))
    with col2:
        st.metric("Securities", _fmt_money(summary["securities_mv"]))
        st.metric("TD Invested", _fmt_money(summary["td_invested"]))
    with col3:
        st.metric("Repo Outstanding", _fmt_money(summary["repo_outstanding"]))
        st.metric("PnL Realized", _fmt_money(summary["pnl_realized"]))
    
    # Show holdings
    st.subheader("Current Holdings")
    for ticker in tickers_ui:
        qty = portfolio.pos_qty.get(ticker, 0.0)
        market_price = prices[ticker]
        bid_price = calculate_effective_price_with_spread(market_price, False)
        ask_price = calculate_effective_price_with_spread(market_price, True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(ticker, f"{qty:,.0f} units")
        with col2:
            st.caption(f"Bid: {_fmt_money(bid_price)} (you receive)")
        with col3:
            st.caption(f"Ask: {_fmt_money(ask_price)} (you pay)")
    
    # Action interface
    st.subheader("Actions")
    
    action_type = st.selectbox("Action Type", [
        "Use Cash", "Repo Securities", "Sell Securities", "Buy Securities", 
        "Invest in TD", "Redeem TD"
    ])
    
    if action_type == "Use Cash":
        amount = st.number_input("Amount", min_value=0.0, step=0.01)
        if st.button("Execute") and amount > 0:
            result = execute_player_action(group_name, "cash", {"amount": amount})
            if result["success"]:
                st.success(f"Used {_fmt_money(result['result']['used'])} cash")
                _safe_rerun()
            else:
                st.error(result["error"])
    
    elif action_type == "Repo Securities":
        ticker = st.selectbox("Security", tickers_ui)
        amount = st.number_input("Amount", min_value=0.0, step=0.01)
        if st.button("Execute") and amount > 0:
            result = execute_player_action(group_name, "repo", {"ticker": ticker, "amount": amount})
            if result["success"]:
                st.success(f"Repo generated {_fmt_money(result['result']['got'])}")
                _safe_rerun()
            else:
                st.error(result["error"])
    
    elif action_type == "Sell Securities":
        ticker = st.selectbox("Security", tickers_ui)
        qty = st.number_input("Quantity", min_value=0.0, step=0.01)
        if st.button("Execute") and qty > 0:
            result = execute_player_action(group_name, "sell", {"ticker": ticker, "qty": qty})
            if result["success"]:
                st.success(f"Sold {result['result']['qty']:,.1f} units for {_fmt_money(result['result']['proceeds'])}")
                _safe_rerun()
            else:
                st.error(result["error"])
    
    elif action_type == "Buy Securities":
        ticker = st.selectbox("Security", tickers_ui)
        qty = st.number_input("Quantity", min_value=0.0, step=0.01)
        if st.button("Execute") and qty > 0:
            result = execute_player_action(group_name, "buy", {"ticker": ticker, "qty": qty})
            if result["success"]:
                st.success(f"Bought {result['result']['qty']:,.1f} units for {_fmt_money(result['result']['cost'])}")
                _safe_rerun()
            else:
                st.error(result["error"])
    
    elif action_type == "Invest in TD":
        amount = st.number_input("Amount", min_value=0.0, step=0.01)
        if st.button("Execute") and amount > 0:
            result = execute_player_action(group_name, "invest_td", {"amount": amount})
            if result["success"]:
                st.success(f"Invested {_fmt_money(result['result']['amount'])} in TD at {result['result']['rate']*100:.2f}%")
                _safe_rerun()
            else:
                st.error(result["error"])
    
    elif action_type == "Redeem TD":
        amount = st.number_input("Amount", min_value=0.0, step=0.01)
        if st.button("Execute") and amount > 0:
            result = execute_player_action(group_name, "redeem_td", {"amount": amount})
            if result["success"]:
                penalty = result['result']['penalty']
                net = result['result']['principal']
                st.success(f"Redeemed TD: {_fmt_money(net)} received (penalty: {_fmt_money(penalty)})")
                _safe_rerun()
            else:
                st.error(result["error"])

# ------------------------
# Main App
# ------------------------
st.title("Liquidity Tranche Simulation")

# Role selector
st.session_state.role = st.sidebar.radio("Role", ["Host", "Player"])

if st.session_state.role == "Host":
    # Host interface
    game_initialized = render_host_setup()
    
    if game_initialized:
        render_host_dashboard()
    else:
        st.info("Please configure and initialize the game using the sidebar.")

else:
    # Player interface
    game_ready, claimed_group = render_player_setup()
    
    if game_ready and claimed_group:
        render_player_interface(claimed_group)
    elif game_ready:
        st.info("Please claim a group to start playing.")
    else:
        st.info("Waiting for the host to initialize the game.")
