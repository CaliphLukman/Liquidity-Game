# streamlit_app.py
import os
import json
import time
import random
from typing import List, Dict, Tuple, Any

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
ACTIONS_QUEUE_PATH  = ".actions_queue.json"  # players enqueue actions: {"type": "apply"|"clear", ...}
SNAPSHOT_PATH       = ".snapshot.json"       # host publishes live snapshot for all

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

# safe adapters (unchanged)
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
    res = execute_sale(portfolio, ticker, qty, px)
    if isinstance(res, dict):
        return {"ticker": res.get("ticker", ticker),
                "qty": float(res.get("qty", qty)),
                "proceeds": float(res.get("proceeds", qty * px)),
                "pnl_delta": float(res.get("pnl_delta", 0.0)),
                "effective_price": float(res.get("effective_price", px))}
    return {"ticker": ticker, "qty": qty, "proceeds": qty*px, "pnl_delta": 0.0, "effective_price": px}

def _safe_buy(portfolio: Portfolio, ticker: str, qty: float, px: float) -> Dict[str, Any]:
    res = execute_buy(portfolio, ticker, qty, px)
    if isinstance(res, dict):
        return {"ticker": res.get("ticker", ticker),
                "qty": float(res.get("qty", qty)),
                "cost": float(res.get("cost", qty * px)),
                "effective_price": float(res.get("effective_price", px))}
    return {"ticker": ticker, "qty": qty, "cost": qty*px, "effective_price": px}

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
    shared = _json_read(SHARED_STATE_PATH, {})
    if not shared.get("initialized"):
        st.sidebar.info("Waiting for Host to upload CSV and start…")
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

            # Clear queue & claims, publish shared session & blank snapshot
            _json_write_atomic(ACTIONS_QUEUE_PATH, [])
            _json_write_atomic(SNAPSHOT_PATH, {})
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
    shared = _json_read(SHARED_STATE_PATH, {})
    if not shared.get("initialized"):
        st.info("Waiting for Host to start the session.")
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
    groups_out = []
    for p in st.session_state.portfolios:
        s = p.summary(prices_all)  # ALL tickers
        positions = {t: float(p.pos_qty.get(t, 0.0)) for t in tickers_ui}
        groups_out.append({
            "name": p.name,
            "summary": {
                "current_account": s["current_account"],
                "securities_mv": s["securities_mv"],
                "repo_outstanding": s["repo_outstanding"],
                "td_invested": s["td_invested"],
                "pnl_realized": s["pnl_realized"],
                "total_mv": s["total_mv"],
            },
            "positions": positions
        })
    _json_write_atomic(SNAPSHOT_PATH, {
        "round": r,
        "tickers": tickers_ui,
        "groups": groups_out,
        "withdrawal": float(st.session_state.withdrawals[r]) if r < len(st.session_state.withdrawals) else 0.0,
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
    # 2) repo
    if repo_tick != "(none)" and repo_amt > 0 and rem_left > 0:
        price = float(px_ui.get(repo_tick, 0.0))
        max_amt = p.pos_qty.get(repo_tick, 0.0) * price
        repo_amt = min(repo_amt, max_amt)
        if repo_amt > 0:
            repo_rate = daily_rates_for_round(r)[0]
            info = _safe_repo_call(p, repo_tick, repo_amt, price, r, repo_rate)
            got = float(info["got"])
            use = min(got, rem_left)
            if use > 0:
                p.current_account -= use
                rem_left -= use
            history.append(("repo", {
                "ticker": repo_tick, "got": round(got, 2), "use": round(use, 2),
                "repo_id": info["repo_id"], "rate": repo_rate
            }))
    # 3) redeem td
    if redeem_amt > 0 and rem_left > 0:
        red = _safe_redeem_td(p, redeem_amt, r)
        principal = float(red["principal"])
        use = min(principal, rem_left)
        if use > 0:
            p.current_account -= use
            rem_left -= use
        history.append(("redeem_td", {
            "principal": round(principal, 2),
            "penalty":   round(float(red["penalty"]), 2),
            "use":       round(use, 2),
            "chunks":    red.get("redeemed", []),
        }))
    # 4) sell
    if sell_tick != "(none)" and sell_qty > 0 and rem_left > 0:
        sell_qty = min(sell_qty, p.pos_qty.get(sell_tick, 0.0))
        if sell_qty > 0:
            sale = _safe_sale(p, sell_tick, sell_qty, px_ui[sell_tick])
            use = min(sale["proceeds"], rem_left)
            if use > 0:
                p.current_account -= use
                rem_left -= use
            history.append(("sell", {
                "ticker": sale["ticker"],
                "qty": round(sale["qty"], 2),
                "proceeds": round(sale["proceeds"], 2),
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

def _host_consume_actions_and_publish(req_all: float):
    actions = _json_read(ACTIONS_QUEUE_PATH, [])
    if not isinstance(actions, list):
        actions = []
    if actions:
        for a in actions:
            gi = int(a.get("group_index", -1))
            rr = int(a.get("round", -1))
            a_type = a.get("type", "apply")
            if rr != r:  # only current round
                continue
            if 0 <= gi < len(st.session_state.portfolios):
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
    if not snapshot:
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

    # choose + claim
    chosen = st.sidebar.selectbox("Select your Group", group_names, index=min(st.session_state.player_group_index, len(group_names)-1))
    st.session_state.player_group_index = group_names.index(chosen)

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
st.caption(f"Today’s rates → Repo: {repo_rate_today*100:.2f}%  •  TD: {td_rate_today*100:.2f}%  •  Early TD penalty: 1.00%")

if role == "Host":
    cols = st.columns(NG)
    for g, c in enumerate(cols):
        if g >= len(st.session_state.portfolios): break
        with c:
            p = st.session_state.portfolios[g]
            rem = compute_remaining_for_group(p.name, r, float(st.session_state.withdrawals[r]))
            reserve = p.market_value(prices_all)
            st.markdown(f"### {p.name}")
            st.markdown(f"<div style='font-size:28px; font-weight:800; color:#006400;'>${reserve:,.0f}</div>", unsafe_allow_html=True)
            for t in tickers_ui:
                st.markdown(f"<div class='ticker-line'>{t}: {p.pos_qty.get(t,0.0):,.0f} @ {prices_ui[t]:,.2f}</div>", unsafe_allow_html=True)
            st.progress(max(0.0, 1 - rem/float(st.session_state.withdrawals[r]) if float(st.session_state.withdrawals[r]) > 0 else 1))
else:
    snapshot = _json_read(SNAPSHOT_PATH, {})
    groups = snapshot.get("groups", [])
    cols = st.columns(min(NG, len(groups)))
    for g, c in enumerate(cols):
        if g >= len(groups): break
        with c:
            G = groups[g]
            s = G["summary"]
            st.markdown(f"### {G['name']}")
            st.markdown(f"<div style='font-size:28px; font-weight:800; color:#006400;'>${s['total_mv']:,.0f}</div>", unsafe_allow_html=True)
            for t in tickers_ui:
                qty = float(G["positions"].get(t, 0.0))
                st.markdown(f"<div class='ticker-line'>{t}: {qty:,.0f}</div>", unsafe_allow_html=True)
            st.progress(1.0 if req_all <= 0 else 0.0)

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
                st.metric("Current Account",        f"{summary['current_account']:,.2f}")
                st.metric("Repo Outstanding",       f"{summary['repo_outstanding']:,.2f}")
            with c2:
                st.metric("Securities Reserve",     f"{summary['securities_mv']:,.2f}")
                st.metric("Term Deposit (Asset)",   f"{summary['td_invested']:,.2f}")
            with c3:
                st.metric("PnL Realized",           f"{summary['pnl_realized']:,.2f}")
                st.metric("Total Reserve",          f"{summary['total_mv']:,.2f}")
            st.markdown(f"**Withdrawal (all groups):** :blue[{req_all:,.2f}]")
            st.markdown(f"**Remaining for {p.name}:** :orange[{rem:,.2f}]")
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
                s = G["summary"]
                st.markdown(f"### {G['name']}{' (You)' if (gi==chosen_idx and you_own) else ''}")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Current Account",        f"{s['current_account']:,.2f}")
                    st.metric("Repo Outstanding",       f"{s['repo_outstanding']:,.2f}")
                with c2:
                    st.metric("Securities Reserve",     f"{s['securities_mv']:,.2f}")
                    st.metric("Term Deposit (Asset)",   f"{s['td_invested']:,.2f}")
                with c3:
                    st.metric("PnL Realized",           f"{s['pnl_realized']:,.2f}")
                    st.metric("Total Reserve",          f"{s['total_mv']:,.2f}")
                st.markdown(f"**Withdrawal (all groups):** :blue[{snapshot.get('withdrawal', 0.0):,.2f}]")

                # Inputs ONLY on your own claimed group; others read-only
                if gi == chosen_idx and you_own:
                    tickers_ui = snapshot.get("tickers", tickers_ui)

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

                    st.number_input("Use cash", min_value=0.0, step=0.01, value=st.session_state.get(cash_key, 0.0), format="%.2f", key=cash_key)
                    st.selectbox("Repo ticker", ["(none)"] + tickers_ui, index=(["(none)"] + tickers_ui).index(st.session_state.get(repo_tick_key, "(none)")), key=repo_tick_key)
                    st.number_input("Repo amount", min_value=0.0, step=0.01, value=st.session_state.get(repo_amt_key, 0.0), format="%.2f", key=repo_amt_key)
                    st.caption(f"Today’s Repo rate: {daily_rates_for_round(r)[0]*100:.2f}%")

                    st.number_input("Redeem Term Deposit", min_value=0.0, step=0.01, value=st.session_state.get(redeem_key, 0.0), format="%.2f", key=redeem_key)
                    st.caption("Early redemption penalty: 1.00%")

                    st.selectbox("Sell ticker", ["(none)"] + tickers_ui, index=(["(none)"] + tickers_ui).index(st.session_state.get(sell_tick_key, "(none)")), key=sell_tick_key)
                    st.number_input("Sell qty", min_value=0.0, step=0.01, value=st.session_state.get(sell_qty_key, 0.0), format="%.2f", key=sell_qty_key)

                    st.number_input("Invest in Term Deposit", min_value=0.0, step=0.01, value=st.session_state.get(invest_key, 0.0), format="%.2f", key=invest_key)
                    st.caption(f"Today’s TD rate (if held to maturity): {daily_rates_for_round(r)[1]*100:.2f}%")

                    st.selectbox("Buy ticker", ["(none)"] + tickers_ui, index=(["(none)"] + tickers_ui).index(st.session_state.get(buy_tick_key, "(none)")), key=buy_tick_key)
                    st.number_input("Buy qty", min_value=0.0, step=0.01, value=st.session_state.get(buy_qty_key, 0.0), format="%.2f", key=buy_qty_key)

                    def _enqueue_player_action(a_type: str):
                        action = {
                            "ts": _now_ts(),
                            "type": a_type,  # "apply" or "clear"
                            "by": st.session_state.player_name.strip(),
                            "group_index": gi,
                            "group_name": G["name"],
                            "round": r,
                            "inputs": {
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
                        }
                        _json_mutate(ACTIONS_QUEUE_PATH, [], lambda q: (q if isinstance(q, list) else []) + [action])
                        if a_type == "apply":
                            # clear local inputs after applying
                            for k in [cash_key, repo_amt_key, redeem_key, invest_key, sell_qty_key, buy_qty_key]:
                                st.session_state[k] = 0.0
                            for k in [repo_tick_key, sell_tick_key, buy_tick_key]:
                                st.session_state[k] = "(none)"

                    b1, b2 = st.columns([2,1])
                    b1.button("Apply Action", on_click=lambda: _enqueue_player_action("apply"))
                    b2.button("Clear Actions", on_click=lambda: _enqueue_player_action("clear"))
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
