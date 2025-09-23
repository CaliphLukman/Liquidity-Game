import os
import json
import uuid
import random
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

# ---- your core modules ----
from game_core import (
    Portfolio, init_portfolios, generate_withdrawal,
    execute_repo, execute_sale, execute_buy,
    execute_invest_td, execute_redeem_td,
    process_maturities
)

# small, stable serializers you already have
from serialize_core import portfolios_to_json, portfolios_from_json

# =========================
# PAGE + THEME (dark-green captions kept)
# =========================
st.set_page_config(page_title="Liquidity Tranche Simulation", layout="wide")
st.markdown("""
<style>
:root{
  --navy:#0B1F3B;         /* sidebar bg */
  --navy-strong:#0B3D91;  /* headings & buttons */
  --white:#FFFFFF;
  --ink:#111827;          /* near-black text */
  --green:#006400;        /* $ totals + captions */
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

/* Sidebar (dark navy, white text) */
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
  color:#EAF2FF !important; opacity:1 !important;
}

/* File uploader */
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]{
  background:#102A53 !important; border:none !important; border-radius:10px !important;
}

/* Number steppers: main = black; sidebar = white */
[data-testid="stNumberInput"] button{
  background:transparent !important; color:#000 !important;
  border:1px solid var(--navy-strong) !important; border-radius:6px !important;
}
section[data-testid="stSidebar"] [data-testid="stNumberInput"] button{
  color:#fff !important; border:1px solid var(--light-navy) !important;
}

/* Buttons */
div.stButton > button{
  background:var(--navy-strong) !important;
  color:#fff !important; border:0 !important; border-radius:8px !important;
  font-weight:700 !important; padding:.45rem .9rem !important;
}
div.stButton > button:hover{ filter:brightness(.92); }
</style>
""", unsafe_allow_html=True)

# =========================
# CONSTANTS / CONFIG
# =========================
BASE_REPO_RATE = 0.015      # 1.5%
BASE_TD_RATE   = 0.0175     # 1.75%
DAILY_SPREAD   = 0.005      # ±50 bps
TD_PENALTY     = 0.01       # 1.0% early redemption penalty
TD_MAT_GAP     = 2          # TDs mature after 2 rounds (handled in game_core)
MAX_GROUPS_UI  = 8          # Host can choose up to 8
TICKERS_SHOW   = 3          # show first 3 columns of CSV

# =========================
# SIMPLE DISK PERSISTENCE (no external deps)
# =========================
DATA_DIR = os.path.join(os.getcwd(), "data")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

STATE_PATH = os.path.join(DATA_DIR, "global_game_state.json")

def _default_state() -> Dict[str, Any]:
    return {
        "initialized": False,
        "rng_seed": 1234,
        "rounds": 3,
        "current_round": 0,
        "last_maturity_round": -1,
        "csv_path": "",
        "tickers_all": [],
        "num_groups": 4,
        "withdrawals": [],   # per round, same for all groups
        "logs": {},          # {group_name: [ {round, request, actions: [...] } ]}
        "claimed_groups": {},# {group_index(str): display_name}
        "portfolios": [],    # serialized via serialize_core
    }

def read_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_PATH):
        return _default_state()
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return _default_state()

def write_state(state: Dict[str, Any]) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_PATH)

def save_uploaded_csv(file) -> str:
    path = os.path.join(UPLOADS_DIR, f"prices_{uuid.uuid4().hex[:8]}.csv")
    with open(path, "wb") as f:
        f.write(file.getbuffer())
    return path

# =========================
# SESSION STATE (UI-only)
# =========================
if "role" not in st.session_state:
    st.session_state.role = "Host"  # Host or Player
if "display_name" not in st.session_state:
    st.session_state.display_name = ""  # Player only
if "player_group_index" not in st.session_state:
    st.session_state.player_group_index = 0  # Player selects which
if "just_loaded" not in st.session_state:
    st.session_state.just_loaded = True  # first render flag

# =========================
# HELPERS
# =========================
def _load_df(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path) if csv_path and os.path.exists(csv_path) else pd.DataFrame()

def price_row_for_round(df: pd.DataFrame, r: int) -> int:
    if df.empty: return 0
    return min(r, len(df) - 1)

def base_prices_for_round(df: pd.DataFrame, r: int, tickers: List[str]) -> Tuple[str, Dict[str, float]]:
    if df.empty:
        return "N/A", {t: 0.0 for t in tickers}
    ix = price_row_for_round(df, r)
    return str(df.loc[ix, "date"]), {t: float(df.loc[ix, t]) for t in tickers}

def daily_rates_for_round(seed: int, r: int) -> Tuple[float, float]:
    rng = random.Random(seed * 991 + r * 7919)
    repo_delta = rng.uniform(-DAILY_SPREAD, DAILY_SPREAD)
    td_delta   = rng.uniform(-DAILY_SPREAD, DAILY_SPREAD)
    return max(0.0, BASE_REPO_RATE + repo_delta), max(0.0, BASE_TD_RATE + td_delta)

def ensure_withdrawal_for_round(state: Dict[str, Any], r: int, portfolios: List[Portfolio], prices_for_mv: Dict[str, float]):
    """Create the round withdrawal (same for all groups) if missing."""
    rounds = int(state["rounds"])
    if len(state["withdrawals"]) < rounds:
        state["withdrawals"] = [0.0 for _ in range(rounds)]
    if r >= rounds: 
        return
    if state["withdrawals"][r] == 0.0:
        req = generate_withdrawal(r, portfolios[0].market_value(prices_for_mv), random.Random(state["rng_seed"] + 10007*r))
        state["withdrawals"][r] = float(req)

def fmt_money(x: float) -> str:
    try:
        return f"${x:,.2f}"
    except Exception:
        return "$0.00"

def compute_remaining_for_group(group_name: str, r: int, req_for_round: float, logs_map: Dict[str, List[dict]]) -> float:
    """Remaining = req - sum of cash used + repo/sell/redeem 'use' in this round."""
    logs = [L for L in logs_map.get(group_name, []) if L["round"] == r+1]
    used = 0.0
    for L in logs:
        for t, d in L["actions"]:
            if t == "cash":
                used += float(d.get("used", 0.0))
            elif t in ("repo","sell","redeem_td"):
                used += float(d.get("use", 0.0))
    return max(0.0, round(req_for_round - used, 2))

# Safe adapters to tolerate minor signature differences in game_core
def _safe_repo_call(portfolio: Portfolio, ticker: str, amount: float, price: float, rnow: int, rate: float):
    try:
        res = execute_repo(portfolio, ticker, amount, price, rnow, rate=rate)
    except TypeError:
        res = execute_repo(portfolio, ticker, amount, rnow, rate=rate)  # older sig
    if isinstance(res, tuple) and len(res) >= 2:
        return {"got": float(res[0]), "repo_id": res[1]}
    if isinstance(res, dict):
        return {"got": float(res.get("got", amount)), "repo_id": res.get("id")}
    return {"got": float(amount), "repo_id": None}

def _safe_redeem_td(portfolio: Portfolio, amount: float, rnow: int):
    try:
        res = execute_redeem_td(portfolio, amount, rnow, penalty=TD_PENALTY)
    except TypeError:
        res = execute_redeem_td(portfolio, amount, rnow, penalty_rate=TD_PENALTY)
    if isinstance(res, dict):
        return {
            "principal": float(res.get("principal", amount)),
            "penalty": float(res.get("penalty", amount * TD_PENALTY)),
            "redeemed": res.get("redeemed", []),
        }
    return {"principal": amount, "penalty": amount * TD_PENALTY, "redeemed": []}

# =========================
# ROLE SWITCH
# =========================
st.sidebar.markdown("### Session")
role = st.sidebar.radio("Role", ["Host", "Player"], index=0 if st.session_state.role == "Host" else 1)
st.session_state.role = role

# =========================
# HOST SIDEBAR
# =========================
if role == "Host":
    st.sidebar.header("Host Setup")
    uploaded = st.sidebar.file_uploader("Bond price CSV (date + ≥3 securities)", type=["csv"])
    seed_in = st.sidebar.number_input("RNG seed", value=1234, step=1)
    rounds_in = st.sidebar.number_input("Rounds", value=3, min_value=1, max_value=10, step=1)
    groups_in = st.sidebar.number_input("Groups (1–8)", value=4, min_value=1, max_value=MAX_GROUPS_UI, step=1)

    c1, c2, c3 = st.sidebar.columns([1,1,1])
    start_clicked = c1.button("Start/Reset", type="primary")
    refresh_clicked = c2.button("Refresh")  # Host-controlled reload from disk
    end_clicked = c3.button("End Game")

else:
    # =========================
    # PLAYER SIDEBAR
    # =========================
    st.sidebar.header("Player Setup")
    state_now = read_state()
    if not state_now["initialized"]:
        st.sidebar.info("Waiting for the Host to start the session...")
        start_clicked = False
        refresh_clicked = False
        end_clicked = False
    else:
        st.sidebar.caption(f"Rounds: {state_now['rounds']} • Groups: {state_now['num_groups']} • RNG seed: {state_now['rng_seed']}")
        display_name = st.sidebar.text_input("Display / Nick name (required)", value=st.session_state.display_name, max_chars=24)
        st.session_state.display_name = display_name.strip()

        # Build group names from current portfolios
        portfolios_live = portfolios_from_json(state_now["portfolios"])
        group_names = [p.name for p in portfolios_live]
        default_idx = min(st.session_state.player_group_index, max(0, len(group_names)-1))
        chosen_name = st.sidebar.selectbox("Select your Group", group_names, index=default_idx)
        chosen_index = group_names.index(chosen_name)
        st.session_state.player_group_index = chosen_index

        # Claim / Lock group (strictly one player per group)
        claimed = state_now.get("claimed_groups", {})
        already_by_me = (claimed.get(str(chosen_index)) == st.session_state.display_name and st.session_state.display_name != "")
        free = (str(chosen_index) not in claimed)

        lock_col, _ = st.sidebar.columns([1,1])
        can_claim = (st.session_state.display_name != "") and (free or already_by_me)
        if lock_col.button("Join / Lock This Group", disabled=not can_claim):
            # re-read state and claim to avoid race
            s = read_state()
            s.setdefault("claimed_groups", {})
            # if already claimed by someone else, deny
            if str(chosen_index) in s["claimed_groups"] and s["claimed_groups"][str(chosen_index)] != st.session_state.display_name:
                st.sidebar.error("Sorry, this group is already taken.")
            else:
                s["claimed_groups"][str(chosen_index)] = st.session_state.display_name
                write_state(s)
                st.sidebar.success(f"You are locked to {chosen_name} as '{st.session_state.display_name}'")

        start_clicked = False
        refresh_clicked = False
        end_clicked = False

# =========================
# HOST START / RESET
# =========================
if role == "Host" and start_clicked:
    if uploaded is None:
        st.sidebar.error("Please upload a CSV with columns: date,BOND_A,BOND_B,BOND_C,...")
        st.stop()
    df = pd.read_csv(uploaded)
    if "date" not in df.columns or len([c for c in df.columns if c != "date"]) < 3:
        st.sidebar.error("CSV must have 'date' and at least 3 security columns.")
        st.stop()

    # Save CSV to disk and create state
    csv_path = save_uploaded_csv(uploaded)
    tickers_all = [c for c in df.columns if c != "date"]

    # init base portfolios
    date0, prices0 = base_prices_for_round(df, 0, tickers_all)
    base_portfolios = init_portfolios(tickers_all, prices0, total_reserve=200000.0)
    cap = min(MAX_GROUPS_UI, len(base_portfolios), int(groups_in))
    portfolios = base_portfolios[:cap]

    # random initial TD allocation round 1
    for p in portfolios:
        frac = random.uniform(0.10, 0.30)
        amt = round(max(0.0, p.current_account) * frac, 2)
        if amt > 0:
            execute_invest_td(p, amt, 0, rate=BASE_TD_RATE)

    # build fresh state
    new_state = _default_state()
    new_state.update({
        "initialized": True,
        "rng_seed": int(seed_in),
        "rounds": int(rounds_in),
        "current_round": 0,
        "last_maturity_round": -1,
        "csv_path": csv_path,
        "tickers_all": tickers_all,
        "num_groups": cap,
        "withdrawals": [0.0 for _ in range(int(rounds_in))],
        "logs": {p.name: [] for p in portfolios},
        "claimed_groups": {},  # nobody is locked yet
        "portfolios": portfolios_to_json(portfolios),
    })
    write_state(new_state)
    st.sidebar.success("Game started. Share the app link with players.")

# =========================
# HOST REFRESH
# =========================
if role == "Host" and refresh_clicked:
    # no-op; just triggers rerun which rereads from disk at render-time
    pass

# =========================
# HOST END GAME
# =========================
if role == "Host" and end_clicked:
    s = read_state()
    if not s["initialized"]:
        st.stop()
    s["current_round"] = s["rounds"]  # trigger end-game screen
    write_state(s)
    st.rerun()

# =========================
# MAIN CONTENT
# =========================
st.title("Liquidity Tranche Simulation")

state = read_state()
if not state["initialized"]:
    if role == "Host":
        st.info("Upload a CSV and click **Start/Reset** to begin, then share this link with players.")
    else:
        st.info("Waiting for the Host to start the session…")
    st.stop()

# Load df & portfolios from state
df = _load_df(state["csv_path"])
if df.empty:
    st.error("CSV not found. Ask the Host to Start/Reset again.")
    st.stop()

portfolios = portfolios_from_json(state["portfolios"])
all_tickers = state["tickers_all"]
tickers = all_tickers[:TICKERS_SHOW] if len(all_tickers) >= TICKERS_SHOW else all_tickers

r = int(state["current_round"])
rng_seed = int(state["rng_seed"])

# END-GAME SCREEN
if r >= int(state["rounds"]):
    st.header("Scoreboard & Logs")
    _, final_px = base_prices_for_round(df, min(state["rounds"]-1, len(df)-1), tickers)
    rows = []
    for p in portfolios:
        ssum = p.summary(final_px)
        rows.append({
            "group": p.name,
            "current_account": fmt_money(ssum["current_account"]),
            "securities_reserve": fmt_money(ssum["securities_mv"]),
            "repo_outstanding": fmt_money(ssum["repo_outstanding"]),
            "term_deposit": fmt_money(ssum["td_invested"]),
            "pnl_realized": fmt_money(ssum["pnl_realized"]),
            "total_reserve": fmt_money(ssum["total_mv"]),
        })
    sb = pd.DataFrame(rows)
    st.dataframe(sb, use_container_width=True)
    # raw (unformatted) exports for CSV/JSON
    raw_rows = []
    for p in portfolios:
        ssum = p.summary(final_px)
        raw_rows.append({
            "group": p.name,
            "current_account": ssum["current_account"],
            "securities_reserve": ssum["securities_mv"],
            "repo_outstanding": ssum["repo_outstanding"],
            "term_deposit": ssum["td_invested"],
            "pnl_realized": ssum["pnl_realized"],
            "total_reserve": ssum["total_mv"],
        })
    raw_df = pd.DataFrame(raw_rows)
    st.download_button("Download scoreboard CSV", raw_df.to_csv(index=False).encode("utf-8"),
                       file_name="scoreboard.csv", mime="text/csv")
    st.download_button("Download logs JSON", json.dumps(state["logs"], indent=2).encode("utf-8"),
                       file_name="logs.json", mime="application/json")
    st.stop()

# ACTIVE ROUND
date_str, prices = base_prices_for_round(df, r, tickers)
repo_rate_today, td_rate_today = daily_rates_for_round(rng_seed, r)

# Settle maturities once per round (on render if not already)
if int(state["last_maturity_round"]) != r:
    for p in portfolios:
        process_maturities(p, r)
    state["last_maturity_round"] = r
    state["portfolios"] = portfolios_to_json(portfolios)
    write_state(state)  # persist maturity changes

# Ensure withdrawal created for this round
ensure_withdrawal_for_round(state, r, portfolios, prices)
req_all = float(state["withdrawals"][r])
# persist in case it was 0 before
write_state(state)

# ======== ROLE-BASED EDIT RIGHTS ========
def _player_can_edit_group(g_index: int) -> bool:
    if st.session_state.role == "Host":
        return False  # host is read-only for groups
    # player must have display name and be locked to this group
    claimed = state.get("claimed_groups", {})
    return claimed.get(str(g_index), None) == st.session_state.display_name and st.session_state.display_name != ""

# ======== APPLY/CLEAR HANDLERS (write-through persistence) ========
def handle_apply(g: int):
    # re-read latest, guard
    s = read_state()
    if s["current_round"] != r:  # host may have advanced
        st.warning("Round changed. Please re-open your action.")
        st.stop()
    portfolios_live = portfolios_from_json(s["portfolios"])

    if not _player_can_edit_group(g):
        st.error("You can only apply actions for your own locked group.")
        st.stop()

    p = portfolios_live[g]
    # current prices (for this round index)
    _, px = base_prices_for_round(df, r, tickers)

    # read inputs
    cash_amt   = float(st.session_state.get(f"cash_{r}_{g}", 0.0) or 0.0)
    repo_amt   = float(st.session_state.get(f"repo_{r}_{g}", 0.0) or 0.0)
    repo_tick  = st.session_state.get(f"repo_t_{r}_{g}", "(none)")
    redeem_amt = float(st.session_state.get(f"redeemtd_{r}_{g}", 0.0) or 0.0)
    invest_amt = float(st.session_state.get(f"investtd_{r}_{g}", 0.0) or 0.0)
    sell_qty   = float(st.session_state.get(f"sell_{r}_{g}", 0.0) or 0.0)
    sell_tick  = st.session_state.get(f"sell_t_{r}_{g}", "(none)")
    buy_qty    = float(st.session_state.get(f"buy_{r}_{g}", 0.0) or 0.0)
    buy_tick   = st.session_state.get(f"buy_t_{r}_{g}", "(none)")

    # remaining
    rem_left = compute_remaining_for_group(p.name, r, req_all, s["logs"])
    history = []

    # 1) Use cash
    if cash_amt > 0 and rem_left > 0:
        use = min(cash_amt, max(0.0, p.current_account), rem_left)
        if use > 0:
            p.current_account -= use
            rem_left -= use
            history.append(("cash", {"used": round(use, 2)}))

    # 2) Repo
    if repo_tick != "(none)" and repo_amt > 0 and rem_left > 0:
        price = px.get(repo_tick, 0.0)
        max_amt = p.pos_qty.get(repo_tick, 0.0) * price
        repo_amt = min(repo_amt, max_amt)
        if repo_amt > 0:
            info = _safe_repo_call(p, repo_tick, repo_amt, price, r, repo_rate_today)
            got = float(info["got"])
            use = min(got, rem_left)
            if use > 0:
                p.current_account -= use
                rem_left -= use
            history.append(("repo", {
                "ticker": repo_tick, "got": round(got, 2), "use": round(use, 2),
                "repo_id": info["repo_id"], "rate": repo_rate_today
            }))

    # 3) Redeem TD
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

    # 4) Sell
    if sell_tick != "(none)" and sell_qty > 0 and rem_left > 0:
        sell_qty = min(sell_qty, p.pos_qty.get(sell_tick, 0.0))
        if sell_qty > 0:
            sale = execute_sale(p, sell_tick, sell_qty, px[sell_tick])
            if isinstance(sale, dict):
                proceeds = float(sale.get("proceeds", sell_qty*px[sell_tick]))
                pnl_delta = float(sale.get("pnl_delta", 0.0))
                eff_px = float(sale.get("effective_price", px[sell_tick]))
            else:
                proceeds = sell_qty*px[sell_tick]
                pnl_delta = 0.0
                eff_px = px[sell_tick]
            use = min(proceeds, rem_left)
            if use > 0:
                p.current_account -= use
                rem_left -= use
            history.append(("sell", {
                "ticker": sell_tick,
                "qty": round(sell_qty, 2),
                "proceeds": round(proceeds, 2),
                "use": round(use, 2),
                "pnl_delta": round(pnl_delta, 2),
                "effective_price": round(eff_px, 6),
            }))

    # 5) Invest TD
    if invest_amt > 0:
        invest_amt = min(invest_amt, max(0.0, p.current_account))
        if invest_amt > 0:
            td_ids = execute_invest_td(p, invest_amt, r, rate=td_rate_today)
            if isinstance(td_ids, dict) and "ids" in td_ids:
                td_ids = list(td_ids["ids"])
            if not isinstance(td_ids, list):
                td_ids = []
            history.append(("invest_td", {
                "amount": round(invest_amt, 2),
                "td_ids": td_ids,
                "rate": td_rate_today
            }))

    # 6) Buy
    if buy_tick != "(none)" and buy_qty > 0:
        buy = execute_buy(p, buy_tick, buy_qty, px[buy_tick])
        if isinstance(buy, dict):
            eff_px = float(buy.get("effective_price", px[buy_tick]))
            cost = float(buy.get("cost", buy_qty*px[buy_tick]))
        else:
            eff_px = px[buy_tick]
            cost = buy_qty*px[buy_tick]
        history.append(("buy", {
            "ticker": buy_tick,
            "qty": round(buy_qty, 2),
            "cost": round(cost, 2),
            "effective_price": round(eff_px, 6),
        }))

    # write logs and portfolios back
    if history:
        s["logs"].setdefault(p.name, [])
        s["logs"][p.name].append({
            "round": r + 1,
            "request": req_all,
            "actions": history
        })
    s["portfolios"] = portfolios_to_json(portfolios_live)
    write_state(s)

    # reset inputs
    for key in [f"cash_{r}_{g}", f"repo_{r}_{g}", f"redeemtd_{r}_{g}", f"investtd_{r}_{g}", f"sell_{r}_{g}", f"buy_{r}_{g}"]:
        st.session_state[key] = 0.0
    for key in [f"repo_t_{r}_{g}", f"sell_t_{r}_{g}", f"buy_t_{r}_{g}"]:
        st.session_state[key] = "(none)"

def handle_clear(g: int):
    s = read_state()
    if s["current_round"] != r:
        st.warning("Round changed. Please re-open your action.")
        st.stop()
    portfolios_live = portfolios_from_json(s["portfolios"])
    if not _player_can_edit_group(g):
        st.error("You can only clear actions for your own locked group.")
        st.stop()
    p = portfolios_live[g]

    # reverse actions for this round
    logs_this_round = [L for L in s["logs"].get(p.name, []) if L["round"] == r + 1]
    for L in logs_this_round:
        for t, d in L["actions"]:
            if t == "cash":
                p.current_account += float(d.get("used", 0.0))
            elif t == "repo":
                got = float(d.get("got", 0.0))
                use = float(d.get("use", 0.0))
                p.current_account -= got
                p.current_account += use
                rid = d.get("repo_id")
                if rid is not None:
                    p.repo_liabilities = [l for l in p.repo_liabilities if l.get("id") != rid]
                else:
                    for i, l in enumerate(list(p.repo_liabilities)):
                        if abs(float(l.get("amount", 0.0)) - got) < 1e-6 and l.get("maturity", 10**9) > r:
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
                        "maturity": int(ch.get("maturity", r + TD_MAT_GAP)),
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

    # drop those logs
    s["logs"][p.name] = [L for L in s["logs"].get(p.name, []) if L["round"] != r + 1]
    s["portfolios"] = portfolios_to_json(portfolios_live)
    write_state(s)

    # reset inputs
    for key in [
        f"cash_{r}_{g}", f"repo_{r}_{g}", f"redeemtd_{r}_{g}",
        f"sell_{r}_{g}", f"investtd_{r}_{g}", f"buy_{r}_{g}"
    ]:
        st.session_state[key] = 0.0
    st.session_state[f"repo_t_{r}_{g}"] = "(none)"
    st.session_state[f"sell_t_{r}_{g}"] = "(none)"
    st.session_state[f"buy_t_{r}_{g}"]  = "(none)"

# =========================
# TOP DASH (everyone sees, read-only)
# =========================
st.subheader(f"Round {r+1} — Date: {date_str}")
st.caption(f"Today’s rates → Repo: {repo_rate_today*100:.2f}% • TD: {td_rate_today*100:.2f}% • Early TD penalty: 1.00%")

cols = st.columns(state["num_groups"])
for g, c in enumerate(cols):
    if g >= len(portfolios): break
    with c:
        p = portfolios[g]
        reserve = p.market_value(prices)
        # per-group rem based on logs
        rem = compute_remaining_for_group(p.name, r, req_all, state["logs"])
        st.markdown(f"### {p.name}")
        st.markdown(f"<div style='font-size:28px; font-weight:800; color:#006400;'>{fmt_money(reserve)}</div>", unsafe_allow_html=True)
        for t in tickers:
            st.markdown(
                f"<div class='ticker-line'>{t}: {p.pos_qty.get(t,0.0):,.0f} @ {prices[t]:,.2f}</div>",
                unsafe_allow_html=True
            )
        st.progress(max(0.0, 1 - (rem/req_all if req_all > 0 else 0)))

# =========================
# GROUP TABS
# =========================
tab_labels = [p.name for p in portfolios[:state["num_groups"]]]
tabs = st.tabs(tab_labels if tab_labels else ["Group 1"])

for g, tab in enumerate(tabs):
    if g >= len(portfolios): break
    with tab:
        p = portfolios[g]
        rem = compute_remaining_for_group(p.name, r, req_all, state["logs"])
        summary = p.summary(prices)

        editable = _player_can_edit_group(g)

        # Host note: read-only
        who = ""
        if role == "Player" and editable:
            who = " (You)"
        elif role == "Host":
            who = " (Host view)"

        st.markdown(f"### {p.name}{who}")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Current Account",        fmt_money(summary['current_account']))
            st.metric("Repo Outstanding",       fmt_money(summary['repo_outstanding']))
        with c2:
            st.metric("Securities Reserve",     fmt_money(summary['securities_mv']))
            st.metric("Term Deposit (Asset)",   fmt_money(summary['td_invested']))
        with c3:
            st.metric("PnL Realized",           fmt_money(summary['pnl_realized']))
            st.metric("Total Reserve",          fmt_money(summary['total_mv']))

        # Next-round projection (read-only)
        def _proj(portfolio: Portfolio, next_round: int) -> Dict[str, float]:
            repo_principal = sum(l["amount"] for l in portfolio.repo_liabilities if l["maturity"] == next_round)
            repo_interest  = sum(l["amount"] * l["rate"] for l in portfolio.repo_liabilities if l["maturity"] == next_round)
            td_principal   = sum(a["amount"] for a in portfolio.td_assets if a["maturity"] == next_round)
            td_interest    = sum(a["amount"] * a["rate"] for a in portfolio.td_assets if a["maturity"] == next_round)
            return {
                "repo_total": repo_principal + repo_interest,
                "repo_interest": repo_interest,
                "td_total": td_principal + td_interest,
                "td_interest": td_interest,
                "net_cash": (td_principal + td_interest) - (repo_principal + repo_interest),
            }
        proj = _proj(p, next_round=r+1)
        with st.expander(f"Next round obligations (Round {r+2})", expanded=False):
            st.markdown(f"**Repo repay next:** {fmt_money(proj['repo_total'])} (Interest {fmt_money(proj['repo_interest'])})")
            st.markdown(f"**TD receipts next:** {fmt_money(proj['td_total'])} (Interest {fmt_money(proj['td_interest'])})")
            st.markdown(f"**Net cash next:** {fmt_money(proj['net_cash'])}")

        st.markdown(f"**Withdrawal (all groups):** :blue[{fmt_money(req_all)}]")
        st.markdown(f"**Remaining for {p.name}:** :orange[{fmt_money(rem)}]")

        # ---- Inputs (players only for their group) ----
        st.number_input("Use cash", min_value=0.0, step=0.01,
                        value=st.session_state.get(f"cash_{r}_{g}", 0.0),
                        format="%.2f", key=f"cash_{r}_{g}", disabled=not editable)

        st.selectbox("Repo ticker", options=["(none)"] + tickers,
                     index=(["(none)"] + tickers).index(st.session_state.get(f"repo_t_{r}_{g}", "(none)")),
                     key=f"repo_t_{r}_{g}", disabled=not editable)
        st.number_input("Repo amount", min_value=0.0, step=0.01,
                        value=st.session_state.get(f"repo_{r}_{g}", 0.0),
                        format="%.2f", key=f"repo_{r}_{g}", disabled=not editable)
        st.caption(f"Today’s Repo rate: {repo_rate_today*100:.2f}%")
        if st.session_state.get(f"repo_t_{r}_{g}", "(none)") != "(none)" and st.session_state.get(f"repo_{r}_{g}", 0.0) > 0:
            rt = st.session_state[f"repo_t_{r}_{g}"]
            ra = float(st.session_state[f"repo_{r}_{g}"] or 0.0)
            if prices.get(rt, 0.0) > 0:
                eq = ra / prices[rt]
                st.caption(f"≈ {eq:,.2f} units of {rt}")

        st.number_input("Redeem Term Deposit", min_value=0.0, step=0.01,
                        value=st.session_state.get(f"redeemtd_{r}_{g}", 0.0),
                        format="%.2f", key=f"redeemtd_{r}_{g}", disabled=not editable)
        st.caption("Early redemption penalty: 1.00%")

        st.selectbox("Sell ticker", options=["(none)"] + tickers,
                     index=(["(none)"] + tickers).index(st.session_state.get(f"sell_t_{r}_{g}", "(none)")),
                     key=f"sell_t_{r}_{g}", disabled=not editable)
        st.number_input("Sell qty", min_value=0.0, step=0.01,
                        value=st.session_state.get(f"sell_{r}_{g}", 0.0),
                        format="%.2f", key=f"sell_{r}_{g}", disabled=not editable)

        st.number_input("Invest in Term Deposit", min_value=0.0, step=0.01,
                        value=st.session_state.get(f"investtd_{r}_{g}", 0.0),
                        format="%.2f", key=f"investtd_{r}_{g}", disabled=not editable)
        st.caption(f"Today’s TD rate (if held to maturity): {td_rate_today*100:.2f}%")

        st.selectbox("Buy ticker", options=["(none)"] + tickers,
                     index=(["(none)"] + tickers).index(st.session_state.get(f"buy_t_{r}_{g}", "(none)")),
                     key=f"buy_t_{r}_{g}", disabled=not editable)
        st.number_input("Buy qty", min_value=0.0, step=0.01,
                        value=st.session_state.get(f"buy_{r}_{g}", 0.0),
                        format="%.2f", key=f"buy_{r}_{g}", disabled=not editable)

        # previews (read-only info)
        try:
            sell_sel = st.session_state.get(f"sell_t_{r}_{g}", "(none)")
            sell_q   = float(st.session_state.get(f"sell_{r}_{g}", 0.0) or 0.0)
            if sell_sel != "(none)" and sell_q > 0:
                raw = prices[sell_sel]
                spec = p.securities[sell_sel]
                half_bps = (spec.bid_ask_bps / 2.0) + 5 * (spec.liquidity_score - 1)
                eff = raw * (1 - half_bps / 10_000.0)
                st.caption(f"Sell Preview: {sell_q:,.2f} × {eff:,.2f} = {fmt_money(sell_q*eff)}")
            buy_sel = st.session_state.get(f"buy_t_{r}_{g}", "(none)")
            buy_q   = float(st.session_state.get(f"buy_{r}_{g}", 0.0) or 0.0)
            if buy_sel != "(none)" and buy_q > 0:
                raw = prices[buy_sel]
                spec = p.securities[buy_sel]
                half_bps = (spec.bid_ask_bps / 2.0) + 5 * (spec.liquidity_score - 1)
                eff_buy = raw * (1 + half_bps / 10_000.0)
                st.caption(f"Buy Preview: {buy_q:,.2f} × {eff_buy:,.2f} = {fmt_money(buy_q*eff_buy)}")
        except Exception:
            pass

        # Buttons
        b1, b2 = st.columns([2,1])
        if role == "Player":
            b1.button("Apply action", key=f"apply_{r}_{g}", on_click=lambda gg=g: handle_apply(gg), disabled=not editable)
            b2.button("Clear Actions", key=f"clear_{r}_{g}", on_click=lambda gg=g: handle_clear(gg), disabled=not editable)
        else:
            # Host cannot edit groups
            b1.button("Apply action", key=f"apply_{r}_{g}", disabled=True)
            b2.button("Clear Actions", key=f"clear_{r}_{g}", disabled=True)

# =========================
# CONTROLS (HOST ONLY)
# =========================
st.divider()
lft, rgt = st.columns([3,1])
with lft:
    st.subheader("Controls")

# everyone computes coverage for display; only host can proceed
all_covered = True
for i in range(min(state["num_groups"], len(portfolios))):
    p = portfolios[i]
    rem_i = compute_remaining_for_group(p.name, r, req_all, state["logs"])
    if rem_i > 0.01:
        all_covered = False
        break

if role == "Host":
    with rgt:
        next_round_btn = st.button("Next Round ▶️")
    if next_round_btn:
        s = read_state()
        if not all_covered:
            st.error("Cover all groups (remaining ≤ $0.01) before moving on.")
        else:
            if s["current_round"] + 1 < s["rounds"]:
                s["current_round"] += 1
            else:
                s["current_round"] = s["rounds"]  # go to end
            write_state(s)
            st.rerun()
else:
    with rgt:
        st.caption("Only the Host can advance rounds.")
