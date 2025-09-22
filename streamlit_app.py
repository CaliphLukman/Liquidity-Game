import streamlit as st
import pandas as pd
import random
import json
from typing import List, Dict, Tuple, Any

from game_core import (
    Portfolio, init_portfolios, generate_withdrawal,
    execute_repo, execute_sale, execute_buy,
    execute_invest_td, execute_redeem_td,
    process_maturities
)

# --- global single-room store helpers ---
from global_store import (
    init_state_if_missing, get_state, update_state,
    save_csv_to_global, read_csv_df,
    set_serialized_portfolios, get_deserialized_portfolios,
    replace_logs, claim_group, release_group
)

# Optional gentle auto-refresh (if installed)
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=3000, key="live_sync")  # refresh every 3s
except Exception:
    pass

st.set_page_config(page_title="Liquidity Tranche Simulation", layout="wide")

# =========================
# CLEAN, HIGH-CONTRAST THEME + dark-green captions
# =========================
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
/* IMPORTANT: don't set .stCaption to ink here */
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

# ------------ Config ------------
BASE_REPO_RATE = 0.015      # 1.5%
BASE_TD_RATE   = 0.0175     # 1.75%
DAILY_SPREAD   = 0.005      # ±50 bps
TD_PENALTY     = 0.01       # 1.0% early redemption penalty
TD_MAT_GAP     = 2          # TDs mature after 2 rounds (handled in game_core)
MAX_GROUPS_UI  = 8          # Host can choose up to 8

# ------------------------
# Local session bootstrap (role & identity)
# ------------------------
def _ensure_session():
    ss = st.session_state
    if "role" not in ss:
        ss.role = "Host"   # default role for first visitor
    if "player_token" not in ss:
        # lightweight identity per browser session
        import uuid
        ss.player_token = uuid.uuid4().hex

_ensure_session()
init_state_if_missing()  # makes sure global_state.json exists

# ------------------------
# Helpers
# ------------------------
def price_row_for_round(r: int) -> int:
    df = st.session_state.price_df
    return min(r, len(df) - 1)

def base_prices_for_round(r: int, df: pd.DataFrame, tickers: List[str]) -> Tuple[str, Dict[str, float]]:
    ix = price_row_for_round(r)
    return str(df.loc[ix, "date"]), {t: float(df.loc[ix, t]) for t in tickers}

def daily_rates_for_round(r: int) -> Tuple[float, float]:
    rng = random.Random(st.session_state.get("rng_seed", 1234) * 991 + r * 7919)
    repo_delta = rng.uniform(-DAILY_SPREAD, DAILY_SPREAD)
    td_delta   = rng.uniform(-DAILY_SPREAD, DAILY_SPREAD)
    return max(0.0, BASE_REPO_RATE + repo_delta), max(0.0, BASE_TD_RATE + td_delta)

def ensure_round_initialized(r: int, prices_for_mv: Dict[str, float], portfolios: List[Portfolio], rng_seed: int, rounds: int, withdrawals: List[float]) -> List[float]:
    """Create the round withdrawal (same for all groups). Returns possibly-updated withdrawals list."""
    if r >= rounds:
        return withdrawals
    inited_rounds = {i for i, v in enumerate(withdrawals) if v != 0.0}
    if r in inited_rounds:
        return withdrawals
    if len(withdrawals) < rounds:
        withdrawals = [0.0 for _ in range(rounds)]
    req = generate_withdrawal(r, portfolios[0].market_value(prices_for_mv),
                              random.Random(rng_seed + 10007*r))
    withdrawals[r] = float(req)
    return withdrawals

def compute_remaining_for_group(group_name: str, r: int, req_for_round: float, logs: Dict[str, List[dict]]) -> float:
    """Remaining = req - sum of cash used + repo/sell/redeem 'use' in this round."""
    logs_r = [L for L in logs.get(group_name, []) if L["round"] == r+1]
    used = 0.0
    for L in logs_r:
        for t, d in L["actions"]:
            if t == "cash":
                used += float(d.get("used", 0.0))
            elif t in ("repo","sell","redeem_td"):
                used += float(d.get("use", 0.0))
    return max(0.0, round(req_for_round - used, 2))

# Safe adapters to tolerate minor signature differences in game_core
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
            return {
                "principal": float(res.get("principal", amount)),
                "penalty": float(res.get("penalty", amount * TD_PENALTY)),
                "redeemed": res.get("redeemed", []),
            }
    except TypeError:
        res = execute_redeem_td(portfolio, amount, rnow, penalty_rate=TD_PENALTY)  # type: ignore
        if isinstance(res, dict):
            return {
                "principal": float(res.get("principal", amount)),
                "penalty": float(res.get("penalty", amount * TD_PENALTY)),
                "redeemed": res.get("redeemed", []),
            }
    return {"principal": amount, "penalty": amount * TD_PENALTY, "redeemed": []}

def _safe_invest_td(portfolio: Portfolio, amount: float, rnow: int, rate: float):
    res = execute_invest_td(portfolio, amount, rnow, rate=rate)
    if isinstance(res, list): return res
    if isinstance(res, dict) and "ids" in res: return list(res["ids"])
    return []

def _safe_sale(portfolio: Portfolio, ticker: str, qty: float, px: float) -> Dict[str, Any]:
    res = execute_sale(portfolio, ticker, qty, px)
    if isinstance(res, dict):
        return {
            "ticker": res.get("ticker", ticker),
            "qty": float(res.get("qty", qty)),
            "proceeds": float(res.get("proceeds", qty * px)),
            "pnl_delta": float(res.get("pnl_delta", 0.0)),
            "effective_price": float(res.get("effective_price", px)),
        }
    return {"ticker": ticker, "qty": qty, "proceeds": qty*px, "pnl_delta": 0.0, "effective_price": px}

def _safe_buy(portfolio: Portfolio, ticker: str, qty: float, px: float) -> Dict[str, Any]:
    res = execute_buy(portfolio, ticker, qty, px)
    if isinstance(res, dict):
        return {
            "ticker": res.get("ticker", ticker),
            "qty": float(res.get("qty", qty)),
            "cost": float(res.get("cost", qty * px)),
            "effective_price": float(res.get("effective_price", px)),
        }
    return {"ticker": ticker, "qty": qty, "cost": qty*px, "effective_price": px}

# ------------------------
# Role selector (top of sidebar)
# ------------------------
st.sidebar.markdown("### Session")
role = st.sidebar.radio("Role", ["Host", "Player"], index=0 if st.session_state.role == "Host" else 1)
st.session_state.role = role
player_token = st.session_state.player_token

# ------------------------
# Load global state every run
# ------------------------
state = get_state()
# state keys: csv_path, rounds, current_round, rng_seed, last_maturity_round, tickers, withdrawals, portfolios(serialized), logs, num_groups, group_claims{}

# ------------------------
# Sidebar — conditional by role
# ------------------------
if role == "Host":
    st.sidebar.header("Host Setup")
    uploaded = st.sidebar.file_uploader("Bond price CSV (date + ≥3 securities)", type=["csv"])
    seed_in = st.sidebar.number_input("RNG seed", value=int(state.get("rng_seed", 1234)), step=1)
    rounds_in = st.sidebar.number_input("Rounds", value=int(state.get("rounds", 3)), min_value=1, max_value=10, step=1)
    groups_in = st.sidebar.number_input("Groups (up to 8)", value=int(state.get("num_groups", 4)), min_value=1, max_value=MAX_GROUPS_UI, step=1)

    c1, c2 = st.sidebar.columns(2)
    start_clicked = c1.button("Start/Reset", type="primary")
    end_clicked   = c2.button("End Game")
else:
    st.sidebar.header("Player Setup")
    uploaded = None
    start_clicked = False
    end_clicked = False

    # Show current config
    st.sidebar.caption(f"Rounds: {state.get('rounds', 3)} • Groups: {state.get('num_groups', 4)} • RNG seed: {state.get('rng_seed', 1234)}")

    # Group claiming UI
    portfolios = get_deserialized_portfolios()
    names = [p.name for p in portfolios] if portfolios else []
    claims = state.get("group_claims", {})
    if names:
        # Build display labels with claim info
        labels = []
        free_indices = []
        my_indices = []
        for i, n in enumerate(names):
            claimed_by = claims.get(str(i))
            if claimed_by is None:
                labels.append(f"{n} (free)")
                free_indices.append(i)
            elif claimed_by == player_token:
                labels.append(f"{n} (you)")
                my_indices.append(i)
            else:
                labels.append(f"{n} (taken)")
        # selection limited to free or mine
        selectable = [i for i in range(len(names)) if (claims.get(str(i)) in (None, player_token))]
        if not selectable:
            st.sidebar.info("All groups are taken. Ask the host to free one.")
            chosen_index = None
        else:
            default_idx = selectable[0]
            chosen_index = st.sidebar.selectbox("Choose your group", selectable, index=selectable.index(default_idx), format_func=lambda i: labels[i])
        c1, c2 = st.sidebar.columns(2)
        if chosen_index is not None:
            if c1.button("Claim / Switch"):
                ok, msg = claim_group(chosen_index, player_token)
                if not ok:
                    st.sidebar.error(msg)
                else:
                    st.experimental_rerun()
        # Allow release if you have any
        yours = [i for i in range(len(names)) if claims.get(str(i)) == player_token]
        if yours:
            idx_to_release = st.sidebar.selectbox("Release my group", yours, format_func=lambda i: names[i])
            if c2.button("Release"):
                release_group(idx_to_release, player_token)
                st.experimental_rerun()
    else:
        st.sidebar.info("Waiting for Host to start…")

with st.sidebar.expander("Game Instructions", expanded=False):
    st.markdown(f"""
- **Withdrawals:** One amount per round, **same for all groups**.
- **Term Deposits (TD):** Assets that mature after **{TD_MAT_GAP} rounds**. Early redemption penalty: **{TD_PENALTY*100:.2f}%**.
- **Rates:** Repo & TD vary daily by **±50 bps** around base and are stored per trade.
- **Initial TD allocation:** In **Round 1** only, each group randomly invests **10–30%** of Current Account into TD.
- **Apply Action:** Cash/Repo/Sell/Redeem TD immediately **deduct** the “used” portion from Current Account toward withdrawal.
- **Clear Actions:** Fully **reverts** this round’s actions for that group.
- **Next Round:** Settle maturities first, then create a new withdrawal (same for all groups).
- Only the **Host** can **Start/Reset**, **Next Round**, and **End Game**.
- Currency: **$**.
""")

# ------------------------
# Start / End (Host only)
# ------------------------
if role == "Host" and start_clicked:
    if uploaded is None:
        st.sidebar.error("Please upload a CSV with columns: date,BOND_A,BOND_B,BOND_C,...")
    else:
        path = save_csv_to_global(uploaded)
        df = read_csv_df()
        if df is None or "date" not in df.columns or len([c for c in df.columns if c != "date"]) < 3:
            st.sidebar.error("CSV must have 'date' and at least 3 security columns.")
        else:
            seed_val = int(seed_in)
            rounds_val = int(rounds_in)
            groups_val = int(groups_in)

            tickers_all = [c for c in df.columns if c != "date"]
            # Init portfolios from game_core (base four), cap to groups_val (<=8)
            date0, prices0 = str(df.loc[0, "date"]), {t: float(df.loc[0, t]) for t in tickers_all}
            base_portfolios = init_portfolios(tickers_all, prices0, total_reserve=200000.0)
            cap = min(MAX_GROUPS_UI, max(1, groups_val), len(base_portfolios))
            portfolios = base_portfolios[:cap]

            # Random initial TD allocation (Round 1 only, 10–30% of CA at base TD rate)
            for p in portfolios:
                frac = random.uniform(0.10, 0.30)
                amt = round(max(0.0, p.current_account) * frac, 2)
                if amt > 0:
                    execute_invest_td(p, amt, 0, rate=BASE_TD_RATE)

            # Reset logs and withdrawals
            logs = {p.name: [] for p in portfolios}
            withdrawals = [0.0 for _ in range(rounds_val)]

            # Write global state
            update_state(
                csv_path=path,
                tickers=tickers_all,
                rounds=rounds_val,
                current_round=0,
                rng_seed=seed_val,
                last_maturity_round=-1,
                withdrawals=withdrawals,
                logs=logs,
                num_groups=cap,
                group_claims={},   # all free
            )
            set_serialized_portfolios(portfolios)
            st.success("Game started. Share this same URL with players.")
            st.experimental_rerun()

if role == "Host" and end_clicked:
    # Move to end state (show scoreboard)
    update_state(current_round=get_state().get("rounds", 0))
    st.experimental_rerun()

# ------------------------
# Main UI
# ------------------------
st.title("Liquidity Tranche Simulation")

# Load df/tickers each run from global state
df = read_csv_df()
tickers_all = state.get("tickers", [])
if df is None or not tickers_all:
    if role == "Host":
        st.info("Upload a CSV and click **Start/Reset** to begin.")
    else:
        st.info("Waiting for the Host to start the session.")
    st.stop()

# Pull most recent state values
state = get_state()  # re-pull in case host just changed
r = int(state.get("current_round", 0))
rounds_total = int(state.get("rounds", 3))
rng_seed = int(state.get("rng_seed", 1234))
withdrawals = list(state.get("withdrawals", [0.0]*rounds_total))
logs_global: Dict[str, List[dict]] = dict(state.get("logs", {}))
portfolios = get_deserialized_portfolios()
NG = int(state.get("num_groups", len(portfolios) or 1))
tickers = tickers_all[:3]

# ==== END-GAME GUARD ====
if r >= rounds_total:
    st.header("Scoreboard & Logs")
    # final prices at last df row or last round
    final_ix = min(rounds_total-1, len(df)-1)
    _, final_px = str(df.loc[final_ix, "date"]), {t: float(df.loc[final_ix, t]) for t in tickers}
    rows = []
    for p in portfolios:
        s = p.summary(final_px)
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
    st.download_button("Download logs JSON", json.dumps(state.get("logs", {}), indent=2).encode("utf-8"),
                       file_name="logs.json", mime="application/json")
    st.stop()

# Active round content
date_str, prices = base_prices_for_round(r, df, tickers)
repo_rate_today, td_rate_today = daily_rates_for_round(r)

# Settle maturities once per round (repos/TDs maturing this round)
if int(state.get("last_maturity_round", -1)) != r:
    for p in portfolios:
        process_maturities(p, r)
    # save portfolios & bump marker
    set_serialized_portfolios(portfolios)
    update_state(last_maturity_round=r)

# Create same-withdrawal-for-all after maturities
withdrawals = ensure_round_initialized(r, prices, portfolios, rng_seed, rounds_total, withdrawals)
if withdrawals != state.get("withdrawals", []):
    update_state(withdrawals=withdrawals)

req_all = float(withdrawals[r])
claims = state.get("group_claims", {})

# ------------------------
# Apply / Clear handlers (permissioned)
# ------------------------
def _player_can_edit_group(index: int) -> bool:
    if role == "Host":
        return True
    return claims.get(str(index)) == player_token

# queued actions via widget callbacks
if "pending_apply" in st.session_state and st.session_state.pending_apply is not None:
    g_apply = int(st.session_state.pending_apply["g"])
    r_apply = int(st.session_state.pending_apply["r"])
    st.session_state.pending_apply = None

    if 0 <= g_apply < len(portfolios) and _player_can_edit_group(g_apply):
        p = portfolios[g_apply]
        rem_left = compute_remaining_for_group(p.name, r_apply, req_all, logs_global)

        # widget keys
        cash_key      = f"cash_{r_apply}_{g_apply}"
        repo_amt_key  = f"repo_{r_apply}_{g_apply}"
        repo_tick_key = f"repo_t_{r_apply}_{g_apply}"
        redeem_key    = f"redeemtd_{r_apply}_{g_apply}"
        invest_key    = f"investtd_{r_apply}_{g_apply}"
        sell_qty_key  = f"sell_{r_apply}_{g_apply}"
        sell_tick_key = f"sell_t_{r_apply}_{g_apply}"
        buy_qty_key   = f"buy_{r_apply}_{g_apply}"
        buy_tick_key  = f"buy_t_{r_apply}_{g_apply}"

        # read inputs
        cash_amt   = float(st.session_state.get(cash_key, 0.0) or 0.0)
        repo_amt   = float(st.session_state.get(repo_amt_key, 0.0) or 0.0)
        repo_tick  = st.session_state.get(repo_tick_key, "(none)")
        redeem_amt = float(st.session_state.get(redeem_key, 0.0) or 0.0)
        invest_amt = float(st.session_state.get(invest_key, 0.0) or 0.0)
        sell_qty   = float(st.session_state.get(sell_qty_key, 0.0) or 0.0)
        sell_tick  = st.session_state.get(sell_tick_key, "(none)")
        buy_qty    = float(st.session_state.get(buy_qty_key, 0.0) or 0.0)
        buy_tick   = st.session_state.get(buy_tick_key, "(none)")

        # current prices for displayed tickers
        px = {t: float(df.loc[price_row_for_round(r_apply), t]) for t in tickers}

        history = []

        # 1) Use cash (immediate payout)
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
                info = _safe_repo_call(p, repo_tick, repo_amt, price, r_apply, repo_rate_today)
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
            red = _safe_redeem_td(p, redeem_amt, r_apply)
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
                sale = _safe_sale(p, sell_tick, sell_qty, px[sell_tick])
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

        # 5) Invest TD
        if invest_amt > 0:
            invest_amt = min(invest_amt, max(0.0, p.current_account))
            if invest_amt > 0:
                td_ids = _safe_invest_td(p, invest_amt, r_apply, td_rate_today)
                history.append(("invest_td", {
                    "amount": round(invest_amt, 2),
                    "td_ids": td_ids,
                    "rate": td_rate_today
                }))

        # 6) Buy
        if buy_tick != "(none)" and buy_qty > 0:
            buy = _safe_buy(p, buy_tick, buy_qty, px[buy_tick])
            history.append(("buy", {
                "ticker": buy["ticker"],
                "qty": round(buy["qty"], 2),
                "cost": round(buy["cost"], 2),
                "effective_price": round(buy["effective_price"], 6),
            }))

        if history:
            logs_global.setdefault(p.name, []).append({
                "round": r_apply + 1,
                "request": req_all,
                "actions": history
            })

        # Save portfolios/logs back to global
        set_serialized_portfolios(portfolios)
        replace_logs(logs_global)

        # Reset inputs BEFORE widgets render
        for key in [cash_key, repo_amt_key, redeem_key, invest_key, sell_qty_key, buy_qty_key]:
            st.session_state[key] = 0.0
        st.session_state[repo_tick_key] = "(none)"
        st.session_state[sell_tick_key] = "(none)"
        st.session_state[buy_tick_key]  = "(none)"

        st.experimental_rerun()

if "pending_clear" in st.session_state and st.session_state.pending_clear is not None:
    g_clear = int(st.session_state.pending_clear["g"])
    r_clear = int(st.session_state.pending_clear["r"])
    st.session_state.pending_clear = None

    if 0 <= g_clear < len(portfolios) and _player_can_edit_group(g_clear):
        p = portfolios[g_clear]
        logs_this_round = [L for L in logs_global.get(p.name, []) if L["round"] == r_clear + 1]

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
                            if abs(float(l.get("amount", 0.0)) - got) < 1e-6 and l.get("maturity", 10**9) > r_clear:
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
                            "maturity": int(ch.get("maturity", r_clear + TD_MAT_GAP)),
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

        # Drop those logs
        logs_global[p.name] = [L for L in logs_global.get(p.name, []) if L["round"] != r_clear + 1]

        # Save
        set_serialized_portfolios(portfolios)
        replace_logs(logs_global)
        st.experimental_rerun()

# ------------------------
# Round dashboard (everyone can see)
# ------------------------
st.subheader(f"Round {r+1} — Date: {date_str}")
st.caption(f"Today’s rates → Repo: {repo_rate_today*100:.2f}%  •  TD: {td_rate_today*100:.2f}%  •  Early TD penalty: 1.00%")

cols = st.columns(NG)
for g, c in enumerate(cols):
    if g >= len(portfolios): break
    with c:
        p = portfolios[g]
        rem = compute_remaining_for_group(p.name, r, req_all, logs_global)
        reserve = p.market_value(prices)

        st.markdown(f"### {p.name}")
        st.markdown(
            f"<div style='font-size:28px; font-weight:800; color:#006400;'>${reserve:,.0f}</div>",
            unsafe_allow_html=True
        )
        for t in tickers:
            st.markdown(
                f"<div class='ticker-line'>{t}: {p.pos_qty.get(t,0.0):,.0f} @ {prices[t]:,.2f}</div>",
                unsafe_allow_html=True
            )
        progress = max(0.0, 1 - rem/req_all) if req_all > 0 else 1.0
        st.progress(progress)

# ------------------------
# Group tabs (inputs disabled unless Host or owning Player)
# ------------------------
tab_labels = [p.name for p in portfolios[:NG]]
tabs = st.tabs(tab_labels if tab_labels else ["Group 1"])
for g, tab in enumerate(tabs):
    if g >= len(portfolios): break
    with tab:
        p = portfolios[g]
        rem = compute_remaining_for_group(p.name, r, req_all, logs_global)

        editable = _player_can_edit_group(g)
        owner = " (You)" if (role == "Player" and editable) else (" (Host)" if role == "Host" else "")
        st.markdown(f"### {p.name}{owner}")
        summary = p.summary(prices)
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

        # Next-round projection
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
            st.markdown(f"**Repo repay next:** {proj['repo_total']:,.2f} (Interest {proj['repo_interest']:,.2f})")
            st.markdown(f"**TD receipts next:** {proj['td_total']:,.2f} (Interest {proj['td_interest']:,.2f})")
            st.markdown(f"**Net cash next:** {proj['net_cash']:,.2f}")

        st.markdown(f"**Withdrawal (all groups):** :blue[{req_all:,.2f}]")
        st.markdown(f"**Remaining for {p.name}:** :orange[{rem:,.2f}]")

        # ---- Inputs (persisted via keys) ----
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
            ra = float(st.session_state[f"repo_{r}_{g}"])
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

        # Optional previews
        try:
            sell_sel = st.session_state.get(f"sell_t_{r}_{g}", "(none)")
            sell_q   = float(st.session_state.get(f"sell_{r}_{g}", 0.0) or 0.0)
            if sell_sel != "(none)" and sell_q > 0:
                raw = prices[sell_sel]
                spec = p.securities[sell_sel]
                half_bps = (spec.bid_ask_bps / 2.0) + 5 * (spec.liquidity_score - 1)
                eff = raw * (1 - half_bps / 10_000.0)
                st.caption(f"Sell Preview: {sell_q:,.2f} × {eff:,.2f} = {sell_q*eff:,.2f}")
            buy_sel = st.session_state.get(f"buy_t_{r}_{g}", "(none)")
            buy_q   = float(st.session_state.get(f"buy_{r}_{g}", 0.0) or 0.0)
            if buy_sel != "(none)" and buy_q > 0:
                raw = prices[buy_sel]
                spec = p.securities[buy_sel]
                half_bps = (spec.bid_ask_bps / 2.0) + 5 * (spec.liquidity_score - 1)
                eff_buy = raw * (1 + half_bps / 10_000.0)
                st.caption(f"Buy Preview: {buy_q:,.2f} × {eff_buy:,.2f} = {buy_q*eff_buy:,.2f}")
        except Exception:
            pass

        cA, cB = st.columns([2,1])
        cA.button("Apply action", key=f"apply_{r}_{g}",
                  on_click=lambda gg=g: st.session_state.update(pending_apply={"g": gg, "r": r}),
                  disabled=not editable)
        cB.button("Clear Actions", key=f"clear_{r}_{g}",
                  on_click=lambda gg=g: st.session_state.update(pending_clear={"g": gg, "r": r}),
                  disabled=not editable)

# ------------------------
# Next Round controls (Host only)
# ------------------------
st.divider()
lft, rgt = st.columns([3,1])
with lft:
    st.subheader("Controls")

all_covered = all(
    compute_remaining_for_group(portfolios[i].name, r, req_all, logs_global) <= 0.01
    for i in range(min(NG, len(portfolios)))
)

if role == "Host":
    with rgt:
        if st.button("Next Round ▶️", key=f"next_round_{r}"):
            if not all_covered:
                st.error("Cover all groups (remaining ≤ $0.01) before moving on.")
            else:
                if r + 1 < rounds_total:
                    update_state(current_round=r+1, last_maturity_round=r)  # maturity will settle at next run
                else:
                    update_state(current_round=rounds_total)
                st.experimental_rerun()
else:
    with rgt:
        st.caption("Only the Host can advance rounds.")
