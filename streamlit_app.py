# streamlit_app.py
import os
import json
import time
import uuid
import random
from typing import List, Dict, Tuple, Any

import pandas as pd
import streamlit as st

# Optional auto-refresh (view only). If not installed, app still runs.
try:
    from streamlit_autorefresh import st_autorefresh
    _AUTOREFRESH_AVAILABLE = True
except Exception:
    _AUTOREFRESH_AVAILABLE = False

from game_core import (
    Portfolio, init_portfolios, generate_withdrawal,
    execute_repo, execute_sale, execute_buy,
    execute_invest_td, execute_redeem_td,
    process_maturities
)

st.set_page_config(page_title="Liquidity Tranche Simulation", layout="wide")

# =========================
# THEME: clean, high-contrast + dark-green captions
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

# ------------ Simple file-based claim ledger (no external DB) ------------
CLAIMS_FILE = "claims.json"
CLAIMS_LOCK = "claims.lock"

def _acquire_lock(timeout: float = 2.5) -> bool:
    """Create an exclusive lock file. Returns True if acquired, False on timeout."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            # O_CREAT|O_EXCL ensures it fails if file exists (atomic)
            fd = os.open(CLAIMS_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except FileExistsError:
            time.sleep(0.05)
    return False

def _release_lock():
    try:
        os.remove(CLAIMS_LOCK)
    except FileNotFoundError:
        pass

def _read_claims() -> Dict[str, Any]:
    if not os.path.exists(CLAIMS_FILE):
        return {"groups": {}}  # {group_name: {"token": "...", "name": "...", "ts": float}}
    try:
        with open(CLAIMS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"groups": {}}

def _write_claims(claims: Dict[str, Any]):
    tmp = f"{CLAIMS_FILE}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(claims, f, indent=2)
    os.replace(tmp, CLAIMS_FILE)

def reset_claims(group_names: List[str]):
    """Host call on Start/Reset: clears claims to empty for fresh session."""
    if _acquire_lock():
        try:
            _write_claims({"groups": {g: None for g in group_names}})
        finally:
            _release_lock()

def claim_group(group_name: str, token: str, player_name: str = "") -> bool:
    """Attempt to claim a group. Returns True on success, False if already claimed by someone else."""
    if _acquire_lock():
        try:
            data = _read_claims()
            groups = data.get("groups", {})
            # Initialize missing group entry if needed
            if group_name not in groups:
                groups[group_name] = None
            current = groups.get(group_name)
            # Already claimed by someone else
            if current and current.get("token") != token:
                return False
            # Claim (or idempotent re-claim by same token)
            groups[group_name] = {"token": token, "name": player_name or "", "ts": time.time()}
            data["groups"] = groups
            _write_claims(data)
            return True
        finally:
            _release_lock()
    return False

def is_claimed(group_name: str) -> bool:
    data = _read_claims()
    entry = data.get("groups", {}).get(group_name)
    return bool(entry)

def claimed_by_me(group_name: str, token: str) -> bool:
    data = _read_claims()
    entry = data.get("groups", {}).get(group_name)
    return bool(entry and entry.get("token") == token)

def claimant_badge(group_name: str) -> str:
    """Returns a short string showing claim status."""
    entry = _read_claims().get("groups", {}).get(group_name)
    if not entry:
        return "Unclaimed"
    # Do not reveal name if you prefer privacy; show 'Claimed' only.
    nm = (entry.get("name") or "").strip()
    return f"Claimed{f' by {nm}' if nm else ''}"

# ------------------------
# Session State bootstrap
# ------------------------
def _init_state():
    ss = st.session_state
    ss.initialized = False
    ss.rng_seed = 1234
    ss.rounds = 3
    ss.current_round = 0
    # Same withdrawal for ALL groups each round
    ss.withdrawals: List[float] = []
    ss.portfolios: List[Portfolio] = []
    ss.logs: Dict[str, List[dict]] = {}
    ss.price_df = None
    # per-round processing guards
    ss.last_maturity_round = -1
    ss.inited_rounds = set()
    # queued actions (processed BEFORE widgets render)
    ss.pending_apply: Dict[str,int] | None = None
    ss.pending_clear: Dict[str,int] | None = None
    # dynamic groups
    ss.num_groups = 4  # default
    # role & player selection
    ss.role = "Host"
    ss.player_group_index = 0
    # host auto-refresh toggle (view-only)
    ss.host_autorefresh = True
    # player auto-refresh (off by default)
    ss.player_autorefresh = False
    # a unique session token to own a claim
    ss.client_token = uuid.uuid4().hex

if "initialized" not in st.session_state:
    _init_state()

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
    # Deterministic per round for a given rng_seed; pure function -> safe with reruns
    rng = random.Random(st.session_state.rng_seed * 991 + r * 7919)
    repo_delta = rng.uniform(-DAILY_SPREAD, DAILY_SPREAD)
    td_delta   = rng.uniform(-DAILY_SPREAD, DAILY_SPREAD)
    return max(0.0, BASE_REPO_RATE + repo_delta), max(0.0, BASE_TD_RATE + td_delta)

def ensure_round_initialized(r: int, prices_for_mv: Dict[str, float]):
    """Create the round withdrawal (same for all groups). Runs once per round."""
    ss = st.session_state
    if r >= ss.rounds or r in ss.inited_rounds:
        return
    if len(ss.withdrawals) < ss.rounds:
        ss.withdrawals = [0.0 for _ in range(ss.rounds)]
    # Deterministic withdrawal using rng_seed; does NOT change on reruns
    req = generate_withdrawal(r, ss.portfolios[0].market_value(prices_for_mv),
                              random.Random(ss.rng_seed + 10007*r))
    ss.withdrawals[r] = float(req)
    ss.inited_rounds.add(r)

def compute_remaining_for_group(group_name: str, r: int, req_for_round: float) -> float:
    """Remaining = req - sum of cash used + repo/sell/redeem 'use' in this round."""
    logs = [L for L in st.session_state.logs.get(group_name, []) if L["round"] == r+1]
    used = 0.0
    for L in logs:
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

# ------------------------
# Sidebar — conditional by role
# ------------------------
if role == "Host":
    st.sidebar.header("Host Setup")
    uploaded = st.sidebar.file_uploader("Bond price CSV (date + ≥3 securities)", type=["csv"])
    seed = st.sidebar.number_input("RNG seed", value=st.session_state.rng_seed, step=1)
    rounds = st.sidebar.number_input("Rounds", value=st.session_state.rounds, min_value=1, max_value=10, step=1)
    groups = st.sidebar.number_input("Groups (up to 8)", value=st.session_state.num_groups, min_value=1, max_value=MAX_GROUPS_UI, step=1)

    # Host-only: auto-refresh toggle (view-only; does NOT alter state)
    if _AUTOREFRESH_AVAILABLE:
        st.session_state.host_autorefresh = st.sidebar.toggle("Auto-refresh host view (3s)", value=st.session_state.host_autorefresh)
        if st.session_state.host_autorefresh:
            st_autorefresh(interval=3000, limit=None, key="host_autorefresh_tick")

    c1, c2 = st.sidebar.columns(2)
    start_clicked = c1.button("Start/Reset", type="primary")
    end_clicked   = c2.button("End Game")
else:
    st.sidebar.header("Player Setup")
    uploaded = None

    # Optional: nickname (not required)
    player_name = st.sidebar.text_input("Your nickname (optional)", value="")

    # Player auto-refresh (off by default)
    if _AUTOREFRESH_AVAILABLE:
        st.session_state.player_autorefresh = st.sidebar.toggle("Auto-refresh player view (5s)", value=st.session_state.player_autorefresh)
        if st.session_state.player_autorefresh:
            st_autorefresh(interval=5000, limit=None, key=f"player_autorefresh_tick")

    if st.session_state.initialized and st.session_state.portfolios:
        available_groups = [p.name for p in st.session_state.portfolios]
        default_idx = min(st.session_state.player_group_index, max(0, len(available_groups)-1))
        player_group_name = st.sidebar.selectbox("Select your Group", available_groups, index=default_idx)
        st.session_state.player_group_index = available_groups.index(player_group_name)

        # Claim UI
        gname = player_group_name
        claimed = is_claimed(gname)
        mine = claimed_by_me(gname, st.session_state.client_token)
        badge = claimant_badge(gname)
        st.sidebar.markdown(f"**Group status:** {badge}")

        if not claimed:
            if st.sidebar.button("Claim this group"):
                ok = claim_group(gname, st.session_state.client_token, player_name)
                if not ok:
                    st.sidebar.error("Someone just claimed this group. Pick another.")
                else:
                    st.sidebar.success(f"You claimed {gname}.")
        elif not mine:
            st.sidebar.warning("This group is already claimed by another player. Choose a different group.")
        else:
            st.sidebar.success(f"You own {gname}.")
    else:
        st.sidebar.info("Waiting for Host to start and upload a CSV…")

    start_clicked = False
    end_clicked = False
    st.sidebar.caption(f"Rounds: {st.session_state.rounds} • Groups: {st.session_state.num_groups} • RNG seed: {st.session_state.rng_seed}")

with st.sidebar.expander("Game Instructions", expanded=False):
    st.markdown(f"""
- **Withdrawals:** One amount per round, **same for all groups**.
- **Term Deposits (TD):** Assets that mature after **{TD_MAT_GAP} rounds**. Early redemption penalty: **{TD_PENALTY*100:.2f}%**.
- **Rates:** Repo & TD vary daily by **±50 bps** around base and are stored per trade.
- **Initial TD allocation:** In **Round 1** only, each group randomly invests **10–30%** of Current Account into TD.
- **Apply Action:** Cash/Repo/Sell/Redeem TD immediately **deduct** the “used” portion from Current Account toward withdrawal.
- **Clear Actions:** Fully **reverts** this round’s actions for that group.
- **Next Round:** Settle maturities first, then create a new withdrawal (same for all groups).
- Players can only **Apply/Clear** for their own claimed group; Host can edit all.
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
        df = pd.read_csv(uploaded)
        if "date" not in df.columns or len([c for c in df.columns if c != "date"]) < 3:
            st.sidebar.error("CSV must have 'date' and at least 3 security columns.")
        else:
            seed_val = int(seed); rounds_val = int(rounds)
            desired_groups = int(groups)

            _init_state()  # resets state on purpose only when Host starts
            st.session_state.initialized   = True
            st.session_state.rng_seed      = seed_val
            st.session_state.rounds        = rounds_val
            st.session_state.price_df      = df.copy()

            tickers_all = [c for c in df.columns if c != "date"]
            date0, prices0 = base_prices_for_round(0, df, tickers_all)

            # Get base portfolios from game_core, then cap/slice to Host’s choice
            base_portfolios = init_portfolios(tickers_all, prices0, total_reserve=200000.0)
            available = len(base_portfolios)
            cap = min(MAX_GROUPS_UI, available, desired_groups)
            if desired_groups > cap:
                st.sidebar.warning(f"Requested {desired_groups} groups, but only {cap} available. Capped automatically.")
            st.session_state.portfolios = base_portfolios[:cap]
            st.session_state.num_groups = cap

            # Reset claims ledger for these groups (fresh game)
            reset_claims([p.name for p in st.session_state.portfolios])

            # Random initial TD allocation (Round 1 only, 10–30% of CA at base TD rate)
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

if role == "Host" and end_clicked and st.session_state.initialized:
    st.session_state.current_round = st.session_state.rounds
    st.rerun()

# ------------------------
# Main UI
# ------------------------
st.title("Liquidity Tranche Simulation")

if not st.session_state.initialized:
    if role == "Host":
        st.info("Upload a CSV and click **Start/Reset** to begin.")
    else:
        st.info("Waiting for the Host to start the session.")
    st.stop()

df = st.session_state.price_df
all_tickers = [c for c in df.columns if c != "date"]
tickers = all_tickers[:3]  # show/use first 3
r = st.session_state.current_round
NG = max(1, int(st.session_state.get("num_groups", len(st.session_state.portfolios) or 1)))

# ==== END-GAME GUARD (before any withdrawals[r] access) ====
if r >= st.session_state.rounds:
    st.header("Scoreboard & Logs")
    _, final_px = base_prices_for_round(min(st.session_state.rounds-1, len(df)-1), df, tickers)
    rows = []
    for p in st.session_state.portfolios:
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
    st.dataframe(
        sb.assign(
            current_account=sb["current_account"].map(lambda v: f"${v:,.2f}"),
            securities_reserve=sb["securities_reserve"].map(lambda v: f"${v:,.2f}"),
            repo_outstanding=sb["repo_outstanding"].map(lambda v: f"${v:,.2f}"),
            term_deposit=sb["term_deposit"].map(lambda v: f"${v:,.2f}"),
            pnl_realized=sb["pnl_realized"].map(lambda v: f"${v:,.2f}"),
            total_reserve=sb["total_reserve"].map(lambda v: f"${v:,.2f}"),
        ),
        use_container_width=True
    )
    st.download_button("Download scoreboard CSV", sb.to_csv(index=False).encode("utf-8"),
                       file_name="scoreboard.csv", mime="text/csv")
    st.download_button("Download logs JSON", json.dumps(st.session_state.logs, indent=2).encode("utf-8"),
                       file_name="logs.json", mime="application/json")
    st.stop()

# Active round content
date_str, prices = base_prices_for_round(r, df, tickers)
repo_rate_today, td_rate_today = daily_rates_for_round(r)

# Settle maturities once per round (repos/TDs maturing this round)
if st.session_state.last_maturity_round != r:
    for p in st.session_state.portfolios:
        process_maturities(p, r)
    st.session_state.last_maturity_round = r

# Create same-withdrawal-for-all after maturities
ensure_round_initialized(r, prices)
req_all = float(st.session_state.withdrawals[r])

# ------------------------
# Handle queued Apply/Clear BEFORE widgets render
# ------------------------
def _player_can_edit_group(g_index: int) -> bool:
    if st.session_state.role == "Host":
        return True
    # Player can edit only if they own the claimed group
    if 0 <= g_index < len(st.session_state.portfolios):
        gname = st.session_state.portfolios[g_index].name
        return claimed_by_me(gname, st.session_state.client_token)
    return False

if st.session_state.pending_apply is not None:
    g_apply = int(st.session_state.pending_apply["g"])
    r_apply = int(st.session_state.pending_apply["r"])

    if 0 <= g_apply < len(st.session_state.portfolios) and _player_can_edit_group(g_apply):
        p = st.session_state.portfolios[g_apply]
        rem_left = compute_remaining_for_group(p.name, r_apply, req_all)

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
        px = {t: float(st.session_state.price_df.loc[price_row_for_round(r_apply), t]) for t in tickers}

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
            st.session_state.logs[p.name].append({
                "round": r_apply + 1,
                "request": req_all,
                "actions": history
            })

        # Reset inputs BEFORE widgets render
        for key in [cash_key, repo_amt_key, redeem_key, invest_key, sell_qty_key, buy_qty_key]:
            st.session_state[key] = 0.0
        st.session_state[repo_tick_key] = "(none)"
        st.session_state[sell_tick_key] = "(none)"
        st.session_state[buy_tick_key]  = "(none)"

    st.session_state.pending_apply = None

if st.session_state.pending_clear is not None:
    g_clear = int(st.session_state.pending_clear["g"])
    r_clear = int(st.session_state.pending_clear["r"])

    if 0 <= g_clear < len(st.session_state.portfolios) and _player_can_edit_group(g_clear):
        p = st.session_state.portfolios[g_clear]
        logs_this_round = [L for L in st.session_state.logs.get(p.name, []) if L["round"] == r_clear + 1]

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
                    amt = float(d.get("amount, 0.0".replace(",", ":").split(":")[1]) if "amount, 0.0" in d else d.get("amount", 0.0))  # defensive
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

        st.session_state.logs[p.name] = [
            L for L in st.session_state.logs.get(p.name, []) if L["round"] != r_clear + 1
        ]

        for key in [
            f"cash_{r_clear}_{g_clear}", f"repo_{r_clear}_{g_clear}",
            f"redeemtd_{r_clear}_{g_clear}", f"sell_{r_clear}_{g_clear}",
            f"investtd_{r_clear}_{g_clear}", f"buy_{r_clear}_{g_clear}"
        ]:
            st.session_state[key] = 0.0
        st.session_state[f"repo_t_{r_clear}_{g_clear}"] = "(none)"
        st.session_state[f"sell_t_{r_clear}_{g_clear}"] = "(none)"
        st.session_state[f"buy_t_{r_clear}_{g_clear}"]  = "(none)"

    st.session_state.pending_clear = None

# ------------------------
# Round dashboard
# ------------------------
st.subheader(f"Round {r+1} — Date: {date_str}")
st.caption(f"Today’s rates → Repo: {repo_rate_today*100:.2f}%  •  TD: {td_rate_today*100:.2f}%  •  Early TD penalty: 1.00%")

cols = st.columns(NG)
for g, c in enumerate(cols):
    if g >= len(st.session_state.portfolios): break
    with c:
        p = st.session_state.portfolios[g]
        rem = compute_remaining_for_group(p.name, r, req_all)
        reserve = p.market_value(prices)

        claim_status = claimant_badge(p.name)

        st.markdown(f"### {p.name}")
        st.caption(f"Status: {claim_status}")
        st.markdown(
            f"<div style='font-size:28px; font-weight:800; color:#006400;'>${reserve:,.0f}</div>",
            unsafe_allow_html=True
        )
        for t in tickers:
            st.markdown(
                f"<div class='ticker-line'>{t}: {p.pos_qty.get(t,0.0):,.0f} @ ${prices[t]:,.2f}</div>",
                unsafe_allow_html=True
            )
        st.progress(max(0.0, 1 - rem/req_all if req_all > 0 else 1))

# ------------------------
# Group tabs (inputs disabled unless Host or owning Player)
# ------------------------
tab_labels = [p.name for p in st.session_state.portfolios[:NG]]
tabs = st.tabs(tab_labels if tab_labels else ["Group 1"])
for g, tab in enumerate(tabs):
    if g >= len(st.session_state.portfolios): break
    with tab:
        p = st.session_state.portfolios[g]
        rem = compute_remaining_for_group(p.name, r, req_all)

        editable = _player_can_edit_group(g)

        st.markdown(f"### {p.name} {'(You)' if (role=='Player' and editable) else ''}")
        summary = p.summary(prices)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Current Account",        f"${summary['current_account']:,.2f}")
            st.metric("Repo Outstanding",       f"${summary['repo_outstanding']:,.2f}")
        with c2:
            st.metric("Securities Reserve",     f"${summary['securities_mv']:,.2f}")
            st.metric("Term Deposit (Asset)",   f"${summary['td_invested']:,.2f}")
        with c3:
            st.metric("PnL Realized",           f"${summary['pnl_realized']:,.2f}")
            st.metric("Total Reserve",          f"${summary['total_mv']:,.2f}")

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
            st.markdown(f"**Repo repay next:** ${proj['repo_total']:,.2f} (Interest ${proj['repo_interest']:,.2f})")
            st.markdown(f"**TD receipts next:** ${proj['td_total']:,.2f} (Interest ${proj['td_interest']:,.2f})")
            st.markdown(f"**Net cash next:** ${proj['net_cash']:,.2f}")

        st.markdown(f"**Withdrawal (all groups):** :blue[${req_all:,.2f}]")
        st.markdown(f"**Remaining for {p.name}:** :orange[${rem:,.2f}]")

        # ---- Inputs (persisted via keys) ----
        st.number_input("Use cash", min_value=0.0, step=0.01,
                        value=st.session_state.get(f"cash_{r}_{g}", 0.0),
                        format="%.2f", key=f"cash_{r}_{g}", disabled=not editable)

        st.selectbox("Repo ticker", options=["(none)"] + tickers,
                     index=(["(none)"] + tickers).index(st.session_state.get(f"repo_t_{r}_{g}", "(none)")),
                     key=f"repo_t_{r}_{g}", disabled=not editable)
        st.number_input("Repo amount ($)", min_value=0.0, step=0.01,
                        value=st.session_state.get(f"repo_{r}_{g}", 0.0),
                        format="%.2f", key=f"repo_{r}_{g}", disabled=not editable)
        st.caption(f"Today’s Repo rate: {repo_rate_today*100:.2f}%")
        if st.session_state.get(f"repo_t_{r}_{g}", "(none)") != "(none)" and st.session_state.get(f"repo_{r}_{g}", 0.0) > 0:
            rt = st.session_state[f"repo_t_{r}_{g}"]
            ra = float(st.session_state[f"repo_{r}_{g}"])
            if prices.get(rt, 0.0) > 0:
                eq = ra / prices[rt]
                st.caption(f"≈ {eq:,.2f} units of {rt}")

        st.number_input("Redeem Term Deposit ($)", min_value=0.0, step=0.01,
                        value=st.session_state.get(f"redeemtd_{r}_{g}", 0.0),
                        format="%.2f", key=f"redeemtd_{r}_{g}", disabled=not editable)
        st.caption("Early redemption penalty: 1.00%")

        st.selectbox("Sell ticker", options=["(none)"] + tickers,
                     index=(["(none)"] + tickers).index(st.session_state.get(f"sell_t_{r}_{g}", "(none)")),
                     key=f"sell_t_{r}_{g}", disabled=not editable)
        st.number_input("Sell qty (units)", min_value=0.0, step=0.01,
                        value=st.session_state.get(f"sell_{r}_{g}", 0.0),
                        format="%.2f", key=f"sell_{r}_{g}", disabled=not editable)

        st.number_input("Invest in Term Deposit ($)", min_value=0.0, step=0.01,
                        value=st.session_state.get(f"investtd_{r}_{g}", 0.0),
                        format="%.2f", key=f"investtd_{r}_{g}", disabled=not editable)
        st.caption(f"Today’s TD rate (if held to maturity): {td_rate_today*100:.2f}%")

        st.selectbox("Buy ticker", options=["(none)"] + tickers,
                     index=(["(none)"] + tickers).index(st.session_state.get(f"buy_t_{r}_{g}", "(none)")),
                     key=f"buy_t_{r}_{g}", disabled=not editable)
        st.number_input("Buy qty (units)", min_value=0.0, step=0.01,
                        value=st.session_state.get(f"buy_{r}_{g}", 0.0),
                        format="%.2f", key=f"buy_{r}_{g}", disabled=not editable)

        # Optional previews (read-only so always shown)
        try:
            sell_sel = st.session_state.get(f"sell_t_{r}_{g}", "(none)")
            sell_q   = float(st.session_state.get(f"sell_{r}_{g}", 0.0) or 0.0)
            if sell_sel != "(none)" and sell_q > 0:
                raw = prices[sell_sel]
                spec = p.securities[sell_sel]
                half_bps = (spec.bid_ask_bps / 2.0) + 5 * (spec.liquidity_score - 1)
                eff = raw * (1 - half_bps / 10_000.0)
                st.caption(f"Sell Preview: {sell_q:,.2f} × ${eff:,.2f} = ${sell_q*eff:,.2f}")
            buy_sel = st.session_state.get(f"buy_t_{r}_{g}", "(none)")
            buy_q   = float(st.session_state.get(f"buy_{r}_{g}", 0.0) or 0.0)
            if buy_sel != "(none)" and buy_q > 0:
                raw = prices[buy_sel]
                spec = p.securities[buy_sel]
                half_bps = (spec.bid_ask_bps / 2.0) + 5 * (spec.liquidity_score - 1)
                eff_buy = raw * (1 + half_bps / 10_000.0)
                st.caption(f"Buy Preview: {buy_q:,.2f} × ${eff_buy:,.2f} = ${buy_q*eff_buy:,.2f}")
        except Exception:
            pass

        b1, b2 = st.columns([2,1])
        b1.button("Apply action", key=f"apply_{r}_{g}",
                  on_click=lambda gg=g: st.session_state.update(pending_apply={"g": gg, "r": r}),
                  disabled=not editable)
        b2.button("Clear Actions", key=f"clear_{r}_{g}",
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
    compute_remaining_for_group(st.session_state.portfolios[i].name, r, req_all) <= 0.01
    for i in range(min(NG, len(st.session_state.portfolios)))
)

if role == "Host":
    with rgt:
        st.button("Next Round ▶️", key=f"next_round_{r}",
                  on_click=lambda: st.session_state.update(_next_round_clicked=True))
else:
    with rgt:
        st.caption("Only the Host can advance rounds.")

if st.session_state.get("_next_round_clicked", False):
    st.session_state._next_round_clicked = False
    if role != "Host":
        st.error("Only the Host can advance rounds.")
    else:
        if not all_covered:
            st.error("Cover all groups (remaining ≤ $0.01) before moving on.")
        else:
            if st.session_state.current_round + 1 < st.session_state.rounds:
                st.session_state.current_round += 1
                st.rerun()
            else:
                st.session_state.current_round = st.session_state.rounds
                st.rerun()
# streamlit_app.py
import streamlit as st
import pandas as pd
import random
import json
import time
import uuid
from typing import List, Dict, Tuple, Any

# --- Optional auto-refresh (works if the package is installed) ---
try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore
    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False

# === Your game core ===
from game_core import (
    Portfolio, init_portfolios, generate_withdrawal,
    execute_repo, execute_sale, execute_buy,
    execute_invest_td, execute_redeem_td,
    process_maturities
)

# =========================
# PAGE & THEME (kept + dark-green captions)
# =========================
st.set_page_config(page_title="Liquidity Tranche Simulation", layout="wide")
st.markdown("""
<style>
:root{
  --navy:#0B1F3B;         /* sidebar bg */
  --navy-strong:#0B3D91;  /* headings & buttons */
  --white:#FFFFFF;
  --ink:#111827;          /* near-black text */
  --green:#006400;        /* big totals & captions */
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
  content:"\\00D7";
  color:#000 !important;
  font-weight:900 !important;
  font-size:14px !important;
  line-height:1 !important;
  display:inline-block !important;
}

/* Number steppers icons */
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

/* Buttons */
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
CLAIM_TIMEOUT  = 120        # seconds of inactivity before a claim expires

# ------------ Shared state (single-room, file-backed) ------------
SHARED_FILE = ".shared_state.json"

def _read_shared() -> Dict[str, Any]:
    try:
        with open(SHARED_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _write_shared(data: Dict[str, Any]) -> None:
    try:
        with open(SHARED_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass

def _mutate_shared(mutator):
    s = _read_shared()
    s2 = mutator(dict(s))
    if s2 is not None:
        _write_shared(s2)

def _now() -> float:
    return time.time()

# ------------ Session State bootstrap ------------
def _init_state():
    ss = st.session_state
    ss.client_id = ss.get("client_id", uuid.uuid4().hex)  # stable per tab
    ss.player_name = ss.get("player_name", "")
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
    ss.pending_apply: Dict[str,int] | None = None
    ss.pending_clear: Dict[str,int] | None = None
    ss.num_groups = 4
    ss.role = "Host"
    ss.player_group_index = 0
    ss.live_refresh = ss.get("live_refresh", True)

if "client_id" not in st.session_state:
    _init_state()

# ------------ Helpers ------------
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
    req = generate_withdrawal(r, ss.portfolios[0].market_value(prices_for_mv),
                              random.Random(ss.rng_seed + 10007*r))
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

# Safe adapters
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

# ------------ Claims (one player per group) ------------
def _get_claims():
    s = _read_shared()
    claims = s.get("claims", {})  # { "0": {"id":..., "name":..., "ts":...}, ... }
    now = _now()
    changed = False
    for k, v in list(claims.items()):
        if now - float(v.get("ts", 0)) > CLAIM_TIMEOUT:
            claims.pop(k, None)
            changed = True
    if changed:
        _mutate_shared(lambda s2: {**s2, "claims": claims, "ts": now})
    return claims

def _claim_group(idx: int, client_id: str, player_name: str) -> bool:
    idx_s = str(idx); now = _now()
    def _apply(s):
        claims = s.get("claims", {})
        ex = claims.get(idx_s)
        if (not ex) or (now - float(ex.get("ts", 0)) > CLAIM_TIMEOUT) or (ex.get("id") == client_id):
            claims[idx_s] = {"id": client_id, "name": player_name or f"Player-{client_id[:4]}", "ts": now}
            return {**s, "claims": claims, "ts": now}
        return s
    before = _read_shared().get("claims", {}).get(idx_s)
    _mutate_shared(_apply)
    after = _read_shared().get("claims", {}).get(idx_s)
    return after and after.get("id") == client_id

def _release_group(idx: int, client_id: str):
    idx_s = str(idx)
    def _apply(s):
        claims = s.get("claims", {})
        if claims.get(idx_s, {}).get("id") == client_id:
            claims.pop(idx_s, None)
            return {**s, "claims": claims, "ts": _now()}
        return s
    _mutate_shared(_apply)

def _heartbeat_claim(idx: int, client_id: str):
    idx_s = str(idx); now = _now()
    def _apply(s):
        claims = s.get("claims", {})
        if claims.get(idx_s, {}).get("id") == client_id:
            claims[idx_s]["ts"] = now
            return {**s, "claims": claims, "ts": now}
        return s
    _mutate_shared(_apply)

def _player_can_edit_group(g_index: int) -> bool:
    if st.session_state.role == "Host":
        return True
    owner = _get_claims().get(str(g_index))
    return owner and owner.get("id") == st.session_state.client_id

# ------------ Role selector (top of sidebar) ------------
st.sidebar.markdown("### Session")
colA, colB = st.sidebar.columns([3,2])
with colA:
    st.session_state.player_name = st.text_input("Your display name", st.session_state.player_name or "").strip()
role = st.sidebar.radio("Role", ["Host", "Player"], index=0 if st.session_state.role == "Host" else 1)
st.session_state.role = role

# Optional live refresh
st.session_state.live_refresh = st.sidebar.checkbox("Live refresh", value=st.session_state.live_refresh)
if st.session_state.live_refresh and HAS_AUTOREFRESH:
    st_autorefresh(interval=3000, limit=None, key="auto_refresh_tick")  # 3s
elif st.session_state.live_refresh and not HAS_AUTOREFRESH:
    st.sidebar.caption("Tip: install `streamlit-autorefresh` for auto updates.")

# ------------ Host or Player sidebars ------------
if role == "Host":
    st.sidebar.header("Host Setup")
    uploaded = st.sidebar.file_uploader("Bond price CSV (date + ≥3 securities)", type=["csv"])
    seed = st.sidebar.number_input("RNG seed", value=st.session_state.rng_seed, step=1)
    rounds = st.sidebar.number_input("Rounds", value=st.session_state.rounds, min_value=1, max_value=10, step=1)
    groups = st.sidebar.number_input("Groups (up to 8)", value=st.session_state.num_groups, min_value=1, max_value=MAX_GROUPS_UI, step=1)
    c1, c2 = st.sidebar.columns(2)
    start_clicked = c1.button("Start / Reset", type="primary")
    end_clicked   = c2.button("End Game")
else:
    st.sidebar.header("Player Setup")
    start_clicked = False
    end_clicked = False
    uploaded = None

with st.sidebar.expander("Game Instructions", expanded=False):
    st.markdown(f"""
- **Withdrawals:** One amount per round, **same for all groups**.
- **TDs:** Mature after **{TD_MAT_GAP} rounds**. Early redemption penalty: **{TD_PENALTY*100:.2f}%**.
- **Rates:** Repo & TD vary daily ±50 bps around base; rate stored per trade.
- **Round 1:** Each group randomly invests **10–30%** of Current Account into TD (auto).
- **Apply/Clear:** Players can **only** edit their own group; Host can edit all.
- **Next Round & End Game:** **Host only**.
- Currency: **$**.
""")

# ------------ Host Start / Reset ------------
if role == "Host" and start_clicked:
    if uploaded is None:
        st.sidebar.error("Please upload a CSV with columns: date,BOND_A,BOND_B,BOND_C,...")
    else:
        df = pd.read_csv(uploaded)
        if "date" not in df.columns or len([c for c in df.columns if c != "date"]) < 3:
            st.sidebar.error("CSV must have 'date' and at least 3 security columns.")
        else:
            # Build a shared snapshot so *all* clients can reconstruct state
            tickers_all = [c for c in df.columns if c != "date"]
            price_payload = {
                "columns": df.columns.tolist(),
                "rows": df.to_dict(orient="records"),
            }
            seed_val = int(seed); rounds_val = int(rounds); desired_groups = int(groups)
            # Initialize portfolios deterministically on each client with same inputs
            date0 = str(df.loc[0, "date"])
            prices0 = {t: float(df.loc[0, t]) for t in tickers_all}

            base_portfolios = init_portfolios(tickers_all, prices0, total_reserve=200000.0)
            available = len(base_portfolios)
            cap = min(MAX_GROUPS_UI, available, desired_groups)
            if desired_groups > cap:
                st.sidebar.warning(f"Requested {desired_groups} groups, capped at {cap}.")
            group_names = [p.name for p in base_portfolios[:cap]]

            # Seed shared file
            def _seed_shared(_):
                return {
                    "initialized": True,
                    "rng_seed": seed_val,
                    "rounds": rounds_val,
                    "current_round": 0,
                    "last_maturity_round": -1,
                    "inited_rounds": [],
                    "num_groups": cap,
                    "tickers": tickers_all,
                    "group_names": group_names,
                    "withdrawals": [0.0 for _ in range(rounds_val)],
                    "logs": {name: [] for name in group_names},
                    "price_data": price_payload,
                    "claims": {},  # fresh claims
                    "ts": _now(),
                }
            _mutate_shared(_seed_shared)

            # Also reflect locally for the host session (for immediate view)
            st.session_state.initialized = True
            st.session_state.rng_seed = seed_val
            st.session_state.rounds = rounds_val
            st.session_state.current_round = 0
            st.session_state.last_maturity_round = -1
            st.session_state.inited_rounds = set()
            st.session_state.num_groups = cap
            st.session_state.logs = {name: [] for name in group_names}
            st.session_state.withdrawals = [0.0 for _ in range(rounds_val)]
            st.session_state.price_df = df.copy()
            st.session_state.portfolios = base_portfolios[:cap]
            # Random initial TD allocation (Round 1)
            for p in st.session_state.portfolios:
                frac = random.uniform(0.10, 0.30)
                amt = round(max(0.0, p.current_account) * frac, 2)
                if amt > 0:
                    execute_invest_td(p, amt, 0, rate=BASE_TD_RATE)
            st.session_state.player_group_index = 0
            st.success("Session initialized for all players.")

# ------------ Host End Game ------------
if role == "Host" and end_clicked:
    def _end(s):
        if not s.get("initialized"):
            return s
        s["current_round"] = s.get("rounds", 0)
        s["ts"] = _now()
        return s
    _mutate_shared(_end)

# ------------ Load shared state (players & host runtime) ------------
S = _read_shared()
if not S.get("initialized"):
    if role == "Host":
        st.info("Upload a CSV and click **Start / Reset** to begin.")
    else:
        st.info("Waiting for the Host to start the session…")
    st.stop()

# Sync local with shared snapshot
st.session_state.initialized = True
st.session_state.rng_seed = int(S.get("rng_seed", 1234))
st.session_state.rounds = int(S.get("rounds", 3))
st.session_state.current_round = int(S.get("current_round", 0))
st.session_state.last_maturity_round = int(S.get("last_maturity_round", -1))
st.session_state.num_groups = int(S.get("num_groups", 4))

# Rebuild price_df from payload
pd_cols = S.get("price_data", {}).get("columns", [])
pd_rows = S.get("price_data", {}).get("rows", [])
if pd_cols and pd_rows:
    st.session_state.price_df = pd.DataFrame(pd_rows, columns=pd_cols)

df = st.session_state.price_df
if df is None or "date" not in df.columns:
    st.error("Price data missing or invalid in shared state.")
    st.stop()

all_tickers = [c for c in df.columns if c != "date"]
tickers = all_tickers[:3]
NG = int(S.get("num_groups", 4))
group_names = S.get("group_names", [f"Group {i+1}" for i in range(NG)])

# Reconstruct portfolios deterministically, then replay all logs
def _reconstruct_portfolios() -> List[Portfolio]:
    date0 = str(df.loc[0, "date"])
    prices0 = {t: float(df.loc[0, t]) for t in all_tickers}
    base = init_portfolios(all_tickers, prices0, total_reserve=200000.0)[:NG]
    # rename base to match group_names if needed
    if len(base) == len(group_names):
        for i, p in enumerate(base):
            p.name = group_names[i]
    # Round 1 random TD allocation (same logic as host)
    for p in base:
        frac = random.uniform(0.10, 0.30)
        amt = round(max(0.0, p.current_account) * frac, 2)
        if amt > 0:
            execute_invest_td(p, amt, 0, rate=BASE_TD_RATE)
    # Replay rounds up to current
    logs = S.get("logs", {})
    for rnum in range(0, int(S.get("current_round", 0)) + 1):
        # settle maturities at start of each round (like host)
        for p in base:
            process_maturities(p, rnum)
        # apply actions of that round
        for p in base:
            Ls = [L for L in logs.get(p.name, []) if L.get("round") == rnum + 1]
            if not Ls:
                continue
            # prices for the round
            row_ix = min(rnum, len(df)-1)
            px = {t: float(df.loc[row_ix, t]) for t in all_tickers}
            # rates for the round
            repo_rate_today, td_rate_today = daily_rates_for_round(rnum)
            for L in Ls:
                for t, d in L.get("actions", []):
                    if t == "cash":
                        used = float(d.get("used", 0.0))
                        p.current_account -= used
                    elif t == "repo":
                        ticker = d.get("ticker"); got = float(d.get("got", 0.0))
                        rate = float(d.get("rate", repo_rate_today))
                        if ticker and px.get(ticker, 0.0) > 0 and got > 0:
                            _safe_repo_call(p, ticker, got, px[ticker], rnum, rate)
                        use = float(d.get("use", 0.0))
                        p.current_account -= use
                    elif t == "redeem_td":
                        principal = float(d.get("principal", 0.0))
                        _safe_redeem_td(p, principal, rnum)  # penalty handled in core
                        use = float(d.get("use", 0.0))
                        p.current_account -= use
                    elif t == "sell":
                        ticker = d.get("ticker"); qty = float(d.get("qty", 0.0))
                        if ticker and qty > 0:
                            _safe_sale(p, ticker, qty, px.get(ticker, 0.0))
                        use = float(d.get("use", 0.0))
                        p.current_account -= use
                    elif t == "invest_td":
                        amt = float(d.get("amount", 0.0))
                        if amt > 0:
                            _safe_invest_td(p, amt, rnum, td_rate_today)
                    elif t == "buy":
                        ticker = d.get("ticker"); qty = float(d.get("qty", 0.0))
                        if ticker and qty > 0:
                            _safe_buy(p, ticker, qty, px.get(ticker, 0.0))
    return base

# Bring logs/withdrawals from shared
st.session_state.logs = S.get("logs", {name: [] for name in group_names})
st.session_state.withdrawals = list(S.get("withdrawals", [0.0 for _ in range(st.session_state.rounds)]))

# Reconstruct portfolios view for everyone
st.session_state.portfolios = _reconstruct_portfolios()

# Current round index
r = st.session_state.current_round

# ==== END-GAME GUARD ====
if r >= st.session_state.rounds:
    st.title("Liquidity Tranche Simulation")
    st.header("Scoreboard & Logs")
    _, final_px = base_prices_for_round(min(st.session_state.rounds-1, len(df)-1), df, tickers)
    rows = []
    for p in st.session_state.portfolios[:NG]:
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
    st.download_button("Download logs JSON", json.dumps(st.session_state.logs, indent=2).encode("utf-8"),
                       file_name="logs.json", mime="application/json")
    st.stop()

# Active round content setup (create same-withdrawal-for-all if missing)
date_str, prices = base_prices_for_round(r, df, tickers)
repo_rate_today, td_rate_today = daily_rates_for_round(r)

# Settle maturities once per round (local view)
if st.session_state.last_maturity_round != r:
    for p in st.session_state.portfolios[:NG]:
        process_maturities(p, r)
    st.session_state.last_maturity_round = r

# Ensure withdrawal for round r exists (host computes -> shared; players read)
if role == "Host":
    if r not in S.get("inited_rounds", []):
        # Use Group 1 MV as base for withdrawal generation
        if st.session_state.portfolios:
            req = generate_withdrawal(r, st.session_state.portfolios[0].market_value(prices),
                                      random.Random(st.session_state.rng_seed + 10007*r))
            def _set_w(s):
                w = s.get("withdrawals", [0.0 for _ in range(st.session_state.rounds)])
                w[r] = float(req)
                inited = set(s.get("inited_rounds", [])); inited.add(r)
                s["withdrawals"] = w
                s["inited_rounds"] = list(inited)
                s["ts"] = _now()
                return s
            _mutate_shared(_set_w)
            S = _read_shared()
            st.session_state.withdrawals = S.get("withdrawals", st.session_state.withdrawals)
req_all = float(S.get("withdrawals", [0.0 for _ in range(st.session_state.rounds)])[r])

# ------------------------
# Role-specific pre-UI: Players choose/claim a group
# ------------------------
if role == "Player":
    if not st.session_state.player_name:
        st.warning("Enter a display name (top of sidebar) so others see who claimed a group.")
    claims = _get_claims()
    names = [p.name for p in st.session_state.portfolios[:NG]]
    labels, selectable = [], []
    for i, name in enumerate(names):
        c = claims.get(str(i))
        if not c:
            labels.append(f"{name} (available)")
            selectable.append(i)
        elif c.get("id") == st.session_state.client_id:
            labels.append(f"{name} (you)")
            selectable.append(i)
        else:
            labels.append(f"{name} (claimed by {c.get('name','player')})")
    if not selectable:
        st.sidebar.warning("All groups are currently claimed. Try again shortly.")
        st.stop()
    # Default to your current claim or first available
    current_idx = st.session_state.player_group_index
    if claims.get(str(current_idx), {}).get("id") != st.session_state.client_id:
        current_idx = selectable[0]
    # Show filtered options (Streamlit selectbox can't disable per-option)
    visible_labels = [labels[i] for i in selectable]
    sel = st.sidebar.selectbox("Select your Group", visible_labels,
                               index=selectable.index(current_idx))
    chosen_idx = selectable[visible_labels.index(sel)]
    # Release previous if switching
    prev = st.session_state.player_group_index
    if prev != chosen_idx:
        _release_group(prev, st.session_state.client_id)
    # Try to claim chosen
    if _claim_group(chosen_idx, st.session_state.client_id, st.session_state.player_name or ""):
        st.session_state.player_group_index = chosen_idx
        _heartbeat_claim(chosen_idx, st.session_state.client_id)
        st.sidebar.caption(f"You control: **{names[chosen_idx]}**")
    else:
        st.sidebar.error("That group was just claimed by someone else. Pick another.")
        st.stop()
    # Keep claim alive
    _heartbeat_claim(st.session_state.player_group_index, st.session_state.client_id)

# ------------------------
# Handle queued Apply/Clear BEFORE widgets render (permissions enforced)
# ------------------------
def _queue_apply(g: int, rnow: int):
    st.session_state.pending_apply = {"g": g, "r": rnow}

def _queue_clear(g: int, rnow: int):
    st.session_state.pending_clear = {"g": g, "r": rnow}

# Apply
if st.session_state.pending_apply is not None:
    g_apply = int(st.session_state.pending_apply["g"])
    r_apply = int(st.session_state.pending_apply["r"])
    st.session_state.pending_apply = None

    if 0 <= g_apply < len(st.session_state.portfolios) and _player_can_edit_group(g_apply):
        p = st.session_state.portfolios[g_apply]
        rem_left = compute_remaining_for_group(p.name, r_apply, req_all)

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
        row_ix = min(r_apply, len(df)-1)
        px = {t: float(df.loc[row_ix, t]) for t in all_tickers}
        repo_rate_today, td_rate_today = daily_rates_for_round(r_apply)

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
            # Save to SHARED logs so all clients see immediately
            def _append_log(s):
                logs = s.get("logs", {})
                name = p.name
                arr = list(logs.get(name, []))
                arr.append({"round": r_apply + 1, "request": req_all, "actions": history})
                logs[name] = arr
                s["logs"] = logs
                s["ts"] = _now()
                return s
            _mutate_shared(_append_log)
            # Update local runtime copy
            st.session_state.logs.setdefault(p.name, []).append({
                "round": r_apply + 1, "request": req_all, "actions": history
            })

        # Reset inputs
        for key in [cash_key, repo_amt_key, redeem_key, invest_key, sell_qty_key, buy_qty_key]:
            st.session_state[key] = 0.0
        st.session_state[repo_tick_key] = "(none)"
        st.session_state[sell_tick_key] = "(none)"
        st.session_state[buy_tick_key]  = "(none)"

# Clear
if st.session_state.pending_clear is not None:
    g_clear = int(st.session_state.pending_clear["g"])
    r_clear = int(st.session_state.pending_clear["r"])
    st.session_state.pending_clear = None

    if 0 <= g_clear < len(st.session_state.portfolios) and _player_can_edit_group(g_clear):
        p = st.session_state.portfolios[g_clear]
        # Remove this round's logs for that group from shared & local
        def _remove_logs(s):
            logs = s.get("logs", {})
            arr = [L for L in logs.get(p.name, []) if L.get("round") != r_clear + 1]
            logs[p.name] = arr
            s["logs"] = logs
            s["ts"] = _now()
            return s
        _mutate_shared(_remove_logs)
        st.session_state.logs[p.name] = [
            L for L in st.session_state.logs.get(p.name, []) if L["round"] != r_clear + 1
        ]

# ========================
# MAIN UI
# ========================
st.title("Liquidity Tranche Simulation")

# Round header
st.subheader(f"Round {r+1} — Date: {date_str}")
st.caption(f"Today’s rates → Repo: {repo_rate_today*100:.2f}%  •  TD: {td_rate_today*100:.2f}%  •  Early TD penalty: 1.00%")

# Round dashboard (everyone can see)
cols = st.columns(NG)
for g, c in enumerate(cols):
    if g >= len(st.session_state.portfolios): break
    with c:
        p = st.session_state.portfolios[g]
        rem = compute_remaining_for_group(p.name, r, req_all)
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
        st.progress(max(0.0, 1 - rem/req_all if req_all > 0 else 1))

# Group tabs (players can edit only their claimed group; host can edit all)
tab_labels = [p.name for p in st.session_state.portfolios[:NG]]
tabs = st.tabs(tab_labels if tab_labels else ["Group 1"])
for g, tab in enumerate(tabs):
    if g >= len(st.session_state.portfolios): break
    with tab:
        p = st.session_state.portfolios[g]
        rem = compute_remaining_for_group(p.name, r, req_all)
        editable = _player_can_edit_group(g)

        tag = ""
        if role == "Player" and editable:
            tag = " (You)"
        st.markdown(f"### {p.name}{tag}")

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

        # Player-specific info: show their own remaining prominently
        if role == "Player" and editable:
            st.markdown(f"**Your group’s remaining withdrawal this round:** :orange[{rem:,.2f}]")
        else:
            st.markdown(f"**Remaining for {p.name}:** :orange[{rem:,.2f}]")
        st.markdown(f"**Withdrawal (all groups):** :blue[{req_all:,.2f}]")

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

        # Optional previews (read-only)
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

        b1, b2 = st.columns([2,1])
        b1.button("Apply action",
                  key=f"apply_{r}_{g}",
                  on_click=_queue_apply,
                  kwargs={"g": g, "r": r},
                  disabled=not editable)
        b2.button("Clear Actions",
                  key=f"clear_{r}_{g}",
                  on_click=_queue_clear,
                  kwargs={"g": g, "r": r},
                  disabled=not editable)

# ------------------------
# Next Round controls (Host only)
# ------------------------
st.divider()
lft, rgt = st.columns([3,1])
with lft:
    st.subheader("Controls")

def _all_covered() -> bool:
    return all(
        compute_remaining_for_group(st.session_state.portfolios[i].name, r, req_all) <= 0.01
        for i in range(min(NG, len(st.session_state.portfolios)))
    )

if role == "Host":
    with rgt:
        if st.button("Next Round ▶️", key=f"next_round_{r}"):
            if not _all_covered():
                st.error("Cover all groups (remaining ≤ $0.01) before moving on.")
            else:
                def _advance(s):
                    if s.get("current_round", 0) + 1 < s.get("rounds", 0):
                        s["current_round"] = s.get("current_round", 0) + 1
                    else:
                        s["current_round"] = s.get("rounds", 0)
                    s["ts"] = _now()
                    return s
                _mutate_shared(_advance)
                st.experimental_rerun()
else:
    with rgt:
        st.caption("Only the Host can advance rounds.")


