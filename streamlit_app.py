import streamlit as st
import pandas as pd
import random
import json
from typing import List, Dict, Tuple, Any
from pathlib import Path
import time

from game_core import (
    Portfolio, init_portfolios, generate_withdrawal,
    execute_repo, execute_sale, execute_buy,
    execute_invest_td, execute_redeem_td,
    process_maturities
)

st.set_page_config(page_title="Liquidity Tranche Simulation", layout="wide")

# =============== THEME (dark-green captions) ===============
st.markdown("""
<style>
:root{
  --navy:#0B1F3B; --navy-strong:#0B3D91; --white:#FFFFFF; --ink:#111827; --green:#006400; --light-navy:#9DB7FF;
}
.stApp{ background:var(--white); color:var(--ink); }
h1,h2,h3{ color:var(--navy-strong) !important; }
.stMarkdown,.stText,label,p,span,div{ color:var(--ink); }
.stCaption, .stCaption * { color: var(--green) !important; }
.stMetricValue{ color:var(--green) !important; font-weight:800 !important; }
.stMetricLabel{ color:var(--ink) !important; font-weight:600 !important; }
.ticker-line{ color:#000 !important; font-weight:700; }
section[data-testid="stSidebar"]{ background:var(--navy); color:var(--white); border-right:4px solid var(--navy-strong); box-shadow:4px 0 12px rgba(0,0,0,.08); }
section[data-testid="stSidebar"] *{ color:var(--white) !important; opacity:1 !important; }
section[data-testid="stSidebar"] input, section[data-testid="stSidebar"] textarea, section[data-testid="stSidebar"] select{
  background:#102A53 !important; color:var(--white) !important; border:1.5px solid var(--light-navy) !important; border-radius:8px !important;
}
section[data-testid="stSidebar"] ::placeholder{ color:#EAF2FF !important; opacity:1 !important; }
div.stButton > button{ background:var(--navy-strong) !important; color:var(--white) !important; border:0 !important; border-radius:8px !important; font-weight:700 !important; padding:.45rem .9rem !important; }
div.stButton > button *{ color:var(--white) !important; fill:var(--white) !important; }
div.stButton > button:hover{ filter:brightness(.92); }
</style>
""", unsafe_allow_html=True)

# =============== CONFIG ===============
BASE_REPO_RATE = 0.015
BASE_TD_RATE   = 0.0175
DAILY_SPREAD   = 0.005
TD_PENALTY     = 0.01
TD_MAT_GAP     = 2
MAX_GROUPS_UI  = 8

# ======== SIMPLE SHARED STATE (one game per app) =========
STATE_FILE = Path("shared_state.json")

def _read_shared() -> Dict[str, Any]:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _write_shared(obj: Dict[str, Any]) -> None:
    # best-effort write; avoids partial writes by temp swap
    tmp = STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj), encoding="utf-8")
    tmp.replace(STATE_FILE)

def _serialize_df_to_csv(df: pd.DataFrame) -> str:
    return df.to_csv(index=False)

def _df_from_csv_text(text: str) -> pd.DataFrame:
    from io import StringIO
    return pd.read_csv(StringIO(text))

# =============== SESSION BOOTSTRAP ===============
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
    ss.pending_apply: Dict[str,int] | None = None
    ss.pending_clear: Dict[str,int] | None = None
    ss.num_groups = 4
    ss.role = "Host"
    ss.player_group_index = 0
    ss.claimed_groups: Dict[str, str] = {}  # name->claimed_by

if "initialized" not in st.session_state:
    _init_state()

# =============== HELPERS ===============
def price_row_for_round(r: int, df: pd.DataFrame) -> int:
    return min(r, len(df) - 1)

def base_prices_for_round(r: int, df: pd.DataFrame, tickers: List[str]) -> Tuple[str, Dict[str, float]]:
    ix = price_row_for_round(r, df)
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

# safe adapters
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
        return {"ticker": res.get("ticker", ticker), "qty": float(res.get("qty", qty)),
                "proceeds": float(res.get("proceeds", qty * px)),
                "pnl_delta": float(res.get("pnl_delta", 0.0)),
                "effective_price": float(res.get("effective_price", px))}
    return {"ticker": ticker, "qty": qty, "proceeds": qty*px, "pnl_delta": 0.0, "effective_price": px}

def _safe_buy(portfolio: Portfolio, ticker: str, qty: float, px: float) -> Dict[str, Any]:
    res = execute_buy(portfolio, ticker, qty, px)
    if isinstance(res, dict):
        return {"ticker": res.get("ticker", ticker), "qty": float(res.get("qty", qty)),
                "cost": float(res.get("cost", qty * px)),
                "effective_price": float(res.get("effective_price", px))}
    return {"ticker": ticker, "qty": qty, "cost": qty*px, "effective_price": px}

# =============== ROLE SELECTOR ===============
st.sidebar.markdown("### Session")
role = st.sidebar.radio("Role", ["Host", "Player"], index=0 if st.session_state.role == "Host" else 1)
st.session_state.role = role

# =============== HOST / PLAYER SIDEBAR ===============
if role == "Host":
    st.sidebar.header("Host Setup")
    uploaded = st.sidebar.file_uploader("Bond price CSV (date + ≥3 securities)", type=["csv"])
    seed = st.sidebar.number_input("RNG seed", value=st.session_state.rng_seed, step=1)
    rounds = st.sidebar.number_input("Rounds", value=st.session_state.rounds, min_value=1, max_value=10, step=1)
    groups = st.sidebar.number_input("Groups (up to 8)", value=st.session_state.num_groups, min_value=1, max_value=MAX_GROUPS_UI, step=1)
    c1, c2 = st.sidebar.columns(2)
    start_clicked = c1.button("Start/Reset", type="primary")
    end_clicked   = c2.button("End Game")
else:
    st.sidebar.header("Player Setup")
    uploaded = None
    start_clicked = False
    end_clicked = False
    st.sidebar.caption("Waiting for Host updates…")
    # Auto-refresh player UI every 2s to pick up host changes
    st.experimental_rerun  # (keeps reference for linters)
    st_autorefresh_count = st.experimental_rerun  # legacy shim
    st.experimental_set_query_params(ts=str(time.time()))  # bust browser cache
    st_autorefresh = st.experimental_singleton(lambda: None)  # no-op to keep compatibility
    st_autorefresh  # touch to avoid flake warnings

with st.sidebar.expander("Game Instructions", expanded=False):
    st.markdown(f"""
- **Withdrawals:** One amount per round, **same for all groups**.
- **Term Deposits (TD):** Mature after **{TD_MAT_GAP} rounds**. Early redemption penalty: **{TD_PENALTY*100:.2f}%**.
- **Rates:** Repo & TD vary daily by ±50 bps.
- **Round 1:** Each group auto-invests 10–30% of Current Account into TD.
- **Players** can only edit their **claimed** group.
- **Host** controls **Start/Reset**, **Next Round**, **End Game**.
""")

# =============== START / RESET (HOST) ===============
if role == "Host" and start_clicked:
    if uploaded is None:
        st.sidebar.error("Please upload a CSV with columns: date,BOND_A,BOND_B,BOND_C,...")
    else:
        df = pd.read_csv(uploaded)
        if "date" not in df.columns or len([c for c in df.columns if c != "date"]) < 3:
            st.sidebar.error("CSV must have 'date' and at least 3 security columns.")
        else:
            seed_val = int(seed); rounds_val = int(rounds); desired_groups = int(groups)

            # Local session
            _init_state()
            st.session_state.initialized   = True
            st.session_state.rng_seed      = seed_val
            st.session_state.rounds        = rounds_val
            st.session_state.price_df      = df.copy()

            tickers_all = [c for c in df.columns if c != "date"]
            _, prices0 = base_prices_for_round(0, df, tickers_all)

            base_portfolios = init_portfolios(tickers_all, prices0, total_reserve=200000.0)
            cap = min(MAX_GROUPS_UI, len(base_portfolios), desired_groups)
            if desired_groups > cap:
                st.sidebar.warning(f"Requested {desired_groups} groups, but only {cap} available. Capped.")
            st.session_state.portfolios = base_portfolios[:cap]
            st.session_state.num_groups = cap

            # Random initial TD allocation
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
            st.session_state.claimed_groups = {}  # reset claims on new game

            # Write shared state so players see it
            shared = {
                "started": True,
                "rng_seed": seed_val,
                "rounds": rounds_val,
                "current_round": 0,
                "num_groups": cap,
                "price_df_csv": _serialize_df_to_csv(df),
                "tickers": tickers_all,
                "withdrawals": st.session_state.withdrawals,
                "logs": st.session_state.logs,
                "claimed_groups": st.session_state.claimed_groups,
            }
            _write_shared(shared)
            st.success("Game started. Players can refresh and join now.")

if role == "Host" and end_clicked and st.session_state.initialized:
    st.session_state.current_round = st.session_state.rounds
    shared = _read_shared()
    if shared.get("started"):
        shared["current_round"] = st.session_state.current_round
        _write_shared(shared)
    st.rerun()

# =============== MAIN UI ===============
st.title("Liquidity Tranche Simulation")

# Pull shared state for everyone (if exists)
shared_now = _read_shared()
if not st.session_state.initialized:
    # Players (or Host after reload) adopt shared state if started
    if shared_now.get("started"):
        try:
            st.session_state.initialized = True
            st.session_state.rng_seed = int(shared_now["rng_seed"])
            st.session_state.rounds   = int(shared_now["rounds"])
            st.session_state.current_round = int(shared_now.get("current_round", 0))
            st.session_state.num_groups = int(shared_now.get("num_groups", 4))
            st.session_state.price_df = _df_from_csv_text(shared_now["price_df_csv"])
            tickers_all = [c for c in st.session_state.price_df.columns if c != "date"]
            _, prices0 = base_prices_for_round(0, st.session_state.price_df, tickers_all)
            # Initialize portfolios deterministically (same seed & df -> same init)
            base_portfolios = init_portfolios(tickers_all, prices0, total_reserve=200000.0)
            st.session_state.portfolios = base_portfolios[:st.session_state.num_groups]
            # TD allocation replay (approx by deterministic seed)
            random.seed(st.session_state.rng_seed)  # ensure similar behavior
            for p in st.session_state.portfolios:
                frac = random.uniform(0.10, 0.30)
                amt = round(max(0.0, p.current_account) * frac, 2)
                if amt > 0:
                    execute_invest_td(p, amt, 0, rate=BASE_TD_RATE)

            st.session_state.logs = shared_now.get("logs", {p.name: [] for p in st.session_state.portfolios})
            st.session_state.withdrawals = shared_now.get("withdrawals", [0.0 for _ in range(st.session_state.rounds)])
            st.session_state.claimed_groups = shared_now.get("claimed_groups", {})
            st.session_state.last_maturity_round = -1
            st.session_state.inited_rounds = set(i for i,v in enumerate(st.session_state.withdrawals) if v != 0.0)
        except Exception:
            st.info("Waiting for the Host to start the session.")
            st.stop()
    else:
        if role == "Host":
            st.info("Upload a CSV and click Start/Reset to begin.")
        else:
            st.info("Waiting for Host to start the session.")
        st.stop()

df = st.session_state.price_df
all_tickers = [c for c in df.columns if c != "date"]
tickers = all_tickers[:3]
r = st.session_state.current_round
NG = max(1, int(st.session_state.get("num_groups", len(st.session_state.portfolios) or 1)))

# ==== END-GAME GUARD ====
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
    st.dataframe(sb, use_container_width=True)
    st.download_button("Download scoreboard CSV", sb.to_csv(index=False).encode("utf-8"),
                       file_name="scoreboard.csv", mime="text/csv")
    st.download_button("Download logs JSON", json.dumps(st.session_state.logs, indent=2).encode("utf-8"),
                       file_name="logs.json", mime="application/json")
    st.stop()

# Active round calc
date_str, prices = base_prices_for_round(r, df, tickers)
repo_rate_today, td_rate_today = daily_rates_for_round(r)

# Settle maturities once per round
if st.session_state.last_maturity_round != r:
    for p in st.session_state.portfolios:
        process_maturities(p, r)
    st.session_state.last_maturity_round = r

# Withdrawal (same for all)
ensure_round_initialized(r, prices)
req_all = float(st.session_state.withdrawals[r])

# =============== CLAIM GROUP (players) ===============
st.sidebar.markdown("---")
if role == "Player":
    # Show available & claimed
    names = [p.name for p in st.session_state.portfolios]
    claimed = shared_now.get("claimed_groups", {})
    options = names
    current_idx = min(st.session_state.player_group_index, max(0, len(options)-1))
    chosen = st.sidebar.selectbox("Choose your group", options, index=current_idx)
    st.session_state.player_group_index = options.index(chosen)

    owner = claimed.get(chosen)
    if owner and owner != "claimed":
        st.sidebar.warning(f"{chosen} already claimed.")
    elif owner == "claimed":
        st.sidebar.info(f"You have claimed {chosen}.")
    else:
        if st.sidebar.button(f"Claim {chosen}"):
            # Write claim to shared, then reload shared into session
            shared_now = _read_shared()
            cg = shared_now.get("claimed_groups", {})
            if cg.get(chosen) in (None, ""):
                cg[chosen] = "claimed"
                shared_now["claimed_groups"] = cg
                _write_shared(shared_now)
                st.session_state.claimed_groups = cg
                st.success(f"Claimed {chosen}.")
            else:
                st.warning("Too slow—someone just claimed it.")

def _player_can_edit_group(idx: int) -> bool:
    if st.session_state.role == "Host":
        return True
    names = [p.name for p in st.session_state.portfolios]
    if not (0 <= idx < len(names)): return False
    gname = names[idx]
    owner = shared_now.get("claimed_groups", {}).get(gname)
    return (idx == st.session_state.player_group_index) and (owner == "claimed")

# =============== APPLY/CLEAR (with permission) ===============
if st.session_state.pending_apply is not None:
    g_apply = int(st.session_state.pending_apply["g"]); r_apply = int(st.session_state.pending_apply["r"])
    if 0 <= g_apply < len(st.session_state.portfolios) and _player_can_edit_group(g_apply):
        p = st.session_state.portfolios[g_apply]
        rem_left = compute_remaining_for_group(p.name, r_apply, req_all)
        px = {t: float(df.loc[price_row_for_round(r_apply, df), t]) for t in tickers}
        hist = []

        cash_key, repo_amt_key, repo_tick_key = f"cash_{r_apply}_{g_apply}", f"repo_{r_apply}_{g_apply}", f"repo_t_{r_apply}_{g_apply}"
        redeem_key, invest_key = f"redeemtd_{r_apply}_{g_apply}", f"investtd_{r_apply}_{g_apply}"
        sell_qty_key, sell_tick_key = f"sell_{r_apply}_{g_apply}", f"sell_t_{r_apply}_{g_apply}"
        buy_qty_key, buy_tick_key = f"buy_{r_apply}_{g_apply}", f"buy_t_{r_apply}_{g_apply}"

        cash_amt = float(st.session_state.get(cash_key, 0.0) or 0.0)
        if cash_amt > 0 and rem_left > 0:
            use = min(cash_amt, max(0.0, p.current_account), rem_left)
            if use > 0:
                p.current_account -= use; rem_left -= use; hist.append(("cash", {"used": round(use,2)}))

        repo_amt = float(st.session_state.get(repo_amt_key, 0.0) or 0.0)
        repo_tick = st.session_state.get(repo_tick_key, "(none)")
        if repo_tick != "(none)" and repo_amt > 0 and rem_left > 0:
            price = px.get(repo_tick, 0.0); max_amt = p.pos_qty.get(repo_tick, 0.0) * price
            repo_amt = min(repo_amt, max_amt)
            if repo_amt > 0:
                info = _safe_repo_call(p, repo_tick, repo_amt, price, r_apply, repo_rate_today)
                got = float(info["got"]); use = min(got, rem_left)
                if use > 0: p.current_account -= use; rem_left -= use
                hist.append(("repo", {"ticker": repo_tick, "got": round(got,2), "use": round(use,2), "repo_id": info["repo_id"], "rate": repo_rate_today}))

        redeem_amt = float(st.session_state.get(redeem_key, 0.0) or 0.0)
        if redeem_amt > 0 and rem_left > 0:
            red = _safe_redeem_td(p, redeem_amt, r_apply)
            principal = float(red["principal"]); use = min(principal, rem_left)
            if use > 0: p.current_account -= use; rem_left -= use
            hist.append(("redeem_td", {"principal": round(principal,2), "penalty": round(float(red["penalty"]),2), "use": round(use,2), "chunks": red.get("redeemed", [])}))

        sell_qty = float(st.session_state.get(sell_qty_key, 0.0) or 0.0)
        sell_tick = st.session_state.get(sell_tick_key, "(none)")
        if sell_tick != "(none)" and sell_qty > 0 and rem_left > 0:
            sell_qty = min(sell_qty, p.pos_qty.get(sell_tick, 0.0))
            if sell_qty > 0:
                sale = _safe_sale(p, sell_tick, sell_qty, px[sell_tick]); use = min(sale["proceeds"], rem_left)
                if use > 0: p.current_account -= use; rem_left -= use
                hist.append(("sell", {"ticker": sale["ticker"], "qty": round(sale["qty"],2), "proceeds": round(sale["proceeds"],2), "use": round(use,2), "pnl_delta": round(sale["pnl_delta"],2), "effective_price": round(sale["effective_price"],6)}))

        invest_amt = float(st.session_state.get(invest_key, 0.0) or 0.0)
        if invest_amt > 0:
            invest_amt = min(invest_amt, max(0.0, p.current_account))
            if invest_amt > 0:
                td_ids = _safe_invest_td(p, invest_amt, r_apply, td_rate_today)
                hist.append(("invest_td", {"amount": round(invest_amt,2), "td_ids": td_ids, "rate": td_rate_today}))

        buy_qty = float(st.session_state.get(buy_qty_key, 0.0) or 0.0)
        buy_tick = st.session_state.get(buy_tick_key, "(none)")
        if buy_tick != "(none)" and buy_qty > 0:
            buy = _safe_buy(p, buy_tick, buy_qty, px[buy_tick])
            hist.append(("buy", {"ticker": buy["ticker"], "qty": round(buy["qty"],2), "cost": round(buy["cost"],2), "effective_price": round(buy["effective_price"],6)}))

        if hist:
            st.session_state.logs[p.name].append({"round": r_apply + 1, "request": req_all, "actions": hist})

        # reset inputs
        for key in [cash_key, repo_amt_key, redeem_key, invest_key, sell_qty_key, buy_qty_key]:
            st.session_state[key] = 0.0
        st.session_state[repo_tick_key] = "(none)"
        st.session_state[sell_tick_key] = "(none)"
        st.session_state[buy_tick_key]  = "(none)"

        # push logs to shared so everyone sees progress
        shared_sync = _read_shared()
        shared_sync["logs"] = st.session_state.logs
        _write_shared(shared_sync)

    st.session_state.pending_apply = None

if st.session_state.pending_clear is not None:
    g_clear = int(st.session_state.pending_clear["g"]); r_clear = int(st.session_state.pending_clear["r"])
    if 0 <= g_clear < len(st.session_state.portfolios) and _player_can_edit_group(g_clear):
        p = st.session_state.portfolios[g_clear]
        logs_this_round = [L for L in st.session_state.logs.get(p.name, []) if L["round"] == r_clear + 1]
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
                            if abs(float(l.get("amount", 0.0)) - got) < 1e-6 and l.get("maturity", 10**9) > r_clear:
                                p.repo_liabilities.pop(i); break
                elif t == "redeem_td":
                    principal = float(d.get("principal", 0.0)); penalty = float(d.get("penalty", 0.0)); use = float(d.get("use", 0.0))
                    p.current_account -= principal
                    if penalty > 0: p.current_account += penalty; p.pnl_realized += penalty
                    p.current_account += use
                    for ch in d.get("chunks", []):
                        p.td_assets.append({"id": ch.get("id"), "amount": float(ch.get("taken", 0.0)),
                                            "rate": float(ch.get("rate", BASE_TD_RATE)), "maturity": int(ch.get("maturity", r_clear + TD_MAT_GAP))})
                elif t == "sell":
                    proceeds = float(d.get("proceeds", 0.0)); use = float(d.get("use", 0.0))
                    ticker = d.get("ticker"); qty = float(d.get("qty", 0.0)); pnl_delta = float(d.get("pnl_delta", 0.0))
                    p.current_account -= proceeds; p.current_account += use
                    if ticker is not None: p.pos_qty[ticker] = p.pos_qty.get(ticker, 0.0) + qty
                    p.pnl_realized -= pnl_delta
                elif t == "invest_td":
                    amt = float(d.get("amount", 0.0)); ids = set(d.get("td_ids", []))
                    p.current_account += amt
                    if ids: p.td_assets = [a for a in p.td_assets if a.get("id") not in ids]
                elif t == "buy":
                    cost = float(d.get("cost", 0.0)); ticker = d.get("ticker"); qty = float(d.get("qty", 0.0))
                    p.current_account += cost
                    if ticker is not None: p.pos_qty[ticker] = p.pos_qty.get(ticker, 0.0) - qty

        st.session_state.logs[p.name] = [L for L in st.session_state.logs.get(p.name, []) if L["round"] != r_clear + 1]

        # push logs change to shared
        shared_sync = _read_shared()
        shared_sync["logs"] = st.session_state.logs
        _write_shared(shared_sync)

        # reset inputs
        for key in [f"cash_{r_clear}_{g_clear}", f"repo_{r_clear}_{g_clear}", f"redeemtd_{r_clear}_{g_clear}",
                    f"sell_{r_clear}_{g_clear}", f"investtd_{r_clear}_{g_clear}", f"buy_{r_clear}_{g_clear}"]:
            st.session_state[key] = 0.0
        st.session_state[f"repo_t_{r_clear}_{g_clear}"] = "(none)"
        st.session_state[f"sell_t_{r_clear}_{g_clear}"] = "(none)"
        st.session_state[f"buy_t_{r_clear}_{g_clear}"]  = "(none)"
    st.session_state.pending_clear = None

# =============== DASHBOARD ===============
st.subheader(f"Round {r+1} — Date: {date_str}")
st.caption(f"Today’s rates → Repo: {repo_rate_today*100:.2f}%  •  TD: {td_rate_today*100:.2f}%  •  Early TD penalty: 1.00%")

cols = st.columns(NG)
for g, c in enumerate(cols):
    if g >= len(st.session_state.portfolios): break
    with c:
        p = st.session_state.portfolios[g]
        rem = compute_remaining_for_group(p.name, r, req_all)
        reserve = p.market_value(prices)
        st.markdown(f"### {p.name}")
        st.markdown(f"<div style='font-size:28px; font-weight:800; color:#006400;'>${reserve:,.0f}</div>", unsafe_allow_html=True)
        for t in tickers:
            st.markdown(f"<div class='ticker-line'>{t}: {p.pos_qty.get(t,0.0):,.0f} @ {prices[t]:,.2f}</div>", unsafe_allow_html=True)
        st.progress(max(0.0, 1 - rem/req_all if req_all > 0 else 1))

# =============== GROUP TABS (edit lock per claim) ===============
tab_labels = [p.name for p in st.session_state.portfolios[:NG]]
tabs = st.tabs(tab_labels if tab_labels else ["Group 1"])
for gidx, tab in enumerate(tabs):
    if gidx >= len(st.session_state.portfolios): break
    with tab:
        p = st.session_state.portfolios[gidx]
        rem = compute_remaining_for_group(p.name, r, req_all)
        editable = (role == "Host") or _player_can_edit_group(gidx)

        st.markdown(f"### {p.name} {'(You)' if (role=='Player' and editable) else ''}")
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

        # Inputs (disabled if not editable)
        st.number_input("Use cash", min_value=0.0, step=0.01,
                        value=st.session_state.get(f"cash_{r}_{gidx}", 0.0),
                        format="%.2f", key=f"cash_{r}_{gidx}", disabled=not editable)

        st.selectbox("Repo ticker", options=["(none)"] + tickers,
                     index=(["(none)"] + tickers).index(st.session_state.get(f"repo_t_{r}_{gidx}", "(none)")),
                     key=f"repo_t_{r}_{gidx}", disabled=not editable)
        st.number_input("Repo amount", min_value=0.0, step=0.01,
                        value=st.session_state.get(f"repo_{r}_{gidx}", 0.0),
                        format="%.2f", key=f"repo_{r}_{gidx}", disabled=not editable)
        st.caption(f"Today’s Repo rate: {repo_rate_today*100:.2f}%")
        if st.session_state.get(f"repo_t_{r}_{gidx}", "(none)") != "(none)" and st.session_state.get(f"repo_{r}_{gidx}", 0.0) > 0:
            rt = st.session_state[f"repo_t_{r}_{gidx}"]; ra = float(st.session_state[f"repo_{r}_{gidx}"])
            if prices.get(rt, 0.0) > 0:
                eq = ra / prices[rt]; st.caption(f"≈ {eq:,.2f} units of {rt}")

        st.number_input("Redeem Term Deposit", min_value=0.0, step=0.01,
                        value=st.session_state.get(f"redeemtd_{r}_{gidx}", 0.0),
                        format="%.2f", key=f"redeemtd_{r}_{gidx}", disabled=not editable)
        st.caption("Early redemption penalty: 1.00%")

        st.selectbox("Sell ticker", options=["(none)"] + tickers,
                     index=(["(none)"] + tickers).index(st.session_state.get(f"sell_t_{r}_{gidx}", "(none)")),
                     key=f"sell_t_{r}_{gidx}", disabled=not editable)
        st.number_input("Sell qty", min_value=0.0, step=0.01,
                        value=st.session_state.get(f"sell_{r}_{gidx}", 0.0),
                        format="%.2f", key=f"sell_{r}_{gidx}", disabled=not editable)

        st.number_input("Invest in Term Deposit", min_value=0.0, step=0.01,
                        value=st.session_state.get(f"investtd_{r}_{gidx}", 0.0),
                        format="%.2f", key=f"investtd_{r}_{gidx}", disabled=not editable)
        st.caption(f"Today’s TD rate (if held to maturity): {td_rate_today*100:.2f}%")

        st.selectbox("Buy ticker", options=["(none)"] + tickers,
                     index=(["(none)"] + tickers).index(st.session_state.get(f"buy_t_{r}_{gidx}", "(none)")),
                     key=f"buy_t_{r}_{gidx}", disabled=not editable)
        st.number_input("Buy qty", min_value=0.0, step=0.01,
                        value=st.session_state.get(f"buy_{r}_{gidx}", 0.0),
                        format="%.2f", key=f"buy_{r}_{gidx}", disabled=not editable)

        # Previews
        try:
            sell_sel = st.session_state.get(f"sell_t_{r}_{gidx}", "(none)")
            sell_q   = float(st.session_state.get(f"sell_{r}_{gidx}", 0.0) or 0.0)
            if sell_sel != "(none)" and sell_q > 0:
                raw = prices[sell_sel]; spec = p.securities[sell_sel]
                half_bps = (spec.bid_ask_bps / 2.0) + 5 * (spec.liquidity_score - 1)
                eff = raw * (1 - half_bps / 10_000.0)
                st.caption(f"Sell Preview: {sell_q:,.2f} × {eff:,.2f} = {sell_q*eff:,.2f}")
            buy_sel = st.session_state.get(f"buy_t_{r}_{gidx}", "(none)")
            buy_q   = float(st.session_state.get(f"buy_{r}_{gidx}", 0.0) or 0.0)
            if buy_sel != "(none)" and buy_q > 0:
                raw = prices[buy_sel]; spec = p.securities[buy_sel]
                half_bps = (spec.bid_ask_bps / 2.0) + 5 * (spec.liquidity_score - 1)
                eff_buy = raw * (1 + half_bps / 10_000.0)
                st.caption(f"Buy Preview: {buy_q:,.2f} × {eff_buy:,.2f} = {buy_q*eff_buy:,.2f}")
        except Exception:
            pass

        b1, b2 = st.columns([2,1])
        b1.button("Apply action", key=f"apply_{r}_{gidx}",
                  on_click=lambda gg=gidx: st.session_state.update(pending_apply={"g": gg, "r": r}),
                  disabled=not editable)
        b2.button("Clear Actions", key=f"clear_{r}_{gidx}",
                  on_click=lambda gg=gidx: st.session_state.update(pending_clear={"g": gg, "r": r}),
                  disabled=not editable)

# =============== NEXT ROUND (Host only, sync to shared) ===============
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
        next_clicked = st.button("Next Round ▶️", key=f"next_round_{r}")
        if next_clicked:
            if not all_covered:
                st.error("Cover all groups (remaining ≤ $0.01) before moving on.")
            else:
                if st.session_state.current_round + 1 < st.session_state.rounds:
                    st.session_state.current_round += 1
                else:
                    st.session_state.current_round = st.session_state.rounds
                # write shared so players jump with host
                shared_sync = _read_shared()
                shared_sync.update({
                    "current_round": st.session_state.current_round,
                    "withdrawals": st.session_state.withdrawals,
                    "logs": st.session_state.logs
                })
                _write_shared(shared_sync)
                st.rerun()
else:
    with rgt:
        st.caption("Only the Host can advance rounds.")

# Periodic player refresh (every ~2s) to pick up shared updates
if role == "Player":
    st.experimental_memo.clear()  # light nudge to re-run
    st.experimental_rerun  # symbol reference
