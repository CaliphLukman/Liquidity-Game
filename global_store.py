# global_store.py
import os
from typing import Any, Dict, List, Tuple, Optional
import pandas as pd

from persist import read_json, write_json, mutate_json, DATA_DIR
from serialize_core import portfolios_to_json, portfolios_from_json

GLOBAL_FILE = "global_state.json"
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

def _read_state() -> Dict[str, Any]:
    return read_json(GLOBAL_FILE, {})

def _write_state(d: Dict[str, Any]) -> None:
    write_json(GLOBAL_FILE, d)

def init_state_if_missing() -> None:
    d = _read_state()
    if not d:
        d = {
            "csv_path": "",
            "tickers": [],
            "rounds": 3,
            "current_round": 0,
            "rng_seed": 1234,
            "last_maturity_round": -1,
            "withdrawals": [0.0, 0.0, 0.0],
            "logs": {},
            "portfolios": [],    # serialized
            "num_groups": 4,
            "group_claims": {},  # index(str) -> player_token
        }
        _write_state(d)

def get_state() -> Dict[str, Any]:
    return _read_state()

def update_state(**kwargs) -> Dict[str, Any]:
    d = _read_state()
    d.update(kwargs)
    _write_state(d)
    return d

def save_csv_to_global(file) -> str:
    # Save uploaded CSV to data/uploads/global.csv
    path = os.path.join(UPLOADS_DIR, "global.csv")
    with open(path, "wb") as f:
        f.write(file.getbuffer())
    d = _read_state()
    d["csv_path"] = path
    # Also set tickers (all non-date columns)
    df = pd.read_csv(path)
    if "date" in df.columns:
        d["tickers"] = [c for c in df.columns if c != "date"]
    _write_state(d)
    return path

def read_csv_df() -> Optional[pd.DataFrame]:
    d = _read_state()
    path = d.get("csv_path", "")
    if not path or not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def set_serialized_portfolios(portfolios) -> None:
    d = _read_state()
    d["portfolios"] = portfolios_to_json(portfolios)
    _write_state(d)

def get_deserialized_portfolios():
    d = _read_state()
    return portfolios_from_json(d.get("portfolios", []))

def replace_logs(logs: Dict[str, List[dict]]) -> None:
    d = _read_state()
    d["logs"] = logs
    _write_state(d)

# ----- Group claims (exclusive selection) -----
def claim_group(index: int, player_token: str) -> Tuple[bool, str]:
    d = _read_state()
    claims = dict(d.get("group_claims", {}))
    key = str(index)
    owner = claims.get(key)
    if owner is None or owner == player_token:
        claims[key] = player_token
        d["group_claims"] = claims
        _write_state(d)
        return True, "claimed"
    else:
        return False, "Group already taken."

def release_group(index: int, player_token: str) -> None:
    d = _read_state()
    claims = dict(d.get("group_claims", {}))
    key = str(index)
    if claims.get(key) == player_token:
        del claims[key]
        d["group_claims"] = claims
        _write_state(d)
