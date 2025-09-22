# room_store.py
import os
import uuid
from typing import Any, Dict, List
import pandas as pd

from persist import read_json, write_json, mutate_json, DATA_DIR
from serialize_core import portfolios_to_json, portfolios_from_json

ROOMS_FILE = "rooms_index.json"  # maps room_id -> metadata (rounds, current_round, params, csv_path, tickers, withdrawals, logs ptr)
ROOM_PREFIX = "room_"            # each room gets a file: room_<id>.json
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

def _room_file(room_id: str) -> str:
    return f"{ROOM_PREFIX}{room_id}.json"

def _load_room(room_id: str) -> Dict[str, Any]:
    return read_json(_room_file(room_id), {})

def _save_room(room_id: str, data: Dict[str, Any]) -> None:
    write_json(_room_file(room_id), data)

def create_room(host_id: str, rounds: int = 3) -> str:
    rid = uuid.uuid4().hex[:6].upper()
    data = {
        "room_id": rid,
        "host_id": host_id,
        "rounds": int(rounds),
        "current_round": 0,
        "params": {"rng_seed": 1234, "last_maturity_round": -1},
        "csv_path": "",
        "tickers": [],
        "withdrawals": [0.0] * int(rounds),
        "portfolios": [],     # serialized
        "logs": {},           # {group_name: [ ... ]}
    }
    _save_room(rid, data)
    # index (optional)
    mutate_json(ROOMS_FILE, {}, lambda idx: {**idx, rid: {"rounds": rounds}})
    return rid

def get_room(room_id: str) -> Dict[str, Any]:
    return _load_room(room_id)

def update_room(room_id: str, **kwargs) -> Dict[str, Any]:
    room = _load_room(room_id)
    room.update(kwargs)
    _save_room(room_id, room)
    return room

def save_csv(room_id: str, file) -> str:
    # Save uploaded CSV to data/uploads/<room_id>.csv
    path = os.path.join(UPLOADS_DIR, f"{room_id}.csv")
    with open(path, "wb") as f:
        f.write(file.getbuffer())
    room = _load_room(room_id)
    room["csv_path"] = path
    # Also set tickers (all non-date columns)
    df = pd.read_csv(path)
    if "date" in df.columns:
        room["tickers"] = [c for c in df.columns if c != "date"]
    _save_room(room_id, room)
    return path

def set_room_portfolios(room_id: str, portfolios) -> None:
    room = _load_room(room_id)
    room["portfolios"] = portfolios_to_json(portfolios)
    _save_room(room_id, room)

def get_room_portfolios(room_id: str):
    room = _load_room(room_id)
    return portfolios_from_json(room.get("portfolios", []))

def replace_logs(room_id: str, logs: Dict[str, List[dict]]) -> None:
    room = _load_room(room_id)
    room["logs"] = logs
    _save_room(room_id, room)
