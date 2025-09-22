# persist.py
import json
import os
from contextlib import contextmanager
from typing import Any, Callable
from filelock import FileLock

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

def _path(name: str) -> str:
    return os.path.join(DATA_DIR, name)

@contextmanager
def _locked(path: str):
    lock = FileLock(path + ".lock")
    lock.acquire(timeout=10)
    try:
        yield
    finally:
        lock.release()

def read_json(name: str, default: Any) -> Any:
    path = _path(name)
    if not os.path.exists(path):
        return default
    with _locked(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception:
                return default

def write_json(name: str, obj: Any) -> None:
    path = _path(name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with _locked(path):
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

def mutate_json(name: str, default: Any, mutator: Callable[[Any], Any]) -> Any:
    obj = read_json(name, default)
    obj = mutator(obj)
    write_json(name, obj)
    return obj
