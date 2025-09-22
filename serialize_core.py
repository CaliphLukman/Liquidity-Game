# serialize_core.py
# Utilities to convert game_core.Portfolio <-> dict for JSON storage.
from typing import Any, Dict, List
from game_core import Portfolio  # uses your existing class

def portfolio_to_dict(p: Portfolio) -> Dict[str, Any]:
    # Adjust fields to match your Portfolio attributes
    return {
        "name": p.name,
        "current_account": float(getattr(p, "current_account", 0.0)),
        "pnl_realized": float(getattr(p, "pnl_realized", 0.0)),
        "pos_qty": dict(getattr(p, "pos_qty", {})),
        "repo_liabilities": list(getattr(p, "repo_liabilities", [])),
        "td_assets": list(getattr(p, "td_assets", [])),
        # It’s fine to expand if you keep extra fields in Portfolio
        "securities_meta": {k: {
            "bid_ask_bps": float(getattr(v, "bid_ask_bps", 0.0)),
            "liquidity_score": float(getattr(v, "liquidity_score", 1.0)),
        } for k, v in getattr(p, "securities", {}).items()} if hasattr(p, "securities") else {},
    }

def dict_to_portfolio(d: Dict[str, Any]) -> Portfolio:
    # Create a minimal Portfolio and then patch fields
    # Your Portfolio likely accepts (name, current_account=..., pos_qty=...)
    # If your __init__ differs, adapt this constructor.
    p = Portfolio(name=d.get("name", "Group"), current_account=d.get("current_account", 0.0))
    p.pnl_realized = d.get("pnl_realized", 0.0)
    p.pos_qty = dict(d.get("pos_qty", {}))
    p.repo_liabilities = list(d.get("repo_liabilities", []))
    p.td_assets = list(d.get("td_assets", []))
    # securities_meta is optional; if your core builds securities elsewhere, you can ignore.
    return p

def portfolios_to_json(portfolios: List[Portfolio]) -> List[Dict[str, Any]]:
    return [portfolio_to_dict(p) for p in portfolios]

def portfolios_from_json(items: List[Dict[str, Any]]) -> List[Portfolio]:
    return [dict_to_portfolio(x) for x in items]
