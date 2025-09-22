#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Liquidity Tranche Liquidity-Management Simulation Game (CLI version)
-------------------------------------------------------------------
Run with:  python liquidity_game.py --prices path/to/bond_prices.csv
"""

import argparse
import json
import random
import pandas as pd

from game_core import (
    Portfolio, init_portfolios, generate_withdrawal,
    execute_sale, execute_repo, max_repo_cash
)

def load_price_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("CSV must include a 'date' column in YYYY-MM-DD format.")
    return df

def settle_round(portfolio: Portfolio, prices: dict, request: float) -> dict:
    remaining = request
    history = []
    print(f"\n=== {portfolio.name}: Withdrawal request = {request:,.2f} ===")
    print("Type 'help' for commands.\n")

    while remaining > 1e-6:
        cmd = input(f"[{portfolio.name}] Need {remaining:,.2f}. Enter action: ").strip()
        if not cmd:
            continue
        parts = cmd.split()
        op = parts[0].lower()

        if op == "help":
            print("Commands:\n  cash AMOUNT\n  repo TICKER AMOUNT\n  sell TICKER QTY\n  status\n  done (when covered)\n")
            continue

        if op == "status":
            print("Status:", portfolio.summary(prices))
            print("Repo capacity:", {k: round(v,2) for k,v in max_repo_cash(portfolio, prices).items()})
            print("Positions:", {k: round(v,2) for k,v in portfolio.pos_qty.items()})
            continue

        if op == "cash" and len(parts) == 2:
            amt = float(parts[1])
            use = min(amt, portfolio.current_account, remaining)
            portfolio.current_account -= use
            remaining -= use
            history.append(("cash", use))
            print(f"Used cash: {use:,.2f}, remaining: {remaining:,.2f}")
            continue

        if op == "repo" and len(parts) == 3:
            ticker = parts[1].upper()
            amt = float(parts[2])
            got = execute_repo(portfolio, ticker, amt, prices)
            if got <= 0:
                print("Repo failed or zero capacity.")
                continue
            use = min(got, remaining)
            portfolio.current_account -= use
            remaining -= use
            history.append(("repo", ticker, got, use))
            print(f"Repo cash: {got:,.2f} (used {use:,.2f}), remaining: {remaining:,.2f}")
            continue

        if op == "sell" and len(parts) == 3:
            ticker = parts[1].upper()
            qty = float(parts[2])
            price = prices.get(ticker, portfolio.securities[ticker].face_price)
            proceeds = execute_sale(portfolio, ticker, qty, price)
            if proceeds <= 0:
                print("Sale failed or zero qty.")
                continue
            use = min(proceeds, remaining)
            portfolio.current_account -= use
            remaining -= use
            history.append(("sell", ticker, qty, proceeds, use))
            print(f"Sale proceeds: {proceeds:,.2f} (used {use:,.2f}), remaining: {remaining:,.2f}")
            continue

        if op == "done":
            if remaining > 0:
                print(f"Still need {remaining:,.2f}.")
            else:
                break
            continue

        print("Unknown command. Type 'help'.")

    print(f">>> {portfolio.name}: Withdrawal request fully covered. <<<")
    return {"request": request, "actions": history}

def simulate_game(price_csv_path: str, seed: int = 1234, rounds: int = 3, out_prefix: str = "game_output"):
    random.seed(seed)
    df = load_price_table(price_csv_path)
    tickers = [c for c in df.columns if c != "date"]
    if len(tickers) < 3:
        raise ValueError("CSV must have at least 3 securities.")

    # Select evenly spaced rows for t0..tN
    step = max(1, (len(df)-1)//rounds)
    round_rows = [i*step for i in range(rounds+1)]
    round_rows[-1] = min(round_rows[-1], len(df)-1)

    portfolios = init_portfolios(tickers[:3])
    scoreboard, logs = [], {p.name: [] for p in portfolios}

    for r in range(rounds):
        t1 = round_rows[r+1]
        prices = {t: df.loc[t1, t] for t in tickers[:3]}
        print(f"\n=== Round {r+1} | Price Date: {df.loc[t1, 'date']} ===")
        print("Prices:", {t: round(prices[t],4) for t in tickers[:3]})

        for p in portfolios:
            mv = p.market_value(prices)
            req = generate_withdrawal(r, mv, random)
            log = settle_round(p, prices, req)
            logs[p.name].append({"round": r+1, "price_date": str(df.loc[t1, 'date']), **log})

    final_prices = {t: df.loc[round_rows[-1], t] for t in tickers[:3]}
    for p in portfolios:
        s = p.summary(final_prices)
        scoreboard.append({"group": p.name, "final_price_date": str(df.loc[round_rows[-1], 'date']), **s})

    score_df = pd.DataFrame(scoreboard)
    score_df.to_csv(f"{out_prefix}_scoreboard.csv", index=False)
    with open(f"{out_prefix}_logs.json", "w") as f:
        json.dump(logs, f, indent=2)

    print("\n=== GAME OVER ===")
    print(score_df.to_string(index=False))
    print(f"Saved: {out_prefix}_scoreboard.csv and {out_prefix}_logs.json")

def main():
    ap = argparse.ArgumentParser(description="Liquidity Tranche Simulation Game (CLI)")
    ap.add_argument("--prices", type=str, required=True, help="CSV: date + ≥3 securities")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out_prefix", type=str, default="game_output")
    args = ap.parse_args()
    simulate_game(args.prices, args.seed, args.rounds, args.out_prefix)

if __name__ == "__main__":
    main()
