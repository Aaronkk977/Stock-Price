#!/usr/bin/env python3
# dca_backtest.py
# 2025‑07‑25 Rubin 專用小工具

import argparse, sys, numpy as np, pandas as pd, yfinance as yf
from math import sqrt

FREQ_MAP = {
    "daily": "B",      # Business day
    "weekly": "W-FRI",
    "monthly": "ME"     # Month end
}

def simulate_dca(ticker, start, end, amount, freq):
    # 1. 抓歷史收盤價
    price = yf.download(ticker, start=start, end=end, auto_adjust=False)["Close"]
    price = price.squeeze()                # ← 這行強制 DataFrame→Series
    price = price.dropna()


    # 2. 產生扣款日索引
    schedule = pd.date_range(start, end, freq=FREQ_MAP[freq])
    schedule = price.reindex(schedule).ffill()          # 若剛好非交易日 → 用最近收盤價
    schedule = schedule.dropna()

    # 3. 模擬定期投入
    shares_bought = amount / schedule                   # 每次投入可買到的股數
    total_shares  = shares_bought.cumsum()
    invested      = amount * np.arange(1, len(schedule)+1)

    # 4. 計算績效
    final_price   = price.iloc[-1]
    final_value   = total_shares.iloc[-1] * final_price
    total_return  = (final_value - invested[-1]) / invested[-1]
    years         = (price.index[-1] - price.index[0]).days / 365.25
    cagr          = (final_value / invested[-1])**(1/years) - 1

    # 年化波動 & Sharpe
    daily_ret = price.pct_change().dropna()
    ann_vol   = daily_ret.std() * sqrt(252)
    sharpe    = (cagr - 0) / ann_vol if ann_vol != 0 else np.nan

    return {
        "invested": invested[-1],
        "shares":   total_shares.iloc[-1],
        "final_val":final_value,
        "total_ret":total_return,
        "cagr":     cagr,
        "sharpe":   sharpe
    }

def prompt_date(name):
    while True:
        d = input(f"請輸入{name} (YYYY-MM-DD): ").strip()
        try:
            pd.to_datetime(d)
            return d
        except:
            print("格式錯誤，請重新輸入")

def prompt_freq():
    while True:
        f = input("請選擇頻率 [daily/weekly/monthly] (預設 monthly): ").strip().lower()
        if f=="" or f in FREQ_MAP:
            return f or "monthly"
        print("只接受 daily、weekly 或 monthly")

def main():
    # p = argparse.ArgumentParser(description="DCA Calculator")
    # p.add_argument("ticker", help="股票代號，如 2330.TW / AAPL")
    # p.add_argument("--start", required=True, help="YYYY-MM-DD")
    # p.add_argument("--end",   required=True, help="YYYY-MM-DD")
    # p.add_argument("--amount", type=float, default=10000, help="每期投入金額")
    # p.add_argument("--freq",   choices=["daily","weekly","monthly"], default="monthly")
    # args = p.parse_args()

    # res = simulate_dca(args.ticker, args.start, args.end, args.amount, args.freq)

    # print(f"\n=== DCA Backtest ({args.ticker}) ===")
    # print(f"期間           : {args.start} → {args.end}")
    # print(f"投入頻率       : {args.freq}  每期投入 {args.amount:,.0f}")
    # print(f"投入期數       : {int(res['invested']/args.amount)}")
    # print(f"總投入         : {res['invested']:,.0f}")
    # print(f"持股數         : {res['shares']:.4f}")
    # print(f"期末市值       : {res['final_val']:,.0f}")
    # print(f"總報酬率       : {res['total_ret']*100:,.2f}%")
    # print(f"CAGR           : {res['cagr']*100:,.2f}%")
    # print(f"Sharpe Ratio   : {res['sharpe']:.2f}\n")

    ticker = input("請輸入股票代號 (如 2330.TW / AAPL): ").strip()
    start  = prompt_date("回測起始日")
    end    = prompt_date("回測結束日")
    amt_in = input("每期投入金額 (預設 10000): ").strip()
    amount = float(amt_in) if amt_in else 10000.0
    freq   = prompt_freq()

    res = simulate_dca(ticker, start, end, amount, freq)

    print(f"""
        === DCA 回測結果 ({ticker}) ===
        期間         : {start} → {end}
        每期投入     : {amount:,.0f}  頻率: {freq}
        期數         : {int(res['invested']/amount)}  總投入: {res['invested']:,.0f}
        持股數       : {res['shares']:.4f}     
        期末市值     : {res['final_val']:,.0f}
        總報酬率     : {res['total_ret']*100:.2f}%
        CAGR         : {res['cagr']*100:.2f}%
        Sharpe Ratio : {res['sharpe']:.2f}
    """)

if __name__ == "__main__":
    sys.exit(main())
