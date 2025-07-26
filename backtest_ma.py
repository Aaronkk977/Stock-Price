import yfinance as yf, pandas as pd
from backtesting import Backtest, Strategy
import matplotlib.pyplot as plt

TICKERS = ["BTC-USD", "ETH-USD", "ADA-USD"]
START   = "2020-01-01"
CASH    = 100_000_000          # 每檔初始資金

fast, slow = 7, 30          # 均線參數
stats_list, curves = [], []  # 存績效與淨值曲線

# ────────────────── 1. 迴圈逐檔回測 ──────────────────
for tk in TICKERS:
    df = yf.download(tk, start=START, auto_adjust=False) #download data from Yahoo Finance
    df = df.droplevel('Ticker', axis=1)              # 扁平欄位
    df = df[['Open','High','Low','Close','Volume']].dropna()
    df['ma_fast'] = df['Close'].rolling(fast).mean()
    df['ma_slow'] = df['Close'].rolling(slow).mean()

    class SmaCross(Strategy):
        def init(self): pass
        def next(self):
            if self.data.ma_fast[-2] < self.data.ma_slow[-2] and self.data.ma_fast[-1] > self.data.ma_slow[-1]:
                if self.position.is_short:
                    self.position.close()
                self.buy()
            elif self.position and self.data.ma_fast[-2] > self.data.ma_slow[-2] and self.data.ma_fast[-1] < self.data.ma_slow[-1]:
                if self.position.is_long:
                    self.position.close()
                self.sell()

    bt     = Backtest(df, SmaCross, cash=CASH, commission=0.0001) # TW STOCK COMMISSION: 0.001425
    result = bt.run()
    stats_list.append(result)               # 收集指標
    curves.append(result['_equity_curve'])  # 收集淨值

# ────────────────── 2. 合併並畫組合淨值 ──────────────────
equity_df = pd.concat({tk: c['Equity'] for tk, c in zip(TICKERS, curves)}, axis=1)
equity_df = equity_df.ffill().dropna(how="all") 
total = equity_df.sum(axis=1)
total.plot(title="Equal‑Weight Portfolio", figsize=(9,4))
plt.ylabel("Equity ($)")
plt.savefig('equity_curve.png', dpi=150)


# ────────────────── 3. 輸出彙總績效表 ──────────────────
summary = pd.DataFrame({
    'Ticker'        : TICKERS,
    'Ann.Return[%]' : [s['Return (Ann.) [%]']        for s in stats_list],
    'MaxDD[%]'      : [s['Max. Drawdown [%]']        for s in stats_list],
    'Sharpe'        : [s['Sharpe Ratio']             for s in stats_list],
    '#Trades'       : [s['# Trades']                 for s in stats_list],
}).set_index('Ticker').round(2)

print("\n=== 個股績效彙總 ===\n", summary)

ann_ret_port = (total.iloc[-1]/total.iloc[0])**(252/len(total)) - 1
print(f"\n組合年化報酬率 ≈ {ann_ret_port*100:.2f}%")
