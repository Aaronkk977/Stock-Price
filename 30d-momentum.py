from openbb import obb   # pip install openbb
import pandas as pd, numpy as np, matplotlib.pyplot as plt

class CryptoBacktester:
    def __init__(self, assets, start='2017-01-01'):
        self.assets = assets

        # new way to load crypto prices
        price_list = []
        for a in assets:
            # 1. 取歷史價格
            resp = obb.crypto.price.historical(
                symbol=a, 
                start_date=start, 
                provider="yfinance"      # safest way to get crypto data
            )
            # 2. 轉 DataFrame → 抽 close 欄
            close = resp.to_df()["close"].rename(a)
            price_list.append(close)

        self.data = pd.concat(price_list, axis=1).dropna()

        self.returns   = self.data.pct_change().dropna()
        self.portfolio = pd.DataFrame(index=self.data.index)

        self.benchmark_ret  = self.data[self.assets[0]].pct_change().fillna(0)  # 以第一個資產為基準
        self.benchmark_curve = (1 + self.benchmark_ret).cumprod()


    def run_strategy(self, signal_fn):
        w = signal_fn(self.returns)
        self.portfolio['strategy_return'] = (w.shift(1)*self.returns).sum(axis=1)
        self.portfolio['cumulative_return'] = (1+self.portfolio['strategy_return']).cumprod()

    def evaluate(self):
        strat = self.portfolio['strategy_return']
        bench = self.benchmark_ret

        def _stats(x):
            ann = (1 + x).prod()**(252/len(x)) - 1
            sharpe = x.mean()/x.std() * np.sqrt(252)
            return ann, sharpe

        ann_s, shp_s = _stats(strat)
        ann_b, shp_b = _stats(bench)

        print(f"=== Strategy ===  Ann: {ann_s:.2%}  Sharpe: {shp_s:.2f}")
        print(f"=== Buy&Hold ===  Ann: {ann_b:.2%}  Sharpe: {shp_b:.2f}")

    def plot(self):
        ax = self.portfolio['cumulative_return'].plot(
            label='Strategy', figsize=(10,4), grid=True)
        self.benchmark_curve.reindex(ax.lines[0].get_xdata()).plot(
            ax=ax, label='Buy & Hold', linestyle='--')
        ax.set_title('Cumulative Return')
        ax.legend()
        save_path = "./backtest_plot.png"
        ax.get_figure().savefig(save_path,
                                dpi=300,               # 解析度
                                bbox_inches='tight')   # 去空白邊
        print(f"圖檔已存為：{save_path}")


def momentum_signal(returns, window=30):
    trailing = returns.rolling(window).sum()
    latest   = trailing.iloc[-1]
    weights  = (latest == latest.max()).astype(int)   # 只買最強者
    return pd.DataFrame([weights]*len(returns), index=returns.index, columns=returns.columns)

bt = CryptoBacktester(['BTC-USD','ETH-USD','BNB-USD', 'XRP-USD', 'SOL-USD', 'ADA-USD', "LTC-USD", 'DOGE-USD'])
bt.run_strategy(momentum_signal)
bt.evaluate();  
bt.plot()