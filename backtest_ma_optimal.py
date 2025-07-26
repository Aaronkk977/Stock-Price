from backtesting import Backtest, Strategy
import pandas as pd, yfinance as yf
from helper.walk_forward import walk_forward

# 1. 準備資料
df = yf.download("BTC-USD", "2017-06-01", auto_adjust=False)
df = df.droplevel('Ticker', axis=1)[['Open','High','Low','Close','Volume']].dropna()

# 2. 策略定義（使用 self.I 計算指標）
class SmaCross(Strategy):
    fast = 1
    slow = 10

    @staticmethod
    def _sma(x, n):
        return pd.Series(x).rolling(n).mean()

    def init(self):
        price = self.data.Close
        self.ma_fast = self.I(self._sma, price, self.fast)
        self.ma_slow = self.I(self._sma, price, self.slow)

    def next(self):
        if self.ma_fast[-2] < self.ma_slow[-2] and self.ma_fast[-1] > self.ma_slow[-1]:
            self.buy()
        elif self.position and self.ma_fast[-2] > self.ma_slow[-2] and self.ma_fast[-1] < self.ma_slow[-1]:
            self.position.close()

if __name__ == "__main__":

    opt_grid = dict(
        fast=range(0, 55, 5),
        slow=range(5, 121, 5),
        maximize='Return (Ann.) [%]',
        constraint=lambda p: p.fast < p.slow
    )

    wf = walk_forward(SmaCross, df, train_years=2, test_years=2, opt_grid=opt_grid) # modified train/test years
    pd.set_option('display.max_colwidth', None)
    cols = ['fast', 'slow', 'Return (Ann.) [%]', 'Sharpe Ratio', 'Max. Drawdown [%]']
    print(wf[cols].to_string(index=False))

    print("\n平均年化報酬 :", wf['Return (Ann.) [%]'].mean(), "%")
    print("平均 Sharpe  :", wf['Sharpe Ratio'].mean())





