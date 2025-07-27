import yfinance as yf
import requests, json, datetime as dt
import pandas as pd
import numpy as np
import ta                # Calculate RSI、Stochastic 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

# ----------------------------#
# 1. 下載資料
# ----------------------------#
ticker = "BTC-USD" # modify as needed
start_date = "2017-01-01"  
print(f"ticker: {ticker}, start_date: {start_date}")

df = yf.download(ticker, start=start_date, end=None, auto_adjust=True)
df = df.droplevel('Ticker', axis=1)  
df = df[['Open','High','Low','Close','Volume']].dropna()

dxy = yf.download("DX-Y.NYB", start=start_date, auto_adjust=True)[["Close"]]
dxy.columns = ["DXY"]
df = df.join(dxy, how="left").ffill()      # 與交易日對齊，缺值用前值填

sp500 = yf.download("^GSPC", start=start_date, auto_adjust=True)[["Close"]]
sp500.columns = ["SP500"]
df = df.join(sp500, how="left").ffill()    # 加入 S&P 500 指數作為參考

# ----------------------------#
# 2. 建立技術指標 (Features)
# ----------------------------#
# (a) 移動平均
df["MA7"]  = df["Close"].rolling(7).mean()
df["MA30"] = df["Close"].rolling(30).mean()
df["MA60"] = df["Close"].rolling(60).mean()
df["MA180"] = df["Close"].rolling(180).mean()

# (b) MA 乖離率
df["MA_dev"] = df["MA7"] / df["MA30"] - 1

# (c) 近期報酬率
df["Ret_1d"]  = df["Close"].pct_change(1)
df["Ret_7d"]  = df["Close"].pct_change(7)
df["Ret_30d"] = df["Close"].pct_change(30)
df["Ret_60d"] = df["Close"].pct_change(60)


# (d) RSI 與 Stochastic
df["RSI14"] = ta.momentum.rsi(df["Close"], window=14)
stoch = ta.momentum.StochasticOscillator(
    high=df["High"], low=df["Low"], close=df["Close"], window=14, smooth_window=3
)
df["Stoch_%K"] = stoch.stoch()
df["Stoch_%D"] = stoch.stoch_signal()

# (e) 成交量標準化（與同檔歷史量比較）
df["Vol_z"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / df["Volume"].rolling(20).std()

# (f) 美元指數
df["DXY_ret_20d"] = df["DXY"].pct_change(20)     
df["DXY_ret_60d"] = df["DXY"].pct_change(60)      # 60 日美元指數漲跌
df["DXY_z"]      = (df["DXY"] - df["DXY"].rolling(60).mean()) / df["DXY"].rolling(60).std()

# (g) S&P 500 指數報酬率
df["SP500_ret_10d"] = df["SP500"].pct_change(10)  # 10 日 S&P 500 漲跌
df["SP500_z"] = (df["SP500"] - df["SP500"].rolling(60).mean()) / df["SP500"].rolling(60).std()


# (h) 減半週期
halving_dates = pd.to_datetime([
    "2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20"   # :contentReference[oaicite:0]{index=0}
])
last_halving = halving_dates.searchsorted(df.index, side="right") - 1
last_halving = halving_dates[last_halving]
cycle_day  = (df.index - last_halving).days                 # 天數 0…1458
cycle_len  = 1458                                           # 約四年
df["halving_phase"]   = cycle_day / cycle_len               # 0~1 連續值
df["halving_phase^5"] = df["halving_phase"]**5              # 捕捉非線性


# ----------------------------#
# 3. 目標變數 (Label)
# ----------------------------#
N = 30                            # modify as needed
df[f"future_ret_{N}d"] = df["Close"].shift(-N).pct_change(N, fill_method=None)
df["y"] = (df[f"future_ret_{N}d"] > 0.20).astype(int)

# ----------------------------#
# 4. 資料清洗
# ----------------------------#
feature_cols = ["MA7", "MA30", "MA60", "MA180", "MA_dev",
                "Ret_1d", "Ret_7d", "Ret_30d", "Ret_60d",
                "RSI14", "Stoch_%K", "Stoch_%D", "Vol_z",
                "DXY_ret_20d", "DXY_ret_60d", "DXY_z", "SP500_ret_10d", "SP500_z",
                "halving_phase", "halving_phase^5"]

df = df.dropna(subset=feature_cols + ["y"])   # 去掉指標缺值
X = df[feature_cols].copy()
y = df["y"].copy()

# 標準化（對 Logistic Regression 很重要）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------#
# 5. 切分訓練 / 測試
# ----------------------------#
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, shuffle=False  # 不隨機打亂才能保持時序
)

# ----------------------------#
# 6. 訓練模型
# ----------------------------#
clf = LogisticRegression(max_iter=7500, class_weight="balanced")
clf.fit(X_train, y_train)

# ----------------------------#
# 7. 評估模型
# ----------------------------#
threshold = 0.31        # modify threshold
y_prob = clf.predict_proba(X_test)[:, 1]
y_pred = (y_prob > threshold).astype(int)

# show predicted probabilities distribution
import matplotlib.pyplot as plt
plt.hist(y_prob, bins=30, edgecolor="k")
plt.title("Distribution of Predicted Probabilities (class=1)")
plt.xlabel("Predicted probability")
plt.ylabel("Frequency")
plt.savefig("predicted_probabilities_distribution.png")

print(classification_report(y_test, y_pred, digits=4))
print("AUC :", round(roc_auc_score(y_test, y_prob), 4))

# ----------------------------#
# 8. (可選) 觀察特徵係數
# ----------------------------#
coef_df = pd.Series(clf.coef_[0], index=feature_cols).sort_values()
print("\nLogistic Regression Coefficients：")
print(coef_df)


# === backtest ===
prob = clf.predict_proba(X_scaled)[:,1]          # 全期間機率

prob_s = pd.Series(prob, index=df.index)

# 2. 區間分層 → 對應倉位（labels 要轉成 float 才能乘報酬）
bins   = [0.0, threshold, 0.5, 1.0]      # 機率分段
labels = [0.0, 0.9, 1.0]              # 各段對應倉位
weight = pd.cut(prob_s, bins=bins, labels=labels).astype(float)

# 3. 形成持倉：隔日開倉 + 持有 30 天
hold_days = 30
position = weight.shift(1).rolling(hold_days).max()


daily_ret = df["Close"].pct_change()
strategy_ret = position * daily_ret

# 年化績效
ann_ret   = (1 + strategy_ret).prod() ** (252 / len(strategy_ret)) - 1
sharpe    = strategy_ret.mean() / strategy_ret.std() * np.sqrt(252)
max_dd    = (equity := (1+strategy_ret).cumprod()).div(equity.cummax()).min()
print("\n=== Strategy Performance ===")
print(f"年化:{ann_ret:.2%}  Sharpe:{sharpe:.2f}  MaxDD:{max_dd:.2%}")
print("\n=== Buy-and-Hold Performance ===")
bh_ret   = (1 + daily_ret).prod() ** (252 / len(daily_ret)) - 1
bh_sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252)
bh_max_dd = (bh_equity := (1+daily_ret).cumprod()).div(bh_equity.cummax()).min()
print(f"年化:{bh_ret:.2%}  Sharpe:{bh_sharpe:.2f}  MaxDD:{bh_max_dd:.2%}")
