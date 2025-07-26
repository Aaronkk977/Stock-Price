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
ticker = "0050.TW" # modify as needed
start_date = "2020-01-01"  

df = yf.download(ticker, start=start_date, end=None, auto_adjust=True)
df = df.droplevel('Ticker', axis=1)  
df = df[['Open','High','Low','Close','Volume']].dropna()

dxy = yf.download("DX-Y.NYB", start=start_date, auto_adjust=True)[["Close"]]
dxy.columns = ["DXY"]
print("DXY:", dxy.iloc[0:5])
df = df.join(dxy, how="left").ffill()      # 與交易日對齊，缺值用前值填

print("df shape:", df.shape)
# ----------------------------#
# 2. 建立技術指標 (Features)
# ----------------------------#
# (a) 移動平均
df["MA5"]  = df["Close"].rolling(5).mean()
df["MA20"] = df["Close"].rolling(20).mean()
df["MA60"] = df["Close"].rolling(60).mean()

# (b) MA 乖離率
df["MA_dev"] = df["MA5"] / df["MA20"] - 1

# (c) 近期報酬率
df["Ret_1d"]  = df["Close"].pct_change(1)
df["Ret_5d"]  = df["Close"].pct_change(5)
df["Ret_20d"] = df["Close"].pct_change(20)

# (d) RSI 與 Stochastic
df["RSI14"] = ta.momentum.rsi(df["Close"], window=14)
stoch = ta.momentum.StochasticOscillator(
    high=df["High"], low=df["Low"], close=df["Close"], window=14, smooth_window=3
)
df["Stoch_%K"] = stoch.stoch()
df["Stoch_%D"] = stoch.stoch_signal()

# (e) 成交量標準化（與同檔歷史量比較）
df["Vol_z"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / df["Volume"].rolling(20).std()

# --- 美元指數 ---
df["DXY_ret_5d"] = df["DXY"].pct_change(5)        # 5 日美元指數漲跌
df["DXY_z"]      = (df["DXY"] - df["DXY"].rolling(60).mean()) / df["DXY"].rolling(60).std()

# ----------------------------#
# 3. 目標變數 (Label)
# ----------------------------#
N = 20                            # modify as needed
df[f"future_ret_{N}d"] = df["Close"].shift(-N).pct_change(N, fill_method=None)
df["y"] = (df[f"future_ret_{N}d"] > 0).astype(int)

# ----------------------------#
# 4. 資料清洗
# ----------------------------#
feature_cols = ["MA5", "MA20", "MA60", "MA_dev",
                "Ret_1d", "Ret_5d", "Ret_20d",
                "RSI14", "Stoch_%K", "Stoch_%D", "Vol_z",
                "DXY_ret_5d","DXY_z"]

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
clf = LogisticRegression(max_iter=5000, class_weight="balanced")
clf.fit(X_train, y_train)

# ----------------------------#
# 7. 評估模型
# ----------------------------#
threshold = 0.037        # modify threshold
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
