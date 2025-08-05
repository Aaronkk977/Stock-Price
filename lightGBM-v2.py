"""
lightGBM_multiclass.py  ────────────────────────────────────────────────
A clean, end‑to‑end example that predicts **short (0) / flat (1) / long (2)**
5‑day BTC returns with an ARIMA residual feature + LightGBM multiclass model.
The script downloads data, engineers features, trains with a walk‑forward
split, evaluates classification metrics, and back‑tests a simple one‑day‑ahead
position strategy.  (Python ≥3.9, LightGBM ≥4.2.0, statsmodels ≥0.14)
"""

# ╭─ Imports ───────────────────────────────────────────────────────────────╮
import pandas as pd, numpy as np, lightgbm as lgb, yfinance as yf, ta
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split
from lightgbm import early_stopping, log_evaluation
import matplotlib.pyplot as plt
from collections import Counter

# ╭─ 1. Download BTC + macro data ──────────────────────────────────────────╮
ticker, start_date = "BTC-USD", "2015-01-01"
df = (
    yf.download(ticker, start=start_date, auto_adjust=True)
    .droplevel("Ticker", axis=1)
    .loc[:, ["Open", "High", "Low", "Close", "Volume"]]
    .dropna()
)

# DXY + S&P 500 (macro context)
dxy = yf.download("DX-Y.NYB", start=start_date, auto_adjust=True)[["Close"]]
sp500 = yf.download("^GSPC", start=start_date, auto_adjust=True)[["Close"]]
dxy.columns, sp500.columns = ["DXY"], ["SP500"]
df = df.join([dxy, sp500], how="left").ffill()

# ╭─ 2. Technical & macro features ─────────────────────────────────────────╮
df["MA7"]   = df["Close"].rolling(7).mean()
df["MA30"]  = df["Close"].rolling(30).mean()
df["MA60"]  = df["Close"].rolling(60).mean()
df["MA180"] = df["Close"].rolling(180).mean()
df["MA_dev"] = df["MA7"] / df["MA30"] - 1

for win in [1, 7, 30, 60]:
    df[f"Ret_{win}d"] = df["Close"].pct_change(win)

df["RSI14"] = ta.momentum.rsi(df["Close"], window=14)
stoch = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"], 14, 3)
df["Stoch_%K"] = stoch.stoch()
df["Stoch_%D"] = stoch.stoch_signal()

df["Vol_z"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / df["Volume"].rolling(20).std()

df["DXY_ret_20d"] = df["DXY"].pct_change(20)
df["DXY_ret_60d"] = df["DXY"].pct_change(60)
df["DXY_z"] = (df["DXY"] - df["DXY"].rolling(60).mean()) / df["DXY"].rolling(60).std()

df["SP500_ret_10d"] = df["SP500"].pct_change(10)
df["SP500_z"] = (df["SP500"] - df["SP500"].rolling(20).mean()) / df["SP500"].rolling(20).std()

# Bitcoin halving cycle (phase in [0,1])
halving_dates = pd.to_datetime(["2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20"])
last_halving = halving_dates.searchsorted(df.index, side="right") - 1
last_halving = halving_dates[last_halving]
df["halving_phase"] = (df.index - last_halving).days / 1458

# ╭─ 2‑b. ARIMA residual feature (walk‑forward) ────────────────────────────╮
df[["ARIMA_pred", "ARIMA_resid"]] = np.nan, np.nan
lookback, order = 365, (5, 1, 0)
for i in range(lookback, len(df)):
    train_y = df["Close"].iloc[i - lookback : i].asfreq("B")  # business‑day frequency
    model = SARIMAX(train_y, order=order, enforce_stationarity=False, enforce_invertibility=False)
    res   = model.fit(disp=False)
    pred  = res.forecast(1).iloc[0]
    df.loc[df.index[i], ["ARIMA_pred", "ARIMA_resid"]] = [pred, df["Close"].iloc[i] - pred]

# ╭─ 3. Label engineering: 5‑day forward return into 3 classes ─────────────╮
pred_horizon = 5
fwd_ret = df["Close"].shift(-pred_horizon) / df["Close"] - 1
upper_th, lower_th = 0.02, -0.02

df["y"] = np.select(
    [fwd_ret < lower_th, (fwd_ret >= lower_th) & (fwd_ret <= upper_th), fwd_ret > upper_th],
    [0, 1, 2],
    default=np.nan,
)

# ╭─ 4. Final dataset ----------------------------------------------------------------╮
feature_cols = [
    "MA7", "MA30", "MA60", "MA180", "MA_dev",
    "Ret_1d", "Ret_7d", "Ret_30d", "Ret_60d",
    "RSI14", "Stoch_%K", "Stoch_%D", "Vol_z",
    "DXY_ret_20d", "DXY_ret_60d", "DXY_z",
    "SP500_ret_10d", "SP500_z", "halving_phase",
    "ARIMA_pred", "ARIMA_resid",
]

# Drop rows with any missing value in features or label
df = df.dropna(subset=feature_cols + ["y"])
X, y = df[feature_cols], df["y"].astype(int)

# ╭─ 5. Walk‑forward hold‑out (80 % / 20 %) -----------------------------------------╮
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, shuffle=False)
# class_weight inversely proportional to freq
cnt = Counter(y_train)
total = sum(cnt.values())
class_weight = {cls: total / (len(cnt) * n) for cls, n in cnt.items()}

# ╭─ 6. LightGBM multiclass model ----------------------------------------------------╮
lgb_params = dict(
    objective="multiclass",
    num_class=3,
    metric=["multi_logloss", "multi_error"],
    learning_rate=0.01,
    num_leaves=64,
    max_depth=3,
    feature_fraction=0.7,
    bagging_fraction=0.7,
    bagging_freq=5,
    min_data_in_leaf=40,
    min_gain_to_split=0.1,
    verbose=-1,
)
model = lgb.LGBMClassifier(**lgb_params)
model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="multi_logloss",
    callbacks=[early_stopping(150), log_evaluation(50)],
)

# ╭─ 7. Validation metrics -----------------------------------------------------------╮
CONF = 0.45
proba_valid = model.predict_proba(X_valid)
y_pred_val = np.full(len(proba_valid), 1)
y_pred_val[proba_valid[:, 2] >= CONF] = 2  # long
y_pred_val[proba_valid[:, 0] >= CONF] = 0  # short
print("\nValidation report (20 % hold‑out):")
print(classification_report(y_valid, y_pred_val, digits=4))
print("Confusion‑matrix:\n", confusion_matrix(y_valid, y_pred_val))

# Optional ROC plot (class‑wise against class 2="Long")
fig, ax = plt.subplots()
for cls in [0, 1, 2]:
    fpr, tpr, _ = roc_curve((y_valid == cls).astype(int), proba_valid[:, cls])
    RocCurveDisplay(fpr=fpr, tpr=tpr, estimator_name=f"class {cls}").plot(ax=ax)
ax.set_title("ROC curves by class (valid set)")
plt.savefig("roc_valid.png", dpi=120)
plt.close()

# ╭─ 8. Simple next‑day strategy back‑test ------------------------------------------╮
proba_all = model.predict_proba(X)
pred_all  = proba_all.argmax(axis=1)

position = (
    pd.Series(pred_all, index=df.index)
    .map({0: -1, 1: 0, 2: 1})  # short / flat / long
    .shift(1)                  # trade executed next day
    .fillna(0.0)
)

daily_ret = df["Close"].pct_change().fillna(0)
strategy_ret = position * daily_ret

equity_curve = (1 + strategy_ret).cumprod()
bench_curve  = (1 + daily_ret).cumprod()

ann_factor = 252 / len(equity_curve)
ann_ret_strategy = equity_curve.iloc[-1] ** ann_factor - 1
ann_ret_bench    = bench_curve.iloc[-1] ** ann_factor - 1
sharpe_strategy  = strategy_ret.mean() / strategy_ret.std() * np.sqrt(252)
max_dd_strategy  = 1 - equity_curve.div(equity_curve.cummax()).min()

print("\n=== Strategy vs Buy&Hold (BTC) ===")
print(
    f"Strategy  →  Ann Ret: {ann_ret_strategy:6.2%}  |  Sharpe: {sharpe_strategy:4.2f}  |  MaxDD: {max_dd_strategy:6.2%}"
)
print(f"Benchmark →  Ann Ret: {ann_ret_bench:6.2%}\n")

# ╭─ 9. Save probability histogram ---------------------------------------------------╮
plt.hist(proba_valid[:, 2], bins=50, edgecolor="k")
plt.title("Predicted P(Long) on validation set")
plt.xlabel("Probability of class = 2 (Long)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("prob_hist.png", dpi=120)
plt.close()
