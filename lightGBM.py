import yfinance as yf, pandas as pd, numpy as np, lightgbm as lgb, ta
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from lightgbm import early_stopping, log_evaluation 

# ---------------- 1. 下載資料 ---------------- #
ticker, start_date = "BTC-USD", "2017-01-01"
df = yf.download(ticker, start=start_date, auto_adjust=True).droplevel("Ticker", axis=1)
df = df[["Open","High","Low","Close","Volume"]].dropna()

dxy   = yf.download("DX-Y.NYB", start=start_date, auto_adjust=True)[["Close"]]
sp500 = yf.download("^GSPC",   start=start_date, auto_adjust=True)[["Close"]]
dxy.columns, sp500.columns = ["DXY"], ["SP500"]
df = df.join([dxy, sp500], how="left").ffill()

# ---------------- 2. 技術指標 ---------------- #
df["MA7"] = df["Close"].rolling(7).mean()
df["MA30"] = df["Close"].rolling(30).mean()
df["MA60"] = df["Close"].rolling(60).mean()
df["MA180"] = df["Close"].rolling(180).mean()
df["MA_dev"] = df["MA7"] / df["MA30"] - 1

df["Ret_1d"]  = df["Close"].pct_change(1)
df["Ret_7d"]  = df["Close"].pct_change(7)
df["Ret_30d"] = df["Close"].pct_change(30)
df["Ret_60d"] = df["Close"].pct_change(60)

df["RSI14"] = ta.momentum.rsi(df["Close"], window=14)
stoch = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"], 14, 3)
df["Stoch_%K"], df["Stoch_%D"] = stoch.stoch(), stoch.stoch_signal()

df["Vol_z"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / df["Volume"].rolling(20).std()

df["DXY_ret_20d"] = df["DXY"].pct_change(20)
df["DXY_ret_60d"] = df["DXY"].pct_change(60)
df["DXY_z"]       = (df["DXY"] - df["DXY"].rolling(60).mean()) / df["DXY"].rolling(60).std()

df["SP500_ret_10d"] = df["SP500"].pct_change(10)
df["SP500_z"]       = (df["SP500"] - df["SP500"].rolling(20).mean()) / df["SP500"].rolling(20).std()

halving_dates = pd.to_datetime(["2012-11-28","2016-07-09","2020-05-11","2024-04-20"])
last_halving = halving_dates.searchsorted(df.index, side="right") - 1
last_halving = halving_dates[last_halving]
cycle_day = (df.index - last_halving).days
df["halving_phase"]   = cycle_day / 1458

# ---------------- 3. 分位標籤 (Top/Bottom 30%) ---------------- #
N = 30
ret_col = f"future_ret_{N}d"
df[ret_col] = df["Close"].shift(-N).pct_change(N, fill_method=None)
q25, q75 = df[ret_col].quantile([0.25, 0.75])
df["y"] = np.where(df[ret_col] >= q75, 1,
           np.where(df[ret_col] <= q25, 0, np.nan))

feature_cols = ["MA7","MA30","MA60","MA180","MA_dev",
                "Ret_1d","Ret_7d","Ret_30d","Ret_60d",
                "RSI14","Stoch_%K","Stoch_%D","Vol_z",
                "DXY_ret_20d","DXY_ret_60d","DXY_z",
                "SP500_ret_10d","SP500_z","halving_phase"]

df = df.dropna(subset=feature_cols + ["y"])
X, y = df[feature_cols], df["y"].astype(int)

# ---------------- 4. 時序切分 ---------------- #
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, shuffle=False)

# ---------------- 5. LightGBM ---------------- #
params = dict(
    objective          = "binary",
    metric             = "auc",
    learning_rate      = 0.01,
    num_leaves         = 24,
    feature_fraction   = 0.3,
    bagging_fraction   = 0.7,
    bagging_freq       = 5,
    min_data_in_leaf   = 40,
    min_gain_to_split  = 0.1,
    verbose            = -1           # 不加 is_unbalance，因類別已近 1:1
)

train_set = lgb.Dataset(X_train, y_train)
valid_set = lgb.Dataset(X_valid, y_valid)

model = lgb.train(
    params,
    train_set,
    num_boost_round=4000,
    valid_names=["train", "valid"],
    valid_sets=[train_set, valid_set],
    callbacks=[
        early_stopping(stopping_rounds=150),   # 
        log_evaluation(period=100)             # == verbose_eval=100
    ],
)

print("Best iter :", model.best_iteration)
print("Train AUC :", model.best_score["train"]["auc"])
print("Valid AUC :", model.best_score["valid"]["auc"])

# ---------------- 6. 測試集評估 ---------------- #
y_prob = model.predict(X_valid)
threshold = 0.3
y_pred = (y_prob > threshold).astype(int)       
print(classification_report(y_valid, y_pred, digits=4))

imp = pd.Series(model.feature_importance(), index=feature_cols).sort_values(ascending=False)
print("\nTop 10 feature importance:")
print(imp.head(10))

# ---------------- 7. 回測：連續倉位 ---------------- #
prob_all = model.predict(X)
weight   = ((prob_all - 0.35).clip(0, 0.25)/0.25) * 1   
position = pd.Series(weight, index=df.index).shift(1).rolling(N).max()

daily_ret = df["Close"].pct_change()
strategy_ret = position * daily_ret

ann_ret = (1+strategy_ret).prod()**(252/len(strategy_ret))-1
sharpe  = strategy_ret.mean()/strategy_ret.std()*np.sqrt(252)
max_dd  = (equity:=(1+strategy_ret).cumprod()).div(equity.cummax()).min()
print(f"\n=== Strategy Performance === \nAnn [%]:{ann_ret:.2%}  Sharpe:{sharpe:.2f}  MaxDD:{max_dd:.2%}")

# compare with buy & hold
daily_ret = df["Close"].pct_change()

equity = (1 + daily_ret).cumprod()

ann_ret = equity.iloc[-1]**(252/len(equity)) - 1
sharpe  = daily_ret.mean() / daily_ret.std() * np.sqrt(252)
max_dd = 1 - (equity / equity.cummax()).min()

print("\n=== Buy & Hold Performance ===")
print(f"Ann [%]:{ann_ret:.2%}  Sharpe:{sharpe:.2f}  MaxDD:{max_dd:.2%}")

# ---------------- 8. 機率分佈圖 ---------------- #
import matplotlib.pyplot as plt
plt.hist(y_prob, bins=60, edgecolor="k")
plt.title("Predicted probability (valid set)")
plt.xlabel("prob(class=1)"); plt.ylabel("count")
plt.savefig("prob_hist.png")


