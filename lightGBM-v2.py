import yfinance as yf, pandas as pd, numpy as np, lightgbm as lgb, ta
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from lightgbm import early_stopping, log_evaluation 
from statsmodels.tsa.statespace.sarimax import SARIMAX # ARIMA 

# === Purged CV & Walk-Forward Helpers ===
def _get_dates_index(X):
    if isinstance(X.index, pd.MultiIndex):
        dates = X.index.get_level_values(0)
    else:
        dates = X.index
    if not isinstance(dates, pd.DatetimeIndex):
        dates = pd.to_datetime(dates)
    return dates

def build_purged_folds(X, n_splits=5, purge_days=20, embargo_days=5):
    dates = _get_dates_index(X).normalize()
    unique_days = dates.unique().sort_values()
    n = len(unique_days)
    fold_bounds = np.linspace(0, n, n_splits + 1, dtype=int)
    folds = []
    for k in range(n_splits):
        val_start = unique_days[fold_bounds[k]]
        val_end   = unique_days[fold_bounds[k+1]-1]
        purge_start = val_start - pd.Timedelta(days=purge_days)
        embargo_end = val_end   + pd.Timedelta(days=embargo_days)
        in_val = (dates >= val_start) & (dates <= val_end)
        in_purge_left  = (dates >= purge_start) & (dates <  val_start)
        in_purge_right = (dates >  val_end)     & (dates <= embargo_end)
        valid_idx = np.where(in_val)[0]
        train_idx = np.where(~(in_val | in_purge_left | in_purge_right))[0]
        if len(valid_idx) == 0 or len(train_idx) == 0:
            continue
        folds.append((train_idx, valid_idx))
    return folds

def generate_walkforward_windows(X, train_years=3, test_months=1, purge_days=20, embargo_days=5):
    dates = _get_dates_index(X).normalize()
    unique_days = dates.unique().sort_values()
    start = unique_days.min()
    while True:
        train_end = start + pd.DateOffset(years=train_years) - pd.Timedelta(days=1)
        test_end  = train_end + pd.DateOffset(months=test_months)
        if test_end > unique_days.max():
            break
        val_start = train_end + pd.Timedelta(days=1)
        val_end   = test_end
        purge_start = val_start - pd.Timedelta(days=purge_days)
        embargo_end = val_end   + pd.Timedelta(days=embargo_days)
        in_train = (dates <= train_end) & (dates < purge_start)
        in_test  = (dates >= val_start) & (dates <= val_end)
        in_right_purge = (dates > val_end) & (dates <= embargo_end)
        train_idx = np.where(in_train & (~in_right_purge))[0]
        test_idx  = np.where(in_test)[0]
        if len(train_idx) == 0 or len(test_idx) == 0:
            break
        yield (train_idx, test_idx)
        start = val_start

# -----------------------------------------------------------------------------------------
# ---------------- 1. 下載資料 ---------------- #
ticker, start_date = "BTC-USD", "2015-01-01"
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

# ---------------- 3. 標籤 (Top/Bottom 30%) ---------------- #
N = 20
ret_col = f"future_ret_{N}d"
df[ret_col] = df["Close"].shift(-N).pct_change(N, fill_method=None)
q30, q70 = df[ret_col].quantile([0.30, 0.70])
df["y"] = np.where(df[ret_col] >= q70, 1, np.where(df[ret_col] <= q30, 0, np.nan))

feature_cols = ["MA7","MA30","MA60","MA180","MA_dev",
                "Ret_1d","Ret_7d","Ret_30d","Ret_60d",
                "RSI14","Stoch_%K","Stoch_%D","Vol_z",
                "DXY_ret_20d","DXY_ret_60d","DXY_z",
                "SP500_ret_10d","SP500_z","halving_phase"]

df = df.dropna(subset=feature_cols + ["y"]).copy()
X, y = df[feature_cols], df["y"].astype(int)

# ---------------- 4. Purged Time-Series CV 取得最佳迭代數 ---------------- #
purge_days = N  # 針對標籤前視窗 (20 天) 做 purge，特徵皆使用歷史不需更長
folds = build_purged_folds(X, n_splits=5, purge_days=purge_days, embargo_days=5)

params = dict(
    objective          = "binary",
    metric             = "auc",
    learning_rate      = 0.01,
    num_leaves         = 16,
    max_depth          = 3,
    feature_fraction   = 0.5,
    bagging_fraction   = 0.6,
    bagging_freq       = 5,
    min_data_in_leaf   = 40,
    min_gain_to_split  = 0.1,
    verbose            = -1,
    seed               = 42
)

dtrain_full = lgb.Dataset(X, label=y)
cv_res = lgb.cv(
    params,
    dtrain_full,
    folds=folds,
    num_boost_round=4000,
    stratified=False,
    seed=42,
    callbacks=[early_stopping(stopping_rounds=150), log_evaluation(period=100)]
)
# Robust metric key detection
cv_keys = list(cv_res.keys())
print("CV result keys:", cv_keys)
mean_keys = [k for k in cv_keys if k.endswith('-mean')]
if not mean_keys:  # fallback any key
    mean_keys = cv_keys[:1]
mean_key = mean_keys[0]
std_key_candidates = [k for k in cv_keys if k.startswith(mean_key.rsplit('-mean',1)[0]) and k.endswith('-stdv')]
std_key = std_key_candidates[0] if std_key_candidates else None
best_rounds = len(cv_res[mean_key])
last_mean = cv_res[mean_key][-1]
last_std  = cv_res[std_key][-1] if std_key else float('nan')
print(f"[Purged CV] best_rounds={best_rounds} {mean_key}={last_mean:.4f} ± {last_std:.4f}")

# ---------------- 5. Walk-Forward 生成連續樣本外預測 ---------------- #
oof_pred = np.full(len(X), np.nan)
for i, (tr_idx, te_idx) in enumerate(generate_walkforward_windows(X, train_years=3, test_months=1,
                                                                  purge_days=purge_days, embargo_days=5)):
    dtr = lgb.Dataset(X.iloc[tr_idx], label=y.iloc[tr_idx])
    model = lgb.train(params | dict(metric='None'), dtr, num_boost_round=best_rounds)
    oof_pred[te_idx] = model.predict(X.iloc[te_idx], num_iteration=model.best_iteration)
    if (i+1) % 6 == 0:
        print(f"Walk window {i+1} done")

oof_series = pd.Series(oof_pred, index=X.index, name='oof_prob')
valid_mask = ~oof_series.isna()

# Use label-based indexing (loc) instead of iloc with boolean mask
y_valid_oof = y.loc[valid_mask]
pred_valid  = oof_series.loc[valid_mask]

fpr, tpr, thresh = roc_curve(y_valid_oof, pred_valid)
best_idx = np.argmax(tpr - fpr)
best_th  = thresh[best_idx]
oof_auc = roc_auc_score(y_valid_oof, pred_valid)
print(f"[Walk-Forward OOF] AUC:{oof_auc:.4f}")
print(f"Best threshold (Youden J) = {best_th:.4f}")

# 分類報告 (僅對有 OOF 的期間)
from sklearn.metrics import classification_report as _cr
print(_cr(y_valid_oof, (pred_valid >= best_th).astype(int)))

# ---------------- 6. Backtest 使用 OOF 機率 (連續倉位) ---------------- #
prob_all = oof_series.ffill()  # 使用 ffill() 取代 fillna(method='ffill')
weight   = ((prob_all - best_th).clip(0, 0.025)/0.025)
position_raw = weight.shift(1).fillna(0.0)
ret_daily = df["Close"].pct_change().fillna(0)
ret_raw   = position_raw * ret_daily
curve_raw = (1 + ret_raw).cumprod()

DD_STOP = 0.20
roll_max   = curve_raw.cummax()
stop_mask  = curve_raw < roll_max * (1 - DD_STOP)
stop_shift = stop_mask.shift(1, fill_value=False)  # 避免後續 fillna(bool) 警告
position_dd = position_raw.copy()
position_dd[stop_shift] = 0
ret_dd  = position_dd * ret_daily
equity  = (1 + ret_dd).cumprod()
ann_ret = equity.iloc[-1]**(252/len(equity)) - 1
sharpe  = ret_dd.mean() / ret_dd.std() * np.sqrt(252)
max_dd  = (equity / equity.cummax()).min()
print(f"\n=== Strategy Performance (Walk-Forward OOF) ===\nAnn [%]: {ann_ret:.2%}  Sharpe: {sharpe:.2f}  MaxDD: {max_dd:.2%}")

# Buy & Hold
bh_daily = df["Close"].pct_change()
Bh_equity = (1 + bh_daily).cumprod()
bh_ann = Bh_equity.iloc[-1]**(252/len(Bh_equity)) - 1
bh_sharpe = bh_daily.mean()/bh_daily.std()*np.sqrt(252)
bh_maxdd = 1 - (Bh_equity / Bh_equity.cummax()).min()
print("\n=== Buy & Hold Performance ===")
print(f"Ann [%]:{bh_ann:.2%}  Sharpe:{bh_sharpe:.2f}  MaxDD:{bh_maxdd:.2%}")

# ---------------- 7. 機率分佈圖 (OOF) ---------------- #
import matplotlib.pyplot as plt
plt.hist(oof_series.loc[valid_mask], bins=60, edgecolor="k")
plt.title("OOF Predicted probability")
plt.xlabel(f"prob(class=1), best_th={best_th:.3f}"); plt.ylabel("count")
plt.savefig("prob_hist.png")


