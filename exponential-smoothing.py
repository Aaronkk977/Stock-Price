import warnings, yfinance as yf, pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_percentage_error, confusion_matrix
import numpy as np, pandas as pd
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")

# 1. 下載 Close 價，保證一維
price = (
    yf.download("BTC-USD", start="2017-01-01", end="2025-08-06",
                auto_adjust=False)["Close"]
      .dropna()
      .asfreq("D")          # 指定日頻
)

# 2. 切分
train, test = price[:-180], price[-180:]

# 3. 找 (p,d,q)
p, d, q = auto_arima(train, seasonal=False, stepwise=True,
                     suppress_warnings=True, max_p=5, max_q=5).order
print(f"Optimal ARIMA order: (p={p}, d={d}, q={q})")

# 4. 滾動預測
history = train.copy()
preds = []

for ts in test.index:
    pred = ARIMA(history, order=(p, d, q)).fit().forecast().iloc[0]
    preds.append(pred)
    history.loc[ts] = test.loc[ts]     # 🔑 保持 history 為 Series

print("MAPE =", mean_absolute_percentage_error(test, preds))

# 6. 繪圖比較 -----------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(price.index, price, label="True Price")
plt.plot(test.index, preds, label="ARIMA forecast", color='orange')
plt.title("BTC-USD Closing Price vs. ARIMA Forecast")
plt.legend()
plt.savefig("arima_forecast.png")

# 7. 未來 n 天預測 (例：預測未來一週)
n_steps = 7
future_model = ARIMA(price, order=(p, d, q)).fit()
future_forecast = future_model.forecast(steps=n_steps)
print("Future 7 days forecast:")
print(future_forecast)
