# walk forward

from backtesting import Backtest, Strategy
import pandas as pd, yfinance as yf

def walk_forward(bt_cls, data, train_years=3, test_years=1,
                 opt_grid=None, cash=1e6, commission=0.0015):
    """
    bt_cls     : 你的 Strategy 類別，如 SmaCross
    data       : 完整 DataFrame (index=日期)
    train_years, test_years : 訓練/測試期長度 (年)
    opt_grid   : dict 傳給 Backtest.optimize；若 None 就不做優化
    回傳       : 每段測試期的 stats 列表
    """
    results = []
    cursor = data.index.min()

    while True:
        train_end = cursor + pd.DateOffset(years=train_years)
        test_end  = train_end + pd.DateOffset(years=test_years)
        train = data.loc[cursor:train_end - pd.Timedelta(days=1)]
        test  = data.loc[train_end:test_end - pd.Timedelta(days=1)]
        if len(test) < 30:      # 測試集不足 30 根 K 線就停
            break

        # 1) 在訓練集優化或直接跑
        bt_train = Backtest(train, bt_cls, cash=cash, commission=commission)
        if opt_grid:
            best = bt_train.optimize(return_heatmap=False, **opt_grid)
            params = {'fast': best._strategy.fast, 'slow': best._strategy.slow}
        else:
            params = {}
        print(f"Training from {train.index[0]} to {train.index[-1]} with params: {params}")

        # 2) 用最佳參數在測試集跑
        bt_test = Backtest(test, bt_cls, cash=cash, commission=commission)
        stats = bt_test.run(**params)   
        stats['fast'] = params.get('fast')
        stats['slow'] = params.get('slow')
        results.append(stats)
        
        cursor = train_end                # 視窗往前推

    return pd.DataFrame(results)