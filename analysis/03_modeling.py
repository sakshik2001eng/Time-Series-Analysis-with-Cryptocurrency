# %% [markdown]
# # 03 - Modelling and evaluation
# We compare three forecasters. The key idea: a model is only useful if it beats the
# NAIVE baseline ("tomorrow = today") on data it has never seen.

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import load_prices
from src.models import (time_split, naive_forecast, arima_forecast, select_arima_order,
                        prophet_forecast, evaluate, walk_forward)
from src.utils import save_fig, ROOT

TICKER, TEST_DAYS, HORIZON = "BTC-USD", 30, 14
close = load_prices(TICKER, years=5)["Close"]

# %% [markdown]
# ## 1. Chronological train/test split
# Train on the past, test on the most recent 30 days. Never shuffle.

# %%
train, test = time_split(close, TEST_DAYS)
print("Train:", train.index.min().date(), "->", train.index.max().date(), len(train))
print("Test: ", test.index.min().date(), "->", test.index.max().date(), len(test))

# %% [markdown]
# ## 2. Choose ARIMA order using AIC
# We use d = 1 (from notebook 02) and search p, q in 0..3. Lower AIC = better trade-off
# between fit and complexity.

# %%
order, aic = select_arima_order(train, max_p=3, max_q=3, d=1)
print("Best ARIMA order:", order, "| AIC:", round(aic, 1))

# %% [markdown]
# ## 3. Forecast the held-out 30 days with each model

# %%
preds = {
    "Naive": naive_forecast(train, TEST_DAYS),
    f"ARIMA{order}": arima_forecast(train, TEST_DAYS, order),
}
try:
    preds["Prophet"] = prophet_forecast(train, TEST_DAYS)
except ImportError:
    print("Prophet not installed -> skipping (pip install prophet)")

table = pd.DataFrame({name: evaluate(test.values, p) for name, p in preds.items()}).T.round(2)
print(table)

plt.figure(figsize=(12, 5))
plt.plot(train.iloc[-90:], label="Train (last 90d)", color="gray")
plt.plot(test, label="Actual", color="black", linewidth=2)
for name, p in preds.items():
    plt.plot(test.index, p, label=name, linestyle="--")
plt.legend(); plt.title(f"{TICKER}: forecasts vs actual"); save_fig("03_forecasts"); plt.show()

# %% [markdown]
# ## 4. Walk-forward validation (the more trustworthy test)
# One 30-day window can be lucky. Here we run 10 different forecast origins, each predicting
# 14 days ahead, and average the results. 'Beat_Naive_%' = share of folds where the model
# had lower RMSE than the naive baseline.

# %%
forecasters = {
    "Naive": naive_forecast,
    f"ARIMA{order}": lambda tr, h: arima_forecast(tr, h, order),
}
wf = walk_forward(close, forecasters, horizon=HORIZON, n_folds=10)
print(wf)

out = ROOT / "reports"
out.mkdir(exist_ok=True)
wf.to_csv(out / "walk_forward_results.csv")
table.to_csv(out / "holdout_results.csv")

# %% [markdown]
# ## 5. Write your conclusion (in the README)
# Be honest. Typical outcome: ARIMA is roughly equal to the naive baseline, because crypto
# prices behave close to a random walk. Saying this clearly, backed by numbers, is exactly
# what a strong data scientist does.
