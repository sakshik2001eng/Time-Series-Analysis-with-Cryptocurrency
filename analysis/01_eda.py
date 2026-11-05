# %% [markdown]
# # 01 - Exploratory Data Analysis (EDA)
# GOAL: understand the data BEFORE modelling. Run cell-by-cell in VS Code (click "Run Cell"),
# or run the whole file with:  python analysis/01_eda.py   (from the project root)

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))          # lets us import from src/ (run from project root)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

from src.data_loader import load_prices, add_features
from src.utils import save_fig

TICKER = "BTC-USD"

# %% [markdown]
# ## 1. Load the data and sanity-check it
# Bad data silently ruins models, so always check shape, missing values and date gaps.

# %%
df = add_features(load_prices(TICKER, years=5), window=30)
print(df.shape)
print(df.head())
print("\nMissing values per column:\n", df.isna().sum())

full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
print("\nMissing calendar days:", len(full_range.difference(df.index)))   # crypto trades 7 days/week
print(df["Close"].describe())

# %% [markdown]
# ## 2. Price over time
# Prices trend and explode over time -> a *non-stationary* series (mean and variance change).
# The log scale makes percentage moves comparable across years.

# %%
fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
ax[0].plot(df["Close"], label="Close"); ax[0].plot(df["MA"], label="30-day MA")
ax[0].set_title(f"{TICKER} price"); ax[0].legend()
ax[1].plot(df["Close"]); ax[1].set_yscale("log"); ax[1].set_title("Log scale")
plt.tight_layout(); save_fig("01_price"); plt.show()

# %% [markdown]
# ## 3. Returns
# Returns (daily % change) are what traders care about and are much closer to stationary.
# Look for FAT TAILS: extreme days happen far more often than a normal distribution predicts.

# %%
ret = df["Return"].dropna()
print(f"Mean daily return: {ret.mean():.4f} | Std: {ret.std():.4f}")
print(f"Skewness: {ret.skew():.2f} | Excess kurtosis: {ret.kurt():.2f}  (normal = 0; high = fat tails)")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(ret); ax[0].set_title("Daily log returns")
ax[1].hist(ret, bins=80); ax[1].set_title("Return distribution")
plt.tight_layout(); save_fig("01_returns"); plt.show()

# %% [markdown]
# ## 4. Volatility clustering
# Calm periods and wild periods come in clusters. This is a famous feature of financial data
# and is the reason volatility models (e.g. GARCH) exist.

# %%
plt.figure(figsize=(12, 4))
plt.plot(df["Volatility"]); plt.title("30-day rolling annualised volatility")
save_fig("01_volatility"); plt.show()

# %% [markdown]
# ## 5. Decomposition: trend + seasonality + residual
# We decompose log-price with a weekly period (7). Crypto may show weak weekday effects;
# if the seasonal part is tiny, that is a finding worth writing in your README.

# %%
dec = seasonal_decompose(np.log(df["Close"]), model="additive", period=7)
fig = dec.plot(); fig.set_size_inches(12, 8)
plt.tight_layout(); save_fig("01_decomposition"); plt.show()
