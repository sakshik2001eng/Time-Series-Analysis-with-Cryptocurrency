# %% [markdown]
# # 02 - Stationarity and ACF/PACF
# WHY: ARIMA assumes the series is stationary after differencing (constant mean/variance,
# no trend). Here we PROVE what differencing we need instead of guessing.

# %%
import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from src.data_loader import load_prices, add_features
from src.utils import save_fig

df = add_features(load_prices("BTC-USD", years=5))
price = df["Close"]
ret = df["Return"].dropna()

# %% [markdown]
# ## Two tests that ask opposite questions
# - ADF:  H0 = "series is NON-stationary".  p < 0.05 -> reject H0 -> stationary.
# - KPSS: H0 = "series IS stationary".      p < 0.05 -> reject H0 -> non-stationary.
# If both agree, you can be confident.

# %%
def stationarity_report(series, name):
    adf_p = adfuller(series.dropna(), autolag="AIC")[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                  # KPSS warns when p-value is off its table
        kpss_p = kpss(series.dropna(), regression="c", nlags="auto")[1]
    verdict = "stationary" if (adf_p < 0.05 and kpss_p >= 0.05) else "NOT clearly stationary"
    print(f"{name:<22} ADF p={adf_p:.4f} | KPSS p={kpss_p:.4f} -> {verdict}")

stationarity_report(price, "Price")
stationarity_report(price.diff(), "Price (1st difference)")
stationarity_report(ret, "Log returns")

# %% [markdown]
# Expected: price is non-stationary, first difference / returns are stationary.
# That tells us ARIMA needs d = 1 (difference once).

# %% [markdown]
# ## ACF / PACF
# - ACF = correlation of the series with its own past at each lag.
# - PACF = same, but removing the effect of shorter lags.
# Rules of thumb: PACF cuts off after lag p -> AR(p); ACF cuts off after lag q -> MA(q).
# For crypto returns you will usually see almost nothing significant: returns are close to
# unpredictable (market efficiency). That is an honest result.

# %%
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
plot_acf(ret, lags=40, ax=ax[0]); plot_pacf(ret, lags=40, ax=ax[1], method="ywm")
plt.tight_layout(); save_fig("02_acf_pacf"); plt.show()

# %% [markdown]
# ## Bonus: ACF of SQUARED returns
# Returns themselves look random, but their SIZE is predictable (volatility clustering).
# Strong autocorrelation here is the evidence.

# %%
plot_acf(ret ** 2, lags=40)
plt.title("ACF of squared returns"); save_fig("02_acf_squared"); plt.show()
