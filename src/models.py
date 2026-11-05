"""Forecasting models and evaluation helpers."""
import logging
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA


# ---------- splitting ----------
def time_split(series: pd.Series, test_size: int):
    """Chronological split. NEVER shuffle a time series: it would leak the future into training."""
    return series.iloc[:-test_size], series.iloc[-test_size:]


# ---------- models ----------
def naive_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    """Baseline: every future day equals the last known price. Any real model must beat this."""
    return np.repeat(float(train.iloc[-1]), horizon)


def arima_forecast(train: pd.Series, horizon: int, order=(1, 1, 1)) -> np.ndarray:
    """ARIMA(p,d,q): p = autoregressive lags, d = differencing, q = moving-average lags."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ARIMA(train.values, order=order).fit()
    return np.asarray(fit.forecast(horizon))


def select_arima_order(train: pd.Series, max_p: int = 3, max_q: int = 3, d: int = 1):
    """Grid-search p and q and keep the order with the lowest AIC (fit vs complexity trade-off)."""
    best_order, best_aic = None, np.inf
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    aic = ARIMA(train.values, order=(p, d, q)).fit().aic
            except Exception:
                continue
            if aic < best_aic:
                best_order, best_aic = (p, d, q), aic
    return best_order, best_aic


def prophet_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    """Prophet: trend + seasonality model. Imported lazily because it is heavy."""
    from prophet import Prophet
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)
    data = pd.DataFrame({"ds": train.index, "y": train.values})
    model = Prophet(daily_seasonality=False)
    model.fit(data)
    future = model.make_future_dataframe(periods=horizon)
    return model.predict(future)["yhat"].iloc[-horizon:].values


# ---------- evaluation ----------
def evaluate(actual, pred) -> dict:
    """RMSE (penalises big misses), MAE (average miss in $), MAPE (average miss in %)."""
    actual, pred = np.asarray(actual), np.asarray(pred)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
        "MAE": float(mean_absolute_error(actual, pred)),
        "MAPE_%": float(mean_absolute_percentage_error(actual, pred) * 100),
    }


def walk_forward(series: pd.Series, forecasters: dict, horizon: int = 14,
                 n_folds: int = 10, min_train: int = 365) -> pd.DataFrame:
    """Walk-forward (rolling-origin) validation.

    A single train/test split can be lucky or unlucky. Here we move the forecast origin
    forward `n_folds` times, always training only on the past, and average the errors.
    `forecasters` is a dict like {"Naive": fn(train, horizon) -> array}.
    Also reports Beat_Naive_%: in what share of folds the model had a lower RMSE than Naive.
    """
    n = len(series)
    if n - horizon * n_folds < min_train:
        raise ValueError("Not enough data: lower n_folds/horizon or load more years.")
    origins = [n - horizon * (n_folds - i) for i in range(n_folds)]

    records = {name: [] for name in forecasters}
    for t in origins:
        train, test = series.iloc[:t], series.iloc[t:t + horizon]
        for name, fn in forecasters.items():
            records[name].append(evaluate(test.values, fn(train, horizon)))

    summary = pd.DataFrame({name: pd.DataFrame(r).mean() for name, r in records.items()}).T
    if "Naive" in records:
        naive_rmse = np.array([r["RMSE"] for r in records["Naive"]])
        for name, r in records.items():
            summary.loc[name, "Beat_Naive_%"] = np.mean(np.array([x["RMSE"] for x in r]) < naive_rmse) * 100
    return summary.round(2)
