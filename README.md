# Time Series Analysis and Forecasting of Cryptocurrency Prices

**Live dashboard:** _add your Streamlit link here_

End-to-end time series project: data collection, exploratory analysis, stationarity testing,
ARIMA/Prophet forecasting, walk-forward validation against a naive baseline, and a deployed
Streamlit dashboard.

## Questions answered
1. What do BTC price, returns and volatility look like over time?
2. Is the series stationary, and what differencing does ARIMA need?
3. Can ARIMA or Prophet beat a naive "tomorrow = today" forecast on unseen data?

## Project structure
```
analysis/01_eda.py            Exploratory analysis (trend, returns, volatility, decomposition)
analysis/02_stationarity.py   ADF/KPSS tests, ACF/PACF
analysis/03_modeling.py       Naive vs ARIMA vs Prophet, walk-forward validation
src/                          Reusable code (data loading, models, metrics)
app.py                        Streamlit dashboard
reports/                      Saved figures and result tables
```

## How to run
```bash
python -m venv venv
venv\Scripts\activate            # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements-analysis.txt
python analysis/01_eda.py
python analysis/02_stationarity.py
python analysis/03_modeling.py
streamlit run app.py
```

## Method
- **Data:** daily BTC-USD closing prices from Yahoo Finance (yfinance), about 5 years.
- **Validation:** chronological split plus walk-forward validation (10 folds, 14-day horizon). No shuffling.
- **Metrics:** RMSE, MAE, MAPE.
- **Baseline:** naive forecast (last observed price).

## Results
_Fill this in after running `03_modeling.py` (see `reports/walk_forward_results.csv`)._

| Model | RMSE | MAPE % | Beat naive % |
|-------|------|--------|--------------|
| Naive | | | n/a |
| ARIMA | | | |
| Prophet | | | |

## Key findings
_Write 3-4 honest bullets, for example:_
- Price is non-stationary (ADF/KPSS); log returns are stationary, so ARIMA uses d = 1.
- Returns show fat tails and volatility clustering.
- ARIMA performed about the same as the naive baseline, consistent with near random-walk behaviour.

## Limitations and next steps
- Univariate models only. Could add external features (volume, sentiment, macro data).
- Try GARCH for volatility forecasting, or an LSTM.
- Not financial advice.
