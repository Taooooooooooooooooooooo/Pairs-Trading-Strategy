# Step 1, download the data

import yfinance as yf

# Download data for two stocks with adjusted prices
df = yf.download(["SPY", "IVV"], start="2020-01-01", end="2025-08-01", auto_adjust=True)

# The 'Close' column now reflects the adjusted closing prices
SPDR_adj = df["Close"]["SPY"]
iShares_adj = df["Close"]["IVV"]

print(SPDR_adj.head())
print(iShares_adj.head())

# Step 2, Eagle-Granger test

import numpy as np
import pandas as pd
import yfinance as yf

from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# 1) Get data (you already did this, but keeping it self-contained)
df = yf.download(["SPY", "IVV"], start="2020-01-01", end="2025-08-01", auto_adjust=True)

SPDR_adj   = df["Close"]["SPY"].rename("SPY")
iShares_adj = df["Close"]["IVV"].rename("IVV")

# Align, log-transform, and drop missing
prices = pd.concat([SPDR_adj, iShares_adj], axis=1).dropna()
logp   = np.log(prices)

# --- Engle–Granger (built-in) ---
# statsmodels coint() implements the augmented Engle–Granger two-step cointegration test
eg_stat, eg_pvalue, eg_crit = coint(logp["SPY"], logp["IVV"], trend='c', autolag='aic')

print("=== Engle–Granger (statsmodels.coint) ===")
print(f"Test statistic: {eg_stat:.4f}")
print(f"p-value       : {eg_pvalue:.6f}")
print(f"Crit values   : 1%={eg_crit[0]:.4f}, 5%={eg_crit[1]:.4f}, 10%={eg_crit[2]:.4f}")
print("Decision (at 5%):", "Cointegrated" if eg_pvalue < 0.05 else "Not cointegrated")
print()

# Step 3 and 4, test and evaluate the strategy

# Hedge ratio via OLS: log(SPY) = alpha + beta*log(IVV) + eps
X = sm.add_constant(logp["IVV"])
ols_res = sm.OLS(logp["SPY"], X).fit()
alpha, beta = ols_res.params["const"], ols_res.params["IVV"]

spread = (logp["SPY"] - (alpha + beta * logp["IVV"]))  # stationary residual (if cointegrated)

LOOKBACK = 60  # rolling window for z-score; tweak as desired
mu  = spread.rolling(LOOKBACK).mean()
sig = spread.rolling(LOOKBACK).std()
zscore = (spread - mu) / sig

# ---------------------------
# Trading rules (entry Z=2, exit Z=0.5)
# Long spread when z <= -2: +1*SPY, -beta*IVV
# Short spread when z >= +2: -1*SPY, +beta*IVV
# Exit when |z| <= 0.5 (flat)
# ---------------------------

entry_high =  2.0
entry_low  = -2.0
exit_band  =  0.5

position = pd.Series(0, index=prices.index, dtype=float)  # +1 = long spread, -1 = short spread, 0 = flat

# Generate signals
state = 0
for t in range(len(zscore)):
    z = zscore.iat[t]
    if np.isnan(z):
        position.iat[t] = state
        continue

    if state == 0:
        if z >= entry_high:
            state = -1  # short spread
        elif z <= entry_low:
            state = +1  # long spread
    else:
        if abs(z) <= exit_band:
            state = 0  # exit to flat

    position.iat[t] = state

# Lag positions by one day to avoid look-ahead in PnL
position = position.shift(1).fillna(0.0)

# ---------------------------
# Step 4: Strategy returns, Sharpe, Drawdown
# ---------------------------

# Build dollar-neutral weights for the two legs based on spread direction
# For +1 (long spread): +1*SPY, -beta*IVV
# For -1 (short spread): -1*SPY, +beta*IVV
w_spy =  position
w_ivv = -position * beta

# Normalize to unit gross exposure (optional but helps comparability)
gross = (abs(w_spy) + abs(w_ivv)).replace(0, 1.0)
w_spy = w_spy / gross
w_ivv = w_ivv / gross

# Daily log returns of assets
ret = np.log(prices).diff().fillna(0.0)
ret_spy = ret["SPY"]
ret_ivv = ret["IVV"]

# Strategy daily return
strategy_ret = w_spy * ret_spy + w_ivv * ret_ivv

# (Optional) subtract costs per turnover
# turnover = (w_spy.diff().abs() + w_ivv.diff().abs()).fillna(0.0)
# cost_per_unit = 0.0002  # 2 bps per leg, example
# strategy_ret = strategy_ret - cost_per_unit * turnover

# Sharpe ratio (annualized, rf≈0)
daily_mean = strategy_ret.mean()
daily_std  = strategy_ret.std(ddof=0)
ann_factor = np.sqrt(252)
sharpe = (daily_mean / daily_std) * ann_factor if daily_std > 0 else np.nan

# Equity curve & drawdown
equity = (1.0 + strategy_ret).cumprod()
running_peak = equity.cummax()
drawdown = equity / running_peak - 1.0
max_drawdown = drawdown.min()

print("=== Pairs Trading Performance (SPY–IVV) ===")
print(f"Hedge ratio beta (SPY~IVV): {beta:.4f}")
print(f"Sharpe (ann.): {sharpe:.3f}")
print(f"Max Drawdown : {max_drawdown:.2%}")
print(f"Trades (non-flat days): {(position!=0).sum()} out of {len(position)}")