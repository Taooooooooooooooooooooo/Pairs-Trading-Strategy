import pandas as pd
import yfinance as yf

def download_option_chain(ticker="SPY"):
    tk = yf.Ticker(ticker)

    # 1) list all expiration dates available for this underlying
    expirations = tk.options  # e.g., ['2025-08-22', '2025-08-29', ...]
    if not expirations:
        raise RuntimeError(f"No option expirations found for {ticker}.")

    all_rows = []

    # 2) loop through expirations and pull calls/puts for each
    for exp in expirations:
        chain = tk.option_chain(exp)   # returns namedtuple(calls=..., puts=...)
        calls = chain.calls.copy()
        puts  = chain.puts.copy()

        # tag with metadata
        calls["expiration"] = exp
        calls["type"] = "call"
        puts["expiration"]  = exp
        puts["type"] = "put"

        all_rows.append(calls)
        all_rows.append(puts)

    # 3) combine into one DataFrame
    df = pd.concat(all_rows, ignore_index=True)

    # 4) (optional) keep a compact set of useful columns if present
    keep = [
        "contractSymbol", "type", "expiration", "strike", "lastPrice",
        "bid", "ask", "volume", "openInterest", "inTheMoney", "impliedVolatility"
    ]
    df = df[[c for c in keep if c in df.columns]]

    return df

if __name__ == "__main__":
    df = download_option_chain("SPY")
    print(df.head())

    # Optional: save to CSV for later IV/surface work
    df.to_csv("option_chain_SPY.csv", index=False)
print("Saved: option_chain_SPY.csv")

 #This code downloads and saves the code as a csv file.

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq
from math import log, sqrt, exp
from datetime import datetime, timezone

# --- Black–Scholes helpers (European) with continuous dividend yield q ---
from math import erf

def _ncdf(x):
    # standard normal CDF without scipy.stats
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))

def bs_price(S, K, T, r, q, sigma, option_type):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan
    d1 = (np.log(S/K) + (r - q + 0.5*sigma*sigma)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    if option_type == "call":
        return S*exp(-q*T)*_ncdf(d1) - K*exp(-r*T)*_ncdf(d2)
    else:  # put
        return K*exp(-r*T)*_ncdf(-d2) - S*exp(-q*T)*_ncdf(-d1)

def bs_vega(S, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S/K) + (r - q + 0.5*sigma*sigma)*T) / (sigma*sqrt(T))
    return S*exp(-q*T)*sqrt(T) * (1.0/np.sqrt(2*np.pi)) * np.exp(-0.5*d1*d1)

def implied_vol(price, S, K, T, r, q, option_type, sigma_lo=1e-6, sigma_hi=5.0):
    # Guardrails: no-arbitrage-ish bounds
    disc = exp(-r*T); dq = exp(-q*T)
    intrinsic = max(0.0, (S - K) if option_type=="call" else (K - S))
    upper = (S if option_type=="call" else K)  # very loose
    # If outside rough bounds, skip
    if not (intrinsic - 1e-8 <= price <= upper + 1e-8):
        return np.nan

    def f(sig): 
        return bs_price(S,K,T,r,q,sig,option_type) - price

    try:
        return brentq(f, sigma_lo, sigma_hi, maxiter=100, xtol=1e-8)
    except ValueError:
        # Try nudging brackets if target is slightly out-of-range due to micro noise
        try:
            return brentq(f, 1e-8, 10.0, maxiter=100, xtol=1e-8)
        except Exception:
            return np.nan

# --- Main routine: attach IV to your option chain DataFrame ---
def compute_iv_surface_inputs(df, ticker="SPY", r=0.0, q=None):
    """
    df: option chain with at least columns:
        ['type','expiration','strike','bid','ask','lastPrice'] (case-insensitive ok)
    r: risk-free rate (annualized, cont. comp assumed). Use e.g. 3M T-bill as proxy.
    q: dividend yield (annualized, cont. comp). If None, try to infer a rough value.
    """
    # Normalize column names
    cols = {c.lower(): c for c in df.columns}
    def col(name): 
        return cols.get(name, name)

    # Spot (S)
    S = float(yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1])

    # Dividend yield q (continuous). If not given, you can insert a recent SPY yield ~1.1%/yr.
    # For rigorous use, pass q explicitly from your data source.
    if q is None:
        q = 0.011  # rough recent SPY dividend yield as a default (adjust as desired)

    # Current time
    now = datetime.now(timezone.utc)

    # Mid price
    df = df.copy()
    if "mid" not in df.columns:
        bid = df[col("bid")] if col("bid") in df.columns else np.nan
        ask = df[col("ask")] if col("ask") in df.columns else np.nan
        last = df[col("lastPrice")] if col("lastPrice") in df.columns else np.nan
        df["mid"] = np.where(np.isfinite(bid) & np.isfinite(ask) & (ask>0),
                             (bid + ask)/2.0,
                             last)

    # Time to expiry (ACT/365)
    exp_series = pd.to_datetime(df[col("expiration")], utc=True, errors="coerce")
    T = (exp_series - now).dt.total_seconds() / (365.0*24*3600)
    df["T"] = T.clip(lower=0.0)

    # Option type normalization: 'call'/'put'
    typ = df[col("type")].str.lower().map({"c":"call","call":"call","p":"put","put":"put"})
    df["otype"] = typ

    # Compute IV per row
    strikes = df[col("strike")].astype(float).values
    prices  = df["mid"].astype(float).values
    Ts      = df["T"].astype(float).values
    types   = df["otype"].values

    ivs = []
    for price_i, K_i, T_i, ty in zip(prices, strikes, Ts, types):
        if not np.isfinite(price_i) or not np.isfinite(K_i) or not np.isfinite(T_i) or ty not in ("call","put"):
            ivs.append(np.nan); continue
        if T_i <= 0:
            ivs.append(np.nan); continue
        iv = implied_vol(price_i, S, K_i, T_i, r, q, ty)
        ivs.append(iv)
    df["iv_bs"] = ivs

    # (optional) Drop rows with missing IV, illiquid/noisy quotes
    # df = df[(df["iv_bs"].notna()) & (df[col("openInterest")] > 0)]

    df.attrs["spot"] = S
    df.attrs["r"] = r
    df.attrs["q"] = q
return df

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, RBFInterpolator
from math import exp

def build_iv_surface(df,
                     iv_col="iv_bs",
                     k_steps=121,
                     T_steps=50,
                     method="griddata",  # or "rbf"
                     rbf_kernel="thin_plate", rbf_smoothing=0.0):
    S = float(df.attrs.get("spot", np.nan))
    r = float(df.attrs.get("r", 0.0))
    q = float(df.attrs.get("q", 0.0))
    now = pd.Timestamp.utcnow()

    df2 = df.copy()
    df2 = df2[df2[iv_col].notna() & df2["T"].notna() & df2["strike"].notna()]
    df2 = df2[(df2["T"] > 0) & (df2[iv_col] > 0)]
    if df2.empty:
        raise ValueError("No valid IV data to interpolate.")

    T = df2["T"].to_numpy(float)
    K = df2["strike"].to_numpy(float)
    F = S * np.exp((r - q) * T)
    k = np.log(K / F)
    w = (df2[iv_col].to_numpy(float)**2) * T

    k_lo, k_hi = np.percentile(k, [2.5, 97.5])
    T_lo, T_hi = T.min(), T.max()
    k_grid = np.linspace(k_lo, k_hi, k_steps)
    T_grid = np.linspace(T_lo, T_hi, T_steps)
    KK, TT = np.meshgrid(k_grid, T_grid)

    pts = np.column_stack([k, T])
    if method == "griddata":
        w_lin = griddata(pts, w, (KK, TT), method="linear")
        w_near = griddata(pts, w, (KK, TT), method="nearest")
        W = np.where(np.isfinite(w_lin), w_lin, w_near)
    else:
        rbf = RBFInterpolator(pts, w, kernel=rbf_kernel, smoothing=rbf_smoothing)
        W = rbf(np.column_stack([KK.ravel(), TT.ravel()])).reshape(KK.shape)

    TT_safe = np.clip(TT, 1e-8, None)
    IV_grid = np.sqrt(np.clip(W, 0, None) / TT_safe)
    F_grid = S * np.exp((r - q) * TT)
    K_grid = F_grid * np.exp(KK)

    return K_grid, TT * 365, IV_grid  # T in days

def plot_iv_surface(K_grid, T_days, IV_grid, title="Implied Vol Surface"):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(K_grid, T_days, IV_grid, cmap="viridis", edgecolor='none')
    ax.set_xlabel("Strike (K)")
    ax.set_ylabel("Days to Expiry")
    ax.set_zlabel("Implied Volatility")
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.6, label="IV")
    plt.show()

    plt.figure(figsize=(9, 5))
    cp = plt.contourf(K_grid, T_days, IV_grid, levels=20, cmap="viridis")
    plt.xlabel("Strike (K)")
    plt.ylabel("Days to Expiry")
    plt.title(f"{title} (Contour)")
    plt.colorbar(cp, label="IV")
    plt.show()
df_iv = pd.read_csv("option_chain_SPY_with_IV.csv")
# If attrs lost, reinstate:
df_iv.attrs["spot"] = 645.31
df_iv.attrs["r"]    = 0.043
df_iv.attrs["q"]    = 0.011

Kg, Td, IVg = build_iv_surface(df_iv, method="griddata")
plot_iv_surface(Kg, Td, IVg, title="SPY IV Surface")

# For a smoother view:
Kg2, Td2, IVg2 = build_iv_surface(df_iv, method="rbf", rbf_smoothing=1e-6)
plot_iv_surface(Kg2, Td2, IVg2, title="SPY IV Surface (smooth RBF)")

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# --- SVI model (raw) on total variance w(k) ---
def svi_w_raw(k, a, b, rho, m, sigma):
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

def fit_svi_slice(k, w, weights=None):
    """
    Fit raw SVI parameters (a,b,rho,m,sigma) for one expiry slice.
    k: array of log-forward moneyness
    w: array of total implied variance (IV^2 * T)
    weights: optional weights (same length as k)
    """
    k = np.asarray(k, float)
    w = np.asarray(w, float)
    if weights is None:
        weights = np.ones_like(w)
    else:
        weights = np.asarray(weights, float)

    # Initial guesses (robust, data-driven)
    a0 = np.maximum(1e-6, np.percentile(w, 10))
    m0 = np.median(k)
    # rough slope from linear fit as seed for b*rho; start b>0 and rho<0 (equity skew)
    slope = np.polyfit(k, w, 1)[0] if len(k) >= 2 else 0.0
    b0 = np.maximum(1e-4, 0.5 * np.std(w))      # magnitude
    rho0 = np.clip(np.sign(slope) * (-0.5), -0.99, 0.99)  # equity: negative skew
    sigma0 = np.maximum(1e-3, 0.1 * (np.percentile(k, 84) - np.percentile(k, 16)))

    x0 = np.array([a0, b0, rho0, m0, sigma0], dtype=float)

    # Bounds: (keep broad but safe)
    kmin, kmax = np.min(k), np.max(k)
    bounds = [
        (1e-10, 5.0),            # a >= 0
        (1e-8,  10.0),           # b > 0
        (-0.999, 0.999),         # rho in (-1,1)
        (kmin-1.0, kmax+1.0),    # m near data range
        (1e-6,  5.0),            # sigma > 0
    ]

    # Weighted least squares objective on total variance
    def obj(x):
        a,b,rho,m,sigma = x
        w_hat = svi_w_raw(k, a,b,rho,m,sigma)
        res = (w_hat - w)
        return np.mean(weights * res * res)

    # A little regularization to avoid degenerate fits (optional)
    def obj_reg(x, lam=1e-8):
        a,b,rho,m,sigma = x
        return obj(x) + lam*(b**2 + sigma**2)

    res = minimize(obj_reg, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 2000})
    return res.x, res.fun, res.success

def prepare_slice_inputs(df_slice, S, r, q):
    """
    For a single expiry slice: compute k and total variance w.
    Requires columns: 'strike', 'T', 'iv_bs'
    """
    T = df_slice["T"].to_numpy(float)
    K = df_slice["strike"].to_numpy(float)
    iv = df_slice["iv_bs"].to_numpy(float)

    # Forward F = S * exp((r - q) * T)
    F = S * np.exp((r - q) * T)
    k = np.log(K / F)
    w = np.maximum(0.0, (iv**2) * T)
    return k, w

def calibrate_svi_per_expiry(df_iv, use_weights=True):
    """
    Calibrate raw-SVI per expiry in df_iv (as built earlier).
    Returns a DataFrame with columns: expiration, a,b,rho,m,sigma, n_pts, loss
    """
    # Pull S,r,q from attrs; fallback to zeros if missing
    S = float(df_iv.attrs.get("spot", np.nan))
    r = float(df_iv.attrs.get("r", 0.0))
    q = float(df_iv.attrs.get("q", 0.0))
    if not np.isfinite(S) or S <= 0:
        raise ValueError("Spot missing. Set df_iv.attrs['spot'] or pass S.")

    # Clean: keep finite IV/T/strike
    data = df_iv.copy()
    data = data[np.isfinite(data["iv_bs"]) & np.isfinite(data["T"]) & np.isfinite(data["strike"])]
    data = data[(data["iv_bs"] > 0) & (data["T"] > 0)]

    out = []
    for exp, grp in data.groupby("expiration"):
        if len(grp) < 5:
            continue  # need enough points
        k, w = prepare_slice_inputs(grp, S, r, q)

        # Optional weights: prefer liquid quotes
        weights = None
        if use_weights:
            if "openInterest" in grp.columns:
                # higher OI -> more weight
                weights = 1.0 + np.asarray(grp["openInterest"].fillna(0.0), float)
            elif {"bid","ask"}.issubset(grp.columns):
                # tighter spreads -> more weight
                spr = (grp["ask"] - grp["bid"]).astype(float).clip(lower=1e-4).to_numpy()
                weights = 1.0 / spr
            else:
                weights = None

        (a,b,rho,m,sigma), loss, ok = fit_svi_slice(k, w, weights)
        out.append({"expiration": exp, "a": a, "b": b, "rho": rho, "m": m, "sigma": sigma,
                    "n_pts": len(grp), "loss": loss, "success": bool(ok)})

    res_df = pd.DataFrame(out).sort_values("expiration")
    return res_df

# ---- Example usage (after you've computed iv_df with compute_iv_surface_inputs) ----
iv_df = pd.read_csv("option_chain_SPY_with_IV.csv")
# If attrs were lost when saving CSV, restore them:
iv_df.attrs["spot"] =  645.31
iv_df.attrs["r"]    =  0.043
iv_df.attrs["q"]    =  0.011
params = calibrate_svi_per_expiry(iv_df, use_weights=True)
print(params[["expiration","a","b","rho","m","sigma","n_pts","loss","success"]].to_string(index=False))

