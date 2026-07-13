# -*- coding: utf-8 -*-
"""
CommodityMacroPro Institutional V1.0
------------------------------------
Daily Yahoo Finance data only. No synthetic prices, no proxy series.
Universe: Gold, Silver, Platinum, WTI Crude Oil, Copper, DXY and US Treasury 10Y yield.

Run:
    streamlit run app.py
"""

import os
import gc
import math
import warnings
from datetime import datetime

# Streamlit Cloud stability: prevent BLAS/OpenMP oversubscription during repeated
# walk-forward Ridge/OLS fits. These variables must be set before NumPy/SciPy import.
for _thread_env in (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
    'BLIS_NUM_THREADS',
):
    os.environ[_thread_env] = '1'
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

# -----------------------------------------------------------------------------
# APPLICATION CONFIGURATION
# -----------------------------------------------------------------------------
APP_VERSION = "1.0.2"
APP_NAME = "CommodityMacroPro Institutional"
TRADING_DAYS = 252
DXY_TICKER = "DX-Y.NYB"
TNX_TICKER = "^TNX"

INSTRUMENTS: Dict[str, Dict[str, str]] = {
    "Gold Futures": {"ticker": "GC=F", "short": "Gold", "type": "commodity", "unit": "USD/oz"},
    "Silver Futures": {"ticker": "SI=F", "short": "Silver", "type": "commodity", "unit": "USD/oz"},
    "Platinum Futures": {"ticker": "PL=F", "short": "Platinum", "type": "commodity", "unit": "USD/oz"},
    "WTI Crude Oil Futures": {"ticker": "CL=F", "short": "WTI Oil", "type": "commodity", "unit": "USD/bbl"},
    "Copper Futures": {"ticker": "HG=F", "short": "Copper", "type": "commodity", "unit": "USD/lb"},
    "US Dollar Index": {"ticker": DXY_TICKER, "short": "DXY", "type": "index", "unit": "Index"},
    "US Treasury 10Y Yield": {"ticker": TNX_TICKER, "short": "UST 10Y", "type": "yield", "unit": "%"},
}
COMMODITY_NAMES = [k for k, v in INSTRUMENTS.items() if v["type"] == "commodity"]
TICKER_TO_NAME = {v["ticker"]: k for k, v in INSTRUMENTS.items()}

st.set_page_config(
    page_title=f"{APP_NAME} V{APP_VERSION}",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# HEDGE-FUND STYLE
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
      :root {
        --hf-navy: #0b1f33;
        --hf-slate: #344054;
        --hf-muted: #667085;
        --hf-border: #dfe3e8;
        --hf-soft: #f6f8fa;
        --hf-white: #ffffff;
        --hf-accent: #214b73;
        --hf-green: #176b45;
        --hf-red: #9f2d2d;
      }
      html, body, [class*="css"] {
        font-family: "Inter", "Aptos", "Segoe UI", Arial, sans-serif;
        color: var(--hf-navy);
      }
      .stApp { background: var(--hf-white); }
      .block-container {
        max-width: 1740px;
        padding-top: 1.0rem;
        padding-bottom: 2.6rem;
        padding-left: 2rem;
        padding-right: 2rem;
      }
      h1, h2, h3, h4, h5, h6 {
        color: var(--hf-navy);
        letter-spacing: -0.025em;
      }
      h1 { font-size: 2.0rem !important; font-weight: 300 !important; }
      h2 { font-size: 1.42rem !important; font-weight: 350 !important; margin-top: 1.3rem !important; }
      h3 { font-size: 1.08rem !important; font-weight: 450 !important; }
      p, label, .stMarkdown, .stCaption { color: var(--hf-slate); }
      hr { border: 0; border-top: 1px solid var(--hf-border); margin: 1.15rem 0 1.25rem 0; }

      .hf-masthead {
        margin: 0.10rem 0 1.0rem 0;
        padding: 0.10rem 0 1.0rem 0;
        border-bottom: 1px solid var(--hf-border);
      }
      .hf-eyebrow {
        font-size: 0.66rem;
        font-weight: 600;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--hf-muted);
        margin-bottom: 0.45rem;
      }
      .hf-title {
        font-size: 2.05rem;
        line-height: 1.12;
        font-weight: 300;
        letter-spacing: -0.045em;
        color: var(--hf-navy);
      }
      .hf-version {
        font-size: 0.70rem;
        font-weight: 600;
        letter-spacing: 0.09em;
        color: var(--hf-accent);
        text-transform: uppercase;
        margin-left: 0.55rem;
      }
      .hf-meta {
        margin-top: 0.45rem;
        font-size: 0.76rem;
        color: var(--hf-muted);
        letter-spacing: 0.035em;
      }
      .instrument-header {
        margin: 1.05rem 0 0.75rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--hf-border);
      }
      .instrument-kicker {
        font-size: 0.62rem;
        font-weight: 600;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--hf-muted);
      }
      .instrument-title {
        margin-top: 0.30rem;
        font-size: 1.60rem;
        font-weight: 350;
        letter-spacing: -0.035em;
        color: var(--hf-navy);
      }
      .instrument-title span {
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        color: var(--hf-accent);
        margin-left: 0.35rem;
      }
      .instrument-subtitle {
        margin-top: 0.25rem;
        font-size: 0.74rem;
        color: var(--hf-muted);
      }
      .section-note {
        border-left: 2px solid var(--hf-accent);
        padding: 0.55rem 0.8rem;
        margin: 0.55rem 0 1.0rem 0;
        font-size: 0.78rem;
        color: var(--hf-slate);
        background: #fafbfc;
      }
      div[data-testid="stMetric"] {
        background: var(--hf-white);
        border: 1px solid var(--hf-border);
        border-radius: 3px;
        padding: 0.70rem 0.80rem 0.64rem 0.80rem;
        box-shadow: none;
      }
      div[data-testid="stMetricLabel"] {
        font-size: 0.64rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.075em !important;
        text-transform: uppercase;
        color: var(--hf-muted) !important;
      }
      div[data-testid="stMetricValue"] {
        font-size: 1.24rem !important;
        font-weight: 400 !important;
        color: var(--hf-navy) !important;
      }
      div[data-testid="stTabs"] button {
        min-height: 2.60rem;
        padding: 0.30rem 0.64rem;
        border-radius: 0;
        font-size: 0.67rem;
        font-weight: 600;
        letter-spacing: 0.065em;
        text-transform: uppercase;
      }
      div[data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--hf-navy);
        border-bottom: 2px solid var(--hf-accent);
      }
      section[data-testid="stSidebar"] {
        border-right: 1px solid var(--hf-border);
        background: #fbfcfd;
      }
      .stButton > button, .stDownloadButton > button {
        border: 1px solid var(--hf-border);
        border-radius: 3px;
        background: var(--hf-white);
        color: var(--hf-navy);
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.035em;
      }
      .stDataFrame { border: 1px solid var(--hf-border); }
      .positive { color: var(--hf-green); }
      .negative { color: var(--hf-red); }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def safe_float(value, default=np.nan) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def fmt_number(value: float, decimals: int = 2, suffix: str = "") -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:,.{decimals}f}{suffix}"


def render_dataframe(
    frame: pd.DataFrame,
    formats: Optional[Dict[str, str]] = None,
    *,
    hide_index: bool = True,
) -> None:
    """Render a numeric table without pandas Styler or matplotlib.

    Streamlit column configuration preserves institutional number formatting while
    avoiding pandas Styler and optional matplotlib styling paths.
    """
    clean = frame.copy().replace([np.inf, -np.inf], np.nan)
    column_config = {}
    for column, fmt in (formats or {}).items():
        if column in clean.columns:
            column_config[column] = st.column_config.NumberColumn(format=fmt)
    st.dataframe(
        clean,
        width="stretch",
        hide_index=hide_index,
        column_config=column_config or None,
    )


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def annualized_volatility(returns: pd.Series) -> float:
    r = pd.Series(returns, dtype=float).dropna()
    return float(r.std(ddof=1) * math.sqrt(TRADING_DAYS)) if len(r) > 1 else np.nan


def max_drawdown(price: pd.Series) -> float:
    p = pd.Series(price, dtype=float).dropna()
    if p.empty:
        return np.nan
    return float((p / p.cummax() - 1.0).min())


def compute_rsi(price: pd.Series, period: int = 14) -> pd.Series:
    delta = price.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def ewma_volatility(log_returns: pd.Series, lam: float = 0.94) -> pd.Series:
    r = pd.Series(log_returns, dtype=float)
    alpha = 1.0 - float(lam)
    variance = r.pow(2).ewm(alpha=alpha, adjust=False, min_periods=20).mean()
    return variance.pow(0.5) * math.sqrt(TRADING_DAYS)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_index()
    out["Analysis Price"] = out["Adj Close"].where(out["Adj Close"].notna(), out["Close"])
    price = out["Analysis Price"].astype(float)
    out["Log Return"] = np.log(price / price.shift(1))
    out["Simple Return"] = price.pct_change()
    out["EMA 20"] = price.ewm(span=20, adjust=False, min_periods=20).mean()
    out["EMA 50"] = price.ewm(span=50, adjust=False, min_periods=50).mean()
    out["EMA 200"] = price.ewm(span=200, adjust=False, min_periods=200).mean()
    out["BB Mid"] = price.rolling(20, min_periods=20).mean()
    bb_std = price.rolling(20, min_periods=20).std(ddof=1)
    out["BB Upper"] = out["BB Mid"] + 2.0 * bb_std
    out["BB Lower"] = out["BB Mid"] - 2.0 * bb_std
    out["ATR"] = compute_atr(out)
    out["ATR %"] = out["ATR"] / price.replace(0.0, np.nan)
    out["RSI"] = compute_rsi(price)
    ema12 = price.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = price.ewm(span=26, adjust=False, min_periods=26).mean()
    out["MACD"] = ema12 - ema26
    out["MACD Signal"] = out["MACD"].ewm(span=9, adjust=False, min_periods=9).mean()
    out["MACD Hist"] = out["MACD"] - out["MACD Signal"]
    out["EWMA Vol 0.94"] = ewma_volatility(out["Log Return"], 0.94)
    out["EWMA Vol 0.97"] = ewma_volatility(out["Log Return"], 0.97)
    out["Rolling Vol 20"] = out["Log Return"].rolling(20, min_periods=20).std(ddof=1) * math.sqrt(TRADING_DAYS)
    out["Rolling Vol 60"] = out["Log Return"].rolling(60, min_periods=40).std(ddof=1) * math.sqrt(TRADING_DAYS)
    out["Rolling Vol 252"] = out["Log Return"].rolling(252, min_periods=126).std(ddof=1) * math.sqrt(TRADING_DAYS)
    out["Momentum 5"] = np.log(price / price.shift(5))
    out["Momentum 20"] = np.log(price / price.shift(20))
    out["Momentum 60"] = np.log(price / price.shift(60))
    out["Return Difference"] = out["Log Return"].diff()
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def download_one_ticker(ticker: str, start_date: str, end_date: str) -> Tuple[pd.DataFrame, str]:
    try:
        raw = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        return pd.DataFrame(), f"{type(exc).__name__}: {exc}"

    if raw is None or raw.empty:
        return pd.DataFrame(), "Yahoo Finance returned no observations."

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]

    raw = raw.copy()
    raw.index = pd.to_datetime(raw.index)
    if getattr(raw.index, "tz", None) is not None:
        raw.index = raw.index.tz_localize(None)
    raw = raw[~raw.index.duplicated(keep="last")].sort_index()

    required = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in required:
        if col not in raw.columns:
            raw[col] = np.nan
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw = raw[required].dropna(subset=["Close"])
    if raw.empty:
        return pd.DataFrame(), "No valid closing-price observations."
    return raw, ""


@st.cache_data(ttl=3600, show_spinner=False)
def download_universe(start_date: str, end_date: str) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    data: Dict[str, pd.DataFrame] = {}
    records: List[Dict[str, object]] = []
    for name, meta in INSTRUMENTS.items():
        ticker = meta["ticker"]
        raw, error = download_one_ticker(ticker, start_date, end_date)
        if not raw.empty:
            enriched = compute_indicators(raw)
            data[ticker] = enriched
            records.append(
                {
                    "Instrument": name,
                    "Ticker": ticker,
                    "Status": "OK",
                    "First Date": enriched.index.min().date(),
                    "Last Date": enriched.index.max().date(),
                    "Observations": int(len(enriched)),
                    "Missing Close": int(enriched["Close"].isna().sum()),
                    "Missing Adj Close": int(enriched["Adj Close"].isna().sum()),
                    "Error": "",
                }
            )
        else:
            records.append(
                {
                    "Instrument": name,
                    "Ticker": ticker,
                    "Status": "FAILED",
                    "First Date": None,
                    "Last Date": None,
                    "Observations": 0,
                    "Missing Close": None,
                    "Missing Adj Close": None,
                    "Error": error,
                }
            )
    return data, pd.DataFrame(records)


def common_log_returns(data: Dict[str, pd.DataFrame], tickers: List[str]) -> pd.DataFrame:
    cols = []
    for ticker in tickers:
        if ticker in data:
            s = data[ticker]["Log Return"].rename(ticker)
            cols.append(s)
    if not cols:
        return pd.DataFrame()
    return pd.concat(cols, axis=1, join="inner").dropna(how="any")


def yield_bp_change(tnx_df: pd.DataFrame) -> pd.Series:
    # Yahoo ^TNX is reported in percentage points (e.g. 4.57%).
    # A 0.01 percentage-point move equals one basis point.
    return tnx_df["Analysis Price"].astype(float).diff() * 100.0


def rolling_relationship(asset_return: pd.Series, dxy_return: pd.Series, window: int) -> pd.DataFrame:
    pair = pd.concat([asset_return.rename("Asset"), dxy_return.rename("DXY")], axis=1, join="inner").dropna()
    corr = pair["Asset"].rolling(window, min_periods=max(20, window // 2)).corr(pair["DXY"])
    cov = pair["Asset"].rolling(window, min_periods=max(20, window // 2)).cov(pair["DXY"])
    var = pair["DXY"].rolling(window, min_periods=max(20, window // 2)).var()
    beta = cov / var.replace(0.0, np.nan)
    alpha_daily = pair["Asset"].rolling(window, min_periods=max(20, window // 2)).mean() - beta * pair["DXY"].rolling(window, min_periods=max(20, window // 2)).mean()
    return pd.DataFrame({f"Corr {window}": corr, f"Beta {window}": beta, f"Alpha Ann {window}": alpha_daily * TRADING_DAYS, f"R2 {window}": corr.pow(2)})


def ewma_correlation(asset_return: pd.Series, dxy_return: pd.Series, span: int = 60) -> pd.Series:
    pair = pd.concat([asset_return.rename("Asset"), dxy_return.rename("DXY")], axis=1, join="inner").dropna()
    cov = pair["Asset"].ewm(span=span, adjust=False, min_periods=30).cov(pair["DXY"])
    var_a = pair["Asset"].ewm(span=span, adjust=False, min_periods=30).var()
    var_d = pair["DXY"].ewm(span=span, adjust=False, min_periods=30).var()
    return cov / np.sqrt(var_a * var_d).replace(0.0, np.nan)


def lead_lag_correlations(asset_return: pd.Series, dxy_return: pd.Series, max_lag: int = 20) -> pd.DataFrame:
    pair = pd.concat([asset_return.rename("Asset"), dxy_return.rename("DXY")], axis=1, join="inner").dropna()
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        # Positive lag: today's DXY return versus commodity return lag days ahead.
        corr = pair["DXY"].corr(pair["Asset"].shift(-lag))
        rows.append({"Lag": lag, "Correlation": corr})
    return pd.DataFrame(rows)


def regression_scenarios(asset_df: pd.DataFrame, dxy_df: pd.DataFrame, tnx_df: pd.DataFrame, window: int = 504) -> Tuple[pd.DataFrame, Dict[str, float]]:
    frame = pd.concat(
        [
            asset_df["Log Return"].rename("Asset"),
            dxy_df["Log Return"].rename("DXY"),
            yield_bp_change(tnx_df).rename("TNX_bp"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    frame = frame.tail(window)
    if len(frame) < 120:
        return pd.DataFrame(), {}
    frame["Interaction"] = frame["DXY"] * frame["TNX_bp"]
    X = np.ascontiguousarray(frame[["DXY", "TNX_bp", "Interaction"]].to_numpy(dtype=np.float64, copy=True))
    y = np.ascontiguousarray(frame["Asset"].to_numpy(dtype=np.float64, copy=True))
    with threadpool_limits(limits=1):
        model = LinearRegression(n_jobs=1).fit(X, y)
        fitted = model.predict(X)
        r2 = float(model.score(X, y))
    resid_std = float(np.std(y - fitted, ddof=1))
    scenarios = [
        ("DXY +1%", 0.01, 0.0),
        ("DXY -1%", -0.01, 0.0),
        ("DXY +2%", 0.02, 0.0),
        ("DXY -2%", -0.02, 0.0),
        ("DXY +1% / UST10Y +10bp", 0.01, 10.0),
        ("DXY -1% / UST10Y -10bp", -0.01, -10.0),
        ("DXY +1% / UST10Y -10bp", 0.01, -10.0),
        ("DXY -1% / UST10Y +10bp", -0.01, 10.0),
    ]
    rows = []
    for label, dxy_shock, bp_shock in scenarios:
        x = np.array([[dxy_shock, bp_shock, dxy_shock * bp_shock]])
        pred = float(model.predict(x)[0])
        rows.append(
            {
                "Scenario": label,
                "Expected Log Return %": pred * 100.0,
                "Approx. Simple Return %": (math.exp(pred) - 1.0) * 100.0,
                "Lower 95% %": (pred - 1.96 * resid_std) * 100.0,
                "Upper 95% %": (pred + 1.96 * resid_std) * 100.0,
            }
        )
    stats = {
        "Intercept": float(model.intercept_),
        "DXY Beta": float(model.coef_[0]),
        "TNX bp Beta": float(model.coef_[1]),
        "Interaction Beta": float(model.coef_[2]),
        "R2": r2,
        "Residual Std": resid_std,
        "Observations": int(len(frame)),
    }
    return pd.DataFrame(rows), stats


def trend_channel(price: pd.Series, lookback: int = 90) -> pd.DataFrame:
    p = pd.Series(price, dtype=float).dropna()
    out = pd.DataFrame(index=price.index, columns=["Mid", "Upper", "Lower"], dtype=float)
    if len(p) < 20:
        return out
    p = p.tail(min(lookback, len(p)))
    x = np.arange(len(p), dtype=float)
    slope, intercept = np.polyfit(x, p.values, 1)
    fitted = intercept + slope * x
    residual_std = float(np.std(p.values - fitted, ddof=1))
    out.loc[p.index, "Mid"] = fitted
    out.loc[p.index, "Upper"] = fitted + 1.5 * residual_std
    out.loc[p.index, "Lower"] = fitted - 1.5 * residual_std
    return out


def anchored_vwap(df: pd.DataFrame, lookback: int = 150) -> Tuple[pd.Series, Optional[pd.Timestamp]]:
    low = df["Low"].astype(float)
    swing_low = (low.shift(2) > low.shift(1)) & (low.shift(1) > low) & (low.shift(-1) > low) & (low.shift(-2) > low)
    recent = df.tail(min(lookback, len(df)))
    candidates = recent.index[swing_low.reindex(recent.index).fillna(False)]
    anchor = candidates[-1] if len(candidates) else recent.index[0]
    mask = df.index >= anchor
    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    volume = df["Volume"].astype(float)
    result = pd.Series(np.nan, index=df.index, dtype=float)
    result.loc[mask] = (typical[mask] * volume[mask]).cumsum() / volume[mask].cumsum().replace(0.0, np.nan)
    return result, anchor


def swing_points(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    sh = (high.shift(2) < high.shift(1)) & (high.shift(1) < high) & (high.shift(-1) < high) & (high.shift(-2) < high)
    sl = (low.shift(2) > low.shift(1)) & (low.shift(1) > low) & (low.shift(-1) > low) & (low.shift(-2) > low)
    return df.loc[sh.fillna(False), ["High"]], df.loc[sl.fillna(False), ["Low"]]


def build_price_chart(df: pd.DataFrame, title: str) -> go.Figure:
    plot_df = df.tail(min(520, len(df))).copy()
    channel = trend_channel(plot_df["Analysis Price"], 90)
    avwap, anchor = anchored_vwap(plot_df, 150)
    swing_high, swing_low = swing_points(plot_df)
    support_20 = plot_df["Low"].rolling(20, min_periods=10).min()
    resistance_20 = plot_df["High"].rolling(20, min_periods=10).max()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.78, 0.22])
    fig.add_trace(
        go.Candlestick(
            x=plot_df.index,
            open=plot_df["Open"],
            high=plot_df["High"],
            low=plot_df["Low"],
            close=plot_df["Close"],
            name="OHLC",
            increasing_line_color="#176b45",
            decreasing_line_color="#9f2d2d",
        ),
        row=1,
        col=1,
    )
    for col, name, color, width, dash in [
        ("EMA 20", "EMA 20", "#2563eb", 1.2, "solid"),
        ("EMA 50", "EMA 50", "#d97706", 1.4, "solid"),
        ("EMA 200", "EMA 200", "#111827", 1.7, "dash"),
    ]:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[col], mode="lines", name=name, line=dict(color=color, width=width, dash=dash)), row=1, col=1)

    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["BB Upper"], mode="lines", name="BB Upper", line=dict(color="rgba(59,130,246,0.45)", width=0.8, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["BB Lower"], mode="lines", name="BB Envelope", line=dict(color="rgba(59,130,246,0.45)", width=0.8, dash="dot"), fill="tonexty", fillcolor="rgba(59,130,246,0.06)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=channel.index, y=channel["Upper"], mode="lines", name="Trend Channel Upper", line=dict(color="#0891b2", width=1.0, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=channel.index, y=channel["Lower"], mode="lines", name="Trend Channel Lower", line=dict(color="#0891b2", width=1.0, dash="dash"), fill="tonexty", fillcolor="rgba(8,145,178,0.06)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=channel.index, y=channel["Mid"], mode="lines", name="Trend Channel Mid", line=dict(color="#0f766e", width=1.2, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=avwap, mode="lines", name=f"Anchored VWAP ({anchor.date() if anchor is not None else 'N/A'})", line=dict(color="#7c3aed", width=1.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=resistance_20, mode="lines", name="20D Resistance", line=dict(color="#a855f7", width=0.9, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=support_20, mode="lines", name="20D Support", line=dict(color="#0f766e", width=0.9, dash="dash")), row=1, col=1)

    for points, col, label, color, pos in [
        (swing_high.tail(4), "High", "Swing High", "#9333ea", "top center"),
        (swing_low.tail(4), "Low", "Swing Low", "#0f766e", "bottom center"),
    ]:
        if not points.empty:
            fig.add_trace(
                go.Scatter(
                    x=points.index,
                    y=points[col],
                    mode="markers+text",
                    text=["SH" if label == "Swing High" else "SL"] * len(points),
                    textposition=pos,
                    name=label,
                    marker=dict(size=8, symbol="diamond", color=color, line=dict(color="white", width=0.7)),
                ),
                row=1,
                col=1,
            )

    volume_colors = np.where(plot_df["Close"] >= plot_df["Open"], "#176b45", "#9f2d2d")
    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df["Volume"], name="Volume", marker_color=volume_colors, opacity=0.55), row=2, col=1)

    fig.update_layout(
        title=dict(text=title, x=0.01, font=dict(size=17, color="#0b1f33")),
        template="plotly_white",
        height=850,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1.0),
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eef2f6", rangeselector=dict(buttons=[
        dict(count=1, label="1M", step="month", stepmode="backward"),
        dict(count=3, label="3M", step="month", stepmode="backward"),
        dict(count=6, label="6M", step="month", stepmode="backward"),
        dict(count=1, label="1Y", step="year", stepmode="backward"),
        dict(step="all", label="ALL"),
    ]), row=1, col=1)
    fig.update_yaxes(title_text="Price / Yield", row=1, col=1, gridcolor="#eef2f6")
    fig.update_yaxes(title_text="Volume", row=2, col=1, gridcolor="#eef2f6")
    return fig


def build_model_frame(asset_df: pd.DataFrame, dxy_df: pd.DataFrame, tnx_df: pd.DataFrame, horizon: int) -> Tuple[pd.DataFrame, List[str], str]:
    frame = pd.concat(
        [
            asset_df[["Analysis Price", "Log Return", "Momentum 5", "Momentum 20", "Momentum 60", "EWMA Vol 0.94", "ATR %", "RSI", "MACD Hist", "Volume"]].add_prefix("Asset "),
            dxy_df[["Log Return", "Momentum 5", "Momentum 20", "Momentum 60", "EWMA Vol 0.94"]].add_prefix("DXY "),
            yield_bp_change(tnx_df).rename("TNX bp 1D"),
        ],
        axis=1,
        join="inner",
    ).sort_index()

    frame["TNX bp 5D"] = frame["TNX bp 1D"].rolling(5, min_periods=5).sum()
    frame["TNX bp 20D"] = frame["TNX bp 1D"].rolling(20, min_periods=20).sum()
    frame["DXY x TNX"] = frame["DXY Log Return"] * frame["TNX bp 1D"]
    frame["Asset Volume Change 5D"] = np.log(frame["Asset Volume"].replace(0.0, np.nan) / frame["Asset Volume"].shift(5).replace(0.0, np.nan))
    frame["DXY Corr 60"] = frame["Asset Log Return"].rolling(60, min_periods=40).corr(frame["DXY Log Return"])
    frame["DXY Beta 60"] = frame["Asset Log Return"].rolling(60, min_periods=40).cov(frame["DXY Log Return"]) / frame["DXY Log Return"].rolling(60, min_periods=40).var().replace(0.0, np.nan)

    target_col = f"Forward {horizon}D Log Return"
    frame[target_col] = np.log(frame["Asset Analysis Price"].shift(-horizon) / frame["Asset Analysis Price"])
    frame["Position"] = np.arange(len(frame), dtype=int)

    feature_cols = [
        "Asset Log Return",
        "Asset Momentum 5",
        "Asset Momentum 20",
        "Asset Momentum 60",
        "Asset EWMA Vol 0.94",
        "Asset ATR %",
        "Asset RSI",
        "Asset MACD Hist",
        "Asset Volume Change 5D",
        "DXY Log Return",
        "DXY Momentum 5",
        "DXY Momentum 20",
        "DXY Momentum 60",
        "DXY EWMA Vol 0.94",
        "TNX bp 1D",
        "TNX bp 5D",
        "TNX bp 20D",
        "DXY x TNX",
        "DXY Corr 60",
        "DXY Beta 60",
    ]
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame, feature_cols, target_col


def fit_scaled_ridge(X: pd.DataFrame, y: pd.Series, alpha: float) -> Tuple[StandardScaler, Ridge]:
    """Fit a numerically stable single-thread Ridge model on contiguous float64 arrays."""
    X_arr = np.ascontiguousarray(X.to_numpy(dtype=np.float64, copy=True))
    y_arr = np.ascontiguousarray(y.to_numpy(dtype=np.float64, copy=True))
    scaler = StandardScaler(copy=True)
    with threadpool_limits(limits=1):
        Xs = scaler.fit_transform(X_arr)
        model = Ridge(alpha=float(alpha), solver="lsqr", tol=1e-7, max_iter=5000)
        model.fit(Xs, y_arr)
    return scaler, model


def macro_regression_prediction(
    train: pd.DataFrame,
    row: pd.DataFrame,
    macro_cols: List[str],
    target_col: str,
) -> float:
    """Transparent DXY/UST10Y regression with bounded native-thread usage."""
    macro_train = train.dropna(subset=macro_cols + [target_col]).tail(756)
    if len(macro_train) < 120:
        return float(train[target_col].tail(756).mean())
    X_train = np.ascontiguousarray(macro_train[macro_cols].to_numpy(dtype=np.float64, copy=True))
    y_train = np.ascontiguousarray(macro_train[target_col].to_numpy(dtype=np.float64, copy=True))
    X_row = np.ascontiguousarray(row[macro_cols].to_numpy(dtype=np.float64, copy=True))
    with threadpool_limits(limits=1):
        macro_model = LinearRegression(n_jobs=1).fit(X_train, y_train)
        prediction = macro_model.predict(X_row)
    return float(prediction[0])


def choose_alpha(X: pd.DataFrame, y: pd.Series, candidates: Tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)) -> float:
    if len(X) < 160:
        return 10.0
    split = int(len(X) * 0.80)
    if split < 100 or len(X) - split < 30:
        return 10.0
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]
    best_alpha = 10.0
    best_rmse = np.inf
    for alpha in candidates:
        scaler, model = fit_scaled_ridge(X_train, y_train, alpha)
        X_val_arr = np.ascontiguousarray(X_val.to_numpy(dtype=np.float64, copy=True))
        y_val_arr = np.ascontiguousarray(y_val.to_numpy(dtype=np.float64, copy=True))
        with threadpool_limits(limits=1):
            pred = model.predict(scaler.transform(X_val_arr))
        rmse = float(np.sqrt(np.mean((y_val_arr - pred) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = float(alpha)
    return best_alpha


def walk_forward_forecast(
    asset_df: pd.DataFrame,
    dxy_df: pd.DataFrame,
    tnx_df: pd.DataFrame,
    horizon: int,
    min_train: int = 400,
    max_oos: int = 180,
) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    horizon = int(horizon)
    max_oos = int(min(max(60, max_oos), 240))
    frame, feature_cols, target_col = build_model_frame(asset_df, dxy_df, tnx_df, horizon)
    usable_features = frame.dropna(subset=feature_cols).copy()
    if len(usable_features) < min_train + horizon + 30:
        return pd.DataFrame(), {}, pd.DataFrame()

    latest_pos = int(usable_features["Position"].iloc[-1])
    realized = usable_features.dropna(subset=[target_col]).copy()
    if len(realized) < min_train + 30:
        return pd.DataFrame(), {}, pd.DataFrame()

    candidate_positions = realized["Position"].tolist()
    candidate_positions = candidate_positions[-max_oos:]
    first_pred_pos = candidate_positions[0]
    initial_train = realized[realized["Position"] <= first_pred_pos - horizon].dropna(subset=feature_cols + [target_col])
    if len(initial_train) < min_train:
        return pd.DataFrame(), {}, pd.DataFrame()

    alpha = choose_alpha(initial_train[feature_cols], initial_train[target_col])
    predictions = []

    for pred_pos in candidate_positions:
        row = usable_features[usable_features["Position"] == pred_pos]
        if row.empty:
            continue
        train = realized[realized["Position"] <= pred_pos - horizon].dropna(subset=feature_cols + [target_col])
        if len(train) < min_train:
            continue

        scaler, ridge = fit_scaled_ridge(train[feature_cols], train[target_col], alpha)
        row_features = np.ascontiguousarray(row[feature_cols].to_numpy(dtype=np.float64, copy=True))
        with threadpool_limits(limits=1):
            ridge_pred = float(ridge.predict(scaler.transform(row_features))[0])

        # Rolling macro regression: transparent DXY + yield relationship benchmark.
        macro_cols = ["DXY Log Return", "DXY Momentum 20", "TNX bp 1D", "TNX bp 20D", "DXY x TNX"]
        macro_pred = macro_regression_prediction(train, row, macro_cols, target_col)

        hist_mean_pred = float(train[target_col].tail(756).mean())
        actual = float(row[target_col].iloc[0])
        predictions.append(
            {
                "Date": row.index[0],
                "Position": pred_pos,
                "Actual": actual,
                "Ridge": ridge_pred,
                "Macro Regression": macro_pred,
                "Historical Mean": hist_mean_pred,
            }
        )
        if len(predictions) % 30 == 0:
            gc.collect()

    pred_df = pd.DataFrame(predictions).set_index("Date") if predictions else pd.DataFrame()
    if pred_df.empty:
        return pred_df, {}, pd.DataFrame()

    model_cols = ["Ridge", "Macro Regression", "Historical Mean"]
    rmse_map = {}
    for col in model_cols:
        rmse_map[col] = float(np.sqrt(np.mean((pred_df["Actual"] - pred_df[col]) ** 2)))
    inv = {k: 1.0 / max(v, 1e-10) for k, v in rmse_map.items()}
    total_inv = sum(inv.values())
    weights = {k: inv[k] / total_inv for k in model_cols}
    pred_df["Ensemble"] = sum(pred_df[col] * weights[col] for col in model_cols)

    benchmark_sse = float(np.sum((pred_df["Actual"] - pred_df["Historical Mean"]) ** 2))
    model_sse = float(np.sum((pred_df["Actual"] - pred_df["Ensemble"]) ** 2))
    oos_r2 = 1.0 - model_sse / benchmark_sse if benchmark_sse > 0 else np.nan
    rmse = float(np.sqrt(np.mean((pred_df["Actual"] - pred_df["Ensemble"]) ** 2)))
    mae = float(np.mean(np.abs(pred_df["Actual"] - pred_df["Ensemble"])))
    hit = float((np.sign(pred_df["Actual"]) == np.sign(pred_df["Ensemble"])).mean())
    corr = float(pred_df["Actual"].corr(pred_df["Ensemble"])) if len(pred_df) > 2 else np.nan
    residual_std = float((pred_df["Actual"] - pred_df["Ensemble"]).std(ddof=1))

    # Latest forecast: only realized targets available at least horizon days before the latest date.
    latest_row = usable_features.iloc[[-1]]
    latest_train = realized[realized["Position"] <= latest_pos - horizon].dropna(subset=feature_cols + [target_col])
    if len(latest_train) < min_train:
        return pred_df, {}, pd.DataFrame()

    scaler, ridge = fit_scaled_ridge(latest_train[feature_cols], latest_train[target_col], alpha)
    latest_features = np.ascontiguousarray(latest_row[feature_cols].to_numpy(dtype=np.float64, copy=True))
    with threadpool_limits(limits=1):
        latest_ridge = float(ridge.predict(scaler.transform(latest_features))[0])
    macro_cols = ["DXY Log Return", "DXY Momentum 20", "TNX bp 1D", "TNX bp 20D", "DXY x TNX"]
    latest_macro = macro_regression_prediction(latest_train, latest_row, macro_cols, target_col)
    latest_mean = float(latest_train[target_col].tail(756).mean())
    latest_ensemble = weights["Ridge"] * latest_ridge + weights["Macro Regression"] * latest_macro + weights["Historical Mean"] * latest_mean
    positive_probability = norm_cdf(latest_ensemble / max(residual_std, 1e-10)) * 100.0

    current_price = float(latest_row["Asset Analysis Price"].iloc[0])
    lower_return = latest_ensemble - 1.96 * residual_std
    upper_return = latest_ensemble + 1.96 * residual_std
    summary = {
        "Horizon": horizon,
        "Forecast Date": latest_row.index[0],
        "Current Price": current_price,
        "Implied Price Target": current_price * math.exp(latest_ensemble),
        "Lower Price Target 95%": current_price * math.exp(lower_return),
        "Upper Price Target 95%": current_price * math.exp(upper_return),
        "Forecast Log Return": latest_ensemble,
        "Forecast Simple Return": math.exp(latest_ensemble) - 1.0,
        "Positive Probability %": positive_probability,
        "Lower 95%": lower_return,
        "Upper 95%": upper_return,
        "OOS R2": oos_r2,
        "RMSE": rmse,
        "MAE": mae,
        "Directional Accuracy": hit,
        "Forecast Correlation": corr,
        "Residual Std": residual_std,
        "Ridge Alpha": alpha,
        "OOS Observations": int(len(pred_df)),
        "Ridge Weight": weights["Ridge"],
        "Macro Weight": weights["Macro Regression"],
        "Historical Mean Weight": weights["Historical Mean"],
    }

    latest_models = pd.DataFrame(
        {
            "Model": ["Ridge", "Macro Regression", "Historical Mean", "Ensemble"],
            "Forecast Log Return %": [latest_ridge * 100.0, latest_macro * 100.0, latest_mean * 100.0, latest_ensemble * 100.0],
            "Ensemble Weight %": [weights["Ridge"] * 100.0, weights["Macro Regression"] * 100.0, weights["Historical Mean"] * 100.0, 100.0],
        }
    )
    return pred_df, summary, latest_models


def regime_snapshot(asset_df: pd.DataFrame, dxy_df: pd.DataFrame, tnx_df: pd.DataFrame) -> Dict[str, str]:
    asset = asset_df.iloc[-1]
    dxy = dxy_df.iloc[-1]
    tnx_bp = yield_bp_change(tnx_df)
    dxy_trend = "Rising USD" if dxy["Analysis Price"] > dxy["EMA 50"] else "Falling USD"
    yield_20 = tnx_bp.rolling(20, min_periods=10).sum().iloc[-1]
    yield_regime = "Rising Yield" if yield_20 > 0 else "Falling Yield"
    vol_series = asset_df["EWMA Vol 0.94"].dropna()
    if len(vol_series) >= 60:
        percentile = float((vol_series <= vol_series.iloc[-1]).mean())
    else:
        percentile = 0.5
    if percentile >= 0.70:
        vol_regime = "High Volatility"
    elif percentile <= 0.30:
        vol_regime = "Low Volatility"
    else:
        vol_regime = "Normal Volatility"
    trend_regime = "Positive Trend" if asset["Analysis Price"] > asset["EMA 50"] else "Negative Trend"
    return {"DXY Regime": dxy_trend, "Yield Regime": yield_regime, "Volatility Regime": vol_regime, "Asset Trend": trend_regime}


def summary_row(name: str, df: pd.DataFrame, dxy_df: Optional[pd.DataFrame]) -> Dict[str, object]:
    meta = INSTRUMENTS[name]
    price = df["Analysis Price"]
    latest = float(price.iloc[-1])
    is_yield = meta["type"] == "yield"
    if is_yield:
        ret_1d = float(price.diff().iloc[-1] * 100.0)
        ret_1m = float(price.diff(20).iloc[-1] * 100.0) if len(price) > 20 else np.nan
        metric_1d = ret_1d
        metric_1m = ret_1m
    else:
        metric_1d = float(price.pct_change().iloc[-1] * 100.0)
        metric_1m = float(price.pct_change(20).iloc[-1] * 100.0) if len(price) > 20 else np.nan
    corr60 = np.nan
    beta60 = np.nan
    if dxy_df is not None and meta["ticker"] != DXY_TICKER:
        relation_series = yield_bp_change(df) if meta["type"] == "yield" else df["Log Return"]
        rel = rolling_relationship(relation_series, dxy_df["Log Return"], 60)
        if not rel.empty:
            corr60 = safe_float(rel["Corr 60"].iloc[-1])
            beta60 = safe_float(rel["Beta 60"].iloc[-1])
    return {
        "Instrument": name,
        "Ticker": meta["ticker"],
        "Latest": latest,
        "1D % / bp": metric_1d,
        "1M % / bp": metric_1m,
        "EWMA Vol %": safe_float(df["EWMA Vol 0.94"].iloc[-1] * 100.0),
        "DXY Corr 60": corr60,
        "DXY Beta 60": beta60,
        "Last Date": df.index[-1].date(),
    }


# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
st.sidebar.markdown("### CommodityMacroPro")
st.sidebar.caption("Yahoo Finance daily market data only. No synthetic series.")
selected_name = st.sidebar.selectbox("Detailed Instrument", COMMODITY_NAMES, index=0)
selected_ticker = INSTRUMENTS[selected_name]["ticker"]

start_date = st.sidebar.date_input("Start Date", pd.Timestamp("2010-01-01"))
end_date = st.sidebar.date_input("End Date", pd.Timestamp.today() + pd.Timedelta(days=1))
log_band_window = st.sidebar.selectbox("Log-Return Difference Band", [20, 60, 120], index=0)
forecast_horizons = st.sidebar.multiselect("Forecast Horizons", [1, 5, 20, 60], default=[1, 5, 20, 60])
max_oos = st.sidebar.slider("Walk-Forward OOS Window", min_value=60, max_value=240, value=120, step=30)

st.sidebar.markdown("---")
st.sidebar.caption("Methodology: Ridge shrinkage, rolling macro regression, historical-mean benchmark, inverse-RMSE forecast combination and regime diagnostics.")
st.sidebar.caption("Cloud numerical engine: single-thread BLAS/OpenMP · shared walk-forward results · stable package pins.")

# -----------------------------------------------------------------------------
# DATA LOAD
# -----------------------------------------------------------------------------
with st.spinner("Retrieving daily Yahoo Finance observations..."):
    market_data, governance = download_universe(str(start_date), str(end_date))

required_tickers = [selected_ticker, DXY_TICKER, TNX_TICKER]
missing_required = [t for t in required_tickers if t not in market_data]
if missing_required:
    st.error("Required Yahoo Finance series could not be loaded: " + ", ".join(missing_required))
    st.dataframe(governance, width="stretch", hide_index=True)
    st.stop()

selected_df = market_data[selected_ticker]
dxy_df = market_data[DXY_TICKER]
tnx_df = market_data[TNX_TICKER]

if len(selected_df) < 500:
    st.warning("The selected instrument has fewer than 500 daily observations. Forecast validation may be unavailable or less stable.")

# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hf-masthead">
      <div class="hf-eyebrow">MK FinTECH LabGEN · Institutional Commodity & Macro Analytics</div>
      <div><span class="hf-title">{APP_NAME}</span><span class="hf-version">Version {APP_VERSION}</span></div>
      <div class="hf-meta">Daily commodities · DXY transmission · UST 10Y yield channel · EWMA risk · Walk-forward predictive analytics</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="instrument-header">
      <div class="instrument-kicker">Selected Research Instrument</div>
      <div class="instrument-title">{selected_name}<span>{selected_ticker}</span></div>
      <div class="instrument-subtitle">Yahoo Finance daily data · {selected_df.index.min().date()} to {selected_df.index.max().date()} · {len(selected_df):,} valid observations</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# PRECOMPUTED SNAPSHOTS
# -----------------------------------------------------------------------------
regime = regime_snapshot(selected_df, dxy_df, tnx_df)
relationship_60 = rolling_relationship(selected_df["Log Return"], dxy_df["Log Return"], 60)
latest_corr60 = safe_float(relationship_60["Corr 60"].iloc[-1]) if not relationship_60.empty else np.nan
latest_beta60 = safe_float(relationship_60["Beta 60"].iloc[-1]) if not relationship_60.empty else np.nan
last_price = float(selected_df["Analysis Price"].iloc[-1])
ret_1d = float(selected_df["Analysis Price"].pct_change().iloc[-1] * 100.0)
ret_20d = float(selected_df["Analysis Price"].pct_change(20).iloc[-1] * 100.0) if len(selected_df) > 20 else np.nan
vol_latest = float(selected_df["EWMA Vol 0.94"].iloc[-1] * 100.0)

kpi_cols = st.columns(8)
kpi_cols[0].metric("Latest Price", fmt_number(last_price, 2))
kpi_cols[1].metric("1D Return", fmt_number(ret_1d, 2, "%"))
kpi_cols[2].metric("20D Return", fmt_number(ret_20d, 2, "%"))
kpi_cols[3].metric("EWMA Vol", fmt_number(vol_latest, 2, "%"))
kpi_cols[4].metric("DXY Corr 60D", fmt_number(latest_corr60, 3))
kpi_cols[5].metric("DXY Beta 60D", fmt_number(latest_beta60, 3))
kpi_cols[6].metric("DXY Regime", regime["DXY Regime"])
kpi_cols[7].metric("Risk Regime", regime["Volatility Regime"])

# -----------------------------------------------------------------------------
# SHARED FORECAST RESULTS — COMPUTED ONCE PER RERUN
# -----------------------------------------------------------------------------
forecast_results: Dict[int, Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]] = {}
if forecast_horizons:
    with st.spinner("Running single-pass walk-forward forecasts..."):
        for _h in sorted(set(int(x) for x in forecast_horizons)):
            forecast_results[_h] = walk_forward_forecast(
                selected_df, dxy_df, tnx_df, _h, min_train=400, max_oos=max_oos
            )
    gc.collect()

# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------
tabs = st.tabs(
    [
        "Executive Dashboard",
        "Smart Price Structure",
        "EWMA Volatility",
        "DXY Relationship",
        "Forecast Laboratory",
        "Log Return Difference ±2σ",
        "Cross-Asset Matrix",
        "Model Validation",
        "Data Governance",
    ]
)

# 1) EXECUTIVE DASHBOARD
with tabs[0]:
    st.subheader("Cross-Market Executive Dashboard")
    st.markdown('<div class="section-note">A consolidated view of commodity prices, DXY sensitivity and current risk state. Yield moves are expressed in basis points; price instruments use percentage returns.</div>', unsafe_allow_html=True)
    rows = []
    for name, meta in INSTRUMENTS.items():
        ticker = meta["ticker"]
        if ticker in market_data:
            rows.append(summary_row(name, market_data[ticker], dxy_df))
    summary_df = pd.DataFrame(rows)
    render_dataframe(
        summary_df,
        {
            "Latest": "%.3f",
            "1D % / bp": "%.2f",
            "1M % / bp": "%.2f",
            "EWMA Vol %": "%.2f",
            "DXY Corr 60": "%.3f",
            "DXY Beta 60": "%.3f",
        },
    )

    commodity_tickers = [INSTRUMENTS[n]["ticker"] for n in COMMODITY_NAMES if INSTRUMENTS[n]["ticker"] in market_data]
    normalized = []
    for ticker in commodity_tickers + [DXY_TICKER]:
        p = market_data[ticker]["Analysis Price"].dropna()
        normalized.append((p / p.iloc[0] * 100.0).rename(TICKER_TO_NAME[ticker]))
    norm_df = pd.concat(normalized, axis=1, join="inner").dropna()
    fig = px.line(norm_df, x=norm_df.index, y=norm_df.columns, title="Normalized Market Performance — Common Daily Sample (Base = 100)")
    fig.update_layout(template="plotly_white", height=560, hovermode="x unified", legend_title_text="")
    fig.update_xaxes(rangeselector=dict(buttons=[dict(count=6, label="6M", step="month", stepmode="backward"), dict(count=1, label="1Y", step="year", stepmode="backward"), dict(count=3, label="3Y", step="year", stepmode="backward"), dict(step="all", label="ALL")]))
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### Current Regime Matrix")
    regime_df = pd.DataFrame([{"Dimension": k, "Current State": v} for k, v in regime.items()])
    st.dataframe(regime_df, width="stretch", hide_index=True)

# 2) SMART PRICE STRUCTURE
with tabs[1]:
    st.subheader("Institutional Smart Price Structure")
    st.markdown('<div class="section-note">Candlestick structure, trend averages, Bollinger envelope, regression trend channel, anchored VWAP, swing points and support/resistance levels are calculated solely from observed daily Yahoo Finance data.</div>', unsafe_allow_html=True)
    st.plotly_chart(build_price_chart(selected_df, f"{selected_name} — Institutional Price Structure"), width="stretch")

    structure = pd.DataFrame(
        {
            "Metric": ["Adjusted / Analysis Price", "EMA 20", "EMA 50", "EMA 200", "RSI", "ATR %", "20D Support", "20D Resistance"],
            "Latest": [
                selected_df["Analysis Price"].iloc[-1],
                selected_df["EMA 20"].iloc[-1],
                selected_df["EMA 50"].iloc[-1],
                selected_df["EMA 200"].iloc[-1],
                selected_df["RSI"].iloc[-1],
                selected_df["ATR %"].iloc[-1] * 100.0,
                selected_df["Low"].rolling(20).min().iloc[-1],
                selected_df["High"].rolling(20).max().iloc[-1],
            ],
        }
    )
    render_dataframe(structure, {"Latest": "%.3f"})

# 3) EWMA VOLATILITY
with tabs[2]:
    st.subheader("EWMA Volatility Laboratory")
    st.markdown('<div class="section-note">EWMA λ=0.94 is the primary daily risk estimate. λ=0.97 and rolling 20D/60D/252D volatility provide persistence and horizon comparisons.</div>', unsafe_allow_html=True)
    vol_df = selected_df[["EWMA Vol 0.94", "EWMA Vol 0.97", "Rolling Vol 20", "Rolling Vol 60", "Rolling Vol 252"]].dropna(how="all") * 100.0
    fig = px.line(vol_df, x=vol_df.index, y=vol_df.columns, title=f"{selected_name} — Annualized Volatility Comparison")
    fig.update_layout(template="plotly_white", height=590, hovermode="x unified", legend_title_text="")
    fig.update_yaxes(title="Annualized Volatility %")
    fig.update_xaxes(rangeselector=dict(buttons=[dict(count=6, label="6M", step="month", stepmode="backward"), dict(count=1, label="1Y", step="year", stepmode="backward"), dict(count=3, label="3Y", step="year", stepmode="backward"), dict(step="all", label="ALL")]))
    st.plotly_chart(fig, width="stretch")

    ewma = selected_df["EWMA Vol 0.94"].dropna()
    vol_percentile = float((ewma <= ewma.iloc[-1]).mean() * 100.0) if not ewma.empty else np.nan
    vol_z = float((ewma.iloc[-1] - ewma.tail(252).mean()) / ewma.tail(252).std(ddof=1)) if len(ewma.tail(252)) > 20 else np.nan
    vol_change_20 = float((ewma.iloc[-1] / ewma.shift(20).iloc[-1] - 1.0) * 100.0) if len(ewma) > 20 and ewma.shift(20).iloc[-1] != 0 else np.nan
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("EWMA λ=0.94", fmt_number(ewma.iloc[-1] * 100.0, 2, "%"))
    c2.metric("Historical Percentile", fmt_number(vol_percentile, 1, "%"))
    c3.metric("252D Vol Z-Score", fmt_number(vol_z, 2))
    c4.metric("20D Vol Change", fmt_number(vol_change_20, 1, "%"))

    top_vol = selected_df[["EWMA Vol 0.94", "Log Return"]].dropna().nlargest(10, "EWMA Vol 0.94").copy()
    top_vol["EWMA Vol %"] = top_vol["EWMA Vol 0.94"] * 100.0
    top_vol["Daily Log Return %"] = top_vol["Log Return"] * 100.0
    st.markdown("#### Highest EWMA Volatility Observations")
    render_dataframe(top_vol[["EWMA Vol %", "Daily Log Return %"]], {"EWMA Vol %": "%.2f", "Daily Log Return %": "%.2f"}, hide_index=False)

# 4) DXY RELATIONSHIP
with tabs[3]:
    st.subheader("DXY Transmission and Lead–Lag Analysis")
    st.markdown('<div class="section-note">The DXY relationship is tested rather than assumed. Pairwise inner joins are used; missing return observations are not forward-filled.</div>', unsafe_allow_html=True)
    rel_parts = [rolling_relationship(selected_df["Log Return"], dxy_df["Log Return"], w) for w in [20, 60, 120, 252]]
    rel = pd.concat(rel_parts, axis=1)
    rel["EWMA Corr 60"] = ewma_correlation(selected_df["Log Return"], dxy_df["Log Return"], 60)

    corr_cols = ["Corr 20", "Corr 60", "Corr 120", "Corr 252", "EWMA Corr 60"]
    corr_fig = px.line(rel[corr_cols], x=rel.index, y=corr_cols, title=f"{selected_name} vs DXY — Dynamic Correlation")
    corr_fig.add_hline(y=0.0, line_dash="dot", line_color="#667085")
    corr_fig.update_layout(template="plotly_white", height=560, hovermode="x unified", legend_title_text="")
    corr_fig.update_yaxes(range=[-1, 1], title="Correlation")
    st.plotly_chart(corr_fig, width="stretch")

    beta_cols = ["Beta 20", "Beta 60", "Beta 120", "Beta 252"]
    beta_fig = px.line(rel[beta_cols], x=rel.index, y=beta_cols, title=f"{selected_name} vs DXY — Rolling Beta")
    beta_fig.add_hline(y=0.0, line_dash="dot", line_color="#667085")
    beta_fig.update_layout(template="plotly_white", height=520, hovermode="x unified", legend_title_text="")
    st.plotly_chart(beta_fig, width="stretch")

    pair = pd.concat([selected_df["Log Return"].rename("Commodity"), dxy_df["Log Return"].rename("DXY")], axis=1, join="inner").dropna().tail(756)
    pair_pct = pair * 100.0
    scatter = go.Figure()
    scatter.add_trace(go.Scatter(
        x=pair_pct["DXY"], y=pair_pct["Commodity"], mode="markers", name="Daily Observations",
        marker=dict(size=6, color="#214b73", opacity=0.45),
        hovertemplate="DXY=%{x:.3f}%<br>Commodity=%{y:.3f}%<extra></extra>",
    ))
    if len(pair_pct) >= 30 and pair_pct["DXY"].std(ddof=1) > 0:
        slope, intercept = np.polyfit(pair_pct["DXY"].values, pair_pct["Commodity"].values, 1)
        x_line = np.linspace(pair_pct["DXY"].min(), pair_pct["DXY"].max(), 120)
        scatter.add_trace(go.Scatter(
            x=x_line, y=intercept + slope * x_line, mode="lines", name=f"OLS Fit (β={slope:.2f})",
            line=dict(color="#9f2d2d", width=2.0),
        ))
    scatter.update_layout(
        title=f"Daily Return Map — {selected_name} vs DXY (Latest 756 Common Observations)",
        template="plotly_white", height=540, xaxis_title="DXY Daily Log Return %",
        yaxis_title=f"{selected_name} Daily Log Return %",
    )
    st.plotly_chart(scatter, width="stretch")

    lag_df = lead_lag_correlations(selected_df["Log Return"], dxy_df["Log Return"], 20)
    lag_fig = go.Figure(go.Bar(x=lag_df["Lag"], y=lag_df["Correlation"], marker_color=np.where(lag_df["Correlation"] >= 0, "#214b73", "#9f2d2d")))
    lag_fig.update_layout(title="Lead–Lag Correlation: Today's DXY Return vs Commodity Return at Lag h", template="plotly_white", height=470, xaxis_title="Lag h (positive = commodity return occurs later)", yaxis_title="Correlation")
    st.plotly_chart(lag_fig, width="stretch")

    scenario_df, regression_stats = regression_scenarios(selected_df, dxy_df, tnx_df)
    st.markdown("#### Conditional DXY / UST10Y Shock Scenarios")
    if scenario_df.empty:
        st.warning("Insufficient common observations for the scenario regression.")
    else:
        left, right = st.columns([1.55, 1.0])
        with left:
            render_dataframe(scenario_df, {c: "%.2f" for c in scenario_df.columns if c != "Scenario"})
        with right:
            stats_table = pd.DataFrame([{"Statistic": k, "Value": v} for k, v in regression_stats.items()])
            render_dataframe(stats_table, {"Value": "%.6f"})

# 5) FORECAST LABORATORY
with tabs[4]:
    st.subheader("Walk-Forward Forecast Laboratory")
    st.markdown('<div class="section-note">Forecasts are generated with information available at each historical decision date. A Ridge model, rolling macro regression and historical-mean benchmark are combined using inverse out-of-sample RMSE weights.</div>', unsafe_allow_html=True)

    forecast_summaries = []
    forecast_outputs: Dict[int, pd.DataFrame] = {}
    model_tables: Dict[int, pd.DataFrame] = {}
    if not forecast_horizons:
        st.info("Select at least one forecast horizon from the sidebar.")
    else:
        for h in sorted(forecast_results):
            pred_df, summary, model_table = forecast_results[h]
            if summary:
                forecast_summaries.append(summary)
                forecast_outputs[h] = pred_df
                model_tables[h] = model_table

        if not forecast_summaries:
            st.warning("The common sample is insufficient for walk-forward validation. Extend the start date or choose an instrument with a longer history.")
        else:
            summary_table = pd.DataFrame(forecast_summaries)
            display = summary_table.copy()
            for col in ["Forecast Log Return", "Forecast Simple Return", "Lower 95%", "Upper 95%", "OOS R2", "RMSE", "MAE", "Directional Accuracy", "Forecast Correlation", "Residual Std", "Ridge Weight", "Macro Weight", "Historical Mean Weight"]:
                if col in display:
                    display[col] = display[col] * 100.0
            render_dataframe(
                display,
                {
                    "Current Price": "%.3f",
                    "Implied Price Target": "%.3f",
                    "Lower Price Target 95%": "%.3f",
                    "Upper Price Target 95%": "%.3f",
                    "Forecast Log Return": "%.2f%%",
                    "Forecast Simple Return": "%.2f%%",
                    "Positive Probability %": "%.1f%%",
                    "Lower 95%": "%.2f%%",
                    "Upper 95%": "%.2f%%",
                    "OOS R2": "%.2f%%",
                    "RMSE": "%.2f%%",
                    "MAE": "%.2f%%",
                    "Directional Accuracy": "%.1f%%",
                    "Forecast Correlation": "%.2f%%",
                    "Residual Std": "%.2f%%",
                    "Ridge Weight": "%.1f%%",
                    "Macro Weight": "%.1f%%",
                    "Historical Mean Weight": "%.1f%%",
                },
            )

            for h in sorted(forecast_outputs):
                pred_df = forecast_outputs[h]
                summary = next(x for x in forecast_summaries if x["Horizon"] == h)
                st.markdown(f"#### {h}-Trading-Day Forecast")
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Expected Return", fmt_number(summary["Forecast Simple Return"] * 100.0, 2, "%"))
                c2.metric("Implied Target", fmt_number(summary["Implied Price Target"], 3))
                c3.metric("Positive Probability", fmt_number(summary["Positive Probability %"], 1, "%"))
                c4.metric("Directional Accuracy", fmt_number(summary["Directional Accuracy"] * 100.0, 1, "%"))
                c5.metric("OOS R² vs Mean", fmt_number(summary["OOS R2"] * 100.0, 2, "%"))

                chart_df = pred_df[["Actual", "Ensemble", "Ridge", "Macro Regression"]].copy() * 100.0
                ffig = px.line(chart_df, x=chart_df.index, y=chart_df.columns, title=f"{h}D Walk-Forward Forecast vs Realized Log Return")
                ffig.update_layout(template="plotly_white", height=480, hovermode="x unified", legend_title_text="")
                ffig.update_yaxes(title="Forward Log Return %")
                st.plotly_chart(ffig, width="stretch")
                render_dataframe(model_tables[h], {"Forecast Log Return %": "%.3f", "Ensemble Weight %": "%.1f"})

# 6) LOG RETURN DIFFERENCE ±2 SIGMA
with tabs[5]:
    st.subheader("Daily Log-Return Difference and Statistical Bands")
    st.markdown('<div class="section-note">Return Difference is Δrₜ = rₜ − rₜ₋₁. The upper and lower control bands are the rolling mean ± 2 standard deviations. Band breaches are observed market outcomes, not generated data.</div>', unsafe_allow_html=True)
    diff = selected_df["Return Difference"]
    mean = diff.rolling(log_band_window, min_periods=max(10, log_band_window // 2)).mean()
    sigma = diff.rolling(log_band_window, min_periods=max(10, log_band_window // 2)).std(ddof=1)
    upper = mean + 2.0 * sigma
    lower = mean - 2.0 * sigma
    band_df = pd.DataFrame({"Return Difference": diff * 100.0, "Rolling Mean": mean * 100.0, "+2 Sigma": upper * 100.0, "-2 Sigma": lower * 100.0}).dropna(how="all")
    upper_breach = diff > upper
    lower_breach = diff < lower

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=band_df.index, y=band_df["+2 Sigma"], mode="lines", name="+2 Sigma", line=dict(color="#9f2d2d", width=1.0, dash="dash")))
    fig.add_trace(go.Scatter(x=band_df.index, y=band_df["-2 Sigma"], mode="lines", name="-2 Sigma", line=dict(color="#176b45", width=1.0, dash="dash"), fill="tonexty", fillcolor="rgba(33,75,115,0.05)"))
    fig.add_trace(go.Scatter(x=band_df.index, y=band_df["Rolling Mean"], mode="lines", name="Rolling Mean", line=dict(color="#214b73", width=1.3)))
    fig.add_trace(go.Scatter(x=band_df.index, y=band_df["Return Difference"], mode="lines", name="Daily Return Difference", line=dict(color="#344054", width=1.0)))
    fig.add_trace(go.Scatter(x=selected_df.index[upper_breach.fillna(False)], y=(diff[upper_breach.fillna(False)] * 100.0), mode="markers", name="Upper Breach", marker=dict(color="#9f2d2d", size=8, symbol="triangle-up")))
    fig.add_trace(go.Scatter(x=selected_df.index[lower_breach.fillna(False)], y=(diff[lower_breach.fillna(False)] * 100.0), mode="markers", name="Lower Breach", marker=dict(color="#176b45", size=8, symbol="triangle-down")))
    fig.update_layout(title=f"{selected_name} — Δ Log Return with {log_band_window}D Mean ± 2σ", template="plotly_white", height=620, hovermode="x unified", legend=dict(orientation="h", y=1.02, x=1.0, xanchor="right"))
    fig.update_yaxes(title="Return Difference %")
    st.plotly_chart(fig, width="stretch")

    future_5 = selected_df["Analysis Price"].shift(-5) / selected_df["Analysis Price"] - 1.0
    future_20 = selected_df["Analysis Price"].shift(-20) / selected_df["Analysis Price"] - 1.0
    breach_stats = []
    for label, mask in [("Upper +2σ Breach", upper_breach), ("Lower -2σ Breach", lower_breach)]:
        resolved = pd.DataFrame({"Mask": mask, "Fwd 5D": future_5, "Fwd 20D": future_20}).dropna()
        resolved = resolved[resolved["Mask"]]
        breach_stats.append(
            {
                "Event": label,
                "Resolved Events": int(len(resolved)),
                "Average Forward 5D %": float(resolved["Fwd 5D"].mean() * 100.0) if len(resolved) else np.nan,
                "Positive Forward 5D %": float((resolved["Fwd 5D"] > 0).mean() * 100.0) if len(resolved) else np.nan,
                "Average Forward 20D %": float(resolved["Fwd 20D"].mean() * 100.0) if len(resolved) else np.nan,
                "Positive Forward 20D %": float((resolved["Fwd 20D"] > 0).mean() * 100.0) if len(resolved) else np.nan,
            }
        )
    render_dataframe(pd.DataFrame(breach_stats), {c: "%.2f" for c in breach_stats[0] if c not in ["Event", "Resolved Events"]})

# 7) CROSS-ASSET MATRIX
with tabs[6]:
    st.subheader("Cross-Asset Dependence Matrix")
    st.markdown('<div class="section-note">All matrices use a common daily sample. The UST10Y series is represented by daily basis-point changes; price instruments use daily log returns.</div>', unsafe_allow_html=True)
    series = []
    for name, meta in INSTRUMENTS.items():
        ticker = meta["ticker"]
        if ticker not in market_data:
            continue
        if meta["type"] == "yield":
            s = yield_bp_change(market_data[ticker]).rename(meta["short"])
        else:
            s = market_data[ticker]["Log Return"].rename(meta["short"])
        series.append(s)
    cross = pd.concat(series, axis=1, join="inner").dropna()
    lookback = min(504, len(cross))
    corr_matrix = cross.tail(lookback).corr()
    heat = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1, title=f"Cross-Asset Correlation Matrix — Latest {lookback} Common Daily Observations")
    heat.update_layout(template="plotly_white", height=650)
    st.plotly_chart(heat, width="stretch")

    rolling_selected = pd.DataFrame(index=cross.index)
    for col in cross.columns:
        if col != INSTRUMENTS[selected_name]["short"]:
            rolling_selected[col] = cross[INSTRUMENTS[selected_name]["short"]].rolling(60, min_periods=40).corr(cross[col])
    roll_fig = px.line(rolling_selected, x=rolling_selected.index, y=rolling_selected.columns, title=f"{selected_name} — 60D Rolling Cross-Asset Correlations")
    roll_fig.update_layout(template="plotly_white", height=560, hovermode="x unified", legend_title_text="")
    roll_fig.update_yaxes(range=[-1, 1], title="Correlation")
    st.plotly_chart(roll_fig, width="stretch")

# 8) MODEL VALIDATION
with tabs[7]:
    st.subheader("Model Validation and Governance")
    st.markdown('<div class="section-note">A model is not accepted merely because it fits the historical sample. The primary evidence is walk-forward, out-of-sample performance against the historical-mean benchmark.</div>', unsafe_allow_html=True)
    validation_rows = []
    if forecast_horizons:
        for h in sorted(forecast_results):
            pred_df, summary, _ = forecast_results[h]
            if summary:
                validation_rows.append(
                {
                        "Horizon": h,
                        "OOS Observations": summary["OOS Observations"],
                        "OOS R2 vs Mean %": summary["OOS R2"] * 100.0,
                        "RMSE %": summary["RMSE"] * 100.0,
                        "MAE %": summary["MAE"] * 100.0,
                        "Directional Accuracy %": summary["Directional Accuracy"] * 100.0,
                        "Forecast Correlation": summary["Forecast Correlation"],
                        "Residual Std %": summary["Residual Std"] * 100.0,
                        "Ridge Alpha": summary["Ridge Alpha"],
                        "Status": "PASS" if summary["OOS R2"] > 0 and summary["Directional Accuracy"] >= 0.52 else "REVIEW",
                    }
                )
    if validation_rows:
        validation_df = pd.DataFrame(validation_rows)
        render_dataframe(validation_df, {
            "OOS R2 vs Mean %": "%.2f",
            "RMSE %": "%.2f",
            "MAE %": "%.2f",
            "Directional Accuracy %": "%.1f",
            "Forecast Correlation": "%.3f",
            "Residual Std %": "%.2f",
            "Ridge Alpha": "%.2f",
        })

        val_fig = px.bar(validation_df, x="Horizon", y="OOS R2 vs Mean %", color="Status", barmode="group", title="Out-of-Sample R² by Forecast Horizon")
        val_fig.add_hline(y=0.0, line_dash="dot", line_color="#667085")
        val_fig.update_layout(template="plotly_white", height=450)
        st.plotly_chart(val_fig, width="stretch")
    else:
        st.info("No validation result is available for the current date range and horizon selection.")

    methodology = pd.DataFrame(
        [
            ("Data", "Yahoo Finance daily observations only; auto_adjust=False"),
            ("Price Return", "Log return of Adj Close where available, otherwise Close"),
            ("UST10Y Transformation", "Daily percentage-point change × 100 = basis-point change"),
            ("Forecast Target", "Forward 1D / 5D / 20D / 60D log return"),
            ("Predictive Engine", "Ridge shrinkage + rolling macro regression + historical mean"),
            ("Combination", "Inverse walk-forward RMSE weights"),
            ("Validation", "Expanding walk-forward; target availability lag enforced"),
            ("No Leakage", "Training targets restricted to dates resolved by each decision date"),
            ("Missing Data", "No forward filling of daily returns"),
            ("Synthetic Data", "Not used"),
        ],
        columns=["Control", "Implementation"],
    )
    st.markdown("#### Methodology Controls")
    st.dataframe(methodology, width="stretch", hide_index=True)

# 9) DATA GOVERNANCE
with tabs[8]:
    st.subheader("Data Governance and Download Center")
    st.markdown('<div class="section-note">The table below records the actual observations retrieved from Yahoo Finance. Failed instruments remain visible and are not replaced with proxies or synthetic series.</div>', unsafe_allow_html=True)
    st.dataframe(governance, width="stretch", hide_index=True)

    selected_export = selected_df.copy()
    selected_export.index.name = "Date"
    st.download_button(
        label=f"Download {selected_ticker} Daily Data CSV",
        data=selected_export.to_csv(index=True).encode("utf-8"),
        file_name=f"{selected_ticker.replace('^', '').replace('=', '_')}_daily_yahoo.csv",
        mime="text/csv",
    )

    all_returns = []
    for name, meta in INSTRUMENTS.items():
        ticker = meta["ticker"]
        if ticker not in market_data:
            continue
        if meta["type"] == "yield":
            s = yield_bp_change(market_data[ticker]).rename(f"{ticker}_BP_CHANGE")
        else:
            s = market_data[ticker]["Log Return"].rename(f"{ticker}_LOG_RETURN")
        all_returns.append(s)
    if all_returns:
        export_returns = pd.concat(all_returns, axis=1)
        export_returns.index.name = "Date"
        st.download_button(
            label="Download Cross-Asset Return Matrix CSV",
            data=export_returns.to_csv(index=True).encode("utf-8"),
            file_name="CommodityMacroPro_cross_asset_daily_returns.csv",
            mime="text/csv",
        )

st.markdown("---")
st.caption(f"By MK FinTECH LabGEN@2026 Istanbul · {APP_NAME} V{APP_VERSION} · Yahoo Finance daily market observations only · Single-thread cloud numerical engine · Not investment advice")
