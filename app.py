# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# SUPERTRENDPRO INSTITUTIONAL V5.0.2 – NO SYNTHETIC DATA + LEADING SIGNAL LAB
# Trend Following + Smart Supertrend + Beta + Risk Metrics
# Expanded BIST Blue-Chip Universe + Capital Gain Leaders Lab
# -------------------------------------------------------------------------
# Save as: app.py
# Run:
#   streamlit run app.py --server.port 8516
# -------------------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore")

import os

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import itertools

# -------------------------------------------------------------------------
# INDICATOR ENGINE: TA-Lib is automatically preferred when installed.
# The user can explicitly select Auto, TA-Lib, or Pandas/NumPy in the sidebar.
# No synthetic prices are ever generated; only indicator formulas can fall back.
# -------------------------------------------------------------------------
try:
    import talib as ta
    TALIB_INSTALLED = True
    TALIB_VERSION = getattr(ta, "__version__", "installed")
    TALIB_IMPORT_ERROR = ""
except Exception as exc:
    ta = None
    TALIB_INSTALLED = False
    TALIB_VERSION = "not installed"
    TALIB_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

# This runtime flag is resolved from the sidebar before any market data is processed.
TALIB_AVAILABLE = False
ACTIVE_INDICATOR_ENGINE = "Pandas / NumPy"

TRADING_DAYS = 252
ROLLING_BETA_WINDOW = 60
ROLLING_VOL_WINDOW = 63
MIN_PRICE_OBS = 120
BENCHMARK_SYMBOL = "XU100.IS"
APP_VERSION = "5.0.2"
APP_RELEASE_NAME = "SupertrendPro Institutional V5.0.2"

st.set_page_config(
    layout="wide",
    page_title="SupertrendPro Institutional V5.0.2",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------------
# STYLING
# -------------------------------------------------------------------------
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
      }

      html, body, [class*="css"] {
        font-family: "Inter", "Aptos", "Segoe UI", Arial, sans-serif;
        color: var(--hf-navy);
      }

      .stApp {
        background: var(--hf-white);
      }

      .block-container {
        max-width: 1680px;
        padding-top: 1.15rem;
        padding-bottom: 2.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
      }

      h1, h2, h3, h4, h5, h6 {
        font-family: "Inter", "Aptos", "Segoe UI", Arial, sans-serif;
        color: var(--hf-navy);
        letter-spacing: -0.025em;
      }

      h1 {
        font-size: 2.00rem !important;
        line-height: 1.18 !important;
        font-weight: 300 !important;
        margin-bottom: 0.35rem !important;
      }

      h2 {
        font-size: 1.45rem !important;
        line-height: 1.25 !important;
        font-weight: 350 !important;
        margin-top: 1.55rem !important;
        margin-bottom: 0.55rem !important;
      }

      h3 {
        font-size: 1.10rem !important;
        line-height: 1.30 !important;
        font-weight: 450 !important;
        margin-top: 1.30rem !important;
        margin-bottom: 0.45rem !important;
      }

      h4 {
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.01em !important;
      }

      p, label, .stMarkdown, .stCaption {
        color: var(--hf-slate);
      }

      hr {
        border: 0;
        border-top: 1px solid var(--hf-border);
        margin: 1.25rem 0 1.35rem 0;
      }

      .hf-masthead {
        margin: 0.15rem 0 1.10rem 0;
        padding: 0.15rem 0 1.05rem 0;
        border-bottom: 1px solid var(--hf-border);
      }

      .hf-eyebrow {
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--hf-muted);
        margin-bottom: 0.48rem;
      }

      .hf-brand-row {
        display: flex;
        align-items: baseline;
        gap: 0.70rem;
        flex-wrap: wrap;
      }

      .hf-title {
        font-size: 2.05rem;
        line-height: 1.12;
        font-weight: 300;
        letter-spacing: -0.045em;
        color: var(--hf-navy);
      }

      .hf-version {
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.08em;
        color: var(--hf-accent);
        text-transform: uppercase;
      }

      .hf-meta {
        margin-top: 0.48rem;
        font-size: 0.78rem;
        font-weight: 400;
        letter-spacing: 0.045em;
        color: var(--hf-muted);
      }

      .instrument-header {
        margin: 1.25rem 0 0.85rem 0;
        padding: 0 0 0.85rem 0;
        border-bottom: 1px solid var(--hf-border);
      }

      .instrument-kicker {
        font-size: 0.64rem;
        line-height: 1;
        font-weight: 600;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--hf-muted);
        margin-bottom: 0.42rem;
      }

      .instrument-title {
        font-size: 1.65rem;
        line-height: 1.20;
        font-weight: 350;
        letter-spacing: -0.035em;
        color: var(--hf-navy);
      }

      .instrument-title span {
        font-size: 0.82rem;
        font-weight: 500;
        letter-spacing: 0.06em;
        color: var(--hf-accent);
        margin-left: 0.35rem;
      }

      .instrument-subtitle {
        margin-top: 0.32rem;
        font-size: 0.76rem;
        font-weight: 400;
        color: var(--hf-muted);
        letter-spacing: 0.025em;
      }

      div[data-testid="stMetric"] {
        background: var(--hf-white);
        border: 1px solid var(--hf-border);
        border-radius: 4px;
        padding: 0.72rem 0.82rem 0.66rem 0.82rem;
        box-shadow: none;
      }

      div[data-testid="stMetricLabel"] {
        font-size: 0.66rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.075em !important;
        text-transform: uppercase;
        color: var(--hf-muted) !important;
      }

      div[data-testid="stMetricValue"] {
        font-size: 1.30rem !important;
        line-height: 1.12 !important;
        font-weight: 400 !important;
        color: var(--hf-navy) !important;
      }

      div[data-testid="stMetricDelta"] {
        font-size: 0.72rem !important;
        font-weight: 500 !important;
      }

      div[data-testid="stTabs"] button {
        min-height: 2.55rem;
        padding: 0.30rem 0.72rem;
        border-radius: 0;
      }

      div[data-testid="stTabs"] button p {
        font-size: 0.68rem !important;
        font-weight: 550 !important;
        letter-spacing: 0.055em !important;
        text-transform: uppercase;
        color: var(--hf-muted) !important;
      }

      div[data-testid="stTabs"] button[aria-selected="true"] {
        border-bottom: 2px solid var(--hf-navy) !important;
      }

      div[data-testid="stTabs"] button[aria-selected="true"] p {
        color: var(--hf-navy) !important;
      }

      section[data-testid="stSidebar"] {
        background: #f8fafc;
        border-right: 1px solid var(--hf-border);
      }

      section[data-testid="stSidebar"] h1,
      section[data-testid="stSidebar"] h2,
      section[data-testid="stSidebar"] h3 {
        font-weight: 400 !important;
      }

      section[data-testid="stSidebar"] label,
      section[data-testid="stSidebar"] p {
        font-size: 0.78rem;
      }

      .small-note {
        font-size: 0.76rem;
        line-height: 1.55;
        color: var(--hf-muted);
      }

      .risk-note,
      .ok-note {
        padding: 0.78rem 0.92rem;
        border-radius: 3px;
        font-size: 0.78rem;
        line-height: 1.50;
      }

      .risk-note {
        background: #fffaf3;
        border: 1px solid #eadbc8;
        color: #6b4f33;
      }

      .ok-note {
        background: #f4f8f6;
        border: 1px solid #cfddd5;
        color: #315244;
      }

      .stButton > button,
      .stDownloadButton > button {
        border-radius: 3px;
        border: 1px solid #cfd6de;
        background: #ffffff;
        color: var(--hf-navy);
        font-size: 0.72rem;
        font-weight: 550;
        letter-spacing: 0.045em;
        text-transform: uppercase;
        box-shadow: none;
      }

      .stButton > button:hover,
      .stDownloadButton > button:hover {
        border-color: var(--hf-accent);
        color: var(--hf-accent);
      }

      div[data-testid="stDataFrame"] {
        border: 1px solid var(--hf-border);
        border-radius: 3px;
        overflow: hidden;
      }

      div[data-testid="stAlert"] {
        border-radius: 3px;
        border-width: 1px;
      }

      .stCaptionContainer,
      [data-testid="stCaptionContainer"] {
        font-size: 0.72rem !important;
        color: var(--hf-muted) !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------------
# MARKET UNIVERSE – EXPANDED BIST BLUE CHIPS + USER'S CAPITAL GAIN LIST
# No synthetic data: tickers are used only to fetch real Yahoo Finance data.
# -------------------------------------------------------------------------
MARKET_DATA: Dict[str, Dict[str, str]] = {
    "Indices": {
        "BIST 100 Index": "XU100.IS",
        "BIST 30 Index": "XU030.IS",
        "BIST Bank Index": "XBANK.IS",
        "BIST Industrial Index": "XUSIN.IS",
        "BIST Holding & Investment Index": "XHOLD.IS",
    },
    "Major Banks & Financials": {
        "Akbank": "AKBNK.IS",
        "Garanti BBVA": "GARAN.IS",
        "Is Bankasi (C)": "ISCTR.IS",
        "Yapi Kredi": "YKBNK.IS",
        "QNB Bank": "QNBTR.IS",
        "VakifBank": "VAKBN.IS",
        "Halkbank": "HALKB.IS",
        "TSKB": "TSKB.IS",
        "Albaraka Turk": "ALBRK.IS",
        "Sekerbank": "SKBNK.IS",
        "Turkiye Sigorta": "TURSG.IS",
        "Anadolu Hayat Emeklilik": "ANHYT.IS",
        "Aksigorta": "AKGRT.IS",
        "Ray Sigorta": "RAYSG.IS",
        "Is Leasing": "ISFIN.IS",
        "Garanti Factoring": "GARFA.IS",
        "Vakif Leasing": "VAKFN.IS",
        "Lider Factoring": "LIDFA.IS",
    },
    "Holdings & Conglomerates": {
        "Koc Holding": "KCHOL.IS",
        "Sabanci Holding": "SAHOL.IS",
        "Anadolu Grubu Holding": "AGHOL.IS",
        "Dogus Otomotiv": "DOAS.IS",
        "GSD Holding": "GSDHO.IS",
        "Bera Holding": "BERA.IS",
        "Eczacibasi Yatirim": "ECZYT.IS",
        "Is Yatirim Menkul": "ISMEN.IS",
    },
    "Transport, Aviation & Tourism": {
        "Turkish Airlines": "THYAO.IS",
        "Pegasus Airlines": "PGSUS.IS",
        "TAV Airports": "TAVHL.IS",
        "Tureks Turizm": "TUREX.IS",
        "Merit Turizm": "MERIT.IS",
    },
    "Industrial Blue Chips": {
        "Tupras": "TUPRS.IS",
        "Aselsan": "ASELS.IS",
        "Eregli Steel": "EREGL.IS",
        "Kardemir D": "KRDMD.IS",
        "Kardemir A": "KRDMA.IS",
        "Ford Otosan": "FROTO.IS",
        "Tofas Auto": "TOASO.IS",
        "Arcelik": "ARCLK.IS",
        "Sisecam": "SISE.IS",
        "Enka Construction": "ENKAI.IS",
        "Brisa": "BRISA.IS",
        "Karsan": "KARSN.IS",
        "Erbosan": "ERBOS.IS",
    },
    "Consumer, Retail & Food": {
        "BIM Markets": "BIMAS.IS",
        "Migros": "MGROS.IS",
        "Coca-Cola Icecek": "CCOLA.IS",
        "Ulker Biskuvi": "ULKER.IS",
        "Mavi Giyim": "MAVI.IS",
        "Desa Deri": "DESA.IS",
        "Kervan Gida": "KRVGD.IS",
        "Konfrut Gida": "KNFRT.IS",
        "Besler Gida": "BESLR.IS",
        "Aygaz": "AYGAZ.IS",
        "Sok Marketler": "SOKM.IS",
        "Anadolu Efes": "AEFES.IS",
    },
    "Technology & Telecom": {
        "Turkcell": "TCELL.IS",
        "Turk Telekom": "TTKOM.IS",
        "Logo Yazilim": "LOGO.IS",
        "Link Bilgisayar": "LINK.IS",
        "Penta Teknoloji": "PENTA.IS",
        "Escort Teknoloji": "ESCOM.IS",
        "Kron Teknoloji": "KRONT.IS",
        "Indeks Bilgisayar": "INDES.IS",
    },
    "Energy, Materials & Construction": {
        "Astor Energy": "ASTOR.IS",
        "Sasa Polyester": "SASA.IS",
        "Hektas": "HEKTS.IS",
        "Petkim": "PETKM.IS",
        "Koza Altin": "KOZAL.IS",
        "Smart Gunes": "SMRTG.IS",
        "Eupower Enerji": "EUPWR.IS",
        "Gesan": "GESAN.IS",
        "Kalekim": "KLKIM.IS",
        "QUA Granite": "QUAGR.IS",
        "Kütahya Porselen": "KUTPO.IS",
    },
    "Real Estate & Other": {
        "Torunlar GYO": "TRGYO.IS",
        "Servet GYO": "SRVGY.IS",
        "Emlak Konut GYO": "EKGYO.IS",
        "Ozak GYO": "OZKGY.IS",
        "MLP Saglik": "MPARK.IS",
    },
}

# User-provided top capital-gain list as a fixed screener basket.
# These snapshot fields are metadata only. Time-series calculations use real Yahoo data.
CAPITAL_GAIN_LEADERS = [
    {"Name": "Tureks Turizm Tasimacilik", "Symbol": "TUREX.IS", "SnapshotPrice": 6.86, "SnapshotGainPct": 65.73, "SnapshotTarget": 11.37, "Rating": "Very Good"},
    {"Name": "Link Bilgisayar", "Symbol": "LINK.IS", "SnapshotPrice": 6.94, "SnapshotGainPct": 60.90, "SnapshotTarget": 11.15, "Rating": "Excellent"},
    {"Name": "Kalekim", "Symbol": "KLKIM.IS", "SnapshotPrice": 27.12, "SnapshotGainPct": 58.26, "SnapshotTarget": 42.89, "Rating": "Very Good"},
    {"Name": "Kutahya Porselen", "Symbol": "KUTPO.IS", "SnapshotPrice": 84.50, "SnapshotGainPct": 57.91, "SnapshotTarget": 133.434, "Rating": "Good"},
    {"Name": "Aygaz", "Symbol": "AYGAZ.IS", "SnapshotPrice": 232.80, "SnapshotGainPct": 56.96, "SnapshotTarget": 365.09, "Rating": "Good"},
    {"Name": "Besler Gida", "Symbol": "BESLR.IS", "SnapshotPrice": 13.18, "SnapshotGainPct": 54.38, "SnapshotTarget": 20.35, "Rating": "Good"},
    {"Name": "QUA Granite", "Symbol": "QUAGR.IS", "SnapshotPrice": 3.49, "SnapshotGainPct": 53.31, "SnapshotTarget": 5.35, "Rating": "Fair"},
    {"Name": "Servet GYO", "Symbol": "SRVGY.IS", "SnapshotPrice": 2.64, "SnapshotGainPct": 51.76, "SnapshotTarget": 3.991, "Rating": "Good"},
    {"Name": "Mavi Giyim", "Symbol": "MAVI.IS", "SnapshotPrice": 37.88, "SnapshotGainPct": 51.57, "SnapshotTarget": 57.35, "Rating": "Very Good"},
    {"Name": "Desa Deri", "Symbol": "DESA.IS", "SnapshotPrice": 10.43, "SnapshotGainPct": 48.08, "SnapshotTarget": 15.46, "Rating": "Very Good"},
    {"Name": "Ford Otosan", "Symbol": "FROTO.IS", "SnapshotPrice": 81.05, "SnapshotGainPct": 47.65, "SnapshotTarget": 119.74, "Rating": "Very Good"},
    {"Name": "Penta Teknoloji", "Symbol": "PENTA.IS", "SnapshotPrice": 13.27, "SnapshotGainPct": 47.57, "SnapshotTarget": 19.58, "Rating": "Very Good"},
    {"Name": "Ray Sigorta", "Symbol": "RAYSG.IS", "SnapshotPrice": 177.90, "SnapshotGainPct": 47.06, "SnapshotTarget": 261.62, "Rating": "Excellent"},
    {"Name": "Is Bankasi C", "Symbol": "ISCTR.IS", "SnapshotPrice": 14.19, "SnapshotGainPct": 46.25, "SnapshotTarget": 20.753, "Rating": "Good"},
    {"Name": "Kardemir D", "Symbol": "KRDMD.IS", "SnapshotPrice": 36.82, "SnapshotGainPct": 46.21, "SnapshotTarget": 53.864, "Rating": "Good"},
    {"Name": "Logo Yazilim", "Symbol": "LOGO.IS", "SnapshotPrice": 138.20, "SnapshotGainPct": 45.47, "SnapshotTarget": 201.04, "Rating": "Very Good"},
    {"Name": "Escort Teknoloji", "Symbol": "ESCOM.IS", "SnapshotPrice": 6.14, "SnapshotGainPct": 45.40, "SnapshotTarget": 8.91, "Rating": "Very Good"},
    {"Name": "Karsan", "Symbol": "KARSN.IS", "SnapshotPrice": 11.96, "SnapshotGainPct": 44.34, "SnapshotTarget": 17.277, "Rating": "Excellent"},
    {"Name": "Konfrut Gida", "Symbol": "KNFRT.IS", "SnapshotPrice": 11.88, "SnapshotGainPct": 42.97, "SnapshotTarget": 16.98, "Rating": "Fair"},
    {"Name": "Erbosan", "Symbol": "ERBOS.IS", "SnapshotPrice": 169.50, "SnapshotGainPct": 40.96, "SnapshotTarget": 238.93, "Rating": "Good"},
    {"Name": "Kardemir A", "Symbol": "KRDMA.IS", "SnapshotPrice": 38.18, "SnapshotGainPct": 40.91, "SnapshotTarget": 53.799, "Rating": "Good"},
    {"Name": "Merit Turizm", "Symbol": "MERIT.IS", "SnapshotPrice": 17.99, "SnapshotGainPct": 39.62, "SnapshotTarget": 25.006, "Rating": "Excellent"},
    {"Name": "GSD Holding", "Symbol": "GSDHO.IS", "SnapshotPrice": 5.57, "SnapshotGainPct": 39.21, "SnapshotTarget": 7.768, "Rating": "Very Good"},
    {"Name": "Lider Faktoring", "Symbol": "LIDFA.IS", "SnapshotPrice": 2.78, "SnapshotGainPct": 38.70, "SnapshotTarget": 3.856, "Rating": "Good"},
    {"Name": "Bera Holding", "Symbol": "BERA.IS", "SnapshotPrice": 14.97, "SnapshotGainPct": 38.26, "SnapshotTarget": 20.684, "Rating": "Fair"},
    {"Name": "Brisa", "Symbol": "BRISA.IS", "SnapshotPrice": 80.95, "SnapshotGainPct": 38.20, "SnapshotTarget": 111.87, "Rating": "Good"},
    {"Name": "Kervan Gida", "Symbol": "KRVGD.IS", "SnapshotPrice": 2.76, "SnapshotGainPct": 37.78, "SnapshotTarget": 3.80, "Rating": "Good"},
    {"Name": "TAV Airports", "Symbol": "TAVHL.IS", "SnapshotPrice": 263.75, "SnapshotGainPct": 36.94, "SnapshotTarget": 360.84, "Rating": "Good"},
    {"Name": "Tofas Auto", "Symbol": "TOASO.IS", "SnapshotPrice": 298.75, "SnapshotGainPct": 36.51, "SnapshotTarget": 407.82, "Rating": "Good"},
    {"Name": "Ulker Biskuvi", "Symbol": "ULKER.IS", "SnapshotPrice": np.nan, "SnapshotGainPct": np.nan, "SnapshotTarget": np.nan, "Rating": "User List"},
]

UNIVERSE_STOCKS: Dict[str, str] = {}
for _category, _mapping in MARKET_DATA.items():
    if _category == "Indices":
        continue
    UNIVERSE_STOCKS.update(_mapping)

CAPITAL_GAIN_SYMBOL_TO_NAME = {row["Symbol"]: row["Name"] for row in CAPITAL_GAIN_LEADERS}
ALL_ANALYSIS_SYMBOLS = sorted(set(UNIVERSE_STOCKS.values()).union(CAPITAL_GAIN_SYMBOL_TO_NAME.keys()))
SYMBOL_TO_NAME = {v: k for k, m in MARKET_DATA.items() if k != "Indices" for v in []}
for cat, mapping in MARKET_DATA.items():
    if cat != "Indices":
        for name, sym in mapping.items():
            SYMBOL_TO_NAME[sym] = name
for row in CAPITAL_GAIN_LEADERS:
    SYMBOL_TO_NAME.setdefault(row["Symbol"], row["Name"])

# -------------------------------------------------------------------------
# INDICATOR FALLBACKS
# -------------------------------------------------------------------------
def _series(x, index=None):
    return pd.Series(x, index=index).astype(float)


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(s: pd.Series, period: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    # Wilder-style ADX approximation; TA-Lib used when available.
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)
    tr = atr(high, low, close, period)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / tr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / tr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def macd_calc(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    m = ema(close, fast) - ema(close, slow)
    sig = ema(m, signal)
    return m, sig, m - sig


def bbands(close: pd.Series, period: int = 20, ndev: float = 2.0):
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std()
    return mid + ndev * sd, mid, mid - ndev * sd

# -------------------------------------------------------------------------
# DATA FETCHING: REAL YAHOO DATA ONLY
# -------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_data(symbol: str, start, end) -> Optional[pd.DataFrame]:
    """Download OHLCV data from Yahoo Finance. Returns None if unavailable.

    Data governance: no proxy, no interpolation, no synthetic fallback.
    """
    try:
        download_start = pd.to_datetime(start) - pd.DateOffset(years=2)
        df = yf.download(
            symbol,
            start=download_start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df is None or df.empty:
            return None
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if not all(c in df.columns for c in required_cols):
            return None
        df = df[required_cols].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None) if getattr(df.index, "tz", None) is not None else pd.to_datetime(df.index)
        df = df.loc[~df.index.duplicated(keep="last")].sort_index()
        df = df.dropna(how="any")
        df = df[(df["Close"] > 0) & (df["High"] >= df["Low"])]
        return df if len(df) > 0 else None
    except Exception:
        return None


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    high, low, close, volume = df["High"], df["Low"], df["Close"], df["Volume"]
    c = close.values.astype(float)
    h = high.values.astype(float)
    l = low.values.astype(float)
    v = volume.values.astype(float)

    if TALIB_AVAILABLE:
        df["RSI"] = ta.RSI(c, timeperiod=14)
        df["EMA_20"] = ta.EMA(c, timeperiod=20)
        df["EMA_50"] = ta.EMA(c, timeperiod=50)
        df["EMA_100"] = ta.EMA(c, timeperiod=100)
        df["EMA_200"] = ta.EMA(c, timeperiod=200)
        df["CCI"] = ta.CCI(h, l, c, timeperiod=20)
        df["ATR"] = ta.ATR(h, l, c, timeperiod=14)
        df["ADX"] = ta.ADX(h, l, c, timeperiod=14)
        macd, macd_signal, macd_hist = ta.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
        upper, middle, lower = ta.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2)
        df["MACD"], df["MACD_SIGNAL"], df["MACD_HIST"] = macd, macd_signal, macd_hist
        df["BB_UPPER"], df["BB_MID"], df["BB_LOWER"] = upper, middle, lower
    else:
        df["RSI"] = rsi(close, 14)
        df["EMA_20"] = ema(close, 20)
        df["EMA_50"] = ema(close, 50)
        df["EMA_100"] = ema(close, 100)
        df["EMA_200"] = ema(close, 200)
        df["CCI"] = cci(high, low, close, 20)
        df["ATR"] = atr(high, low, close, 14)
        df["ADX"] = adx(high, low, close, 14)
        m, sig, hist = macd_calc(close)
        df["MACD"], df["MACD_SIGNAL"], df["MACD_HIST"] = m, sig, hist
        up, mid, lo = bbands(close, 20, 2)
        df["BB_UPPER"], df["BB_MID"], df["BB_LOWER"] = up, mid, lo

    df["Return"] = df["Close"].pct_change()
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Dollar_Volume"] = df["Close"] * df["Volume"]
    df["Vol_20D_Ann"] = df["Return"].rolling(20).std() * np.sqrt(TRADING_DAYS)
    df["Vol_63D_Ann"] = df["Return"].rolling(63).std() * np.sqrt(TRADING_DAYS)
    df["Momentum_20D"] = df["Close"].pct_change(20)
    df["Momentum_63D"] = df["Close"].pct_change(63)
    df["Momentum_126D"] = df["Close"].pct_change(126)
    df["Momentum_252D"] = df["Close"].pct_change(252)
    df["High_252D"] = df["Close"].rolling(252).max()
    df["Low_252D"] = df["Close"].rolling(252).min()
    df["Pct_From_52W_High"] = df["Close"] / df["High_252D"] - 1
    df["Pct_From_52W_Low"] = df["Close"] / df["Low_252D"] - 1
    df["Drawdown"] = df["Close"] / df["Close"].cummax() - 1
    df["ATR_Pct"] = df["ATR"] / df["Close"]
    df = df.dropna(subset=["Close", "RSI", "EMA_50", "EMA_200", "ATR", "ADX", "MACD", "MACD_SIGNAL"])
    return df


def compute_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    """Robust Supertrend implementation.

    Critical fix versus the previous version:
    - final upper/lower bands are explicitly initialized at the first valid ATR row.
    - trend can flip both ways after initialization.
    - the line does not remain NaN after the warm-up window.

    trend = 1 means bullish; trend = -1 means bearish; trend = 0 means warm-up.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    if TALIB_AVAILABLE:
        atr_series = pd.Series(
            ta.ATR(high.values, low.values, close.values, timeperiod=period),
            index=df.index,
            dtype=float,
        )
    else:
        atr_series = atr(high, low, close, period)

    hl2 = (high + low) / 2.0
    basic_ub = hl2 + multiplier * atr_series
    basic_lb = hl2 - multiplier * atr_series

    final_ub = pd.Series(np.nan, index=df.index, dtype=float)
    final_lb = pd.Series(np.nan, index=df.index, dtype=float)
    trend = pd.Series(0, index=df.index, dtype=int)
    st_line = pd.Series(np.nan, index=df.index, dtype=float)

    first_valid = atr_series.first_valid_index()
    if first_valid is None:
        return st_line, trend

    start_pos = df.index.get_loc(first_valid)

    for i in range(start_pos, len(df)):
        if pd.isna(basic_ub.iloc[i]) or pd.isna(basic_lb.iloc[i]):
            continue

        if i == start_pos or pd.isna(final_ub.iloc[i - 1]) or pd.isna(final_lb.iloc[i - 1]):
            final_ub.iloc[i] = basic_ub.iloc[i]
            final_lb.iloc[i] = basic_lb.iloc[i]
            trend.iloc[i] = 1
            st_line.iloc[i] = final_lb.iloc[i]
            continue

        prev_final_ub = final_ub.iloc[i - 1]
        prev_final_lb = final_lb.iloc[i - 1]
        prev_close = close.iloc[i - 1]

        if (basic_ub.iloc[i] < prev_final_ub) or (prev_close > prev_final_ub):
            final_ub.iloc[i] = basic_ub.iloc[i]
        else:
            final_ub.iloc[i] = prev_final_ub

        if (basic_lb.iloc[i] > prev_final_lb) or (prev_close < prev_final_lb):
            final_lb.iloc[i] = basic_lb.iloc[i]
        else:
            final_lb.iloc[i] = prev_final_lb

        prev_trend = trend.iloc[i - 1] if trend.iloc[i - 1] in (1, -1) else 1

        if prev_trend == 1 and close.iloc[i] < final_lb.iloc[i]:
            trend.iloc[i] = -1
        elif prev_trend == -1 and close.iloc[i] > final_ub.iloc[i]:
            trend.iloc[i] = 1
        else:
            trend.iloc[i] = prev_trend

        st_line.iloc[i] = final_lb.iloc[i] if trend.iloc[i] == 1 else final_ub.iloc[i]

    return st_line, trend


# -------------------------------------------------------------------------
# RISK METRICS
# -------------------------------------------------------------------------
def max_drawdown(equity: pd.Series) -> float:
    eq = pd.Series(equity).dropna()
    if eq.empty:
        return np.nan
    return float((eq / eq.cummax() - 1).min())


def safe_cagr(ret: pd.Series) -> float:
    r = pd.Series(ret).dropna()
    if r.empty:
        return np.nan
    total = (1 + r).prod() - 1
    years = len(r) / TRADING_DAYS
    if years <= 0 or 1 + total <= 0:
        return np.nan
    return float((1 + total) ** (1 / years) - 1)


def tail_risk(r: pd.Series, alpha: float = 0.95):
    x = pd.Series(r).dropna()
    if len(x) < 20:
        return np.nan, np.nan
    var = -np.quantile(x, 1 - alpha)
    cvar = -x[x <= np.quantile(x, 1 - alpha)].mean()
    return float(var), float(cvar)


def compute_return_metrics(ret: pd.Series, benchmark_ret: Optional[pd.Series] = None, name: str = "Series") -> Dict[str, float]:
    r = pd.Series(ret).dropna().astype(float)
    if r.empty:
        return {"Name": name}
    eq = (1 + r).cumprod()
    cagr = safe_cagr(r)
    vol = float(r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else np.nan
    downside = r[r < 0].std() * np.sqrt(TRADING_DAYS) if (r < 0).sum() > 2 else np.nan
    sharpe = cagr / vol if vol and vol > 0 else np.nan
    sortino = cagr / downside if downside and downside > 0 else np.nan
    mdd = max_drawdown(eq)
    calmar = cagr / abs(mdd) if mdd and mdd < 0 else np.nan
    dd_series = eq / eq.cummax() - 1.0
    ulcer = float(np.sqrt(np.mean(np.square(dd_series * 100)))) if len(dd_series) else np.nan
    gains = float(r[r > 0].sum())
    losses_abs = float(abs(r[r < 0].sum()))
    omega = gains / losses_abs if losses_abs > 0 else np.nan
    recovery_days = 0
    longest_recovery_days = 0
    for underwater in (dd_series < 0):
        recovery_days = recovery_days + 1 if underwater else 0
        longest_recovery_days = max(longest_recovery_days, recovery_days)
    var95, cvar95 = tail_risk(r, 0.95)
    var99, cvar99 = tail_risk(r, 0.99)

    beta = alpha = tracking_error = info_ratio = corr = np.nan
    if benchmark_ret is not None:
        b = pd.Series(benchmark_ret).reindex(r.index).dropna()
        common = r.index.intersection(b.index)
        rr, bb = r.reindex(common).dropna(), b.reindex(common).dropna()
        common = rr.index.intersection(bb.index)
        rr, bb = rr.reindex(common), bb.reindex(common)
        if len(rr) > 20 and bb.var() > 0:
            beta = float(np.cov(rr, bb, ddof=1)[0, 1] / np.var(bb, ddof=1))
            alpha = float((rr.mean() - beta * bb.mean()) * TRADING_DAYS)
            active = rr - bb
            tracking_error = float(active.std() * np.sqrt(TRADING_DAYS))
            info_ratio = float(active.mean() * TRADING_DAYS / tracking_error) if tracking_error > 0 else np.nan
            corr = float(rr.corr(bb))

    return {
        "Name": name,
        "Total Return %": ((1 + r).prod() - 1) * 100,
        "CAGR %": cagr * 100 if pd.notna(cagr) else np.nan,
        "Ann Vol %": vol * 100 if pd.notna(vol) else np.nan,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Omega": omega,
        "Ulcer Index": ulcer,
        "Longest Drawdown Days": int(longest_recovery_days),
        "Max Drawdown %": mdd * 100 if pd.notna(mdd) else np.nan,
        "VaR 95% %": var95 * 100 if pd.notna(var95) else np.nan,
        "CVaR 95% %": cvar95 * 100 if pd.notna(cvar95) else np.nan,
        "VaR 99% %": var99 * 100 if pd.notna(var99) else np.nan,
        "CVaR 99% %": cvar99 * 100 if pd.notna(cvar99) else np.nan,
        "Beta vs XU100": beta,
        "Alpha Ann %": alpha * 100 if pd.notna(alpha) else np.nan,
        "Tracking Error %": tracking_error * 100 if pd.notna(tracking_error) else np.nan,
        "Information Ratio": info_ratio,
        "Correlation vs XU100": corr,
    }


def compute_stats(df: pd.DataFrame, trades: list, index_returns: Optional[pd.Series] = None):
    df = df.copy()
    df["BH_Equity"] = (1 + df["Return"].fillna(0)).cumprod()
    df["Strategy_Equity"] = (1 + df["Strategy_Return"].fillna(0)).cumprod()
    bh = compute_return_metrics(df["Return"], index_returns, name="Buy & Hold")
    stg = compute_return_metrics(df["Strategy_Return"], index_returns, name="Strategy")

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        wins = trades_df[trades_df["Return"] > 0]
        losses = trades_df[trades_df["Return"] <= 0]
        gross_profit = wins["Return"].sum()
        gross_loss = losses["Return"].sum()
        trade_stats = {
            "trade_count": len(trades_df),
            "win_rate": len(wins) / len(trades_df) * 100,
            "avg_trade": trades_df["Return"].mean() * 100,
            "avg_win": wins["Return"].mean() * 100 if not wins.empty else np.nan,
            "avg_loss": losses["Return"].mean() * 100 if not losses.empty else np.nan,
            "profit_factor": gross_profit / abs(gross_loss) if gross_loss < 0 else np.nan,
            "avg_hold": trades_df["HoldingDays"].mean(),
        }
    else:
        trade_stats = {"trade_count": 0, "win_rate": 0.0, "avg_trade": 0.0, "avg_win": np.nan, "avg_loss": np.nan, "profit_factor": np.nan, "avg_hold": np.nan}

    pos_mask = df["Return"] > 0
    neg_mask = df["Return"] < 0
    up_capture = (df.loc[pos_mask, "Strategy_Return"].sum() / df.loc[pos_mask, "Return"].sum() * 100) if pos_mask.any() and df.loc[pos_mask, "Return"].sum() != 0 else np.nan
    down_capture = (df.loc[neg_mask, "Strategy_Return"].sum() / df.loc[neg_mask, "Return"].sum() * 100) if neg_mask.any() and df.loc[neg_mask, "Return"].sum() != 0 else np.nan
    directional = ((np.sign(df["Return"]) == np.sign(df["Strategy_Return"])).mean() * 100) if len(df) else np.nan

    stats = {
        "bh_total_pct": bh.get("Total Return %", np.nan),
        "strat_total_pct": stg.get("Total Return %", np.nan),
        "bh_annual_pct": bh.get("CAGR %", np.nan),
        "strat_annual_pct": stg.get("CAGR %", np.nan),
        "bh_mdd_pct": bh.get("Max Drawdown %", np.nan),
        "strat_mdd_pct": stg.get("Max Drawdown %", np.nan),
        "sharpe": stg.get("Sharpe", np.nan),
        "sortino": stg.get("Sortino", np.nan),
        "calmar": stg.get("Calmar", np.nan),
        "omega": stg.get("Omega", np.nan),
        "ulcer_index": stg.get("Ulcer Index", np.nan),
        "longest_dd_days": stg.get("Longest Drawdown Days", np.nan),
        "var95_pct": stg.get("VaR 95% %", np.nan),
        "cvar95_pct": stg.get("CVaR 95% %", np.nan),
        "var99_pct": stg.get("VaR 99% %", np.nan),
        "cvar99_pct": stg.get("CVaR 99% %", np.nan),
        "beta_asset": bh.get("Beta vs XU100", np.nan),
        "beta_strategy": stg.get("Beta vs XU100", np.nan),
        "alpha_strategy_pct": stg.get("Alpha Ann %", np.nan),
        "tracking_error_pct": stg.get("Tracking Error %", np.nan),
        "information_ratio": stg.get("Information Ratio", np.nan),
        "up_capture_pct": up_capture,
        "down_capture_pct": down_capture,
        "directional_match_pct": directional,
        "corr_bh": df["BH_Equity"].corr(df["Strategy_Equity"]),
        "exposure_pct": float(df.get("Position", pd.Series(0, index=df.index)).mean() * 100) if len(df) else np.nan,
        "buy_signal_count": int((df.get("Signal", pd.Series(0, index=df.index)) == 1).sum()),
        "sell_signal_count": int((df.get("Signal", pd.Series(0, index=df.index)) == -1).sum()),
        "entry_eligible_days": int(df.get("Entry_Eligible", pd.Series(False, index=df.index)).sum()) if len(df) else 0,
        "active_position_now": bool(df.get("Position", pd.Series(0, index=df.index)).iloc[-1] == 1) if len(df) else False,
    }
    stats.update(trade_stats)
    return trades_df, stats


def add_rolling_beta(df: pd.DataFrame, index_returns: Optional[pd.Series]) -> pd.DataFrame:
    df = df.copy()
    if index_returns is None or len(index_returns) == 0:
        df["Rolling_Beta_Asset"] = np.nan
        df["Rolling_Beta_Strategy"] = np.nan
        return df
    pair = pd.DataFrame({
        "asset": df["Return"],
        "strategy": df["Strategy_Return"],
        "index": index_returns.reindex(df.index),
    }).dropna()
    if len(pair) >= ROLLING_BETA_WINDOW:
        roll_var = pair["index"].rolling(ROLLING_BETA_WINDOW).var()
        df["Rolling_Beta_Asset"] = (pair["asset"].rolling(ROLLING_BETA_WINDOW).cov(pair["index"]) / roll_var).reindex(df.index).ffill()
        df["Rolling_Beta_Strategy"] = (pair["strategy"].rolling(ROLLING_BETA_WINDOW).cov(pair["index"]) / roll_var).reindex(df.index).ffill()
    else:
        df["Rolling_Beta_Asset"] = np.nan
        df["Rolling_Beta_Strategy"] = np.nan
    return df

# -------------------------------------------------------------------------
# BACKTESTS
# -------------------------------------------------------------------------
def backtest_macd_atr_trailing(
    df: pd.DataFrame,
    start_date,
    atr_mult_stop: float = 2.0,
    use_rsi_exit: bool = False,
    rsi_exit_level: float = 30.0,
    use_ema_filter: bool = False,
    use_adx_filter: bool = False,
    adx_threshold: float = 10.0,
    use_macd_exit: bool = False,
    fastperiod: int = 8,
    slowperiod: int = 21,
    signalperiod: int = 9,
    market_filter=None,
    index_returns: Optional[pd.Series] = None,
    transaction_cost_bps: float = 8.0,
    slippage_bps: float = 4.0,
):
    df = df.copy()
    df = df[df.index >= pd.Timestamp(start_date)].copy()
    if df.empty:
        return df, pd.DataFrame(), {}

    m, sig, hist = macd_calc(df["Close"], fastperiod, slowperiod, signalperiod)
    if TALIB_AVAILABLE:
        m0, s0, h0 = ta.MACD(df["Close"].values, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
        m, sig, hist = pd.Series(m0, index=df.index), pd.Series(s0, index=df.index), pd.Series(h0, index=df.index)
    df["MACD"], df["MACD_SIGNAL"], df["MACD_HIST"] = m, sig, hist

    bull_cross = (df["MACD"] > df["MACD_SIGNAL"]) & (df["MACD"].shift(1) <= df["MACD_SIGNAL"].shift(1))
    bear_cross = (df["MACD"] < df["MACD_SIGNAL"]) & (df["MACD"].shift(1) >= df["MACD_SIGNAL"].shift(1))

    # FIX: do not require a fresh crossover after the selected backtest start date.
    # If the backtest starts while MACD is already bullish, the strategy may enter.
    # Otherwise many valid histories appear as if the strategy never worked.
    df["Filter_Trend_Pass"] = (df["MACD"] > df["MACD_SIGNAL"]).fillna(False)
    df["Filter_EMA200_Pass"] = ((df["Close"] > df["EMA_200"]).fillna(False) if use_ema_filter else True)
    df["Filter_ADX_Pass"] = ((df["ADX"] > adx_threshold).fillna(False) if use_adx_filter else True)
    if market_filter is not None:
        df["Filter_Market_Pass"] = market_filter.reindex(df.index).ffill().fillna(False)
    else:
        df["Filter_Market_Pass"] = True
    entry_state = (
        df["Filter_Trend_Pass"]
        & df["Filter_EMA200_Pass"]
        & df["Filter_ADX_Pass"]
        & df["Filter_Market_Pass"]
    )
    entry_long = entry_state

    exit_rule = pd.Series(False, index=df.index)
    if use_macd_exit:
        exit_rule |= bear_cross.fillna(False)
    if use_rsi_exit:
        exit_rule |= (df["RSI"] < rsi_exit_level).fillna(False)

    return _run_trailing_backtest(df, entry_long, exit_rule, atr_mult_stop, "MACD/RSI_EXIT", index_returns, transaction_cost_bps, slippage_bps)


def backtest_supertrend_trailing(
    df: pd.DataFrame,
    start_date,
    st_period: int = 10,
    st_mult: float = 2.5,
    use_adx_filter: bool = False,
    adx_threshold: float = 10.0,
    use_ema_filter: bool = False,
    atr_mult_stop: float = 2.0,
    market_filter=None,
    index_returns: Optional[pd.Series] = None,
    transaction_cost_bps: float = 8.0,
    slippage_bps: float = 4.0,
):
    df = df.copy()
    st_line, st_dir = compute_supertrend(df, st_period, st_mult)
    df["ST_Line"] = st_line
    df["ST_Dir"] = st_dir
    df = df[df.index >= pd.Timestamp(start_date)].copy()
    if df.empty:
        return df, pd.DataFrame(), {}
    df["ST_Dir_prev"] = df["ST_Dir"].shift(1).fillna(0)

    # FIX: state-based entry instead of flip-only entry.
    # The previous version waited only for ST_Dir to flip from non-bullish to bullish.
    # If the chosen start date occurred during an already bullish regime, Strategy_Return
    # stayed flat and the Backtest & Risk tab looked broken.
    df["Filter_Trend_Pass"] = (df["ST_Dir"] == 1).fillna(False)
    df["Filter_EMA200_Pass"] = ((df["Close"] > df["EMA_200"]).fillna(False) if use_ema_filter else True)
    df["Filter_ADX_Pass"] = ((df["ADX"] > adx_threshold).fillna(False) if use_adx_filter else True)
    if market_filter is not None:
        df["Filter_Market_Pass"] = market_filter.reindex(df.index).ffill().fillna(False)
    else:
        df["Filter_Market_Pass"] = True
    entry_state = (
        df["Filter_Trend_Pass"]
        & df["Filter_EMA200_Pass"]
        & df["Filter_ADX_Pass"]
        & df["Filter_Market_Pass"]
    )
    entry_long = entry_state

    exit_rule = ((df["ST_Dir"] == -1) & (df["ST_Dir_prev"] == 1)).fillna(False)
    return _run_trailing_backtest(df, entry_long.fillna(False), exit_rule, atr_mult_stop, "SUPERTREND_FLIP", index_returns, transaction_cost_bps, slippage_bps)


def _run_trailing_backtest(df: pd.DataFrame, entry_long: pd.Series, exit_rule: pd.Series, atr_mult_stop: float, exit_label: str, index_returns: Optional[pd.Series], transaction_cost_bps: float = 8.0, slippage_bps: float = 4.0):
    df = df.copy()
    entry_long = pd.Series(entry_long, index=df.index).reindex(df.index).fillna(False).astype(bool)
    exit_rule = pd.Series(exit_rule, index=df.index).reindex(df.index).fillna(False).astype(bool)

    position = 0
    signals, positions, atr_stops = [], [], []
    entry_price = None
    entry_index = None
    peak_price = None
    stop_level = np.nan
    trades = []

    for i, (idx, row) in enumerate(df.iterrows()):
        price = float(row["Close"])
        atr_value = float(row["ATR"])
        buy = bool(entry_long.iloc[i])
        exit_today = bool(exit_rule.iloc[i])
        hit_stop = False

        if position == 1:
            peak_price = max(peak_price if peak_price is not None else price, price)
            new_stop = peak_price - atr_mult_stop * atr_value
            stop_level = new_stop if np.isnan(stop_level) else max(stop_level, new_stop)
            hit_stop = price <= stop_level

        if position == 0:
            if buy:
                position = 1
                entry_price = price
                entry_index = idx
                peak_price = price
                stop_level = price - atr_mult_stop * atr_value
                signal = 1
            else:
                signal = 0
        else:
            reason = "ATR_TRAILING_STOP" if hit_stop else (exit_label if exit_today else None)
            if reason is not None:
                position = 0
                signal = -1
                ret = price / entry_price - 1.0 if entry_price else np.nan
                trades.append({
                    "EntryDate": entry_index,
                    "ExitDate": idx,
                    "EntryPrice": entry_price,
                    "ExitPrice": price,
                    "Return": ret,
                    "ReturnPct": ret * 100,
                    "HoldingDays": (idx - entry_index).days if entry_index is not None else np.nan,
                    "ExitReason": reason,
                })
                entry_price = None
                entry_index = None
                peak_price = None
                stop_level = np.nan
            else:
                signal = 0

        signals.append(signal)
        positions.append(position)
        atr_stops.append(stop_level if position == 1 else np.nan)

    df["Signal"] = signals
    df["Position"] = positions
    df["ATR_Stop"] = atr_stops
    df["Entry_Eligible"] = entry_long
    df["Exit_Rule"] = exit_rule
    df["Days_In_Market"] = df["Position"].expanding().sum()
    df["Return"] = df["Close"].pct_change().fillna(0.0)
    df["Gross_Strategy_Return"] = df["Position"].shift(1).fillna(0) * df["Return"]
    turnover = df["Position"].diff().abs().fillna(df["Position"].abs())
    one_way_cost = (float(transaction_cost_bps) + float(slippage_bps)) / 10000.0
    df["Trading_Cost"] = turnover * one_way_cost
    df["Strategy_Return"] = df["Gross_Strategy_Return"] - df["Trading_Cost"]
    df["Turnover"] = turnover
    df["BH_Equity"] = (1 + df["Return"]).cumprod()
    df["Strategy_Equity"] = (1 + df["Strategy_Return"]).cumprod()
    df["Strategy_Drawdown"] = df["Strategy_Equity"] / df["Strategy_Equity"].cummax() - 1
    df = add_rolling_beta(df, index_returns)
    trades_df, stats = compute_stats(df, trades, index_returns=index_returns)
    return df, trades_df, stats

# -------------------------------------------------------------------------
# SCREENER HELPERS
# -------------------------------------------------------------------------
def technical_grade(last: pd.Series) -> Tuple[float, str]:
    score = 0.0
    reasons = []
    if last["Close"] > last["EMA_200"]:
        score += 20; reasons.append("Price>EMA200")
    if last["EMA_50"] > last["EMA_200"]:
        score += 15; reasons.append("EMA50>EMA200")
    if last["MACD"] > last["MACD_SIGNAL"]:
        score += 12; reasons.append("MACD+")
    if 45 <= last["RSI"] <= 70:
        score += 15; reasons.append("Healthy RSI")
    elif last["RSI"] > 70:
        score += 5; reasons.append("Overbought RSI")
    if last["ADX"] >= 20:
        score += 12; reasons.append("Trend ADX")
    if pd.notna(last.get("Momentum_63D", np.nan)) and last["Momentum_63D"] > 0:
        score += 10; reasons.append("3M Momentum+")
    if pd.notna(last.get("Momentum_126D", np.nan)) and last["Momentum_126D"] > 0:
        score += 8; reasons.append("6M Momentum+")
    if pd.notna(last.get("Pct_From_52W_High", np.nan)) and last["Pct_From_52W_High"] > -0.15:
        score += 8; reasons.append("Near 52W high")
    return min(score, 100.0), ", ".join(reasons)


def analyze_symbol(symbol: str, name: str, start_date, end_date, index_returns: Optional[pd.Series], min_obs: int = MIN_PRICE_OBS):
    raw = get_data(symbol, start_date, end_date)
    if raw is None or len(raw) < min_obs:
        return None, {"Name": name, "Symbol": symbol, "Status": "Excluded", "Reason": f"Insufficient Yahoo data (<{min_obs} rows)"}
    ind = compute_indicators(raw)
    if len(ind) < min_obs:
        return None, {"Name": name, "Symbol": symbol, "Status": "Excluded", "Reason": "Insufficient indicator-ready rows"}
    last = ind.iloc[-1]
    returns = ind["Return"].dropna()
    bench = index_returns.reindex(returns.index).dropna() if index_returns is not None else None
    metrics = compute_return_metrics(returns, bench, name=name)
    tech_score, reasons = technical_grade(last)

    row = {
        "Name": name,
        "Symbol": symbol,
        "Last Close": last["Close"],
        "RSI": last["RSI"],
        "ADX": last["ADX"],
        "ATR %": last["ATR_Pct"] * 100,
        "20D Momentum %": last.get("Momentum_20D", np.nan) * 100,
        "3M Momentum %": last.get("Momentum_63D", np.nan) * 100,
        "6M Momentum %": last.get("Momentum_126D", np.nan) * 100,
        "1Y Momentum %": last.get("Momentum_252D", np.nan) * 100,
        "From 52W High %": last.get("Pct_From_52W_High", np.nan) * 100,
        "From 52W Low %": last.get("Pct_From_52W_Low", np.nan) * 100,
        "Ann Vol %": metrics.get("Ann Vol %", np.nan),
        "CAGR %": metrics.get("CAGR %", np.nan),
        "Max Drawdown %": metrics.get("Max Drawdown %", np.nan),
        "Sharpe": metrics.get("Sharpe", np.nan),
        "Sortino": metrics.get("Sortino", np.nan),
        "Calmar": metrics.get("Calmar", np.nan),
        "Beta vs XU100": metrics.get("Beta vs XU100", np.nan),
        "VaR 95% %": metrics.get("VaR 95% %", np.nan),
        "CVaR 95% %": metrics.get("CVaR 95% %", np.nan),
        "Avg Daily TL Volume": ind["Dollar_Volume"].tail(60).mean(),
        "Technical Score": tech_score,
        "Signal Drivers": reasons,
        "Status": "OK",
        "Reason": "",
    }
    return ind, row


def run_universe_scan(name_to_symbol: Dict[str, str], start_date, end_date, index_returns: Optional[pd.Series], min_obs: int = MIN_PRICE_OBS):
    rows, excluded, data_map = [], [], {}
    progress = st.progress(0.0)
    items = list(name_to_symbol.items())
    for i, (name, symbol) in enumerate(items):
        ind, row = analyze_symbol(symbol, name, start_date, end_date, index_returns, min_obs=min_obs)
        if ind is not None:
            data_map[symbol] = ind
            rows.append(row)
        else:
            excluded.append(row)
        progress.progress((i + 1) / max(len(items), 1))
    progress.empty()
    return pd.DataFrame(rows), pd.DataFrame(excluded), data_map


def smart_score_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    x = df.copy()
    # Composite score: trend quality + risk-adjusted return + liquidity + drawdown control.
    for col in ["Technical Score", "Sharpe", "Sortino", "3M Momentum %", "6M Momentum %", "Avg Daily TL Volume", "Max Drawdown %"]:
        if col not in x.columns:
            x[col] = np.nan
    def rank_pct(s, higher=True):
        return s.rank(pct=True, ascending=not higher) * 100
    x["Composite Score"] = (
        0.30 * x["Technical Score"] +
        0.20 * rank_pct(x["Sharpe"].fillna(-999), True) +
        0.15 * rank_pct(x["3M Momentum %"].fillna(-999), True) +
        0.15 * rank_pct(x["6M Momentum %"].fillna(-999), True) +
        0.10 * rank_pct(x["Avg Daily TL Volume"].fillna(0), True) +
        0.10 * rank_pct(x["Max Drawdown %"].fillna(-999), True)
    )
    x["Action Lens"] = np.select(
        [x["Composite Score"] >= 75, x["Composite Score"] >= 55, x["Composite Score"] >= 35],
        ["Leadership Watch", "Constructive", "Neutral / Validate",],
        default="High Risk / Weak"
    )
    return x.sort_values("Composite Score", ascending=False)

# -------------------------------------------------------------------------
# PORTFOLIO LAB
# -------------------------------------------------------------------------
def run_equal_weight_portfolio(symbols: List[str], start_date, end_date, idx_ind: Optional[pd.DataFrame] = None, min_len: int = 120):
    if not symbols:
        return None
    prices = {}
    for sym in symbols:
        raw = get_data(sym, start_date, end_date)
        if raw is None or len(raw) < min_len:
            continue
        ind = compute_indicators(raw)
        if len(ind) >= min_len:
            prices[sym] = ind["Close"]
    if len(prices) < 2:
        return None
    px_df = pd.concat(prices, axis=1, join="inner").dropna()
    if px_df.shape[0] < 60:
        return None
    ret = px_df.pct_change().dropna()
    port_ret = ret.mean(axis=1)
    idx_ret = None
    if idx_ind is not None and "Close" in idx_ind.columns:
        idx_ret = idx_ind["Close"].reindex(ret.index).ffill().pct_change().dropna()
        common = port_ret.index.intersection(idx_ret.index)
        port_ret, idx_ret, ret = port_ret.reindex(common), idx_ret.reindex(common), ret.reindex(common)
    return {
        "prices": px_df,
        "returns": ret,
        "port_ret": port_ret,
        "idx_ret": idx_ret,
        "eq_port": (1 + port_ret).cumprod(),
        "eq_index": (1 + idx_ret).cumprod() if idx_ret is not None else None,
        "corr": ret.corr(),
        "asset_total_ret": (1 + ret).prod() - 1,
        "metrics_port": compute_return_metrics(port_ret, idx_ret, "Equal Weight Basket"),
        "metrics_index": compute_return_metrics(idx_ret, None, "XU100") if idx_ret is not None else {},
    }

# -------------------------------------------------------------------------
# PLOT HELPERS
# -------------------------------------------------------------------------
def clean_fig(fig, height=520):
    fig.update_layout(
        template="plotly_white",
        height=height,
        hovermode="x unified",
        margin=dict(l=18, r=18, t=55, b=22),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eef2f6", rangeslider_visible=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eef2f6")
    return fig


def strategy_chart(p: pd.DataFrame, title: str):
    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.025,
        row_heights=[0.42, 0.16, 0.14, 0.14, 0.14],
        subplot_titles=(title, "MACD", "RSI", "ATR %", "Strategy Drawdown"),
    )
    fig.add_trace(go.Scatter(x=p.index, y=p["Close"], mode="lines", name="Adjusted Close", line=dict(width=1.7)), row=1, col=1)
    for col, name, dash in [("EMA_50", "EMA 50", "dot"), ("EMA_200", "EMA 200", "solid"), ("BB_UPPER", "BB Upper", "dash"), ("BB_LOWER", "BB Lower", "dash")]:
        if col in p.columns:
            fig.add_trace(go.Scatter(x=p.index, y=p[col], mode="lines", name=name, line=dict(width=1.0, dash=dash)), row=1, col=1)
    if "ATR_Stop" in p.columns:
        fig.add_trace(go.Scatter(x=p.index, y=p["ATR_Stop"], mode="lines", name="ATR Stop", line=dict(width=1.2, dash="dot")), row=1, col=1)
    buys = p[p.get("Signal", 0) == 1]
    sells = p[p.get("Signal", 0) == -1]
    if not buys.empty:
        fig.add_trace(go.Scatter(x=buys.index, y=buys["Close"], mode="markers", name="BUY", marker=dict(symbol="triangle-up", size=11, line=dict(width=1, color="black"))), row=1, col=1)
    if not sells.empty:
        fig.add_trace(go.Scatter(x=sells.index, y=sells["Close"], mode="markers", name="SELL", marker=dict(symbol="triangle-down", size=11, line=dict(width=1, color="black"))), row=1, col=1)
    fig.add_trace(go.Scatter(x=p.index, y=p["MACD"], mode="lines", name="MACD", line=dict(width=1.3)), row=2, col=1)
    fig.add_trace(go.Scatter(x=p.index, y=p["MACD_SIGNAL"], mode="lines", name="MACD Signal", line=dict(width=1.1, dash="dot")), row=2, col=1)
    fig.add_trace(go.Bar(x=p.index, y=p["MACD_HIST"], name="MACD Hist"), row=2, col=1)
    fig.add_trace(go.Scatter(x=p.index, y=p["RSI"], mode="lines", name="RSI", line=dict(width=1.4)), row=3, col=1)
    fig.add_hrect(y0=70, y1=100, opacity=0.08, line_width=0, row=3, col=1)
    fig.add_hrect(y0=0, y1=30, opacity=0.08, line_width=0, row=3, col=1)
    fig.add_trace(go.Scatter(x=p.index, y=p["ATR_Pct"] * 100, mode="lines", name="ATR %", line=dict(width=1.3)), row=4, col=1)
    dd_col = "Strategy_Drawdown" if "Strategy_Drawdown" in p.columns else "Drawdown"
    fig.add_trace(go.Scatter(x=p.index, y=p[dd_col] * 100, mode="lines", name="Drawdown %", fill="tozeroy"), row=5, col=1)
    fig.update_yaxes(title_text="TRY", row=1, col=1)
    fig.update_yaxes(title_text="%", row=4, col=1)
    fig.update_yaxes(title_text="%", row=5, col=1)
    return clean_fig(fig, height=1040)


def equity_risk_chart(p: pd.DataFrame):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.42, 0.28, 0.30], subplot_titles=("Equity Curves", "Rolling 60D Beta vs XU100", "Return Distribution"))
    fig.add_trace(go.Scatter(x=p.index, y=p["BH_Equity"], name="Buy & Hold", mode="lines", line=dict(width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=p.index, y=p["Strategy_Equity"], name="Strategy", mode="lines", line=dict(width=2.0)), row=1, col=1)
    if "Rolling_Beta_Asset" in p.columns:
        fig.add_trace(go.Scatter(x=p.index, y=p["Rolling_Beta_Asset"], name="Asset Beta", mode="lines"), row=2, col=1)
        fig.add_trace(go.Scatter(x=p.index, y=p["Rolling_Beta_Strategy"], name="Strategy Beta", mode="lines"), row=2, col=1)
    fig.add_trace(go.Histogram(x=p["Strategy_Return"] * 100, nbinsx=80, name="Strategy Daily Return %"), row=3, col=1)
    return clean_fig(fig, height=880)


def risk_return_bubble(df: pd.DataFrame, title: str):
    if df.empty:
        return go.Figure()
    plot_df = df.copy()
    plot_df["Bubble"] = np.sqrt(plot_df["Avg Daily TL Volume"].clip(lower=0).fillna(0))
    fig = px.scatter(
        plot_df,
        x="Ann Vol %",
        y="CAGR %",
        size="Bubble",
        hover_name="Name",
        hover_data=["Symbol", "Sharpe", "Max Drawdown %", "Beta vs XU100", "Technical Score", "Composite Score"],
        text="Symbol",
        title=title,
    )
    fig.update_traces(textposition="top center", marker=dict(opacity=0.72, line=dict(width=0.7, color="#475467")))
    return clean_fig(fig, height=650)


def momentum_bar(df: pd.DataFrame, title: str, n: int = 20):
    if df.empty:
        return go.Figure()
    cols = ["Name", "Symbol", "3M Momentum %", "6M Momentum %", "1Y Momentum %", "Composite Score"]
    tmp = df[[c for c in cols if c in df.columns]].head(n).sort_values("Composite Score")
    fig = go.Figure()
    fig.add_trace(go.Bar(y=tmp["Symbol"], x=tmp["3M Momentum %"], orientation="h", name="3M"))
    fig.add_trace(go.Bar(y=tmp["Symbol"], x=tmp["6M Momentum %"], orientation="h", name="6M"))
    fig.update_layout(barmode="group", title=title, xaxis_title="Momentum %", yaxis_title="Ticker")
    return clean_fig(fig, height=620)


def corr_heatmap(corr: pd.DataFrame, title: str):
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1, colorbar=dict(title="Corr")))
    fig.update_layout(title=title)
    return clean_fig(fig, height=650)


def _institutional_css_gradient(series: pd.Series) -> list[str]:
    """Matplotlib-free red/neutral/green gradient for Streamlit Cloud.

    pandas Styler.background_gradient() imports matplotlib at render time. The
    app intentionally avoids that optional dependency, so this helper creates
    equivalent institutional table shading with plain CSS only.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return ["" for _ in series]

    lo = float(finite.min())
    hi = float(finite.max())
    span = hi - lo
    if not np.isfinite(span) or span <= 1e-12:
        return [
            "background-color: rgba(226,232,240,0.45); color: #0f172a;"
            if pd.notna(value) else ""
            for value in numeric
        ]

    styles: list[str] = []
    for value in numeric:
        if pd.isna(value):
            styles.append("")
            continue
        z = float(np.clip((value - lo) / span, 0.0, 1.0))
        if z < 0.5:
            # Pale institutional red -> neutral grey.
            w = z / 0.5
            r = round(254 * (1 - w) + 241 * w)
            g = round(226 * (1 - w) + 245 * w)
            b = round(226 * (1 - w) + 249 * w)
        else:
            # Neutral grey -> pale institutional green.
            w = (z - 0.5) / 0.5
            r = round(241 * (1 - w) + 220 * w)
            g = round(245 * (1 - w) + 252 * w)
            b = round(249 * (1 - w) + 231 * w)
        styles.append(
            f"background-color: rgb({r},{g},{b}); color: #0f172a; "
            "font-weight: 500;"
        )
    return styles


def style_smart_table(df: pd.DataFrame):
    fmt_cols = {c: "{:.2f}" for c in df.select_dtypes(include=[np.number]).columns}
    pct_cols = [c for c in df.columns if "%" in c or c in ["ATR %", "From 52W High %", "From 52W Low %"]]
    for c in pct_cols:
        if c in fmt_cols:
            fmt_cols[c] = "{:.2f}%"

    sty = df.style.format(fmt_cols, na_rep="N/A")
    gradient_columns = [
        "Composite Score", "Technical Score", "Sharpe", "Sortino",
        "CAGR %", "3M Momentum %", "6M Momentum %",
    ]
    for column in gradient_columns:
        if column in df.columns:
            sty = sty.apply(_institutional_css_gradient, subset=[column], axis=0)
    return sty


# -------------------------------------------------------------------------
# VECTORISED LEADING SIGNAL LAB
# Classic MA crossover method + advanced multi-confirmation signal engine.
# Signals are generated at close and applied from the next bar to prevent
# look-ahead bias. No synthetic data is used.
# -------------------------------------------------------------------------
def _signal_perf_metrics(ret: pd.Series, benchmark_ret: Optional[pd.Series] = None) -> Dict[str, float]:
    r = pd.Series(ret).replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return {
            "Total Return %": np.nan, "CAGR %": np.nan, "Ann Vol %": np.nan,
            "Sharpe": np.nan, "Sortino": np.nan, "Max Drawdown %": np.nan,
            "Win Rate %": np.nan, "Positive Days": 0, "Active Days": 0,
            "Beta vs XU100": np.nan, "Information Ratio": np.nan,
        }
    eq=(1+r).cumprod()
    total=eq.iloc[-1]-1
    years=max(len(r)/TRADING_DAYS, 1/TRADING_DAYS)
    cagr=(1+total)**(1/years)-1 if total > -1 else np.nan
    vol=r.std(ddof=1)*np.sqrt(TRADING_DAYS)
    sharpe=(r.mean()*TRADING_DAYS/vol) if vol and np.isfinite(vol) and vol>0 else np.nan
    downside=r[r<0].std(ddof=1)*np.sqrt(TRADING_DAYS)
    sortino=(r.mean()*TRADING_DAYS/downside) if downside and np.isfinite(downside) and downside>0 else np.nan
    dd=eq/eq.cummax()-1
    active=r[r!=0]
    beta=ir=np.nan
    if benchmark_ret is not None:
        pair=pd.concat([r.rename('strategy'), pd.Series(benchmark_ret).rename('benchmark')], axis=1).dropna()
        if len(pair)>2 and pair['benchmark'].var(ddof=1)>0:
            beta=pair['strategy'].cov(pair['benchmark'])/pair['benchmark'].var(ddof=1)
            active_ret=pair['strategy']-pair['benchmark']
            te=active_ret.std(ddof=1)*np.sqrt(TRADING_DAYS)
            ir=(active_ret.mean()*TRADING_DAYS/te) if te and np.isfinite(te) and te>0 else np.nan
    return {
        "Total Return %": total*100, "CAGR %": cagr*100, "Ann Vol %": vol*100,
        "Sharpe": sharpe, "Sortino": sortino, "Max Drawdown %": dd.min()*100,
        "Win Rate %": (active.gt(0).mean()*100 if not active.empty else np.nan),
        "Positive Days": int(active.gt(0).sum()), "Active Days": int(active.size),
        "Beta vs XU100": beta, "Information Ratio": ir,
    }

def run_leading_signal_lab(
    df: pd.DataFrame,
    mode: str = "Classic SMA Crossover",
    fast_window: int = 20,
    slow_window: int = 50,
    breakout_window: int = 20,
    entry_score: int = 4,
    exit_score: int = 2,
    use_volume_confirmation: bool = True,
    market_regime: Optional[pd.Series] = None,
    benchmark_ret: Optional[pd.Series] = None,
    transaction_cost_bps: float = 8.0,
    slippage_bps: float = 4.0,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    x=df.copy().sort_index()
    close=x['Close'].astype(float)
    x['SMA_Fast']=close.rolling(fast_window, min_periods=fast_window).mean()
    x['SMA_Slow']=close.rolling(slow_window, min_periods=slow_window).mean()
    x['EMA_Fast']=close.ewm(span=fast_window, adjust=False, min_periods=fast_window).mean()
    x['EMA_Slow']=close.ewm(span=slow_window, adjust=False, min_periods=slow_window).mean()
    x['Prior_High']=close.rolling(breakout_window, min_periods=breakout_window).max().shift(1)
    x['Volume_Median_20']=x['Volume'].rolling(20, min_periods=20).median()
    x['Return_Lab']=close.pct_change().fillna(0.0)

    if mode == "Classic SMA Crossover":
        x['Trend_Pass']=x['SMA_Fast'] > x['SMA_Slow']
        x['Breakout_Pass']=False
        x['MACD_Pass']=False
        x['RSI_Pass']=False
        x['Volume_Pass']=True
        x['Market_Pass']=True if market_regime is None else market_regime.reindex(x.index).ffill().fillna(False)
        desired=(x['Trend_Pass'] & x['Market_Pass']).astype(int)
        x['Signal_Score']=x['Trend_Pass'].astype(int)+x['Market_Pass'].astype(int)
    else:
        x['Trend_Pass']=(x['EMA_Fast'] > x['EMA_Slow']) & (close > x['EMA_Slow'])
        x['Breakout_Pass']=close > x['Prior_High']
        x['MACD_Pass']=(x['MACD_HIST'] > 0) & (x['MACD_HIST'] > x['MACD_HIST'].shift(1))
        x['RSI_Pass']=(x['RSI'] >= 50) & (x['RSI'] <= 75)
        x['Volume_Pass']=(x['Volume'] > x['Volume_Median_20']) if use_volume_confirmation else True
        x['Market_Pass']=True if market_regime is None else market_regime.reindex(x.index).ffill().fillna(False)
        score_cols=['Trend_Pass','Breakout_Pass','MACD_Pass','RSI_Pass','Volume_Pass','Market_Pass']
        x['Signal_Score']=sum(x[c].astype(int) for c in score_cols)
        state=0
        desired_vals=[]
        for score in x['Signal_Score'].fillna(0).astype(int):
            if state==0 and score>=entry_score:
                state=1
            elif state==1 and score<=exit_score:
                state=0
            desired_vals.append(state)
        desired=pd.Series(desired_vals,index=x.index,dtype=int)

    x['Desired_Position']=desired.astype(int)
    # Decision at today's close; exposure begins on the next bar.
    x['Position_Lab']=x['Desired_Position'].shift(1).fillna(0).astype(int)
    x['Position_Change']=x['Position_Lab'].diff().abs().fillna(x['Position_Lab'].abs())
    one_way_cost=(transaction_cost_bps+slippage_bps)/10000.0
    x['Trading_Cost_Lab']=x['Position_Change']*one_way_cost
    x['Gross_Strategy_Return_Lab']=x['Position_Lab']*x['Return_Lab']
    x['Strategy_Return_Lab']=x['Gross_Strategy_Return_Lab']-x['Trading_Cost_Lab']
    x['BuyHold_Equity_Lab']=(1+x['Return_Lab']).cumprod()
    x['Strategy_Equity_Lab']=(1+x['Strategy_Return_Lab']).cumprod()
    x['Signal_Event']=np.select(
        [x['Desired_Position'].eq(1)&x['Desired_Position'].shift(1).fillna(0).eq(0),
         x['Desired_Position'].eq(0)&x['Desired_Position'].shift(1).fillna(0).eq(1)],
        ['AL','SAT'], default='')
    x['Leading_Action']=np.where(x['Signal_Event'].ne(''),x['Signal_Event'],np.where(x['Desired_Position'].eq(1),'TUT','BEKLE'))
    metrics=_signal_perf_metrics(x['Strategy_Return_Lab'], benchmark_ret)
    metrics['Signal Count']=int(x['Signal_Event'].isin(['AL','SAT']).sum())
    metrics['Buy Signals']=int(x['Signal_Event'].eq('AL').sum())
    metrics['Sell Signals']=int(x['Signal_Event'].eq('SAT').sum())
    metrics['Exposure %']=float(x['Position_Lab'].mean()*100)
    return x,metrics

def leading_signal_chart(df: pd.DataFrame, title: str) -> go.Figure:
    fig=make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=0.04,row_heights=[0.52,0.23,0.25],subplot_titles=("Price, Averages and Signal Events","Signal Confirmation Score","Strategy vs Buy & Hold"))
    fig.add_trace(go.Scatter(x=df.index,y=df['Close'],mode='lines',name='Close'),row=1,col=1)
    for col,name in [('SMA_Fast','Fast SMA'),('SMA_Slow','Slow SMA'),('EMA_Fast','Fast EMA'),('EMA_Slow','Slow EMA')]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index,y=df[col],mode='lines',name=name,line=dict(width=1.1)),row=1,col=1)
    buys=df[df['Signal_Event']=='AL']; sells=df[df['Signal_Event']=='SAT']
    fig.add_trace(go.Scatter(x=buys.index,y=buys['Close'],mode='markers',name='AL',marker=dict(symbol='triangle-up',size=12)),row=1,col=1)
    fig.add_trace(go.Scatter(x=sells.index,y=sells['Close'],mode='markers',name='SAT',marker=dict(symbol='triangle-down',size=12)),row=1,col=1)
    fig.add_trace(go.Bar(x=df.index,y=df['Signal_Score'],name='Confirmation Score'),row=2,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df['Strategy_Equity_Lab'],mode='lines',name='Signal Strategy'),row=3,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df['BuyHold_Equity_Lab'],mode='lines',name='Buy & Hold'),row=3,col=1)
    fig.update_layout(title=title,hovermode='x unified',height=900,template='plotly_white',margin=dict(l=20,r=20,t=60,b=20))
    return fig


# -------------------------------------------------------------------------
# INSTITUTIONAL LEADING SIGNAL ENGINE V5.0.2
# -------------------------------------------------------------------------
def _safe_percentile_rank(series: pd.Series, window: int = 252) -> pd.Series:
    s = pd.Series(series, dtype=float)
    return s.rolling(window, min_periods=max(40, window // 4)).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

def build_institutional_signal_engine(
    df: pd.DataFrame,
    benchmark_close: Optional[pd.Series] = None,
    benchmark_returns: Optional[pd.Series] = None,
    forward_horizon: int = 60,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Explainable 100-point decision engine using only observed market data.

    The current score uses information available through each close. Historical
    probability estimates use only rows whose forward outcome is already known.
    """
    x = df.copy().sort_index()
    close = x['Close'].astype(float)
    ret = close.pct_change()

    # Core derived series
    ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    ema200 = close.ewm(span=200, adjust=False, min_periods=200).mean()
    roc20 = close.pct_change(20)
    roc60 = close.pct_change(60)
    vol20 = ret.rolling(20, min_periods=20).std() * np.sqrt(TRADING_DAYS)
    vol60 = ret.rolling(60, min_periods=40).std() * np.sqrt(TRADING_DAYS)
    drawdown = close / close.cummax() - 1.0
    atr_pct = x['ATR'].astype(float) / close.replace(0, np.nan) if 'ATR' in x else pd.Series(np.nan, index=x.index)

    volume = x['Volume'].astype(float)
    vol_med20 = volume.rolling(20, min_periods=20).median()
    obv = (np.sign(ret.fillna(0.0)) * volume).cumsum()
    obv_ma20 = obv.rolling(20, min_periods=20).mean()

    # Relative strength against XU100
    if benchmark_close is not None:
        bclose = pd.Series(benchmark_close, dtype=float).reindex(x.index).ffill()
        rs = close / bclose.replace(0, np.nan)
        rs20 = rs.pct_change(20)
        rs60 = rs.pct_change(60)
    else:
        rs20 = pd.Series(0.0, index=x.index)
        rs60 = pd.Series(0.0, index=x.index)

    # Rolling beta/alpha proxies
    if benchmark_returns is not None:
        bret = pd.Series(benchmark_returns, dtype=float).reindex(x.index)
        pair = pd.concat([ret.rename('asset'), bret.rename('bench')], axis=1)
        cov = pair['asset'].rolling(60, min_periods=40).cov(pair['bench'])
        var = pair['bench'].rolling(60, min_periods=40).var()
        beta60 = cov / var.replace(0, np.nan)
        alpha60 = (pair['asset'].rolling(60, min_periods=40).mean() - beta60 * pair['bench'].rolling(60, min_periods=40).mean()) * TRADING_DAYS
        market_pass = (bclose > bclose.ewm(span=200, adjust=False, min_periods=200).mean()).astype(float) if benchmark_close is not None else pd.Series(0.5, index=x.index)
    else:
        beta60 = pd.Series(np.nan, index=x.index)
        alpha60 = pd.Series(np.nan, index=x.index)
        market_pass = pd.Series(0.5, index=x.index)

    # 1) Trend: 20 points
    trend_score = (
        7.0 * (close > ema50).astype(float) +
        7.0 * (ema50 > ema200).astype(float) +
        6.0 * ((x.get('ADX', pd.Series(0, index=x.index)) > 18) & (x.get('ST_Dir', pd.Series(1, index=x.index)) >= 0)).astype(float)
    )

    # 2) Momentum: 20 points
    rsi = x.get('RSI', pd.Series(50.0, index=x.index)).astype(float)
    macd_hist = x.get('MACD_HIST', pd.Series(0.0, index=x.index)).astype(float)
    momentum_score = (
        6.0 * (roc20 > 0).astype(float) +
        5.0 * (roc60 > 0).astype(float) +
        5.0 * ((rsi >= 50) & (rsi <= 72)).astype(float) +
        4.0 * ((macd_hist > 0) & (macd_hist > macd_hist.shift(1))).astype(float)
    )

    # 3) Relative strength: 15 points
    relative_score = 8.0 * (rs20 > 0).astype(float) + 7.0 * (rs60 > 0).astype(float)

    # 4) Volume/flow: 15 points
    volume_score = (
        7.0 * (volume > vol_med20).astype(float) +
        5.0 * (obv > obv_ma20).astype(float) +
        3.0 * (volume.pct_change(5) > 0).astype(float)
    )

    # 5) Volatility quality: 10 points; reward controlled, non-expanding risk
    vol_rank = _safe_percentile_rank(vol20, 252)
    volatility_score = (
        5.0 * (vol20 <= vol60).astype(float) +
        3.0 * (vol_rank <= 0.75).astype(float) +
        2.0 * (atr_pct <= atr_pct.rolling(60, min_periods=30).median()).astype(float)
    )

    # 6) Risk quality: 10 points
    risk_score = (
        4.0 * (drawdown > -0.15).astype(float) +
        3.0 * ((beta60.isna()) | (beta60 <= 1.25)).astype(float) +
        3.0 * ((alpha60.isna()) | (alpha60 > 0)).astype(float)
    )

    # 7) Market regime: 10 points
    market_score = 10.0 * market_pass.clip(0, 1)

    factors = {
        'Trend Score': trend_score,
        'Momentum Score': momentum_score,
        'Relative Strength Score': relative_score,
        'Volume Score': volume_score,
        'Volatility Score': volatility_score,
        'Risk Score': risk_score,
        'Market Regime Score': market_score,
    }
    for name, series in factors.items():
        x[name] = pd.Series(series, index=x.index).fillna(0.0)
    x['Institutional Score'] = sum(x[name] for name in factors).clip(0, 100)

    # Confidence rewards broad factor agreement and stable recent score.
    # Use a plain dictionary and iterate over .items(). Iterating directly
    # over a pandas Series returns its VALUES (20, 20, 15, ...), not its index.
    # The previous code therefore attempted x[20] and raised KeyError.
    factor_max = {
        'Trend Score': 20.0,
        'Momentum Score': 20.0,
        'Relative Strength Score': 15.0,
        'Volume Score': 15.0,
        'Volatility Score': 10.0,
        'Risk Score': 10.0,
        'Market Regime Score': 10.0,
    }

    missing_factor_columns = [name for name in factor_max if name not in x.columns]
    if missing_factor_columns:
        raise ValueError(
            "Institutional signal engine is missing factor columns: "
            + ", ".join(missing_factor_columns)
        )

    normalized = pd.DataFrame(
        {
            name: pd.to_numeric(x[name], errors='coerce').fillna(0.0) / maximum
            for name, maximum in factor_max.items()
        },
        index=x.index,
    )
    agreement = 1.0 - normalized.std(axis=1).clip(0, 0.5) / 0.5
    stability = 1.0 - (x['Institutional Score'].rolling(20, min_periods=5).std() / 25.0).clip(0, 1)
    x['Confidence Score'] = (100.0 * (0.6 * agreement + 0.4 * stability)).clip(0, 100)

    x['Recommendation'] = pd.cut(
        x['Institutional Score'],
        bins=[-np.inf, 25, 40, 60, 75, np.inf],
        labels=['STRONG SELL', 'SELL', 'HOLD', 'BUY', 'STRONG BUY'],
    ).astype(str)

    # Historical empirical probabilities from resolved forward outcomes only.
    x[f'Forward {forward_horizon}D Return'] = close.shift(-forward_horizon) / close - 1.0
    if benchmark_close is not None:
        bench_fwd = bclose.shift(-forward_horizon) / bclose - 1.0
        x[f'Forward {forward_horizon}D Active Return'] = x[f'Forward {forward_horizon}D Return'] - bench_fwd
    else:
        x[f'Forward {forward_horizon}D Active Return'] = np.nan

    current_score = float(x['Institutional Score'].iloc[-1])
    resolved = x.iloc[:-forward_horizon].dropna(subset=[f'Forward {forward_horizon}D Return']) if len(x) > forward_horizon else x.iloc[0:0]
    band = 7.5
    peers = resolved[(resolved['Institutional Score'] >= current_score - band) & (resolved['Institutional Score'] <= current_score + band)]
    if len(peers) < 20 and not resolved.empty:
        peers = resolved.assign(_dist=(resolved['Institutional Score'] - current_score).abs()).nsmallest(min(60, len(resolved)), '_dist')

    positive_prob = float((peers[f'Forward {forward_horizon}D Return'] > 0).mean() * 100) if len(peers) else np.nan
    plus10_prob = float((peers[f'Forward {forward_horizon}D Return'] >= 0.10).mean() * 100) if len(peers) else np.nan
    active_col = f'Forward {forward_horizon}D Active Return'
    outperform_prob = float((peers[active_col] > 0).mean() * 100) if len(peers) and peers[active_col].notna().any() else np.nan

    latest = x.iloc[-1]
    summary = {
        'Institutional Score': float(latest['Institutional Score']),
        'Confidence Score': float(latest['Confidence Score']),
        'Recommendation': str(latest['Recommendation']),
        f'Positive Return Probability {forward_horizon}D %': positive_prob,
        f'+10% Probability {forward_horizon}D %': plus10_prob,
        f'Outperform XU100 Probability {forward_horizon}D %': outperform_prob,
        'Historical Analog Count': int(len(peers)),
    }

    factor_names = list(factor_max.keys())
    contribution = pd.DataFrame({
        'Factor': factor_names,
        'Score': [float(latest[name]) for name in factor_names],
        'Maximum': [float(factor_max[name]) for name in factor_names],
    })
    contribution['Contribution %'] = contribution['Score'] / contribution['Maximum'] * 100.0
    return x, contribution, summary

def institutional_score_chart(score_df: pd.DataFrame) -> go.Figure:
    score_df = score_df.copy().sort_index()
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.50, 0.22, 0.28],
        specs=[[{'secondary_y': True}], [{'secondary_y': False}], [{'secondary_y': False}]],
        subplot_titles=(
            'Adjusted Price Structure — Candles, Trend Channel, Anchored VWAP, Swing Structure, Targets & Volume',
            'Institutional Score and Confidence',
            'Factor Contribution History',
        ),
    )

    regime_color_map = {
        'STRONG BUY': 'rgba(22,163,74,0.10)',
        'BUY': 'rgba(132,204,22,0.08)',
        'HOLD': 'rgba(245,158,11,0.07)',
        'SELL': 'rgba(249,115,22,0.08)',
        'STRONG SELL': 'rgba(220,38,38,0.10)',
    }
    regime_marker_map = {
        'STRONG BUY': '#16a34a',
        'BUY': '#84cc16',
        'HOLD': '#f59e0b',
        'SELL': '#f97316',
        'STRONG SELL': '#dc2626',
    }

    # --- derived price-structure helpers ---
    resistance_20 = score_df['High'].rolling(20, min_periods=10).max()
    support_20 = score_df['Low'].rolling(20, min_periods=10).min()
    resistance_60 = score_df['High'].rolling(60, min_periods=30).max()
    support_60 = score_df['Low'].rolling(60, min_periods=30).min()
    volume_colors = np.where(score_df['Close'] >= score_df['Open'], '#16a34a', '#dc2626')
    regime = score_df['Recommendation'].fillna('HOLD').astype(str) if 'Recommendation' in score_df.columns else pd.Series('HOLD', index=score_df.index)

    # --- dynamic trend channel (recent regression channel) ---
    channel_lookback = int(min(90, max(40, len(score_df))))
    trend_mid = pd.Series(np.nan, index=score_df.index, dtype=float)
    trend_upper = pd.Series(np.nan, index=score_df.index, dtype=float)
    trend_lower = pd.Series(np.nan, index=score_df.index, dtype=float)
    channel_slice = score_df.iloc[-channel_lookback:].copy()
    if len(channel_slice) >= 20:
        x_idx = np.arange(len(channel_slice), dtype=float)
        slope, intercept = np.polyfit(x_idx, channel_slice['Close'].astype(float).values, 1)
        fitted = intercept + slope * x_idx
        resid_std = float(np.nanstd(channel_slice['Close'].astype(float).values - fitted, ddof=1)) if len(channel_slice) > 2 else 0.0
        atr_pad = float(channel_slice['ATR'].tail(20).median()) if 'ATR' in channel_slice.columns and channel_slice['ATR'].notna().any() else 0.0
        channel_width = max(resid_std * 1.40, atr_pad * 1.20, 1e-9)
        trend_mid.loc[channel_slice.index] = fitted
        trend_upper.loc[channel_slice.index] = fitted + channel_width
        trend_lower.loc[channel_slice.index] = fitted - channel_width
    else:
        channel_width = np.nan

    # --- anchored VWAP anchored to latest swing low or fallback recent support ---
    low = score_df['Low'].astype(float)
    high = score_df['High'].astype(float)
    close = score_df['Close'].astype(float)
    volume = score_df['Volume'].astype(float)
    typical_price = (high + low + close) / 3.0
    swing_low_mask = (low.shift(2) > low.shift(1)) & (low.shift(1) > low) & (low.shift(-1) > low) & (low.shift(-2) > low)
    swing_high_mask = (high.shift(2) < high.shift(1)) & (high.shift(1) < high) & (high.shift(-1) < high) & (high.shift(-2) < high)
    recent_window = min(150, len(score_df))
    recent_index = score_df.index[-recent_window:]
    recent_swing_lows = score_df.index[swing_low_mask.fillna(False) & score_df.index.isin(recent_index)]
    recent_swing_highs = score_df.index[swing_high_mask.fillna(False) & score_df.index.isin(recent_index)]
    if len(recent_swing_lows) > 0:
        anchor_date = recent_swing_lows[-1]
    else:
        anchor_date = score_df.index[max(0, len(score_df) - 60)]
    anchor_mask = score_df.index >= anchor_date
    anchored_vwap = pd.Series(np.nan, index=score_df.index, dtype=float)
    cum_pv = (typical_price[anchor_mask] * volume[anchor_mask]).cumsum()
    cum_vol = volume[anchor_mask].cumsum().replace(0, np.nan)
    anchored_vwap.loc[anchor_mask] = cum_pv / cum_vol

    # --- regime shading on price panel ---
    if len(score_df) > 1:
        group_id = (regime != regime.shift(1)).cumsum()
        temp = score_df.copy()
        temp['_group_id'] = group_id.values
        for _, seg in temp.groupby('_group_id'):
            label = str(seg['Recommendation'].iloc[0]) if 'Recommendation' in seg.columns else 'HOLD'
            fig.add_vrect(
                x0=seg.index[0], x1=seg.index[-1],
                fillcolor=regime_color_map.get(label, 'rgba(148,163,184,0.08)'),
                opacity=0.32, line_width=0, row=1, col=1
            )

    # --- row 1: price panel ---
    fig.add_trace(
        go.Candlestick(
            x=score_df.index, open=score_df['Open'], high=score_df['High'], low=score_df['Low'], close=score_df['Close'],
            name='Adjusted OHLC', increasing_line_color='#16a34a', decreasing_line_color='#dc2626',
            increasing_fillcolor='rgba(22,163,74,0.75)', decreasing_fillcolor='rgba(220,38,38,0.75)',
            whiskerwidth=0.4, opacity=0.95,
        ),
        row=1, col=1, secondary_y=False
    )

    # Bollinger envelope
    if 'BB_UPPER' in score_df.columns and 'BB_LOWER' in score_df.columns:
        fig.add_trace(
            go.Scatter(x=score_df.index, y=score_df['BB_UPPER'], mode='lines', line=dict(width=1.0, dash='dot', color='rgba(59,130,246,0.50)'), name='BB Upper'),
            row=1, col=1, secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=score_df.index, y=score_df['BB_LOWER'], mode='lines', line=dict(width=1.0, dash='dot', color='rgba(59,130,246,0.50)'), fill='tonexty', fillcolor='rgba(59,130,246,0.08)', name='Bollinger Envelope'),
            row=1, col=1, secondary_y=False
        )

    # Trend averages
    ema_specs = [('EMA_20', 'EMA 20', '#2563eb', 1.4, 'solid'), ('EMA_50', 'EMA 50', '#f59e0b', 1.6, 'solid'), ('EMA_200', 'EMA 200', '#111827', 1.8, 'dash')]
    for col, name, color, width, dash in ema_specs:
        if col in score_df.columns:
            fig.add_trace(go.Scatter(x=score_df.index, y=score_df[col], mode='lines', name=name, line=dict(color=color, width=width, dash=dash)), row=1, col=1, secondary_y=False)

    # Dynamic trend channel
    if trend_mid.notna().any():
        fig.add_trace(go.Scatter(x=score_df.index, y=trend_upper, mode='lines', name='Trend Channel Upper', line=dict(color='#0891b2', width=1.2, dash='dash')), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=score_df.index, y=trend_lower, mode='lines', name='Trend Channel Lower', line=dict(color='#0891b2', width=1.2, dash='dash'), fill='tonexty', fillcolor='rgba(8,145,178,0.08)'), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=score_df.index, y=trend_mid, mode='lines', name='Trend Channel Mid', line=dict(color='#0f766e', width=1.4, dash='dot')), row=1, col=1, secondary_y=False)

    # Anchored VWAP
    if anchored_vwap.notna().any():
        fig.add_trace(go.Scatter(x=score_df.index, y=anchored_vwap, mode='lines', name='Anchored VWAP', line=dict(color='#7c3aed', width=2.0)), row=1, col=1, secondary_y=False)

    # Support / resistance bands
    fig.add_trace(go.Scatter(x=score_df.index, y=resistance_20, mode='lines', name='20D Resistance', line=dict(color='#a855f7', width=1.1, dash='dash')), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=score_df.index, y=support_20, mode='lines', name='20D Support', line=dict(color='#0f766e', width=1.1, dash='dash')), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=score_df.index, y=resistance_60, mode='lines', name='60D Resistance', line=dict(color='#c026d3', width=1.0, dash='dot')), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=score_df.index, y=support_60, mode='lines', name='60D Support', line=dict(color='#14b8a6', width=1.0, dash='dot')), row=1, col=1, secondary_y=False)

    # Buy / Sell markers based on recommendation regime changes
    rec_prev = regime.shift(1).fillna('HOLD')
    buy_mask = regime.isin(['BUY', 'STRONG BUY']) & ~rec_prev.isin(['BUY', 'STRONG BUY'])
    sell_mask = regime.isin(['SELL', 'STRONG SELL']) & ~rec_prev.isin(['SELL', 'STRONG SELL'])
    hold_mask = (regime == 'HOLD') & (rec_prev != 'HOLD')
    if buy_mask.any():
        fig.add_trace(go.Scatter(x=score_df.index[buy_mask], y=score_df.loc[buy_mask, 'Low'] * 0.985, mode='markers', name='BUY Signal', marker=dict(symbol='triangle-up', size=12, color='#16a34a', line=dict(width=1.1, color='white')), hovertemplate='Date=%{x}<br>BUY Signal<br>Adjusted Price=%{y:.2f}<extra></extra>'), row=1, col=1, secondary_y=False)
    if sell_mask.any():
        fig.add_trace(go.Scatter(x=score_df.index[sell_mask], y=score_df.loc[sell_mask, 'High'] * 1.015, mode='markers', name='SELL Signal', marker=dict(symbol='triangle-down', size=12, color='#dc2626', line=dict(width=1.1, color='white')), hovertemplate='Date=%{x}<br>SELL Signal<br>Adjusted Price=%{y:.2f}<extra></extra>'), row=1, col=1, secondary_y=False)
    if hold_mask.any():
        fig.add_trace(go.Scatter(x=score_df.index[hold_mask], y=score_df.loc[hold_mask, 'Close'], mode='markers', name='HOLD / Neutral', marker=dict(symbol='circle', size=8, color='#f59e0b', line=dict(width=0.8, color='white')), hovertemplate='Date=%{x}<br>HOLD / Neutral<br>Adjusted Close=%{y:.2f}<extra></extra>'), row=1, col=1, secondary_y=False)

    # Swing high / low labels (last few only for readability)
    swing_high_points = score_df.loc[swing_high_mask.fillna(False), ['High']].tail(4)
    swing_low_points = score_df.loc[swing_low_mask.fillna(False), ['Low']].tail(4)
    if not swing_high_points.empty:
        fig.add_trace(go.Scatter(x=swing_high_points.index, y=swing_high_points['High'] * 1.01, mode='markers+text', name='Swing High', text=['SH'] * len(swing_high_points), textposition='top center', marker=dict(symbol='diamond', size=9, color='#9333ea', line=dict(width=0.8, color='white')), hovertemplate='Date=%{x}<br>Swing High=%{y:.2f}<extra></extra>'), row=1, col=1, secondary_y=False)
    if not swing_low_points.empty:
        fig.add_trace(go.Scatter(x=swing_low_points.index, y=swing_low_points['Low'] * 0.99, mode='markers+text', name='Swing Low', text=['SL'] * len(swing_low_points), textposition='bottom center', marker=dict(symbol='diamond', size=9, color='#0f766e', line=dict(width=0.8, color='white')), hovertemplate='Date=%{x}<br>Swing Low=%{y:.2f}<extra></extra>'), row=1, col=1, secondary_y=False)

    # Latest regime marker
    latest = score_df.iloc[-1]
    latest_regime = str(latest.get('Recommendation', 'HOLD'))
    fig.add_trace(go.Scatter(x=[score_df.index[-1]], y=[latest['Close']], mode='markers', name=f'Latest Regime: {latest_regime}', marker=dict(size=13, color=regime_marker_map.get(latest_regime, '#64748b'), line=dict(width=1.2, color='white')), hovertemplate='Date=%{x}<br>Adjusted Close=%{y:.2f}<br>Recommendation=' + latest_regime + '<extra></extra>'), row=1, col=1, secondary_y=False)

    # Price target box near the right edge
    latest_close = float(latest['Close'])
    target_floor = float(np.nanmax([latest_close, resistance_20.iloc[-1] if pd.notna(resistance_20.iloc[-1]) else np.nan, anchored_vwap.iloc[-1] if pd.notna(anchored_vwap.iloc[-1]) else np.nan]))
    target_ceiling = float(np.nanmax([target_floor, resistance_60.iloc[-1] if pd.notna(resistance_60.iloc[-1]) else np.nan, trend_upper.iloc[-1] if pd.notna(trend_upper.iloc[-1]) else np.nan]))
    support_floor = float(np.nanmin([support_20.iloc[-1] if pd.notna(support_20.iloc[-1]) else latest_close, support_60.iloc[-1] if pd.notna(support_60.iloc[-1]) else latest_close, trend_lower.iloc[-1] if pd.notna(trend_lower.iloc[-1]) else latest_close]))
    target_box_x0 = score_df.index[max(0, len(score_df) - min(18, len(score_df)))]
    target_box_x1 = score_df.index[-1]
    if np.isfinite(target_floor) and np.isfinite(target_ceiling) and target_ceiling >= target_floor:
        fig.add_hrect(y0=target_floor, y1=target_ceiling, fillcolor='rgba(22,163,74,0.10)', line_color='rgba(22,163,74,0.55)', line_width=1, row=1, col=1)
        fig.add_annotation(x=target_box_x1, y=target_ceiling, xref='x1', yref='y1', text=f'Target Box<br>{target_floor:,.2f} – {target_ceiling:,.2f}', showarrow=True, arrowhead=1, ax=35, ay=-25, bgcolor='rgba(236,253,243,0.95)', bordercolor='rgba(22,163,74,0.55)', font=dict(size=11, color='#065f46'), row=1, col=1)
    if np.isfinite(support_floor):
        fig.add_annotation(x=target_box_x1, y=support_floor, xref='x1', yref='y1', text=f'Risk Pivot<br>{support_floor:,.2f}', showarrow=True, arrowhead=1, ax=30, ay=25, bgcolor='rgba(255,247,237,0.95)', bordercolor='rgba(249,115,22,0.55)', font=dict(size=11, color='#9a3412'), row=1, col=1)

    # Volume bars on secondary axis
    fig.add_trace(go.Bar(x=score_df.index, y=score_df['Volume'], name='Volume', opacity=0.22, marker_color=volume_colors, hovertemplate='Date=%{x}<br>Volume=%{y:,.0f}<extra></extra>'), row=1, col=1, secondary_y=True)

    # --- row 2: institutional score panel ---
    fig.add_trace(go.Scatter(x=score_df.index, y=score_df['Institutional Score'], mode='lines', name='Institutional Score', line=dict(color='#111827', width=2.4), fill='tozeroy', fillcolor='rgba(17,24,39,0.08)'), row=2, col=1)
    fig.add_trace(go.Scatter(x=score_df.index, y=score_df['Confidence Score'], mode='lines', name='Confidence Score', line=dict(color='#7c3aed', width=2.0, dash='dot')), row=2, col=1)
    score_bands = [(0, 25, 'rgba(220,38,38,0.08)'), (25, 40, 'rgba(249,115,22,0.08)'), (40, 60, 'rgba(245,158,11,0.08)'), (60, 75, 'rgba(132,204,22,0.08)'), (75, 100, 'rgba(22,163,74,0.08)')]
    for y0, y1, color in score_bands:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=color, line_width=0, row=2, col=1)
    for y in (25, 40, 60, 75):
        fig.add_hline(y=y, line_dash='dot', line_width=1, line_color='rgba(100,116,139,0.7)', row=2, col=1)

    # --- row 3: factor contribution history ---
    factor_cols = ['Trend Score','Momentum Score','Relative Strength Score','Volume Score','Volatility Score','Risk Score','Market Regime Score']
    factor_color_map = {'Trend Score': '#2563eb', 'Momentum Score': '#7c3aed', 'Relative Strength Score': '#14b8a6', 'Volume Score': '#f59e0b', 'Volatility Score': '#ef4444', 'Risk Score': '#64748b', 'Market Regime Score': '#16a34a'}
    for col in factor_cols:
        if col in score_df.columns:
            fig.add_trace(go.Scatter(x=score_df.index, y=score_df[col], mode='lines', stackgroup='one', name=col.replace(' Score',''), line=dict(width=0.9, color=factor_color_map.get(col)), hovertemplate='Date=%{x}<br>' + col.replace(' Score','') + '=%{y:.2f}<extra></extra>'), row=3, col=1)

    # --- mini performance header ---
    score_now = float(latest.get('Institutional Score', np.nan))
    confidence_now = float(latest.get('Confidence Score', np.nan))
    regime_now = str(latest.get('Recommendation', 'HOLD'))
    ret_20 = float(score_df['Close'].pct_change(20).iloc[-1] * 100) if len(score_df) > 20 and pd.notna(score_df['Close'].pct_change(20).iloc[-1]) else np.nan
    ret_63 = float(score_df['Close'].pct_change(63).iloc[-1] * 100) if len(score_df) > 63 and pd.notna(score_df['Close'].pct_change(63).iloc[-1]) else np.nan
    ret_252 = float(score_df['Close'].pct_change(252).iloc[-1] * 100) if len(score_df) > 252 and pd.notna(score_df['Close'].pct_change(252).iloc[-1]) else np.nan
    avwap_now = float(anchored_vwap.dropna().iloc[-1]) if anchored_vwap.notna().any() else np.nan
    perf_text = (
        f"<b>Adjusted Close:</b> {latest_close:,.2f} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"<b>20D:</b> {ret_20:,.1f}% &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"<b>3M:</b> {ret_63:,.1f}% &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"<b>1Y:</b> {ret_252:,.1f}% &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"<b>Anchored VWAP:</b> {avwap_now:,.2f} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"<b>Score:</b> {score_now:,.1f}/100 &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"<b>Confidence:</b> {confidence_now:,.1f}% &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"<b>Regime:</b> {regime_now}"
    )
    fig.add_annotation(xref='paper', yref='paper', x=0.0, y=1.11, showarrow=False, align='left', text=perf_text, font=dict(size=12, color='#111827'), bgcolor='rgba(255,255,255,0.92)', bordercolor='rgba(203,213,225,0.95)', borderwidth=1, borderpad=6)

    fig.update_layout(height=1110, template='plotly_white', hovermode='x unified', margin=dict(l=20, r=20, t=100, b=20), legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1.0), xaxis_rangeslider_visible=False, bargap=0.0)
    fig.update_xaxes(showgrid=True, gridcolor='#eef2f6')
    fig.update_yaxes(showgrid=True, gridcolor='#eef2f6', title_text='Adjusted Price', row=1, col=1, secondary_y=False)
    fig.update_yaxes(showgrid=False, title_text='Volume', row=1, col=1, secondary_y=True)
    fig.update_yaxes(range=[0, 100], title_text='Score', row=2, col=1)
    fig.update_yaxes(title_text='Points', row=3, col=1)
    return fig

# -------------------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------------------
st.sidebar.title("SupertrendPro Institutional")
st.sidebar.caption("Version 5.0.2 · Real Yahoo Finance daily data · No synthetic price series")

selected_category = st.sidebar.selectbox("Select Sector / Category:", list(MARKET_DATA.keys()), index=2)
ticker_options = MARKET_DATA[selected_category]
selected_asset_name = st.sidebar.selectbox("Select Asset:", list(ticker_options.keys()))
ticker_symbol = ticker_options[selected_asset_name]

st.sidebar.markdown("---")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today") + pd.Timedelta(days=1))

st.sidebar.markdown("---")
st.sidebar.subheader("Indicator Engine")
_indicator_options = ["Auto — Prefer TA-Lib", "TA-Lib", "Pandas / NumPy"]
_indicator_default = 0
indicator_engine_mode = st.sidebar.selectbox(
    "Calculation Engine",
    options=_indicator_options,
    index=_indicator_default,
    help=(
        "Auto uses TA-Lib when the package is installed and falls back to the internal "
        "Pandas/NumPy formulas only when TA-Lib cannot be imported."
    ),
)

if indicator_engine_mode == "Pandas / NumPy":
    TALIB_AVAILABLE = False
    ACTIVE_INDICATOR_ENGINE = "Pandas / NumPy"
elif indicator_engine_mode == "TA-Lib":
    TALIB_AVAILABLE = bool(TALIB_INSTALLED)
    ACTIVE_INDICATOR_ENGINE = "TA-Lib" if TALIB_AVAILABLE else "Pandas / NumPy fallback"
else:
    TALIB_AVAILABLE = bool(TALIB_INSTALLED)
    ACTIVE_INDICATOR_ENGINE = "TA-Lib" if TALIB_AVAILABLE else "Pandas / NumPy fallback"

st.sidebar.caption(
    f"Active engine: {ACTIVE_INDICATOR_ENGINE}"
    + (f" · TA-Lib {TALIB_VERSION}" if TALIB_AVAILABLE else "")
)

st.sidebar.markdown("---")
st.sidebar.subheader("Execution Assumptions")
transaction_cost_bps = st.sidebar.number_input("Transaction Cost (bps per side)", min_value=0.0, max_value=100.0, value=8.0, step=1.0)
slippage_bps = st.sidebar.number_input("Slippage (bps per side)", min_value=0.0, max_value=100.0, value=4.0, step=1.0)
st.sidebar.caption("Net strategy returns deduct costs only when position changes. No synthetic prices are used.")

use_index_filter_global = st.sidebar.checkbox("Use BIST 100 Regime Filter (XU100 > EMA200)", value=False)
strategy_choice = st.sidebar.radio("Select Strategy Variant:", ["MACD + ATR Trailing", "Smart Supertrend", "Smart Supertrend + Optimizer"], index=1)

if strategy_choice.startswith("MACD"):
    st.sidebar.subheader("MACD + ATR Parameters")
    atr_mult_stop_macd = st.sidebar.slider("ATR Trailing Stop Multiplier", 0.5, 6.0, 2.0, 0.5)
    use_rsi_exit = st.sidebar.checkbox("Use RSI Exit Filter", False)
    rsi_exit_level = st.sidebar.slider("RSI Exit Threshold", 20, 50, 30)
    use_macd_exit = st.sidebar.checkbox("Use MACD Exit (Bear Cross)", False)
    use_ema_macd = st.sidebar.checkbox("Use EMA200 Filter (Entry)", False)
    use_adx_macd = st.sidebar.checkbox("Use ADX Filter (Entry)", False)
    adx_threshold_macd = st.sidebar.slider("ADX Threshold (MACD)", 5, 40, 10)
    macd_fast = st.sidebar.slider("MACD Fast Period", 5, 20, 8, 1)
    macd_slow = st.sidebar.slider("MACD Slow Period", 10, 40, 21, 1)
    macd_signal = st.sidebar.slider("MACD Signal Period", 5, 20, 9, 1)
else:
    st.sidebar.subheader("Smart Supertrend Parameters")
    st_period = st.sidebar.slider("Supertrend Period", 7, 50, 10)
    st_mult = st.sidebar.slider("Supertrend Multiplier", 1.0, 6.0, 2.5, 0.5)
    use_adx_filter = st.sidebar.checkbox("Use ADX Filter", False)
    adx_threshold = st.sidebar.slider("ADX Threshold", 5, 40, 10)
    use_ema_filter = st.sidebar.checkbox("Use EMA200 Filter (Entry)", False)
    atr_mult_stop_st = st.sidebar.slider("ATR Trailing Stop Multiplier", 0.5, 6.0, 2.0, 0.5)

# -------------------------------------------------------------------------
# MAIN DATA LOAD
# -------------------------------------------------------------------------
st.markdown(
    """
    <div class="hf-masthead">
      <div class="hf-eyebrow">MK FinTECH LabGEN · Institutional Analytics</div>
      <div class="hf-brand-row">
        <div class="hf-title">SupertrendPro Institutional</div>
        <div class="hf-version">Version 5.0.2</div>
      </div>
      <div class="hf-meta">Trend · Risk · Diagnostics · Leading Signal Engine · No Synthetic Data</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("Yahoo Finance daily OHLCV · Net-of-cost backtests · Educational analytics, not investment advice")

engine_col1, engine_col2, engine_col3 = st.columns(3)
engine_col1.metric("Market Data Engine", "Yahoo Finance")
engine_col2.metric("Indicator Engine", ACTIVE_INDICATOR_ENGINE)
engine_col3.metric("TA-Lib Version", TALIB_VERSION if TALIB_INSTALLED else "Unavailable")

if indicator_engine_mode == "TA-Lib" and not TALIB_INSTALLED:
    st.warning(
        "TA-Lib was explicitly selected but could not be imported. "
        "The app has switched to its internal Pandas/NumPy indicator formulas. "
        f"Import detail: {TALIB_IMPORT_ERROR or 'unknown error'}"
    )
elif indicator_engine_mode.startswith("Auto") and TALIB_INSTALLED:
    st.success(f"TA-Lib {TALIB_VERSION} is active for supported indicators. Pandas/NumPy remains available as a fallback.")
elif indicator_engine_mode.startswith("Auto") and not TALIB_INSTALLED:
    st.info(
        "TA-Lib is not available in this deployment, so Auto mode selected the internal "
        "Pandas/NumPy engine. Add TA-Lib to requirements.txt and reboot the app to activate it."
    )
else:
    st.info("Pandas/NumPy indicator calculations were selected manually. Price data still comes only from Yahoo Finance.")

with st.spinner("Fetching XU100 benchmark for regime filter and beta calculations..."):
    idx_raw = get_data(BENCHMARK_SYMBOL, start_date, end_date)
idx_ind, index_returns, index_regime = None, None, None
if idx_raw is not None and len(idx_raw) > 260:
    idx_ind = compute_indicators(idx_raw)
    index_returns = idx_ind["Close"].pct_change().fillna(0.0)
    index_regime = idx_ind["Close"] > idx_ind["EMA_200"]
else:
    st.warning("XU100 benchmark data is unavailable or insufficient. Beta, alpha and regime filter may be disabled.")
    use_index_filter_global = False

with st.spinner(f"Fetching real Yahoo data for {selected_asset_name} ({ticker_symbol})..."):
    data_raw = get_data(ticker_symbol, start_date, end_date)
if data_raw is None or len(data_raw) < 260:
    st.error(f"Insufficient Yahoo Finance data for {selected_asset_name} ({ticker_symbol}). No synthetic fallback is used.")
    st.stop()

data = compute_indicators(data_raw)
if strategy_choice.startswith("MACD"):
    plot_data, trades_df, stats = backtest_macd_atr_trailing(
        data, start_date=start_date,
        atr_mult_stop=atr_mult_stop_macd,
        use_rsi_exit=use_rsi_exit,
        rsi_exit_level=rsi_exit_level,
        use_ema_filter=use_ema_macd,
        use_adx_filter=use_adx_macd,
        adx_threshold=adx_threshold_macd,
        use_macd_exit=use_macd_exit,
        fastperiod=macd_fast,
        slowperiod=macd_slow,
        signalperiod=macd_signal,
        market_filter=index_regime if use_index_filter_global else None,
        index_returns=index_returns,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
    )
else:
    plot_data, trades_df, stats = backtest_supertrend_trailing(
        data, start_date=start_date,
        st_period=st_period,
        st_mult=st_mult,
        use_adx_filter=use_adx_filter,
        adx_threshold=adx_threshold,
        use_ema_filter=use_ema_filter,
        atr_mult_stop=atr_mult_stop_st,
        market_filter=index_regime if use_index_filter_global else None,
        index_returns=index_returns,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
    )

last = plot_data.iloc[-1]
trend_state = "BULLISH" if last["Close"] > last["EMA_200"] else "BEARISH"
tech_score, tech_reasons = technical_grade(last)

st.markdown(
    f"""
    <div class="instrument-header">
      <div class="instrument-kicker">Selected Instrument</div>
      <div class="instrument-title">{selected_asset_name}<span>{ticker_symbol}</span></div>
      <div class="instrument-subtitle">Institutional technical, risk, backtest and decision analytics</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("Version 5.0.2 · Strategy diagnostics · Leading signal lab · Cloud-stable deployment")

# Top KPIs
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Last Price", f"₺{last['Close']:.2f}")
k2.metric("RSI", f"{last['RSI']:.1f}")
k3.metric("Trend vs EMA200", trend_state)
k4.metric("Strategy Return", f"{stats.get('strat_total_pct', np.nan):.1f}%")
k5.metric("Strategy MaxDD", f"{stats.get('strat_mdd_pct', np.nan):.1f}%")
k6.metric("Technical Score", f"{tech_score:.0f}/100")

st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "Strategy & Signal",
    "Market Data",
    "Technical Analytics",
    "Backtest & Risk",
    "Strategy Diagnostics",
    "Blue-Chip Screener",
    "Capital Gain Leaders",
    "Portfolio Lab",
    "Leading Signal Lab",
    "Institutional Decision Engine",
])

# -------------------------------------------------------------------------
# TAB 1
# -------------------------------------------------------------------------
with tab1:
    st.plotly_chart(strategy_chart(plot_data, f"{selected_asset_name} ({ticker_symbol}) — {strategy_choice}"), width="stretch", theme=None)
    st.markdown(f"<div class='ok-note'><b>Signal Drivers:</b> {tech_reasons or 'No strong technical driver detected.'}</div>", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# TAB 2
# -------------------------------------------------------------------------
with tab2:
    st.subheader("Market Data — OHLCV, Signals, Risk & Rolling Beta")
    cols = ["Open", "High", "Low", "Close", "Volume", "RSI", "EMA_50", "EMA_200", "MACD", "MACD_SIGNAL", "ATR_Pct", "ADX", "ST_Dir", "Filter_Trend_Pass", "Filter_EMA200_Pass", "Filter_ADX_Pass", "Filter_Market_Pass", "Entry_Eligible", "Exit_Rule", "Signal", "Position", "ATR_Stop", "Return", "Gross_Strategy_Return", "Trading_Cost", "Turnover", "Strategy_Return", "Rolling_Beta_Asset", "Rolling_Beta_Strategy", "Drawdown"]
    show = plot_data[[c for c in cols if c in plot_data.columns]].sort_index(ascending=False).copy()
    st.dataframe(style_smart_table(show.head(800)), width="stretch", height=620)
    csv = show.to_csv(index=True).encode("utf-8")
    st.download_button("Download current asset table as CSV", csv, file_name=f"{ticker_symbol.replace('.','_')}_smart_table.csv", mime="text/csv")
    if not trades_df.empty:
        st.subheader("Trade Log")
        st.dataframe(style_smart_table(trades_df.sort_values("EntryDate", ascending=False)), width="stretch")

# -------------------------------------------------------------------------
# TAB 3
# -------------------------------------------------------------------------
with tab3:
    st.subheader("Technical Analytics — Candlestick, Bollinger Bands, Supertrend, RSI and MACD")
    ts = plot_data.copy()
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.58, 0.22, 0.20], subplot_titles=("Candlestick + Bollinger + EMA + Supertrend", "MACD", "RSI"))
    fig.add_trace(go.Candlestick(x=ts.index, open=ts["Open"], high=ts["High"], low=ts["Low"], close=ts["Close"], name="OHLC"), row=1, col=1)
    for c, n, dash in [("BB_UPPER", "BB Upper", "dash"), ("BB_MID", "BB Mid", "dot"), ("BB_LOWER", "BB Lower", "dash"), ("EMA_50", "EMA50", "dot"), ("EMA_200", "EMA200", "solid"), ("ST_Line", "Supertrend", "solid")]:
        if c in ts.columns:
            fig.add_trace(go.Scatter(x=ts.index, y=ts[c], name=n, mode="lines", line=dict(width=1.1, dash=dash)), row=1, col=1)
    fig.add_trace(go.Scatter(x=ts.index, y=ts["MACD"], name="MACD", mode="lines"), row=2, col=1)
    fig.add_trace(go.Scatter(x=ts.index, y=ts["MACD_SIGNAL"], name="Signal", mode="lines", line=dict(dash="dot")), row=2, col=1)
    fig.add_trace(go.Bar(x=ts.index, y=ts["MACD_HIST"], name="Hist"), row=2, col=1)
    fig.add_trace(go.Scatter(x=ts.index, y=ts["RSI"], name="RSI", mode="lines"), row=3, col=1)
    fig.add_hrect(y0=70, y1=100, opacity=0.08, line_width=0, row=3, col=1)
    fig.add_hrect(y0=0, y1=30, opacity=0.08, line_width=0, row=3, col=1)
    st.plotly_chart(clean_fig(fig, height=900), width="stretch", theme=None)

# -------------------------------------------------------------------------
# TAB 4
# -------------------------------------------------------------------------
with tab4:
    st.subheader("Backtest, Beta and Institutional Risk Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Strategy CAGR", f"{stats.get('strat_annual_pct', np.nan):.2f}%")
    c2.metric("Sharpe / Sortino", f"{stats.get('sharpe', np.nan):.2f} / {stats.get('sortino', np.nan):.2f}")
    c3.metric("Beta vs XU100", f"{stats.get('beta_strategy', np.nan):.2f}")
    c4.metric("IR / TE", f"{stats.get('information_ratio', np.nan):.2f} / {stats.get('tracking_error_pct', np.nan):.1f}%")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("VaR 95% / CVaR 95%", f"{stats.get('var95_pct', np.nan):.2f}% / {stats.get('cvar95_pct', np.nan):.2f}%")
    c6.metric("VaR 99% / CVaR 99%", f"{stats.get('var99_pct', np.nan):.2f}% / {stats.get('cvar99_pct', np.nan):.2f}%")
    c7.metric("Up / Down Capture", f"{stats.get('up_capture_pct', np.nan):.1f}% / {stats.get('down_capture_pct', np.nan):.1f}%")
    c8.metric("Win Rate / Closed Trades", f"{stats.get('win_rate', np.nan):.1f}% / {stats.get('trade_count', 0)}")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Exposure", f"{stats.get('exposure_pct', np.nan):.1f}%")
    d2.metric("Buy / Sell Signals", f"{stats.get('buy_signal_count', 0)} / {stats.get('sell_signal_count', 0)}")
    d3.metric("Entry-Eligible Days", f"{stats.get('entry_eligible_days', 0)}")
    d4.metric("Open Position Now", "YES" if stats.get('active_position_now', False) else "NO")

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Omega Ratio", f"{stats.get('omega', np.nan):.2f}")
    e2.metric("Ulcer Index", f"{stats.get('ulcer_index', np.nan):.2f}")
    e3.metric("Longest Drawdown", f"{stats.get('longest_dd_days', np.nan):.0f} days")
    total_cost_pct = plot_data.get("Trading_Cost", pd.Series(0.0, index=plot_data.index)).sum() * 100
    e4.metric("Cumulative Trading Costs", f"{total_cost_pct:.2f}%")

    benchmark_aligned_obs = int(index_returns.reindex(plot_data.index).notna().sum()) if index_returns is not None else 0
    diagnostic_rows = [
        {"Check": "Valid observations", "Value": f"{len(plot_data):,}", "Status": "PASS" if len(plot_data) >= 120 else "REVIEW"},
        {"Check": "Entry-eligible days", "Value": f"{int(stats.get('entry_eligible_days', 0)):,}", "Status": "PASS" if stats.get('entry_eligible_days', 0) > 0 else "FAIL"},
        {"Check": "Buy signals", "Value": f"{int(stats.get('buy_signal_count', 0)):,}", "Status": "PASS" if stats.get('buy_signal_count', 0) > 0 else "FAIL"},
        {"Check": "Market exposure", "Value": f"{stats.get('exposure_pct', np.nan):.1f}%", "Status": "PASS" if stats.get('exposure_pct', 0) > 0 else "FAIL"},
        {"Check": "Non-zero net returns", "Value": f"{int((plot_data['Strategy_Return'].abs() > 1e-12).sum()):,}", "Status": "PASS" if (plot_data['Strategy_Return'].abs() > 1e-12).any() else "FAIL"},
        {"Check": "Benchmark alignment", "Value": f"{benchmark_aligned_obs:,}", "Status": "PASS" if benchmark_aligned_obs >= 60 else "REVIEW"},
    ]
    st.markdown("#### Strategy Execution Audit")
    st.dataframe(pd.DataFrame(diagnostic_rows), width="stretch", hide_index=True)

    if stats.get('buy_signal_count', 0) == 0 and stats.get('entry_eligible_days', 0) == 0:
        st.warning("Strategy produced no eligible entry days under the current filters. Try disabling EMA200, ADX, or BIST100 regime filter, or use a longer backtest window.")
    elif stats.get('trade_count', 0) == 0 and stats.get('active_position_now', False):
        st.info("The strategy is active but has not closed a trade yet in the selected window. Closed-trade statistics will remain zero until an exit occurs.")

    st.plotly_chart(equity_risk_chart(plot_data), width="stretch", theme=None)

    risk_table = pd.DataFrame([compute_return_metrics(plot_data["Return"], index_returns, "Buy & Hold"), compute_return_metrics(plot_data["Strategy_Return"], index_returns, "Strategy")])
    st.dataframe(style_smart_table(risk_table), width="stretch")

    if strategy_choice == "Smart Supertrend + Optimizer":
        st.markdown("---")
        st.subheader("Smart Supertrend Recent-Window Optimization")
        opt_window = st.slider("Optimization Window Days", 90, 540, 180, 30)
        if st.button("Run Optimization Grid"):
            rows = []
            combos = list(itertools.product([7, 10, 14, 20, 30], [1.5, 2.0, 2.5, 3.0, 4.0], [5, 10, 15, 20]))
            prog = st.progress(0.0)
            opt_start = plot_data.index[-1] - pd.Timedelta(days=opt_window)
            for i, (per, mult, adx_th) in enumerate(combos):
                bt, tr, stt = backtest_supertrend_trailing(data, opt_start, per, mult, True, adx_th, use_ema_filter, atr_mult_stop_st, index_regime if use_index_filter_global else None, index_returns)
                rows.append({"Period": per, "Multiplier": mult, "ADX": adx_th, "Return %": stt.get("strat_total_pct", np.nan), "Sharpe": stt.get("sharpe", np.nan), "MaxDD %": stt.get("strat_mdd_pct", np.nan), "Trades": stt.get("trade_count", 0)})
                prog.progress((i + 1) / len(combos))
            prog.empty()
            opt_df = pd.DataFrame(rows).sort_values(["Sharpe", "Return %"], ascending=False)
            st.dataframe(style_smart_table(opt_df.head(30)), width="stretch")
            pivot = opt_df[opt_df["ADX"] == opt_df.iloc[0]["ADX"]].pivot_table(index="Period", columns="Multiplier", values="Return %", aggfunc="mean")
            st.plotly_chart(clean_fig(go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, colorbar=dict(title="Return %"))).update_layout(title="Return Heatmap for Best ADX Bucket"), height=500), width="stretch")

# -------------------------------------------------------------------------
# TAB 5: STRATEGY DIAGNOSTICS
# -------------------------------------------------------------------------
with tab5:
    st.subheader("Strategy Diagnostics — Filter Constraint Analysis")
    st.caption("Every row is calculated from real Yahoo Finance observations. Disabled filters are treated as PASS and are clearly labelled below.")

    diagnostic_columns = [
        "Filter_Trend_Pass", "Filter_EMA200_Pass", "Filter_ADX_Pass",
        "Filter_Market_Pass", "Entry_Eligible"
    ]
    diag_df = plot_data[[c for c in diagnostic_columns if c in plot_data.columns]].copy()
    total_obs = max(len(diag_df), 1)
    labels = {
        "Filter_Trend_Pass": "Trend condition passed",
        "Filter_EMA200_Pass": "EMA200 filter passed",
        "Filter_ADX_Pass": "ADX filter passed",
        "Filter_Market_Pass": "BIST100 regime passed",
        "Entry_Eligible": "FINAL ENTRY ELIGIBLE",
    }
    active_flags = {
        "Filter_Trend_Pass": True,
        "Filter_EMA200_Pass": bool(use_ema_macd) if strategy_choice.startswith("MACD") else bool(use_ema_filter),
        "Filter_ADX_Pass": bool(use_adx_macd) if strategy_choice.startswith("MACD") else bool(use_adx_filter),
        "Filter_Market_Pass": bool(use_index_filter_global),
        "Entry_Eligible": True,
    }
    rows = []
    for col in diagnostic_columns:
        if col not in diag_df.columns:
            continue
        passed = int(diag_df[col].fillna(False).astype(bool).sum())
        rows.append({
            "Constraint": labels[col],
            "Filter status": "ACTIVE" if active_flags[col] else "OFF (not restrictive)",
            "Days passed": passed,
            "Pass rate %": passed / total_obs * 100,
            "Days blocked": total_obs - passed,
        })
    constraint_table = pd.DataFrame(rows)
    st.dataframe(
        constraint_table.style.format({"Pass rate %": "{:.2f}%"}),
        width="stretch",
        hide_index=True,
    )

    if not constraint_table.empty:
        fig_diag = go.Figure(go.Bar(
            x=constraint_table["Days passed"],
            y=constraint_table["Constraint"],
            orientation="h",
            text=constraint_table["Pass rate %"].map(lambda x: f"{x:.1f}%"),
            textposition="auto",
        ))
        fig_diag.update_layout(
            title="Filter Funnel — Number of Trading Days Passing Each Condition",
            template="plotly_white", height=430,
            xaxis_title="Trading days passed", yaxis_title="",
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(fig_diag, width="stretch", theme=None)

    st.markdown("#### Daily decision audit")
    audit_cols = [
        "Close", "EMA_200", "ADX", "ST_Dir", "MACD", "MACD_SIGNAL",
        "Filter_Trend_Pass", "Filter_EMA200_Pass", "Filter_ADX_Pass",
        "Filter_Market_Pass", "Entry_Eligible", "Signal", "Position"
    ]
    audit = plot_data[[c for c in audit_cols if c in plot_data.columns]].sort_index(ascending=False).head(500)
    st.dataframe(style_smart_table(audit), width="stretch", height=600)

    eligible = int(plot_data.get("Entry_Eligible", pd.Series(False, index=plot_data.index)).sum())
    if eligible == 0:
        st.error("FINAL ENTRY ELIGIBLE = 0. At least one active filter blocks every possible entry day. Compare the pass rates above and disable or relax the smallest active pass-rate filter first.")
    else:
        st.success(f"FINAL ENTRY ELIGIBLE = {eligible} trading days ({eligible / total_obs * 100:.2f}% of the test sample).")

# -------------------------------------------------------------------------
# TAB 5: BLUE-CHIP UNIVERSE SCREENER
# -------------------------------------------------------------------------
with tab6:
    st.subheader("BIST Blue-Chip Universe Screener")
    st.markdown("<div class='small-note'>Universe includes banks, QNB, Garanti, YKBNK, Koç Holding, Sabancı Holding, Pegasus, industrials, telecom, consumer and energy names. Calculations use real Yahoo daily data only.</div>", unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    min_obs_scan = col_a.slider("Minimum valid observations", 60, 756, 180, 30, key="blue_min_obs")
    selected_groups = col_b.multiselect("Universe Groups", [k for k in MARKET_DATA.keys() if k != "Indices"], default=["Major Banks & Financials", "Holdings & Conglomerates", "Transport, Aviation & Tourism", "Industrial Blue Chips"])
    scan_limit = col_c.slider("Max names to show", 10, 100, 40, 5)

    selected_map = {}
    for group in selected_groups:
        selected_map.update(MARKET_DATA[group])

    if st.button("Run Blue-Chip Universe Scan"):
        with st.spinner("Scanning selected BIST universe with real Yahoo data..."):
            scan_df, excluded_df, data_map = run_universe_scan(selected_map, start_date, end_date, index_returns, min_obs=min_obs_scan)
            scan_df = smart_score_table(scan_df)
            st.session_state["blue_scan_df"] = scan_df
            st.session_state["blue_excluded_df"] = excluded_df
            st.session_state["blue_data_symbols"] = list(data_map.keys())

    scan_df = st.session_state.get("blue_scan_df", pd.DataFrame())
    if scan_df is not None and not scan_df.empty:
        st.success(f"Scan complete: {len(scan_df)} valid names. Excluded names are listed below, if any.")
        show_cols = ["Name", "Symbol", "Action Lens", "Composite Score", "Technical Score", "Last Close", "RSI", "ADX", "3M Momentum %", "6M Momentum %", "CAGR %", "Ann Vol %", "Sharpe", "Max Drawdown %", "Beta vs XU100", "Avg Daily TL Volume", "Signal Drivers"]
        st.dataframe(style_smart_table(scan_df[[c for c in show_cols if c in scan_df.columns]].head(scan_limit)), width="stretch", height=620)
        st.plotly_chart(risk_return_bubble(scan_df, "Blue-Chip Risk / Return / Liquidity Map"), width="stretch", theme=None)
        st.plotly_chart(momentum_bar(scan_df, "Top Blue-Chip Momentum Profile", n=min(20, len(scan_df))), width="stretch", theme=None)
        st.download_button("Download blue-chip screener CSV", scan_df.to_csv(index=False).encode("utf-8"), "bist_blue_chip_screener.csv", "text/csv")
    excluded_df = st.session_state.get("blue_excluded_df", pd.DataFrame())
    if excluded_df is not None and not excluded_df.empty:
        with st.expander("Data Quality / Exclusion Log"):
            st.dataframe(excluded_df, width="stretch")

# -------------------------------------------------------------------------
# TAB 6: CAPITAL GAIN LEADERS LAB
# -------------------------------------------------------------------------
with tab7:
    st.subheader("Capital Gain Leaders — High-Momentum Basket")
    st.markdown("<div class='risk-note'><b>No synthetic data rule:</b> the snapshot gain table is only a user-provided watchlist/metadata layer. All prices, returns, beta, volatility and signals below are recalculated from real Yahoo Finance OHLCV. If Yahoo data is missing, the stock is excluded and logged.</div>", unsafe_allow_html=True)
    cap_meta = pd.DataFrame(CAPITAL_GAIN_LEADERS)
    st.markdown("#### User-Provided Snapshot Watchlist")
    st.dataframe(style_smart_table(cap_meta), width="stretch", height=320)

    col1, col2, col3 = st.columns(3)
    min_obs_cap = col1.slider("Minimum valid observations", 40, 756, 120, 20, key="cap_min_obs")
    cap_top_n = col2.slider("Top N rows", 5, 40, 30, 5, key="cap_topn")
    rank_basis = col3.selectbox("Sort by", ["Composite Score", "3M Momentum %", "6M Momentum %", "Sharpe", "Technical Score", "SnapshotGainPct"], index=0)

    if st.button("Run Capital Gain Leaders Scan"):
        cap_map = {row["Name"]: row["Symbol"] for row in CAPITAL_GAIN_LEADERS}
        with st.spinner("Scanning capital gain leaders using real Yahoo data..."):
            cap_df, cap_excl, cap_data_map = run_universe_scan(cap_map, start_date, end_date, index_returns, min_obs=min_obs_cap)
            cap_df = smart_score_table(cap_df)
            if not cap_df.empty:
                cap_df = cap_df.merge(cap_meta[["Symbol", "SnapshotPrice", "SnapshotGainPct", "SnapshotTarget", "Rating"]], on="Symbol", how="left")
                if rank_basis in cap_df.columns:
                    cap_df = cap_df.sort_values(rank_basis, ascending=False)
            st.session_state["cap_df"] = cap_df
            st.session_state["cap_excl"] = cap_excl
            st.session_state["cap_data_symbols"] = list(cap_data_map.keys())

    cap_df = st.session_state.get("cap_df", pd.DataFrame())
    if cap_df is not None and not cap_df.empty:
        show_cols = ["Name", "Symbol", "Action Lens", "Composite Score", "SnapshotGainPct", "SnapshotTarget", "Rating", "Last Close", "RSI", "ADX", "3M Momentum %", "6M Momentum %", "1Y Momentum %", "From 52W High %", "ATR %", "Ann Vol %", "Sharpe", "Max Drawdown %", "Beta vs XU100", "VaR 95% %", "Avg Daily TL Volume", "Signal Drivers"]
        st.markdown("#### Capital Gain Leaders — Smart Ranking")
        st.dataframe(style_smart_table(cap_df[[c for c in show_cols if c in cap_df.columns]].head(cap_top_n)), width="stretch", height=650)
        c1, c2, c3 = st.columns(3)
        top = cap_df.iloc[0]
        c1.metric("Top Composite", f"{top['Name']} ({top['Symbol']})", f"{top['Composite Score']:.1f}")
        c2.metric("Best 3M Momentum", f"{cap_df.sort_values('3M Momentum %', ascending=False).iloc[0]['Symbol']}", f"{cap_df['3M Momentum %'].max():.1f}%")
        c3.metric("Highest Risk Vol", f"{cap_df.sort_values('Ann Vol %', ascending=False).iloc[0]['Symbol']}", f"{cap_df['Ann Vol %'].max():.1f}%")
        st.plotly_chart(risk_return_bubble(cap_df, "Capital Gain Leaders — Risk / Return / Liquidity Map"), width="stretch", theme=None)
        st.plotly_chart(momentum_bar(cap_df, "Capital Gain Leaders — Momentum Comparison", n=min(25, len(cap_df))), width="stretch", theme=None)
        st.download_button("Download capital gain leaders CSV", cap_df.to_csv(index=False).encode("utf-8"), "bist_capital_gain_leaders_scan.csv", "text/csv")
    cap_excl = st.session_state.get("cap_excl", pd.DataFrame())
    if cap_excl is not None and not cap_excl.empty:
        with st.expander("Capital Gain Leaders — Exclusion Log"):
            st.dataframe(cap_excl, width="stretch")

# -------------------------------------------------------------------------
# TAB 7: MINI PORTFOLIO LAB
# -------------------------------------------------------------------------
with tab8:
    st.subheader("Equal-Weight Portfolio Lab vs XU100")
    source_choice = st.radio("Choose selection source", ["Manual Universe", "Top Blue-Chip Scan", "Top Capital Gain Leaders"], horizontal=True)
    if source_choice == "Manual Universe":
        all_names = list(UNIVERSE_STOCKS.keys())
        default_names = ["Akbank", "Garanti BBVA", "Yapi Kredi", "Koc Holding", "Sabanci Holding", "Pegasus Airlines", "Turkish Airlines", "Ford Otosan"]
        chosen_names = st.multiselect("Select stocks", all_names, default=[x for x in default_names if x in all_names])
        chosen_symbols = [UNIVERSE_STOCKS[n] for n in chosen_names]
    elif source_choice == "Top Blue-Chip Scan":
        scan_df = st.session_state.get("blue_scan_df", pd.DataFrame())
        if scan_df.empty:
            st.warning("Run the Blue-Chip Universe Scan first, or switch to Manual Universe.")
            chosen_symbols = []
        else:
            topn = st.slider("Top N from blue-chip scan", 3, min(20, len(scan_df)), min(8, len(scan_df)))
            chosen_symbols = scan_df.head(topn)["Symbol"].tolist()
            st.write(chosen_symbols)
    else:
        cap_df = st.session_state.get("cap_df", pd.DataFrame())
        if cap_df.empty:
            st.warning("Run the Capital Gain Leaders Scan first, or switch to Manual Universe.")
            chosen_symbols = []
        else:
            topn = st.slider("Top N from capital gain leaders", 3, min(20, len(cap_df)), min(8, len(cap_df)), key="cap_port_topn")
            chosen_symbols = cap_df.head(topn)["Symbol"].tolist()
            st.write(chosen_symbols)

    if st.button("Run Equal-Weight Portfolio Backtest"):
        with st.spinner("Building equal-weight portfolio from real Yahoo data..."):
            portfolio = run_equal_weight_portfolio(chosen_symbols, start_date, end_date, idx_ind=idx_ind, min_len=MIN_PRICE_OBS)
            st.session_state["portfolio_result"] = portfolio

    portfolio = st.session_state.get("portfolio_result", None)
    if portfolio is not None:
        if portfolio is None:
            st.error("Portfolio could not be created. Too few valid Yahoo Finance histories.")
        else:
            pmet = portfolio["metrics_port"]
            imet = portfolio.get("metrics_index", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Portfolio CAGR", f"{pmet.get('CAGR %', np.nan):.2f}%")
            c2.metric("Portfolio Vol", f"{pmet.get('Ann Vol %', np.nan):.2f}%")
            c3.metric("Sharpe", f"{pmet.get('Sharpe', np.nan):.2f}")
            c4.metric("MaxDD", f"{pmet.get('Max Drawdown %', np.nan):.2f}%")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=portfolio["eq_port"].index, y=portfolio["eq_port"], mode="lines", name="Equal-Weight Basket"))
            if portfolio["eq_index"] is not None:
                fig.add_trace(go.Scatter(x=portfolio["eq_index"].index, y=portfolio["eq_index"], mode="lines", name="XU100"))
            fig.update_layout(title="Mini Portfolio Equity Curve vs XU100", yaxis_title="Normalized Equity")
            st.plotly_chart(clean_fig(fig, height=560), width="stretch", theme=None)

            st.markdown("#### Portfolio vs Benchmark Metrics")
            st.dataframe(style_smart_table(pd.DataFrame([pmet, imet])), width="stretch")
            st.markdown("#### Component Total Returns")
            comp = portfolio["asset_total_ret"].mul(100).sort_values(ascending=False).reset_index()
            comp.columns = ["Symbol", "Total Return %"]
            st.dataframe(style_smart_table(comp), width="stretch")
            st.plotly_chart(corr_heatmap(portfolio["corr"], "Portfolio Component Correlation Matrix"), width="stretch", theme=None)



# -------------------------------------------------------------------------
# TAB 9: LEADING AL/SAT SIGNAL LAB
# -------------------------------------------------------------------------
with tab9:
    st.subheader("Leading Signal Lab — Vectorised Backtest Without Zipline")
    st.markdown(
        "<div class='ok-note'><b>Methodology:</b> The classic mode reproduces the transparent moving-average crossover approach used as a practical alternative to Zipline. The advanced mode adds trend, prior-high breakout, MACD acceleration, RSI regime, volume and optional XU100 regime confirmation. All decisions are generated at the close and applied from the next trading bar, preventing look-ahead bias.</div>",
        unsafe_allow_html=True,
    )
    c1,c2,c3,c4=st.columns(4)
    signal_mode=c1.selectbox("Signal Method",["Classic SMA Crossover","Advanced Multi-Confirmation"],index=1,key="lead_mode")
    lead_fast=c2.slider("Fast Window",5,80,20,1,key="lead_fast")
    lead_slow=c3.slider("Slow Window",20,250,60,5,key="lead_slow")
    lead_breakout=c4.slider("Breakout Lookback",10,100,20,5,key="lead_breakout")
    if lead_fast>=lead_slow:
        st.error("Fast Window must be smaller than Slow Window.")
    else:
        d1,d2,d3,d4=st.columns(4)
        lead_entry=d1.slider("Advanced Entry Score",2,6,4,1,key="lead_entry",disabled=(signal_mode=="Classic SMA Crossover"))
        lead_exit=d2.slider("Advanced Exit Score",0,5,2,1,key="lead_exit",disabled=(signal_mode=="Classic SMA Crossover"))
        lead_volume=d3.checkbox("Use Volume Confirmation",True,key="lead_volume",disabled=(signal_mode=="Classic SMA Crossover"))
        lead_market=d4.checkbox("Use XU100 Regime Confirmation",False,key="lead_market")
        if lead_entry<=lead_exit and signal_mode=="Advanced Multi-Confirmation":
            st.warning("Entry Score should normally be greater than Exit Score to avoid excessive switching.")
        lab_df,lab_metrics=run_leading_signal_lab(
            plot_data,mode=signal_mode,fast_window=lead_fast,slow_window=lead_slow,
            breakout_window=lead_breakout,entry_score=lead_entry,exit_score=lead_exit,
            use_volume_confirmation=lead_volume,
            market_regime=(index_regime if lead_market else None),benchmark_ret=index_returns,
            transaction_cost_bps=transaction_cost_bps,slippage_bps=slippage_bps,
        )
        latest_action=str(lab_df['Leading_Action'].iloc[-1])
        latest_event=str(lab_df['Signal_Event'].iloc[-1]) or "No new event"
        latest_score=float(lab_df['Signal_Score'].iloc[-1])
        k1,k2,k3,k4,k5=st.columns(5)
        k1.metric("Current Leading Action",latest_action)
        k2.metric("Latest Event",latest_event)
        k3.metric("Confirmation Score",f"{latest_score:.0f}")
        k4.metric("Strategy CAGR",f"{lab_metrics.get('CAGR %',np.nan):.2f}%")
        k5.metric("Strategy MaxDD",f"{lab_metrics.get('Max Drawdown %',np.nan):.2f}%")
        st.plotly_chart(leading_signal_chart(lab_df,f"{selected_asset_name} ({ticker_symbol}) — {signal_mode}"),width="stretch",theme=None)
        st.markdown("#### Signal Strategy Performance")
        metric_order=['Total Return %','CAGR %','Ann Vol %','Sharpe','Sortino','Max Drawdown %','Win Rate %','Beta vs XU100','Information Ratio','Exposure %','Signal Count','Buy Signals','Sell Signals']
        metric_df=pd.DataFrame([{'Metric':m,'Value':lab_metrics.get(m,np.nan)} for m in metric_order])
        st.dataframe(style_smart_table(metric_df),width="stretch",hide_index=True)
        st.markdown("#### Latest Signal Decisions and Confirmations")
        lead_cols=['Close','SMA_Fast','SMA_Slow','EMA_Fast','EMA_Slow','RSI','MACD_HIST','Volume','Trend_Pass','Breakout_Pass','MACD_Pass','RSI_Pass','Volume_Pass','Market_Pass','Signal_Score','Signal_Event','Leading_Action','Position_Lab','Strategy_Return_Lab']
        lead_show=lab_df[[c for c in lead_cols if c in lab_df.columns]].tail(250).sort_index(ascending=False)
        st.dataframe(style_smart_table(lead_show),width="stretch",height=620)
        st.download_button("Download Leading Signal Lab CSV",lab_df.to_csv(index=True).encode('utf-8'),file_name=f"{ticker_symbol.replace('.','_')}_leading_signal_lab.csv",mime='text/csv')
        st.caption("AL/SAT outputs are model signals, not guaranteed forecasts or investment advice. Their consistency must be judged through out-of-sample testing, turnover, drawdown and stability across parameter ranges.")

# -------------------------------------------------------------------------
# TAB 10: INSTITUTIONAL DECISION ENGINE
# -------------------------------------------------------------------------
with tab10:
    st.subheader("Institutional Leading Signal Engine — Explainable 100-Point Decision Score")
    st.markdown(
        "<div class='ok-note'><b>Methodology:</b> Seven transparent factors are scored from observed Yahoo Finance data only: Trend (20), Momentum (20), Relative Strength vs XU100 (15), Volume/Flow (15), Volatility Quality (10), Risk Quality (10), and Market Regime (10). Historical probabilities are empirical outcomes from resolved past observations with similar scores; they are not synthetic forecasts.</div>",
        unsafe_allow_html=True,
    )
    h1,h2,h3 = st.columns(3)
    decision_horizon = h1.selectbox("Probability Horizon", [20, 40, 60, 90], index=2, key="decision_horizon")
    h2.caption("Current signal uses information available through the latest close.")
    h3.caption("Forward outcomes are used only for historical validation rows whose horizons are complete.")

    score_df, factor_df, decision = build_institutional_signal_engine(
        plot_data,
        benchmark_close=(idx_ind['Close'] if idx_ind is not None and 'Close' in idx_ind else None),
        benchmark_returns=index_returns,
        forward_horizon=int(decision_horizon),
    )

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Institutional Score", f"{decision['Institutional Score']:.1f}/100")
    c2.metric("Confidence", f"{decision['Confidence Score']:.1f}%")
    c3.metric("Final Recommendation", decision['Recommendation'])
    c4.metric("Historical Analogs", f"{decision['Historical Analog Count']}")

    p1,p2,p3 = st.columns(3)
    p1.metric(f"Positive Return Probability ({decision_horizon}D)", f"{decision.get(f'Positive Return Probability {decision_horizon}D %', np.nan):.1f}%")
    p2.metric(f"+10% Probability ({decision_horizon}D)", f"{decision.get(f'+10% Probability {decision_horizon}D %', np.nan):.1f}%")
    p3.metric(f"Outperform XU100 Probability ({decision_horizon}D)", f"{decision.get(f'Outperform XU100 Probability {decision_horizon}D %', np.nan):.1f}%")

    st.plotly_chart(institutional_score_chart(score_df), width="stretch", theme=None)

    st.markdown("#### Current Factor Scorecard")
    factor_display = factor_df.copy()
    factor_display['Score'] = factor_display['Score'].round(2)
    factor_display['Maximum'] = factor_display['Maximum'].round(2)
    factor_display['Contribution %'] = factor_display['Contribution %'].round(1)
    st.dataframe(
        factor_display,
        width="stretch",
        hide_index=True,
        column_config={
            'Score': st.column_config.NumberColumn(format='%.2f'),
            'Maximum': st.column_config.NumberColumn(format='%.2f'),
            'Contribution %': st.column_config.ProgressColumn(min_value=0, max_value=100, format='%.1f%%'),
        },
    )

    st.markdown("#### Diagnostics 2.0 — Latest Daily Decisions")
    decision_cols = [
        'Close','Institutional Score','Confidence Score','Recommendation',
        'Trend Score','Momentum Score','Relative Strength Score','Volume Score',
        'Volatility Score','Risk Score','Market Regime Score',
        f'Forward {decision_horizon}D Return', f'Forward {decision_horizon}D Active Return',
    ]
    decision_table = score_df[[c for c in decision_cols if c in score_df.columns]].tail(300).sort_index(ascending=False).copy()
    st.dataframe(style_smart_table(decision_table), width="stretch", height=650)
    st.download_button(
        "Download Institutional Decision Engine CSV",
        score_df.to_csv(index=True).encode('utf-8'),
        file_name=f"{ticker_symbol.replace('.','_')}_institutional_decision_engine.csv",
        mime='text/csv',
    )
    st.caption("Scores and empirical probabilities are decision-support outputs, not guaranteed forecasts or investment advice. Validate stability across horizons, assets, transaction costs and out-of-sample periods.")

