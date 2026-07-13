# SupertrendPro Institutional V5.0.2 — Matplotlib-Free Table Fix

Fixes Streamlit Cloud error:
`ImportError: background_gradient requires matplotlib.`

The institutional table gradient is now generated with plain CSS through
`pandas.Styler.apply`; matplotlib is no longer required. Existing calculations,
tabs, charts, Yahoo Finance data flow, TA-Lib selection and hedge-fund styling
are preserved.
