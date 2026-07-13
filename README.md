# CommodityMacroPro Institutional V1.0.2

Cloud-stable Streamlit application for daily Yahoo Finance analysis of Gold, Silver, Platinum, WTI Crude Oil, Copper, DXY and US Treasury 10Y yield.

## Hotfix 1.0.2

- Commodity project only; no equity/SupertrendPro code.
- Removed every pandas `Styler` table path.
- No `background_gradient` call and no matplotlib dependency.
- Tables use Streamlit `column_config.NumberColumn` formatting.
- Preserves the nine institutional tabs, Yahoo Finance-only policy, EWMA volatility, DXY relationship, log-return difference bands and walk-forward forecast engine.
- Keeps cloud-stable NumPy 1.26.4 and PyArrow 18.1.0 pins.
