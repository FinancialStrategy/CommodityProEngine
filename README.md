# CommodityMacroPro Institutional V1.0

A hedge-fund style Streamlit application for daily Yahoo Finance analysis of:

- Gold Futures (`GC=F`)
- Silver Futures (`SI=F`)
- Platinum Futures (`PL=F`)
- WTI Crude Oil Futures (`CL=F`)
- Copper Futures (`HG=F`)
- US Dollar Index (`DX-Y.NYB`)
- US Treasury 10Y Yield (`^TNX`)

## Core tabs

1. Executive Dashboard
2. Smart Price Structure
3. EWMA Volatility
4. DXY Relationship
5. Forecast Laboratory
6. Log Return Difference ±2σ
7. Cross-Asset Matrix
8. Model Validation
9. Data Governance

## Data discipline

- Yahoo Finance daily observations only
- `auto_adjust=False`
- No synthetic prices
- No proxy fallback
- No forward filling of returns
- UST10Y is transformed to daily basis-point changes

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
