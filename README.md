# CommodityMacroPro Institutional V1.0.1

Streamlit Cloud stability update for the Yahoo Finance commodity/DXY/UST10Y research engine.

## Stability changes

- Forecast and validation tabs now share one walk-forward calculation per horizon.
- BLAS/OpenMP numerical libraries are restricted to one thread before NumPy/SciPy import.
- Ridge uses the stable LSQR solver on contiguous float64 arrays.
- Macro regressions run with bounded native threads.
- Walk-forward OOS window is capped at 240 observations and defaults to 120.
- Memory cleanup is applied during long walk-forward runs.
- Streamlit `width="stretch"` replaces deprecated `use_container_width=True`.
- Cloud packages are pinned to a tested conservative numerical stack, including NumPy 1.26.4 and PyArrow 18.1.0.

## Deployment

Upload `app.py` and `requirements.txt` to the root of the Streamlit Cloud repository, commit, then reboot the app.
