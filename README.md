# Derived Algebraic Geometry Engine for ETFs

Applies Jacob Lurie's derived algebraic geometry to ETF portfolios. Computes the **Massey product obstruction** from the simplicial cohomology of the correlation distance complex. High obstruction indicates higher‑order algebraic structure – a novel alpha signal.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Rips complex from distance matrix (1 - |correlation|)
- Cohomology via Hodge Laplacian of the 2‑skeleton
- Score = triangle participation × (1 + cohomology dimension)
- Walk‑forward backtest validates predictive power
- Three‑tab Streamlit dashboard (auto best, manual, backtest)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-derived-algebraic-geometry-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (runtime ~10‑30 minutes)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- The obstruction class is the most abstract invariant in our suite.
- High score suggests the ETF lies in a highly non‑trivial derived moduli – potentially "arbitrage‑free" in a higher categorical sense.
- This engine extends the MOTIVIC‑COHOMOLOGY repo to the derived setting.

## Requirements

See `requirements.txt`.
