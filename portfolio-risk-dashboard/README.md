# Portfolio Risk Dashboard

This is a compact, finance-ready toolkit to analyze a user-provided portfolio
of tickers and weights.

**Features**
- Download adjusted price data with `yfinance`
- Compute returns, risk stats (vol, Sharpe, drawdown)
- Historical & parametric VaR / Expected Shortfall
- Correlation heatmap
- Efficient frontier & tangency portfolio (mean–variance)
- Clean, one-run script that saves plots and a summary report

> **Quick start** (one-liner):  
> ```bash
> streamlit run src/app_streamlit.py
> ```

## Inputs
Provide a CSV with columns: `ticker,weight`. Weights should sum to 1 (the script will re-normalize if they don’t). Example in `data/sample_portfolio.csv`.

## Outputs
The script writes to `reports/`:
- `risk_summary.csv` – metrics for assets & portfolio
- `plots/` – PNG charts (efficient frontier, correlation heatmap, cumulative returns, drawdown)
- `allocations.csv` – cleaned weights used
- ` frontier.csv` – frontier points (mu, sigma, sharpe)

## Roadmap (nice-to-haves)
- Factor model (Fama-French) exposures
- Rolling betas & regime plots
- Resampled frontier (Michaud)
- Streamlit front-end
