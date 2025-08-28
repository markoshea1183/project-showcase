import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from utils import annualize_return, annualize_vol, sharpe_ratio, max_drawdown, hist_var, hist_es, gaussian_var, gaussian_es
from optimizer import min_variance, max_sharpe, efficient_frontier

def load_portfolio(csv_path: str) -> pd.Series:
    df = pd.read_csv(csv_path)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0) # handle invalid parsings
    df = df.groupby("ticker", as_index=False)["weight"].sum() # handle duplicate tickers
    w = df.set_index("ticker")["weight"]
    if w.sum() <= 0:
        raise ValueError("Weights must sum to a positive number.")
    w = w / w.sum()
    return w

def fetch_prices(tickers, start="2015-01-01", end=None) -> pd.DataFrame:
    px = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame()
    px = px.dropna(how="all").dropna(axis=1, how="any")
    return px

def compute_metrics(rets: pd.DataFrame, weights: pd.Series, rf_annual=0.0) -> pd.DataFrame:
    # align weights and returns
    w = weights.reindex(rets.columns).fillna(0.0)
    port_ret = (rets * w.values).sum(axis=1)
    cum_port = (1 + port_ret).cumprod()
    # asset metrics
    rows = []
    for col in rets.columns:
        r = rets[col]
        rows.append({
            "name": col,
            "ann_return": annualize_return(r),
            "ann_vol": annualize_vol(r),
            "sharpe": sharpe_ratio(r, rf=rf_annual),
            "mdd": max_drawdown((1 + r).cumprod()),
            "hist_VaR_5%": hist_var(r, 0.05),
            "hist_ES_5%": hist_es(r, 0.05),
        })
    # portfolio metrics
    rows.append({
        "name": "PORTFOLIO",
        "ann_return": annualize_return(port_ret),
        "ann_vol": annualize_vol(port_ret),
        "sharpe": sharpe_ratio(port_ret, rf=rf_annual),
        "mdd": max_drawdown(cum_port),
        "hist_VaR_5%": hist_var(port_ret, 0.05),
        "hist_ES_5%": hist_es(port_ret, 0.05),
    })
    return pd.DataFrame(rows).set_index("name"), port_ret

def plot_cumulative_returns(rets: pd.DataFrame, weights: pd.Series, outpath: str):
    w = weights.reindex(rets.columns).fillna(0.0)
    port = (rets * w.values).sum(axis=1)
    cum = (1 + port).cumprod()
    plt.figure()
    cum.plot()
    plt.title("Portfolio Cumulative Return")
    plt.ylabel("Growth of $1")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_drawdown(rets: pd.DataFrame, weights: pd.Series, outpath: str):
    w = weights.reindex(rets.columns).fillna(0.0)
    port = (rets * w.values).sum(axis=1)
    cum = (1 + port).cumprod()
    roll_max = cum.cummax()
    dd = cum / roll_max - 1.0
    plt.figure()
    dd.plot()
    plt.title("Portfolio Drawdown")
    plt.ylabel("Drawdown")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_corr_heatmap(rets: pd.DataFrame, outpath: str):
    corr = rets.corr()
    plt.figure()
    plt.imshow(corr.values, interpolation="nearest")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_frontier(exp_returns: pd.Series, cov: pd.DataFrame, rf: float, outpath: str):
    from optimizer import max_sharpe, min_variance, efficient_frontier
    mu = exp_returns.values
    covm = cov.values
    df_frontier, _ = efficient_frontier(mu, covm, points=60)
    w_tan = max_sharpe(mu, covm, rf)
    w_min = min_variance(mu, covm)

    mu_tan = float(np.dot(w_tan, mu))
    sigma_tan = float(np.sqrt(np.dot(w_tan, np.dot(covm, w_tan))))
    mu_min = float(np.dot(w_min, mu))
    sigma_min = float(np.sqrt(np.dot(w_min, np.dot(covm, w_min))))

    plt.figure()
    plt.scatter(df_frontier["sigma"], df_frontier["mu"], s=10)
    plt.scatter([sigma_tan], [mu_tan], marker="*", s=150, label="Tangency")
    plt.scatter([sigma_min], [mu_min], marker="o", s=100, label="Min-Var")
    plt.xlabel("Volatility (σ)")
    plt.ylabel("Expected Return (μ)")
    plt.title("Efficient Frontier")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    return df_frontier

def main(args):
    os.makedirs("reports/plots", exist_ok=True)
    w = load_portfolio(args.portfolio)
    px = fetch_prices(w.index, start=args.start, end=args.end)
    # daily log returns
    rets = np.log(px / px.shift(1)).dropna()

    # risk-free annual (convert to daily for mean estimates if using Gaussian VaR)
    rf_annual = args.rf

    # ex-ante expected returns (simple: trailing mean * 252)
    exp_daily = rets.mean()
    exp_annual = exp_daily * 252.0

    # covariance (annualized)
    cov_daily = rets.cov()
    cov_annual = cov_daily * 252.0

    summary, port_ret = compute_metrics(rets, w, rf_annual=rf_annual)
    summary.to_csv("reports/risk_summary.csv")
    w.to_csv("reports/allocations.csv", header=["weight"])

    # plots
    plot_cumulative_returns(rets, w, "reports/plots/cumulative.png")
    plot_drawdown(rets, w, "reports/plots/drawdown.png")
    plot_corr_heatmap(rets, "reports/plots/corr_heatmap.png")
    df_frontier = plot_frontier(exp_annual, cov_annual, rf_annual, "reports/plots/efficient_frontier.png")
    df_frontier.to_csv("reports/frontier.csv", index=False)

    print("Done. See reports/ for outputs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Portfolio Risk Dashboard")
    parser.add_argument("--portfolio", type=str, default="data/sample_portfolio.csv", help="CSV with columns ticker,weight")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--rf", type=float, default=0.02, help="Annual risk-free rate (as a decimal)")
    args = parser.parse_args()
    main(args)
