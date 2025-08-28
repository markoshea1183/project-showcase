import io, os, time, zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from dashboard import fetch_prices
from utils import (
    annualize_return, annualize_vol, sharpe_ratio, max_drawdown,
    hist_var, hist_es
)
from optimizer import min_variance, max_sharpe, efficient_frontier

REPORT_DIR = "reports"

# ---------- helpers ----------
def ensure_reports_dir():
    os.makedirs(os.path.join(REPORT_DIR, "plots"), exist_ok=True)

def normalize_weights(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
    df = df.groupby("ticker", as_index=False)["weight"].sum()
    if df["weight"].sum() <= 0:
        raise ValueError("Weights must sum to a positive number.")
    df["weight"] = df["weight"] / df["weight"].sum()
    return df

def compute_core(px: pd.DataFrame, weights_df: pd.DataFrame, rf_annual: float = 0.02):
    rets = np.log(px / px.shift(1)).dropna()
    w = weights_df.set_index("ticker")["weight"].reindex(rets.columns).fillna(0.0)
    port = (rets * w.values).sum(axis=1)

    exp_daily = rets.mean()
    exp_annual = exp_daily * 252.0
    cov_daily = rets.cov()
    cov_annual = cov_daily * 252.0

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
    rows.append({
        "name": "PORTFOLIO",
        "ann_return": annualize_return(port),
        "ann_vol": annualize_vol(port),
        "sharpe": sharpe_ratio(port, rf=rf_annual),
        "mdd": max_drawdown((1 + port).cumprod()),
        "hist_VaR_5%": hist_var(port, 0.05),
        "hist_ES_5%": hist_es(port, 0.05),
    })
    metrics = pd.DataFrame(rows).set_index("name")
    return rets, w, port, exp_annual, cov_annual, metrics

def make_onepage_figure(rets: pd.DataFrame, w: pd.Series, exp_annual: pd.Series, cov_annual: pd.DataFrame, rf_annual: float):
    port = (rets * w.values).sum(axis=1)
    cum = (1 + port).cumprod()
    roll_max = cum.cummax()
    dd = cum / roll_max - 1.0

    corr = rets.corr()
    mu = exp_annual.values
    cov = cov_annual.values
    df_frontier, _ = efficient_frontier(mu, cov, points=60)
    w_tan = max_sharpe(mu, cov, rf_annual)
    w_min = min_variance(mu, cov)
    mu_tan = float(np.dot(w_tan, mu)); sig_tan = float(np.sqrt(np.dot(w_tan, cov @ w_tan)))
    mu_min = float(np.dot(w_min, mu)); sig_min = float(np.sqrt(np.dot(w_min, cov @ w_min)))

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.2)

    ax1 = fig.add_subplot(gs[0, 0])
    cum.plot(ax=ax1)
    ax1.set_title("Portfolio Cumulative Return"); ax1.set_ylabel("Growth of $1")

    ax2 = fig.add_subplot(gs[0, 1])
    dd.plot(ax=ax2)
    ax2.set_title("Portfolio Drawdown"); ax2.set_ylabel("Drawdown")

    ax3 = fig.add_subplot(gs[1, 0])
    im = ax3.imshow(corr.values, interpolation="nearest")
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    ax3.set_xticks(range(len(corr.columns))); ax3.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax3.set_yticks(range(len(corr.index)));   ax3.set_yticklabels(corr.index)
    ax3.set_title("Correlation Heatmap")

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(df_frontier["sigma"], df_frontier["mu"], s=10)
    ax4.scatter([sig_tan], [mu_tan], marker="*", s=150, label="Tangency")
    ax4.scatter([sig_min], [mu_min], marker="o", s=100, label="Min-Var")
    ax4.set_xlabel("Volatility (σ)"); ax4.set_ylabel("Expected Return (μ)")
    ax4.set_title("Efficient Frontier"); ax4.legend()

    fig.suptitle("Portfolio Risk Dashboard", y=0.98, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig

def bundle_results_zip(metrics: pd.DataFrame, weights_df: pd.DataFrame, fig) -> bytes:
    """Return an in-memory .zip with CSVs and PNG."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # metrics
        zf.writestr("risk_summary.csv", metrics.to_csv())
        zf.writestr("allocations.csv", weights_df.to_csv(index=False))
        # figure
        png_bytes = io.BytesIO()
        fig.savefig(png_bytes, format="png", bbox_inches="tight")
        zf.writestr("plots/dashboard_onepage.png", png_bytes.getvalue())
    buf.seek(0)
    return buf.getvalue()

# ---------- Streamlit App ----------
st.set_page_config(page_title="Portfolio Risk Dashboard", layout="wide")
ensure_reports_dir()

st.title("📈 Portfolio Risk Dashboard")

# Sidebar inputs
with st.sidebar:
    st.header("Inputs")
    start = st.text_input("Start date (YYYY-MM-DD)", "2015-01-01")
    end = st.text_input("End date (optional)", "")
    rf = st.number_input("Risk-free (annual)", value=0.02, step=0.005, format="%.3f")

    st.write("---")
    st.caption("Upload a portfolio CSV (ticker,weight) or edit table below.")
    uploaded = st.file_uploader("Portfolio CSV", type=["csv"])

# Portfolio editor (dataframe)
if "portfolio_df" not in st.session_state:
    st.session_state.portfolio_df = pd.DataFrame(
        [{"ticker": "SPY", "weight": 0.4},
         {"ticker": "EFA", "weight": 0.2},
         {"ticker": "IEMG", "weight": 0.15},
         {"ticker": "AGG", "weight": 0.2},
         {"ticker": "GLD", "weight": 0.05}]
    )

if uploaded is not None:
    try:
        st.session_state.portfolio_df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

st.subheader("Portfolio (edit cells directly)")
edited = st.data_editor(
    st.session_state.portfolio_df,
    num_rows="dynamic",
    use_container_width=True
)
st.session_state.portfolio_df = edited

colA, colB, colC = st.columns([1,1,2])
with colA:
    save_csv_name = st.text_input("Save CSV as…", "portfolio.csv")
    if st.button("💾 Save Portfolio CSV"):
        try:
            df_norm = normalize_weights(st.session_state.portfolio_df)
            path = os.path.join(REPORT_DIR, save_csv_name)
            df_norm.to_csv(path, index=False)
            st.success(f"Saved to {path}")
        except Exception as e:
            st.error(str(e))
with colB:
    run_clicked = st.button("▶️ Run Diagnostics", type="primary")

# Results
if run_clicked:
    try:
        df_norm = normalize_weights(st.session_state.portfolio_df)
        tickers = df_norm["ticker"].tolist()
        px = fetch_prices(tickers, start=start or "2015-01-01", end=end or None)
        df_norm = df_norm[df_norm["ticker"].isin(px.columns)]

        rets, w, port, exp_annual, cov_annual, metrics = compute_core(px, df_norm, rf_annual=rf)

        # One-page figure
        fig = make_onepage_figure(rets, w, exp_annual, cov_annual, rf)
        st.pyplot(fig, clear_figure=False)

        st.subheader("Risk Metrics")
        st.dataframe(metrics.style.format({
            "ann_return": "{:.4f}",
            "ann_vol": "{:.4f}",
            "sharpe": "{:.3f}",
            "mdd": "{:.4f}",
            "hist_VaR_5%": "{:.4f}",
            "hist_ES_5%": "{:.4f}",
        }))

        # Downloads
        zip_bytes = bundle_results_zip(metrics, df_norm, fig)
        st.download_button(
            "⬇️ Download Results (.zip)",
            data=zip_bytes,
            file_name=f"portfolio_dashboard_{time.strftime('%Y%m%d-%H%M%S')}.zip",
            mime="application/zip",
        )

        # Also optionally save to reports/ on disk
        tsdir = os.path.join(REPORT_DIR, f"STREAMLIT_run_{time.strftime('%Y%m%d-%H%M%S')}")
        os.makedirs(os.path.join(tsdir, "plots"), exist_ok=True)
        metrics.to_csv(os.path.join(tsdir, "risk_summary.csv"))
        df_norm.to_csv(os.path.join(tsdir, "allocations.csv"), index=False)
        fig.savefig(os.path.join(tsdir, "plots", "dashboard_onepage.png"), bbox_inches="tight")

    except Exception as e:
        st.error(str(e))
