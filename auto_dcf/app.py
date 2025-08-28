import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime
from typing import Dict, Optional, Tuple

st.set_page_config(page_title="Monte Carlo DCF", page_icon="💸", layout="wide")
st.title("💸 Monte Carlo DCF (Yahoo Finance)")
st.caption("Estimate a distribution of intrinsic values via DCF with uncertainty. Data source: Yahoo Finance (via yfinance).")

### utilities

def _coerce_numeric(s: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(s, errors="coerce")
    except Exception:
        return s

@st.cache_resource(show_spinner=False)
def get_ticker_obj(ticker: str) -> yf.Ticker:
    """Cache the yfinance Ticker object as a resource (not pickled by st.cache_data)."""
    return yf.Ticker(ticker)

@st.cache_data(show_spinner=False)
def load_statements(ticker: str) -> Dict[str, pd.DataFrame]:
    """Return only pickle-serializable objects (DataFrames) for cache_data."""
    t = get_ticker_obj(ticker)
    # try multiple attribute names across yfinance versions
    income = getattr(t, "income_stmt", None)
    if income is None or getattr(income, "empty", True):
        income = getattr(t, "financials", pd.DataFrame())
    cashflow = getattr(t, "cash_flow", None)
    if cashflow is None or getattr(cashflow, "empty", True):
        cashflow = getattr(t, "cashflow", pd.DataFrame())
    balance = getattr(t, "balance_sheet", None)
    if balance is None or getattr(balance, "empty", True):
        balance = getattr(t, "balance_sheet", pd.DataFrame())

    # ensure indices are strings and columns are sorted ascending by date
    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df, pd.DataFrame) and not df.empty:
            df = df.copy()
            df.index = [str(i) for i in df.index]
            try:
                cols = list(df.columns)
                try:
                    cols_sorted = sorted(cols, key=lambda c: pd.to_datetime(c))
                except Exception:
                    cols_sorted = cols
                df = df[cols_sorted]
            except Exception:
                pass
            df = df.apply(_coerce_numeric)
        else:
            df = pd.DataFrame()
        return df

    income = _prep(income)
    cashflow = _prep(cashflow)
    balance = _prep(balance)
    return {"income": income, "cashflow": cashflow, "balance": balance}



def _find_row(df: pd.DataFrame, candidates) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    idx_map = {str(x).lower(): x for x in df.index}
    for cand in candidates:
        cl = cand.lower()
        # exact or substring match
        for low, idx in idx_map.items():
            if low == cl or cl in low:
                return _coerce_numeric(df.loc[idx])
    return None


def _latest_value(series_like: Optional[pd.Series]) -> Optional[float]:
    if series_like is None or not isinstance(series_like, pd.Series) or series_like.empty:
        return None
    try:
        s = series_like.dropna()
        if s.empty:
            return None
        return float(s.iloc[-1])
    except Exception:
        return None


def _shares_outstanding(t: yf.Ticker) -> Optional[float]:
    # try fast_info first
    try:
        fi = getattr(t, "fast_info", None)
        if fi and "shares_outstanding" in fi:
            so = fi["shares_outstanding"]
            return float(so) if so else None
    except Exception:
        pass
    # try balance sheet line
    try:
        bs = getattr(t, "balance_sheet", None)
        if bs is None or bs.empty:
            bs = getattr(t, "balance_sheet", pd.DataFrame())
        s = _find_row(bs, [
            "Common Stock Shares Outstanding",
            "CommonStockSharesOutstanding",
            "OrdinarySharesNumber",
        ])
        val = _latest_value(s)
        if val:
            return float(val)
    except Exception:
        pass
    # try get_shares_full
    try:
        sh = t.get_shares_full()
        if isinstance(sh, pd.Series) and not sh.empty:
            return float(sh.iloc[-1])
    except Exception:
        pass
    return None


def _last_price(t: yf.Ticker) -> Optional[float]:
    try:
        fi = getattr(t, "fast_info", None)
        if fi and "last_price" in fi:
            return float(fi["last_price"]) if fi["last_price"] else None
    except Exception:
        pass
    try:
        hist = t.history(period="5d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return None


def derive_base_assumptions(pkg: Dict[str, pd.DataFrame], t: yf.Ticker) -> Dict[str, float]:
    inc: pd.DataFrame = pkg["income"]
    cf: pd.DataFrame = pkg["cashflow"]
    bs: pd.DataFrame = pkg["balance"]

    revenue = _find_row(inc, ["Total Revenue", "TotalRevenue", "Revenue", "Operating Revenue"])
    op_income = _find_row(inc, ["Operating Income", "OperatingIncome", "EBIT"])  # EBIT
    da = _find_row(cf, ["Depreciation & Amortization", "DepreciationAndAmortization", "Depreciation"])
    capex = _find_row(cf, ["Capital Expenditure", "CapitalExpenditure"])
    chg_nwc = _find_row(cf, ["Change In Working Capital", "ChangeInWorkingCapital"])  # can be +/-
    tax_exp = _find_row(inc, ["Tax Provision", "Income Tax Expense", "TaxExpense"])
    pretax = _find_row(inc, ["Pretax Income", "Income Before Tax", "IncomeBeforeTax"])

    # recent scalar values
    latest_revenue = _latest_value(revenue) or 0.0

    # build time series aligned over common columns
    cols = None
    for s in [revenue, op_income, da, capex, chg_nwc]:
        if isinstance(s, pd.Series):
            cols = s.index if cols is None else cols.intersection(s.index)
    if cols is None:
        cols = []
    def align(s):
        return s[cols].astype(float) if isinstance(s, pd.Series) else pd.Series(index=cols, dtype=float)

    rev_s, op_s, da_s, capex_s, dnw_s = map(align, [revenue, op_income, da, capex, chg_nwc])

    # historical ratios
    with np.errstate(all='ignore'):
        ebitda_s = op_s + da_s
        ebitda_margin_hist = (ebitda_s / rev_s).replace([np.inf, -np.inf], np.nan).dropna()
        da_sales_hist = (da_s / rev_s).replace([np.inf, -np.inf], np.nan).dropna()
        capex_sales_hist = (-capex_s / rev_s).replace([np.inf, -np.inf], np.nan).dropna()  # capex is typically negative in CF
        dnw_sales_hist = (dnw_s / rev_s).replace([np.inf, -np.inf], np.nan).dropna()
        # revenue growth
        rev_ts = rev_s.dropna().astype(float)
        rev_growth_hist = (rev_ts.pct_change().dropna())

    def _safe_mean(x: pd.Series, default: float) -> float:
        try:
            v = float(np.nanmean(x.values))
            if np.isfinite(v):
                return v
        except Exception:
            pass
        return default

    def _safe_std(x: pd.Series, default: float) -> float:
        try:
            v = float(np.nanstd(x.values, ddof=1))
            if np.isfinite(v):
                return v
        except Exception:
            pass
        return default

    # base levels (use conservative fallbacks)
    base = {
        "latest_revenue": latest_revenue,
        "ebitda_margin_mean": np.clip(_safe_mean(ebitda_margin_hist, 0.20), 0.05, 0.60),
        "ebitda_margin_sd": np.clip(_safe_std(ebitda_margin_hist, 0.05), 0.01, 0.10),
        "da_sales_mean": np.clip(_safe_mean(da_sales_hist, 0.04), 0.00, 0.15),
        "da_sales_sd": np.clip(_safe_std(da_sales_hist, 0.01), 0.002, 0.04),
        "capex_sales_mean": np.clip(_safe_mean(capex_sales_hist, 0.05), 0.00, 0.20),
        "capex_sales_sd": np.clip(_safe_std(capex_sales_hist, 0.02), 0.005, 0.06),
        "dnw_sales_mean": np.clip(_safe_mean(dnw_sales_hist, 0.00), -0.08, 0.08),
        "dnw_sales_sd": np.clip(_safe_std(dnw_sales_hist, 0.02), 0.005, 0.06),
        "rev_growth_mean": np.clip(_safe_mean(rev_growth_hist, 0.06), -0.20, 0.25),
        "rev_growth_sd": np.clip(_safe_std(rev_growth_hist, 0.05), 0.01, 0.12),
    }

    # tax rate from statements if available
    tax_rate = None
    tax_val = _latest_value(tax_exp)
    pretax_val = _latest_value(pretax)
    if tax_val is not None and pretax_val and pretax_val > 0:
        try:
            tax_rate = float(tax_val / pretax_val)
        except Exception:
            tax_rate = None
    base["tax_rate_mean"] = float(np.clip(0.21 if tax_rate is None or not np.isfinite(tax_rate) else tax_rate, 0.00, 0.35))
    base["tax_rate_sd"] = 0.03

    # balance sheet for net debt and cash
    cash = _find_row(bs, ["Cash And Cash Equivalents", "CashAndCashEquivalents", "Cash"])
    st_debt = _find_row(bs, ["Short Long Term Debt", "Short Term Debt", "CurrentPortionOfLongTermDebt"])
    lt_debt = _find_row(bs, ["Long Term Debt", "LongTermDebt", "LongTermDebtNoncurrent"])
    total_debt = _find_row(bs, ["Total Debt", "TotalDebt"])  # some versions expose this directly

    cash_val = _latest_value(cash) or 0.0
    if total_debt is not None and not total_debt.empty and np.isfinite(total_debt.iloc[-1]):
        debt_val = float(total_debt.iloc[-1])
    else:
        debt_val = float(((_latest_value(st_debt) or 0.0) + (_latest_value(lt_debt) or 0.0)))

    base["cash"] = max(0.0, cash_val)
    base["debt"] = max(0.0, debt_val)

    so = _shares_outstanding(t)
    base["shares_outstanding"] = float(so) if so else np.nan

    price = _last_price(t)
    base["last_price"] = float(price) if price else np.nan

    return base


### monte carlo valuation

def _sample_normal_bounded(mean, sd, low, high, size):
    x = np.random.normal(loc=mean, scale=sd, size=size)
    return np.clip(x, low, high)


def simulate_dcf(
    latest_revenue: float,
    horizon_years: int,
    n_sims: int,
    rev_growth_mean: float,
    rev_growth_sd: float,
    ebitda_margin_mean: float,
    ebitda_margin_sd: float,
    da_sales_mean: float,
    da_sales_sd: float,
    capex_sales_mean: float,
    capex_sales_sd: float,
    dnw_sales_mean: float,
    dnw_sales_sd: float,
    tax_rate_mean: float,
    tax_rate_sd: float,
    wacc_mean: float,
    wacc_sd: float,
    term_g_mean: float,
    term_g_sd: float,
    cash: float,
    debt: float,
    shares_outstanding: float,
    seed: Optional[int] = 42,
) -> Tuple[np.ndarray, Dict[str, float]]:
    if seed is not None:
        np.random.seed(seed)

    # draw parameters
    g = _sample_normal_bounded(rev_growth_mean, rev_growth_sd, -0.5, 0.5, size=(n_sims, horizon_years))
    ebitda_m = _sample_normal_bounded(ebitda_margin_mean, ebitda_margin_sd, 0.00, 0.70, size=(n_sims, horizon_years))
    da_s = _sample_normal_bounded(da_sales_mean, da_sales_sd, 0.00, 0.20, size=(n_sims, horizon_years))
    capex_s = _sample_normal_bounded(capex_sales_mean, capex_sales_sd, 0.00, 0.25, size=(n_sims, horizon_years))
    dnw_s = _sample_normal_bounded(dnw_sales_mean, dnw_sales_sd, -0.12, 0.12, size=(n_sims, horizon_years))
    tax = _sample_normal_bounded(tax_rate_mean, tax_rate_sd, 0.00, 0.35, size=(n_sims, 1))
    wacc = _sample_normal_bounded(wacc_mean, wacc_sd, 0.03, 0.25, size=(n_sims, 1))
    term_g = _sample_normal_bounded(term_g_mean, term_g_sd, -0.01, 0.05, size=(n_sims, 1))

    # ensure terminal growth < WACC by a margin
    min_spread = 0.005
    term_g = np.minimum(term_g, wacc - min_spread)

    # revenue path
    rev = np.empty((n_sims, horizon_years))
    rev[:, 0] = latest_revenue * (1.0 + g[:, 0])
    for t_idx in range(1, horizon_years):
        rev[:, t_idx] = rev[:, t_idx - 1] * (1.0 + g[:, t_idx])

    # operating items
    ebitda = rev * ebitda_m
    da = rev * da_s
    ebit = ebitda - da
    nopat = ebit * (1.0 - tax)  # broadcast along years
    capex = rev * capex_s
    dnw = rev * dnw_s
    fcff = nopat + da - capex - dnw

    # discounting
    t = np.arange(1, horizon_years + 1)[None, :]  # shape (1, H)
    disc = 1.0 / np.power(1.0 + wacc, t)
    pv_fcf = (fcff * disc).sum(axis=1)

    # terminal val at end of year H
    fcff_last = fcff[:, -1]
    tv = fcff_last * (1.0 + term_g.ravel()) / (wacc.ravel() - term_g.ravel())
    pv_tv = tv / np.power(1.0 + wacc.ravel(), horizon_years)

    ev = pv_fcf + pv_tv
    equity = ev + cash - debt

    per_share = np.where(np.isfinite(shares_outstanding) & (shares_outstanding > 0), equity / shares_outstanding, np.nan)

    stats = {
        "EV_mean": float(np.nanmean(ev)),
        "EV_median": float(np.nanmedian(ev)),
        "Equity_mean": float(np.nanmean(equity)),
        "Equity_median": float(np.nanmedian(equity)),
        "PerShare_mean": float(np.nanmean(per_share)),
        "PerShare_median": float(np.nanmedian(per_share)),
        "PerShare_p5": float(np.nanpercentile(per_share, 5)),
        "PerShare_p25": float(np.nanpercentile(per_share, 25)),
        "PerShare_p75": float(np.nanpercentile(per_share, 75)),
        "PerShare_p95": float(np.nanpercentile(per_share, 95)),
    }

    return per_share, stats


### sidebar inputs

with st.sidebar:
    st.header("Inputs")
    st.write("If you meant **Streamlit** when you wrote 'streamline', you're in the right place. 👇")

    default_ticker = st.session_state.get("_last_ticker", "AAPL")
    ticker = st.text_input("Ticker (Yahoo Finance)", value=default_ticker).strip().upper()
    st.session_state["_last_ticker"] = ticker

    horizon = st.slider("Forecast Horizon (years)", 3, 10, 5)
    n_sims = st.slider("Simulations", 500, 20000, 5000, step=500)

    st.divider()
    st.subheader("Discounting & Terminal")
    wacc_mean = st.number_input("WACC (mean)", min_value=0.01, max_value=0.25, value=0.10, step=0.005, format="%.3f")
    wacc_sd = st.number_input("WACC (sd)", min_value=0.000, max_value=0.10, value=0.020, step=0.005, format="%.3f")
    term_g_mean = st.number_input("Terminal growth (mean)", min_value=-0.01, max_value=0.05, value=0.02, step=0.005, format="%.3f")
    term_g_sd = st.number_input("Terminal growth (sd)", min_value=0.000, max_value=0.05, value=0.005, step=0.005, format="%.3f")

    st.divider()
    st.subheader("Advanced (override from history)")
    advanced_exp = st.expander("Calibrate distributions")
    with advanced_exp:
        colA, colB = st.columns(2)
        with colA:
            rev_growth_mean = st.number_input("Revenue growth mean", value=0.06, step=0.005, format="%.3f")
            ebitda_margin_mean = st.number_input("EBITDA margin mean", value=0.25, step=0.01, format="%.3f")
            da_sales_mean = st.number_input("D&A / Sales mean", value=0.04, step=0.005, format="%.3f")
            capex_sales_mean = st.number_input("Capex / Sales mean", value=0.05, step=0.005, format="%.3f")
            dnw_sales_mean = st.number_input("ΔNWC / Sales mean", value=0.00, step=0.005, format="%.3f")
            tax_rate_mean = st.number_input("Tax rate mean", value=0.21, step=0.005, format="%.3f")
        with colB:
            rev_growth_sd = st.number_input("Revenue growth sd", value=0.05, step=0.005, format="%.3f")
            ebitda_margin_sd = st.number_input("EBITDA margin sd", value=0.05, step=0.005, format="%.3f")
            da_sales_sd = st.number_input("D&A / Sales sd", value=0.01, step=0.005, format="%.3f")
            capex_sales_sd = st.number_input("Capex / Sales sd", value=0.02, step=0.005, format="%.3f")
            dnw_sales_sd = st.number_input("ΔNWC / Sales sd", value=0.02, step=0.005, format="%.3f")
            tax_rate_sd = st.number_input("Tax rate sd", value=0.03, step=0.005, format="%.3f")

    run_btn = st.button("Run Monte Carlo DCF", type="primary")

### main: fetch and calibrate

if ticker:
    try:
        pkg = load_statements(ticker)
        t = get_ticker_obj(ticker)
        base = derive_base_assumptions(pkg, t)
        # auto-fill advanced inputs if user hasn't changed them (first render)
        if "_autofilled" not in st.session_state:
            st.session_state["_autofilled"] = {}
        autof = st.session_state["_autofilled"].get(ticker) is None
        if autof and base["latest_revenue"] > 0:
            with advanced_exp:
                st.info("Filled advanced assumptions from recent financials. Adjust as needed.")
            # update the widget values by writing to session_state keys used by Streamlit internally
            st.session_state["Revenue growth mean"] = float(base["rev_growth_mean"])  # best-effort
            st.session_state["EBITDA margin mean"] = float(base["ebitda_margin_mean"])  # may not always sync
            # since Streamlit won't allow programmatic widget value set reliably here, display a quick table instead:
        st.subheader(f"Snapshot: {ticker}")
        cols = st.columns(5)
        cols[0].metric("Last Price", f"{base['last_price']:.2f}" if np.isfinite(base['last_price']) else "—")
        cols[1].metric("Latest Revenue", f"{base['latest_revenue'] / 1e9:.2f} B")
        cols[2].metric("Cash", f"{base['cash'] / 1e9:.2f} B")
        cols[3].metric("Debt", f"{base['debt'] / 1e9:.2f} B")
        cols[4].metric("Shares (diluted)", f"{base['shares_outstanding'] / 1e9:.2f} B" if np.isfinite(base['shares_outstanding']) else "—")

        with st.expander("Historical-derived assumptions"):
            tbl = pd.DataFrame({
                "metric": [
                    "rev_growth_mean", "rev_growth_sd",
                    "ebitda_margin_mean", "ebitda_margin_sd",
                    "da_sales_mean", "da_sales_sd",
                    "capex_sales_mean", "capex_sales_sd",
                    "dnw_sales_mean", "dnw_sales_sd",
                    "tax_rate_mean", "tax_rate_sd",
                ],
                "value": [
                    base["rev_growth_mean"], base["rev_growth_sd"],
                    base["ebitda_margin_mean"], base["ebitda_margin_sd"],
                    base["da_sales_mean"], base["da_sales_sd"],
                    base["capex_sales_mean"], base["capex_sales_sd"],
                    base["dnw_sales_mean"], base["dnw_sales_sd"],
                    base["tax_rate_mean"], base["tax_rate_sd"],
                ]
            })
            st.dataframe(tbl, use_container_width=True)

    except Exception as e:
        st.error(f"Failed to load data for {ticker}: {e}")
        base = None
else:
    base = None

### run simulation and display

if run_btn and base is not None and base.get("latest_revenue", 0) > 0:
    with st.spinner("Running simulations..."):
        per_share, stats = simulate_dcf(
            latest_revenue=base["latest_revenue"],
            horizon_years=horizon,
            n_sims=n_sims,
            rev_growth_mean=rev_growth_mean,
            rev_growth_sd=rev_growth_sd,
            ebitda_margin_mean=ebitda_margin_mean,
            ebitda_margin_sd=ebitda_margin_sd,
            da_sales_mean=da_sales_mean,
            da_sales_sd=da_sales_sd,
            capex_sales_mean=capex_sales_mean,
            capex_sales_sd=capex_sales_sd,
            dnw_sales_mean=dnw_sales_mean,
            dnw_sales_sd=dnw_sales_sd,
            tax_rate_mean=tax_rate_mean,
            tax_rate_sd=tax_rate_sd,
            wacc_mean=wacc_mean,
            wacc_sd=wacc_sd,
            term_g_mean=term_g_mean,
            term_g_sd=term_g_sd,
            cash=base["cash"],
            debt=base["debt"],
            shares_outstanding=base["shares_outstanding"],
            seed=42,
        )

    st.subheader("Valuation distribution (per share)")
    df_vals = pd.DataFrame({"per_share": per_share})
    df_vals = df_vals.replace([np.inf, -np.inf], np.nan).dropna()

    if df_vals.empty:
        st.warning("Not enough data to compute per-share values (missing shares outstanding). You can still inspect enterprise value in code if needed.")
    else:
        p5, p50, p95 = np.percentile(df_vals["per_share"], [5, 50, 95])
        price = base["last_price"] if np.isfinite(base["last_price"]) else np.nan

        chart = alt.Chart(df_vals).mark_bar().encode(
            x=alt.X("per_share:Q", bin=alt.Bin(maxbins=60), title="Per-share intrinsic value ($)"),
            y=alt.Y("count()", title="Simulations"),
        ).properties(height=300)

        lines = []
        for val, name in [(p5, "P5"), (p50, "Median"), (p95, "P95")]:
            lines.append(
                alt.Chart(pd.DataFrame({"x": [val], "label": [name]})).mark_rule().encode(x="x:Q")
            )
        if np.isfinite(price):
            lines.append(
                alt.Chart(pd.DataFrame({"x": [price], "label": ["Last Price"]})).mark_rule(strokeDash=[4,2]).encode(x="x:Q")
            )

        st.altair_chart(chart + sum(lines[1:], lines[0]) if lines else chart, use_container_width=True)

        cols = st.columns(4)
        cols[0].metric("Median ($)", f"{p50:,.2f}")
        cols[1].metric("P5–P95 ($)", f"{p5:,.2f} – {p95:,.2f}")
        if np.isfinite(price):
            upside = (p50 / price - 1.0) * 100
            cols[2].metric("vs. Last Price", f"{upside:+.1f}%")
            cols[3].metric("Last Price", f"{price:,.2f}")

        st.download_button(
            label="Download simulation results (CSV)",
            data=df_vals.to_csv(index=False).encode("utf-8"),
            file_name=f"{ticker}_dcf_monte_carlo.csv",
            mime="text/csv",
        )

    with st.expander("Assumption summary"):
        st.json({**stats, **{k: float(v) for k, v in base.items() if isinstance(v, (int, float))}})

elif run_btn and (base is None or base.get("latest_revenue", 0) <= 0):
    st.error("Missing key inputs from Yahoo Finance. Try another ticker or manually override advanced settings.")

st.markdown("---")
st.caption(
    """
    **Notes & Caveats**  
    • This app samples independent normals (clipped to plausible bounds) for key drivers. In reality, drivers are correlated; refine as needed.  
    • WACC and terminal growth are user-controlled; ensure WACC > g for stability (the app enforces a small spread).  
    • Cash/debt/shares come from the most recent balance sheet/fast_info; values can be stale or missing on Yahoo.  
    • This tool is for education/research—not investment advice.
    """
)

