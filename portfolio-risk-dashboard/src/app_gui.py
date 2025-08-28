# src/app_gui.py
import os
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_pdf import PdfPages

from dashboard import fetch_prices  # reuse your existing function
from utils import (
    annualize_return, annualize_vol, sharpe_ratio, max_drawdown,
    hist_var, hist_es
)
from optimizer import min_variance, max_sharpe, efficient_frontier

REPORT_DIR = "reports"

def ensure_reports_dir():
    os.makedirs(os.path.join(REPORT_DIR, "plots"), exist_ok=True)

def normalize_weights(df):
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
    df = df.groupby("ticker", as_index=False)["weight"].sum()
    if df["weight"].sum() <= 0:
        raise ValueError("Weights must sum to a positive number.")
    df["weight"] = df["weight"] / df["weight"].sum()
    return df

def compute_core(px, weights_df, rf_annual=0.02):
    # returns & weights alignment
    rets = np.log(px / px.shift(1)).dropna()
    w = weights_df.set_index("ticker")["weight"].reindex(rets.columns).fillna(0.0)
    port = (rets * w.values).sum(axis=1)

    # annualized stats (exp returns & cov for frontier)
    exp_daily = rets.mean()
    exp_annual = exp_daily * 252.0
    cov_daily = rets.cov()
    cov_annual = cov_daily * 252.0

    # asset + portfolio metrics
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

def make_onepage_figure(rets, w, exp_annual, cov_annual, rf_annual):
    """Return a single matplotlib Figure with 4 subplots:
       1) Cumulative returns of the portfolio
       2) Drawdown
       3) Correlation heatmap
       4) Efficient frontier with min-var & tangency
    """
    # portfolio series
    port = (rets * w.values).sum(axis=1)
    cum = (1 + port).cumprod()
    roll_max = cum.cummax()
    dd = cum / roll_max - 1.0

    # correlation
    corr = rets.corr()

    # frontier
    mu = exp_annual.values
    cov = cov_annual.values
    df_frontier, _ = efficient_frontier(mu, cov, points=60)
    w_tan = max_sharpe(mu, cov, rf_annual)
    w_min = min_variance(mu, cov)
    mu_tan = float(np.dot(w_tan, mu)); sig_tan = float(np.sqrt(np.dot(w_tan, cov @ w_tan)))
    mu_min = float(np.dot(w_min, mu)); sig_min = float(np.sqrt(np.dot(w_min, cov @ w_min)))

    # build figure
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.2)

    # 1) Cumulative
    ax1 = fig.add_subplot(gs[0, 0])
    cum.plot(ax=ax1)
    ax1.set_title("Portfolio Cumulative Return")
    ax1.set_ylabel("Growth of $1")

    # 2) Drawdown
    ax2 = fig.add_subplot(gs[0, 1])
    dd.plot(ax=ax2)
    ax2.set_title("Portfolio Drawdown")
    ax2.set_ylabel("Drawdown")

    # 3) Correlation Heatmap
    ax3 = fig.add_subplot(gs[1, 0])
    im = ax3.imshow(corr.values, interpolation="nearest")
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    ax3.set_xticks(range(len(corr.columns)))
    ax3.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax3.set_yticks(range(len(corr.index)))
    ax3.set_yticklabels(corr.index)
    ax3.set_title("Correlation Heatmap")

    # 4) Frontier
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(df_frontier["sigma"], df_frontier["mu"], s=10)
    ax4.scatter([sig_tan], [mu_tan], marker="*", s=150, label="Tangency")
    ax4.scatter([sig_min], [mu_min], marker="o", s=100, label="Min-Var")
    ax4.set_xlabel("Volatility (σ)")
    ax4.set_ylabel("Expected Return (μ)")
    ax4.set_title("Efficient Frontier")
    ax4.legend()

    fig.suptitle("Portfolio Risk Dashboard", y=0.98, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig

class PortfolioApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Portfolio Risk Dashboard")
        self.geometry("1100x800")
        ensure_reports_dir()

        # shared state
        self.rf_annual = tk.DoubleVar(value=0.02)
        self.start_date = tk.StringVar(value="2015-01-01")
        self.end_date = tk.StringVar(value="")
        self.portfolio_df = pd.DataFrame(columns=["ticker", "weight"])
        self.last_results = {}  # rets, w, exp_annual, cov_annual, metrics, fig

        # frames
        self.container = ttk.Frame(self)
        self.container.pack(fill="both", expand=True)

        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        self.minsize(900, 600)
        self.grid_propagate(True)

        self.frames = {}
        for F in (InputFrame, ResultsFrame):
            frame = F(self.container, self)
            self.frames[F.__name__] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show("InputFrame")

    def show(self, name):
        self.frames[name].tkraise()

    def run_diagnostics(self):
        try:
            df = normalize_weights(self.portfolio_df.copy())
            tickers = df["ticker"].tolist()
            px = fetch_prices(tickers, start=self.start_date.get() or "2015-01-01",
                              end=self.end_date.get() or None)
            # Align to available tickers only
            df = df[df["ticker"].isin(px.columns)]

            rets, w, port, exp_annual, cov_annual, metrics = compute_core(px, df, rf_annual=self.rf_annual.get())
            fig = make_onepage_figure(rets, w, exp_annual, cov_annual, self.rf_annual.get())
            self.last_results = dict(rets=rets, w=w, exp_annual=exp_annual,
                                     cov_annual=cov_annual, metrics=metrics, fig=fig, px=px, weights=df)
            self.frames["ResultsFrame"].render(fig, metrics)
            self.show("ResultsFrame")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def save_portfolio_csv(self):
        try:
            df = normalize_weights(self.portfolio_df.copy())
            path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                initialfile="portfolio.csv",
                filetypes=[("CSV", "*.csv")],
                title="Save Portfolio CSV"
            )
            if path:
                df.to_csv(path, index=False)
                messagebox.showinfo("Saved", f"Saved portfolio to {path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def save_results_bundle(self):
        if not self.last_results:
            messagebox.showwarning("Nothing to save", "Run diagnostics first.")
            return
        ts = time.strftime("%Y%m%d-%H%M%S")
        outdir = os.path.join(REPORT_DIR, f"GUI_run_{ts}")
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)

        # Save metrics & allocations & frontier
        metrics = self.last_results["metrics"]
        metrics.to_csv(os.path.join(outdir, "risk_summary.csv"))
        self.last_results["weights"].to_csv(os.path.join(outdir, "allocations.csv"), index=False)

        # Save plots: one-page PNG + PDF
        fig = self.last_results["fig"]
        fig.savefig(os.path.join(outdir, "plots", "dashboard_onepage.png"), bbox_inches="tight")

        with PdfPages(os.path.join(outdir, "dashboard_onepage.pdf")) as pdf:
            pdf.savefig(fig, bbox_inches="tight")

        messagebox.showinfo("Saved", f"Results saved to {outdir}")

class InputFrame(ttk.Frame):
    def __init__(self, parent, app: PortfolioApp):
        super().__init__(parent)
        self.app = app

        # Controls
        top = ttk.Frame(self)
        top.pack(fill="x", padx=12, pady=8)

        ttk.Label(top, text="Start (YYYY-MM-DD):").pack(side="left")
        ttk.Entry(top, textvariable=app.start_date, width=12).pack(side="left", padx=6)

        ttk.Label(top, text="End (optional):").pack(side="left")
        ttk.Entry(top, textvariable=app.end_date, width=12).pack(side="left", padx=6)

        ttk.Label(top, text="Risk-free (annual):").pack(side="left")
        ttk.Entry(top, textvariable=app.rf_annual, width=8).pack(side="left", padx=6)

        # Table for tickers/weights
        mid = ttk.Frame(self)
        mid.pack(fill="both", expand=True, padx=12, pady=8)

        self.tree = ttk.Treeview(mid, columns=("ticker", "weight"), show="headings", height=18)
        self.tree.heading("ticker", text="Ticker")
        self.tree.heading("weight", text="Weight")
        self.tree.pack(side="left", fill="both", expand=True)

        sb = ttk.Scrollbar(mid, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=sb.set)
        sb.pack(side="left", fill="y")

        # Buttons for editing rows
        right = ttk.Frame(self)
        right.pack(fill="x", padx=12, pady=8)

        self.ticker_var = tk.StringVar()
        self.weight_var = tk.StringVar()

        ttk.Entry(right, textvariable=self.ticker_var, width=12).pack(side="left", padx=6)
        ttk.Entry(right, textvariable=self.weight_var, width=12).pack(side="left", padx=6)

        ttk.Button(right, text="Add / Update Row", command=self.add_update_row).pack(side="left", padx=6)
        ttk.Button(right, text="Delete Row", command=self.delete_row).pack(side="left", padx=6)
        ttk.Button(right, text="Save Portfolio CSV", command=app.save_portfolio_csv).pack(side="left", padx=6)

        bottom = ttk.Frame(self)
        bottom.pack(fill="x", padx=12, pady=12)
        ttk.Button(bottom, text="Run Diagnostics", command=app.run_diagnostics).pack(side="left")
        ttk.Button(bottom, text="Exit", command=self.quit_program).pack(side="right")

        self.refresh_table()

    def refresh_table(self):
        # reload tree from app.portfolio_df
        for r in self.tree.get_children():
            self.tree.delete(r)
        for _, row in self.app.portfolio_df.iterrows():
            self.tree.insert("", "end", values=(row["ticker"], row["weight"]))

    def add_update_row(self):
        t = self.ticker_var.get().strip().upper()
        try:
            w = float(self.weight_var.get())
        except:
            messagebox.showerror("Invalid weight", "Weight must be a number.")
            return
        if not t:
            messagebox.showerror("Missing ticker", "Enter a ticker.")
            return

        df = self.app.portfolio_df.copy()
        if (df["ticker"] == t).any():
            df.loc[df["ticker"] == t, "weight"] = w
        else:
            df = pd.concat([df, pd.DataFrame([{"ticker": t, "weight": w}])], ignore_index=True)
        self.app.portfolio_df = df
        self.refresh_table()
        self.ticker_var.set(""); self.weight_var.set("")

    def delete_row(self):
        sel = self.tree.selection()
        if not sel:
            return
        vals = self.tree.item(sel[0], "values")
        t = vals[0]
        df = self.app.portfolio_df.copy()
        df = df[df["ticker"] != t]
        self.app.portfolio_df = df
        self.refresh_table()

    def quit_program(self):
        self.winfo_toplevel().destroy()

class ResultsFrame(ttk.Frame):
    def __init__(self, parent, app: PortfolioApp):
        super().__init__(parent)
        self.app = app

        # Toolbar
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=12, pady=8)
        ttk.Button(bar, text="Back / Edit", command=lambda: app.show("InputFrame")).pack(side="left")
        ttk.Button(bar, text="Re-run", command=app.run_diagnostics).pack(side="left", padx=6)
        ttk.Button(bar, text="Save Results", command=app.save_results_bundle).pack(side="left", padx=6)
        ttk.Button(bar, text="Exit", command=self.quit_program).pack(side="right")

        # Split pane: canvas (left) and metrics (right)
        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)

        self.canvas_frame = ttk.Frame(body)
        self.canvas_frame.pack(side="left", fill="both", expand=True)

        self.table_frame = ttk.Frame(body, width=350)
        self.table_frame.pack(side="left", fill="y")

        self.tree = ttk.Treeview(self.table_frame, columns=("ann_return","ann_vol","sharpe","mdd","VaR5","ES5"),
                                 show="headings", height=25)
        for c, lbl in [
            ("ann_return","AnnRet"),
            ("ann_vol","AnnVol"),
            ("sharpe","Sharpe"),
            ("mdd","MaxDD"),
            ("VaR5","VaR 5%"),
            ("ES5","ES 5%"),
        ]:
            self.tree.heading(c, text=lbl)
        self.tree.pack(fill="both", expand=True, padx=8, pady=8)

        self._tk_figure = None
        self._canvas = None

    def render(self, fig, metrics: pd.DataFrame):
        # Draw figure
        if self._canvas:
            self._canvas.get_tk_widget().destroy()
        self._tk_figure = fig
        self._canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)

        # Fill metrics
        for r in self.tree.get_children():
            self.tree.delete(r)
        m = metrics.copy()
        m = m.rename(columns={
            "hist_VaR_5%": "VaR5",
            "hist_ES_5%": "ES5"
        })
        # format nicely
        fmt = lambda x: f"{x:.4f}" if isinstance(x, (float, int)) else str(x)
        for idx, row in m.iterrows():
            self.tree.insert("", "end", values=(
                fmt(row["ann_return"]),
                fmt(row["ann_vol"]),
                fmt(row["sharpe"]),
                fmt(row["mdd"]),
                fmt(row["VaR5"]),
                fmt(row["ES5"]),
            ), tags=(idx,))
        # Make PORTFOLIO row bold-ish
        self.tree.tag_configure("PORTFOLIO", background="#f0f0f0")

    def quit_program(self):
        self.winfo_toplevel().destroy()

def main():
    app = PortfolioApp()
    # Preload with a small example
    app.portfolio_df = pd.DataFrame([
        {"ticker": "SPY", "weight": 0.4},
        {"ticker": "EFA", "weight": 0.2},
        {"ticker": "IEMG", "weight": 0.15},
        {"ticker": "AGG", "weight": 0.2},
        {"ticker": "GLD", "weight": 0.05},
    ])
    app.frames["InputFrame"].refresh_table()
    app.mainloop()

if __name__ == "__main__":
    main()
