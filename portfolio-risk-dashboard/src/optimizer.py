import numpy as np
import pandas as pd
from scipy.optimize import minimize

def portfolio_perf(weights: np.ndarray, exp_returns: np.ndarray, cov: np.ndarray, rf: float = 0.0):
    mu = float(np.dot(weights, exp_returns))
    sigma = float(np.sqrt(np.dot(weights, np.dot(cov, weights))))
    sharpe = (mu - rf) / sigma if sigma > 0 else np.nan
    return mu, sigma, sharpe

def _min_var_objective(weights, cov):
    return float(np.dot(weights, np.dot(cov, weights)))

def _neg_sharpe_objective(weights, exp_returns, cov, rf):
    mu, sigma, _ = portfolio_perf(weights, exp_returns, cov, rf)
    return float(-(mu - rf) / sigma) if sigma > 0 else 1e6

def min_variance(exp_returns: np.ndarray, cov: np.ndarray):
    n = len(exp_returns)
    w0 = np.repeat(1/n, n)
    bounds = [(0.0, 1.0)] * n
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0})
    res = minimize(_min_var_objective, w0, args=(cov,), method="SLSQP", bounds=bounds, constraints=cons)
    return res.x

def max_sharpe(exp_returns: np.ndarray, cov: np.ndarray, rf: float = 0.0):
    n = len(exp_returns)
    w0 = np.repeat(1/n, n)
    bounds = [(0.0, 1.0)] * n
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0})
    res = minimize(_neg_sharpe_objective, w0, args=(exp_returns, cov, rf), method="SLSQP", bounds=bounds, constraints=cons)
    return res.x

def efficient_frontier(exp_returns: np.ndarray, cov: np.ndarray, points: int = 50):
    # Trace the frontier by targeting portfolio volatility (naive grid) or return (here: return grid)
    mu_min, mu_max = float(np.min(exp_returns)), float(np.max(exp_returns))
    targets = np.linspace(mu_min, mu_max, points)
    n = len(exp_returns)
    bounds = [(0.0, 1.0)] * n
    cons_sum = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    mus, sigmas = [], []
    weights_list = []
    for t in targets:
        cons_mu = {"type": "eq", "fun": lambda w, t=t: np.dot(w, exp_returns) - t}
        w0 = np.repeat(1/n, n)
        res = minimize(lambda w: np.dot(w, np.dot(cov, w)), w0, method="SLSQP", bounds=bounds, constraints=(cons_sum, cons_mu))
        w = res.x
        weights_list.append(w)
        mu = float(np.dot(w, exp_returns))
        sigma = float(np.sqrt(np.dot(w, np.dot(cov, w))))
        mus.append(mu); sigmas.append(sigma)
    df = pd.DataFrame({"mu": mus, "sigma": sigmas})
    return df, np.array(weights_list)
