import numpy as np
import pandas as pd

def annualize_return(daily_ret: pd.Series, periods_per_year: int = 252) -> float:
    ar = (1 + daily_ret).prod() ** (periods_per_year / len(daily_ret)) - 1
    return float(ar)

def annualize_vol(daily_ret: pd.Series, periods_per_year: int = 252) -> float:
    return float(daily_ret.std(ddof=0) * np.sqrt(periods_per_year))

def sharpe_ratio(daily_ret: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    er = annualize_return(daily_ret, periods_per_year) - rf
    vol = annualize_vol(daily_ret, periods_per_year)
    return float(er / vol) if vol > 0 else np.nan

def max_drawdown(cum: pd.Series) -> float:
    roll_max = cum.cummax()
    dd = cum / roll_max - 1.0
    return float(dd.min())

def hist_var(returns: pd.Series, alpha: float = 0.05) -> float:
    # historical VaR (left tail, returns negative are losses)
    return float(np.percentile(returns.dropna(), 100 * alpha))

def hist_es(returns: pd.Series, alpha: float = 0.05) -> float:
    cutoff = hist_var(returns, alpha)
    tail = returns[returns <= cutoff]
    return float(tail.mean()) if len(tail) else np.nan

def gaussian_var(mean: float, std: float, alpha: float = 0.05) -> float:
    # parametric (gaussian) VaR using inverse CDF
    from math import sqrt
    from scipy.stats import norm
    return float(mean + std * norm.ppf(alpha))

def gaussian_es(mean: float, std: float, alpha: float = 0.05) -> float:
    from scipy.stats import norm
    z = norm.ppf(alpha)
    return float(mean - std * norm.pdf(z) / alpha)
