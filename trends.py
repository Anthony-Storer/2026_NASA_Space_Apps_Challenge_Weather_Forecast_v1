"""
Long-term trend statistics for NASA POWER daily data.

Daily values are first summarized into one number per year (or per season):
day-to-day weather is strongly autocorrelated, so testing trends on raw daily
values would give wildly overconfident p-values. On the yearly series:

  * Ordinary least squares (OLS) slope with a confidence interval and a
    t-test. Optionally adjusted for lag-1 autocorrelation of the residuals
    using an effective sample size, n_eff = n (1 - r1) / (1 + r1)
    (Santer et al. 2000, J. Geophys. Res. 105, 7337).
  * Mann-Kendall test: a non-parametric test for a monotonic trend, with the
    standard correction for tied values (Kendall 1975; Gilbert 1987), and
    optionally a variance correction for lag-1 autocorrelation (AR(1) form
    of Hamed & Rao 1998 / Yue & Wang 2004).
  * Sen's slope (Theil-Sen estimator): the median of all pairwise slopes,
    with its confidence interval. Robust to outliers.

Extremes are counted relative to a baseline period (1991-2020 when the data
covers it): days above the baseline's 90th percentile for that calendar month
(below the 10th for "low" days); for precipitation, days above the baseline's
95th percentile of wet-day amounts, and dry days.
"""

from __future__ import annotations

import calendar

import numpy as np
import pandas as pd
from scipy import stats

import weather_analysis as wa

SEASONS: dict[str, tuple[int, ...]] = {
    "Winter (Dec–Feb)": (12, 1, 2),
    "Spring (Mar–May)": (3, 4, 5),
    "Summer (Jun–Aug)": (6, 7, 8),
    "Fall (Sep–Nov)": (9, 10, 11),
}
DEFAULT_BASELINE = (1991, 2020)   # current WMO climate normal period
MIN_YEARS = 10


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
def _aggregate(values: pd.Series, groups: np.ndarray, expected: pd.Series,
               how: str, min_coverage: float) -> pd.Series:
    valid = values.notna()
    g = values[valid].groupby(groups[valid.to_numpy()])
    count = g.count()
    mean = g.mean()
    expected = expected.reindex(count.index)
    agg = mean * expected if how == "sum" else mean   # fill gaps with the period mean for totals
    return agg[count >= min_coverage * expected].astype(float)


def annual_series(daily: pd.Series, how: str = "mean", min_coverage: float = 0.95) -> pd.Series:
    """One value per calendar year (mean, or total for how='sum').

    Years with less than min_coverage of their days present are dropped, so a
    partial first/last year (e.g. the current year) can't bias the trend.
    """
    years = daily.index.year
    yrs = np.unique(years)
    expected = pd.Series([366 if calendar.isleap(y) else 365 for y in yrs], index=yrs)
    out = _aggregate(daily, np.asarray(years), expected, how, min_coverage)
    out.index.name = "year"
    return out


def seasonal_series(daily: pd.Series, months: tuple[int, ...], how: str = "mean",
                    min_coverage: float = 0.9) -> pd.Series:
    """One value per season-year. December counts toward the following winter."""
    def season_year(idx: pd.DatetimeIndex) -> np.ndarray:
        y = np.asarray(idx.year)
        if 12 in months and 1 in months:
            y = y + (np.asarray(idx.month) == 12)
        return y

    sel = daily[daily.index.month.isin(months)]
    full = pd.date_range(f"{daily.index.min().year - 1}-01-01", f"{daily.index.max().year + 1}-12-31")
    full = full[full.month.isin(months)]
    expected = pd.Series(1, index=full).groupby(season_year(full)).sum()
    out = _aggregate(sel, season_year(sel.index), expected, how, min_coverage)
    out.index.name = "year"
    return out


def choose_baseline(years: pd.Index | np.ndarray, preferred=DEFAULT_BASELINE,
                    min_years: int = 20) -> tuple[int, int]:
    """Use 1991-2020 if at least min_years of it are available, else the whole record."""
    years = np.asarray(years)
    inside = years[(years >= preferred[0]) & (years <= preferred[1])]
    if len(inside) >= min_years:
        return int(inside.min()), int(inside.max())
    return int(years.min()), int(years.max())


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
def ols_trend(x, y, conf: float = 0.95, adjust_autocorr: bool = True) -> dict:
    """OLS slope, CI, t-test; optionally with an effective-sample-size correction."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    xm = x.mean()
    sxx = float(np.sum((x - xm) ** 2))
    b = float(np.sum((x - xm) * (y - y.mean())) / sxx)
    a = float(y.mean() - b * xm)
    fitted = a + b * x
    e = y - fitted
    r1 = float(np.sum(e[1:] * e[:-1]) / np.sum(e**2)) if n > 2 else 0.0

    def _stats(n_used: float):
        dof = max(n_used - 2, 1)
        s2 = float(np.sum(e**2) / dof)
        se = np.sqrt(s2 / sxx)
        t = b / se if se > 0 else np.inf
        p = float(2 * stats.t.sf(abs(t), dof))
        tcrit = float(stats.t.ppf(0.5 + conf / 2, dof))
        return s2, se, t, p, tcrit, dof

    _, _, _, p_naive, _, _ = _stats(n)
    n_eff = n * (1 - r1) / (1 + r1) if (adjust_autocorr and r1 > 0) else n
    n_eff = float(min(max(n_eff, 3), n))
    s2, se, t, p, tcrit, dof = _stats(n_eff)
    half = tcrit * np.sqrt(s2 * (1 / n + (x - xm) ** 2 / sxx))
    return {
        "slope": b, "intercept": a, "se": float(se),
        "ci_low": b - tcrit * se, "ci_high": b + tcrit * se,
        "t": float(t), "dof": float(dof), "p": p, "p_naive": p_naive,
        "r1": r1, "n": n, "n_eff": n_eff,
        "fitted": fitted, "band_low": fitted - half, "band_high": fitted + half,
    }


def mann_kendall(y, adjust_autocorr: bool = False) -> dict:
    """Mann-Kendall trend test (two-sided) with tie correction.

    With adjust_autocorr, the variance of S is inflated for positive lag-1
    autocorrelation r1 of the Sen-detrended series, assuming AR(1):
        n / n* = 1 + 2 * sum_{k=1}^{n-1} (1 - k/n) * r1**k
    (variance-correction approach of Hamed & Rao 1998 and Yue & Wang 2004,
    restricted to an AR(1) structure).
    """
    y = np.asarray(y, float)
    n = len(y)
    diff = y[None, :] - y[:, None]               # diff[i, j] = y_j - y_i
    s = float(np.sign(diff[np.triu_indices(n, 1)]).sum())
    _, tie_counts = np.unique(y, return_counts=True)
    ties = tie_counts[tie_counts > 1]
    var_s = (n * (n - 1) * (2 * n + 5) - np.sum(ties * (ties - 1) * (2 * ties + 5))) / 18.0

    def _zp(v):
        if v <= 0 or s == 0:
            return 0.0, 1.0
        z = (s - np.sign(s)) / np.sqrt(v)
        return float(z), float(2 * stats.norm.sf(abs(z)))

    z_naive, p_naive = _zp(var_s)
    factor, r1 = 1.0, 0.0
    if adjust_autocorr and n > 3:
        x = np.arange(n, dtype=float)
        slope = stats.theilslopes(y, x).slope
        e = y - slope * x
        e = e - e.mean()
        r1 = float(np.sum(e[1:] * e[:-1]) / np.sum(e**2)) if np.any(e) else 0.0
        if r1 > 0:
            k = np.arange(1, n)
            factor = float(1 + 2 * np.sum((1 - k / n) * r1**k))
    z, p = _zp(var_s * factor)
    return {"S": s, "var_S": float(var_s * factor), "Z": z, "p": p, "p_naive": p_naive,
            "Z_naive": z_naive, "variance_factor": factor, "r1": r1,
            "tau": s / (n * (n - 1) / 2)}


def sens_slope(x, y, conf: float = 0.95) -> dict:
    """Theil-Sen slope with confidence interval (scipy implementation)."""
    res = stats.theilslopes(y, x, alpha=conf)
    return {"slope": float(res.slope), "intercept": float(res.intercept),
            "ci_low": float(res.low_slope), "ci_high": float(res.high_slope)}


def trend_test(series: pd.Series, conf: float = 0.95, adjust_autocorr: bool = True,
               baseline: tuple[int, int] | None = None) -> dict | None:
    """All trend statistics for a yearly series. Slopes are per decade."""
    s = series.dropna()
    if len(s) < MIN_YEARS:
        return None
    x, y = s.index.to_numpy(float), s.to_numpy(float)
    ols = ols_trend(x, y, conf, adjust_autocorr)
    mk = mann_kendall(y, adjust_autocorr)
    sen = sens_slope(x, y, conf)
    if baseline is None:
        baseline = choose_baseline(s.index)
    base = s.loc[baseline[0]:baseline[1]]
    base_mean = float(base.mean()) if len(base) else float(s.mean())
    return {
        "first_year": int(x.min()), "last_year": int(x.max()), "years": len(s),
        "mean": float(y.mean()), "baseline_mean": base_mean,
        "ols_per_decade": ols["slope"] * 10,
        "ols_ci_low": ols["ci_low"] * 10, "ols_ci_high": ols["ci_high"] * 10,
        "ols_t": ols["t"], "ols_dof": ols["dof"],
        "ols_p": ols["p"], "ols_p_naive": ols["p_naive"],
        "r1": ols["r1"], "n_eff": ols["n_eff"],
        "mk_tau": mk["tau"], "mk_z": mk["Z"], "mk_p": mk["p"], "mk_p_naive": mk["p_naive"],
        "mk_variance_factor": mk["variance_factor"],
        "sen_per_decade": sen["slope"] * 10,
        "sen_ci_low": sen["ci_low"] * 10, "sen_ci_high": sen["ci_high"] * 10,
        "total_change": ols["slope"] * (x.max() - x.min()),
        "pct_per_decade": (ols["slope"] * 10 / base_mean * 100) if base_mean else np.nan,
        "_ols": ols, "_sen": sen, "_x": x,
    }


def verdict(t: dict | None, alpha: float, words: tuple[str, str] = ("increase", "decrease")) -> str:
    """Plain-language result, requiring both tests to agree for 'significant'."""
    if t is None:
        return "Not enough years"
    up = t["ols_per_decade"] > 0
    direction = words[0] if up else words[1]
    both = t["ols_p"] < alpha and t["mk_p"] < alpha
    one = t["ols_p"] < alpha or t["mk_p"] < alpha
    if both:
        return f"Significant {direction}"
    if one:
        return f"Possible {direction} (tests disagree)"
    return "No significant trend"


def format_p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


# --------------------------------------------------------------------------
# Extremes
# --------------------------------------------------------------------------
def extremes_counts(daily: pd.Series, var: wa.Variable, system: str,
                    baseline: tuple[int, int] | None = None,
                    min_coverage: float = 0.95) -> tuple[pd.DataFrame, dict]:
    """Days per year that are extreme relative to a baseline period.

    Returns (DataFrame indexed by year with columns 'high' and 'low', info dict).
    """
    daily = daily.asfreq("D")
    years_all = np.unique(daily.dropna().index.year)
    if baseline is None:
        baseline = choose_baseline(years_all)
    in_base = (daily.index.year >= baseline[0]) & (daily.index.year <= baseline[1])

    if var.kind == "precipitation":
        wet = wa.convert_amount(1.0, var, system)
        base_wet = daily[in_base & (daily >= wet)]
        heavy = float(base_wet.quantile(0.95))
        high = daily > heavy
        low = daily < wet
        info = {
            "high_label": "Heavy-precipitation days",
            "high_def": f"days above {heavy:.2f} {wa.unit(var, system)}: the baseline's "
                        "95th percentile of wet-day amounts",
            "low_label": "Dry days",
            "low_def": f"days with less than {wet:.2f} {wa.unit(var, system)}",
        }
    else:
        seasonal, _ = wa.fit_seasonal(daily, n_harmonics=3, trend=False,
                                      fit_mask=np.asarray(in_base))
        anom = daily - seasonal
        month = daily.index.month
        base_anom = anom[in_base]
        p90 = base_anom.groupby(base_anom.index.month).quantile(0.90)
        p10 = base_anom.groupby(base_anom.index.month).quantile(0.10)
        high = anom > p90.reindex(month).to_numpy()
        low = anom < p10.reindex(month).to_numpy()
        hi_word, lo_word = ("warm", "cold") if var.kind == "temperature" else ("high", "low")
        info = {
            "high_label": f"Unusually {hi_word} days",
            "high_def": "days above the baseline's 90th percentile for that time of year",
            "low_label": f"Unusually {lo_word} days",
            "low_def": "days below the baseline's 10th percentile for that time of year",
        }

    present = daily.notna()
    years = daily.index.year
    count = present.groupby(years).sum()
    expected = pd.Series([366 if calendar.isleap(y) else 365 for y in count.index], index=count.index)
    keep = count >= min_coverage * expected
    out = pd.DataFrame({
        "high": (high & present).groupby(years).sum(),
        "low": (low & present).groupby(years).sum(),
    })[keep].astype(float)
    # Scale slightly incomplete years up to a full year so missing days don't look like a trend.
    out = out.mul((expected / count)[keep], axis=0)
    out.index.name = "year"
    info["baseline"] = baseline
    return out, info
