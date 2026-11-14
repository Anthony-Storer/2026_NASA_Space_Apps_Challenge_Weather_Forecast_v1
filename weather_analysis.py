"""
Separate daily temperatures into a predictable part and an unpredictable part.

Data source: NASA POWER daily point API (no API key needed).
    https://power.larc.nasa.gov/docs/services/api/temporal/daily/

Model, applied separately to each city:

    temperature = seasonal cycle + long-term trend      (calendar-predictable)
                + persistence                            (predictable from recent days)
                + residual                               (the unpredictable portion)

  1. Seasonal + trend: least-squares fit of a linear trend plus annual
     harmonics (sin/cos of 1, 2, 3 cycles per year).
  2. Persistence: an autoregressive model AR(p) on the de-seasonalized
     anomaly, with p chosen by BIC. It says how much of today's departure
     from normal could be guessed from the last few days' departures.
  3. Residual: whatever is left. This is the day-to-day "surprise" that
     neither the calendar nor the recent past can explain.

Out-of-sample testing: pass train_end to decompose() and every coefficient is
estimated from days up to that date only. Residuals after it are honest
forecast errors, not in-sample fit. forecast_horizon() runs the model forward
1..N days from each test-period origin to show how fast predictability fades.

"Unpredictable" here means unpredictable from the temperature record itself.
A real weather forecast also uses pressure, wind and satellite data, so it
explains part of this residual.

Run it from the command line for a text summary:
    python weather_analysis.py
    python weather_analysis.py --start 1991-01-01 --test-from 2016-01-01 --units F --csv out.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
from scipy import stats

POWER_DAILY_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
POWER_FILL_VALUE = -999.0

CITIES: dict[str, tuple[float, float]] = {
    # name: (latitude, longitude)
    "San Francisco": (37.7749, -122.4194),
    "New York": (40.7128, -74.0060),
    "Dallas": (32.7767, -96.7970),
}

DAYS_PER_YEAR = 365.2425


# --------------------------------------------------------------------------
# Variables
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Variable:
    """One NASA POWER daily parameter and how to present it.

    Values arrive in metric units (community=RE). US value = metric * us_scale + us_offset.
    """
    code: str
    label: str              # long name for headings
    short: str              # short name for sentences
    group: str
    kind: str               # temperature | humidity | precipitation | wind | solar | pressure
    metric_unit: str
    us_unit: str
    us_scale: float = 1.0
    us_offset: float = 0.0
    annual: str = "mean"    # how a year is summarized: "mean" or "sum"
    continuous: bool = True # False -> skip the day-to-day predictability model
    pct_change: bool = False  # also report trends as % of the baseline mean
    note: str = ""


VARIABLES: dict[str, Variable] = {v.code: v for v in [
    Variable("T2M", "Temperature, daily mean", "temperature", "Temperature",
             "temperature", "°C", "°F", 9 / 5, 32),
    Variable("T2M_MAX", "Temperature, daily high", "daily high temperature", "Temperature",
             "temperature", "°C", "°F", 9 / 5, 32),
    Variable("T2M_MIN", "Temperature, daily low", "daily low temperature", "Temperature",
             "temperature", "°C", "°F", 9 / 5, 32),
    Variable("T2MDEW", "Dew point", "dew point", "Moisture",
             "temperature", "°C", "°F", 9 / 5, 32,
             note="Dew point measures how much moisture is in the air; higher feels muggier."),
    Variable("RH2M", "Relative humidity", "relative humidity", "Moisture",
             "humidity", "%", "%"),
    Variable("PRECTOTCORR", "Precipitation", "precipitation", "Moisture",
             "precipitation", "mm", "in", 1 / 25.4, annual="sum", continuous=False,
             pct_change=True,
             note="Bias-corrected daily totals; years are summarized as annual totals."),
    Variable("WS10M", "Wind speed (10 m)", "wind speed", "Wind & pressure",
             "wind", "m/s", "mph", 2.236936, pct_change=True),
    Variable("PS", "Surface air pressure", "surface pressure", "Wind & pressure",
             "pressure", "kPa", "inHg", 0.2953006),
    Variable("ALLSKY_SFC_SW_DWN", "Solar radiation (all-sky)", "solar radiation",
             "Sunlight", "solar", "kWh/m²/day", "kWh/m²/day", pct_change=True,
             note="Sunlight reaching the ground, from satellite data. NASA POWER's solar record starts in 1984."),
]}


def unit(var: Variable, system: str) -> str:
    return var.us_unit if system == "US" else var.metric_unit


def annual_unit(var: Variable, system: str) -> str:
    """Unit of a yearly summary: totals for precipitation, otherwise the daily unit."""
    u = unit(var, system)
    return f"{u}/yr" if var.annual == "sum" else u


def convert(series: pd.Series, var: Variable, system: str) -> pd.Series:
    """Metric -> requested unit system (no-op for metric)."""
    if system != "US":
        return series
    return series * var.us_scale + var.us_offset


def convert_amount(x: float, var: Variable, system: str) -> float:
    """Convert a metric amount or difference (no offset), e.g. a 1 mm wet-day threshold."""
    return x * var.us_scale if system == "US" else x


def trend_words(var: Variable) -> tuple[str, str]:
    """Nouns for an upward / downward long-term trend."""
    if var.kind == "temperature":
        return "warming", "cooling"
    if var.kind == "precipitation":
        return "wetting", "drying"
    return "increase", "decrease"


def change_words(var: Variable) -> tuple[str, str]:
    """Words for above / below expectations."""
    if var.kind == "temperature":
        return "warmer", "colder"
    if var.kind == "precipitation":
        return "wetter", "drier"
    return "higher", "lower"


# --------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------
def parse_power_json(payload: dict, parameter: str = "T2M") -> pd.Series:
    """Turn a NASA POWER daily JSON response into a daily pd.Series (metric units).

    The POWER response stores values under
    payload["properties"]["parameter"][parameter] as {"YYYYMMDD": value}.
    Missing days come back as -999 and are converted to NaN.
    """
    try:
        values = payload["properties"]["parameter"][parameter]
    except (KeyError, TypeError) as exc:
        messages = payload.get("messages") or payload.get("errors") or payload
        raise ValueError(f"Unexpected NASA POWER response: {messages}") from exc

    s = pd.Series(values, dtype="float64")
    s.index = pd.to_datetime(s.index, format="%Y%m%d")
    s = s.sort_index()
    s[s <= POWER_FILL_VALUE + 1e-9] = np.nan
    s.name = parameter
    # NASA POWER lags real time by a few days, and some records start late;
    # drop missing days at both ends.
    first_valid, last_valid = s.first_valid_index(), s.last_valid_index()
    if last_valid is None:
        raise ValueError(f"NASA POWER returned no valid values for {parameter}.")
    return s.loc[first_valid:last_valid]


def _power_request(lat, lon, start, end, parameters: list[str], timeout: float) -> dict:
    fmt = lambda d: pd.Timestamp(d).strftime("%Y%m%d")  # noqa: E731
    params = {
        "parameters": ",".join(parameters),
        "community": "RE",
        "latitude": lat,
        "longitude": lon,
        "start": fmt(start),
        "end": fmt(end),
        "format": "JSON",
        "time-standard": "LST",
    }
    resp = requests.get(POWER_DAILY_URL, params=params, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(
            f"NASA POWER request failed ({resp.status_code}): {resp.text[:500]}"
        )
    return resp.json()


def fetch_daily(
    lat: float,
    lon: float,
    start: date | str,
    end: date | str,
    parameters: list[str] | None = None,
    timeout: float = 240,
) -> pd.DataFrame:
    """Download several daily NASA POWER parameters for one point (metric units).

    Tries one combined request first. If that fails, falls back to one request
    per parameter and keeps whatever succeeds; failures are listed in
    df.attrs["errors"].
    """
    parameters = list(parameters or VARIABLES)
    cols, errors = {}, {}
    try:
        payload = _power_request(lat, lon, start, end, parameters, timeout)
        for p in parameters:
            try:
                cols[p] = parse_power_json(payload, p)
            except ValueError as exc:
                errors[p] = str(exc)
    except Exception as exc:  # noqa: BLE001 - fall back to single-parameter requests
        errors = {}
        first_error = exc
        for p in parameters:
            try:
                cols[p] = parse_power_json(_power_request(lat, lon, start, end, [p], timeout), p)
            except Exception as exc_p:  # noqa: BLE001
                errors[p] = str(exc_p)
        if not cols:
            raise RuntimeError(f"NASA POWER download failed: {first_error}") from first_error
    if not cols:
        raise RuntimeError(f"NASA POWER returned no usable parameters: {errors}")
    df = pd.DataFrame(cols).asfreq("D")
    df.attrs["errors"] = errors
    return df


def fetch_daily_temps(
    lat: float,
    lon: float,
    start: date | str,
    end: date | str,
    parameter: str = "T2M",
    timeout: float = 120,
) -> pd.Series:
    """Download one daily parameter (metric units) for one point from NASA POWER."""
    return parse_power_json(_power_request(lat, lon, start, end, [parameter], timeout), parameter)


# --------------------------------------------------------------------------
# Decomposition
# --------------------------------------------------------------------------
def _seasonal_design(index: pd.DatetimeIndex, n_harmonics: int, trend: bool):
    t_years = (index - index[0]).days.to_numpy() / DAYS_PER_YEAR
    cols = [np.ones_like(t_years)]
    names = ["intercept"]
    if trend:
        cols.append(t_years)
        names.append("trend_per_year")
    for k in range(1, n_harmonics + 1):
        cols += [np.sin(2 * np.pi * k * t_years), np.cos(2 * np.pi * k * t_years)]
        names += [f"sin{k}", f"cos{k}"]
    return np.column_stack(cols), names


def fit_seasonal(
    series: pd.Series,
    n_harmonics: int = 3,
    trend: bool = True,
    fit_mask: np.ndarray | None = None,
):
    """Least-squares seasonal cycle (+ optional linear trend).

    Coefficients are estimated only on days where fit_mask is True (default:
    all days), then the curve is evaluated over the whole index, so with a
    training mask the test days get a genuine out-of-sample "normal".
    Returns (fitted values over the whole index, coefficient dict).
    """
    X, names = _seasonal_design(series.index, n_harmonics, trend)
    ok = series.notna().to_numpy(copy=True)
    if fit_mask is not None:
        ok &= np.asarray(fit_mask, dtype=bool)
    beta, *_ = np.linalg.lstsq(X[ok], series.to_numpy()[ok], rcond=None)
    fitted = pd.Series(X @ beta, index=series.index, name="seasonal")
    return fitted, dict(zip(names, beta))


def _lag_matrix(x: pd.Series, p: int) -> pd.DataFrame:
    return pd.concat({f"lag{k}": x.shift(k) for k in range(1, p + 1)}, axis=1)


def fit_ar(anomaly: pd.Series, max_lag: int = 7, fit_mask: np.ndarray | None = None) -> dict:
    """Fit AR(p) to the anomaly by OLS, choosing p in 1..max_lag by BIC.

    Only days where fit_mask is True are used as targets for fitting (default:
    all days). Gaps in the record are handled by dropping any day whose lags
    are missing. Returns order, coefficients, intercept, and the one-day-ahead
    prediction for every day.
    """
    target = anomaly if fit_mask is None else anomaly.where(fit_mask)
    # Choose the order on a common sample so the BIC values are comparable.
    full = pd.concat([target.rename("y"), _lag_matrix(anomaly, max_lag)], axis=1).dropna()
    y_all = full["y"].to_numpy()
    n = len(y_all)
    if n < 50:
        raise ValueError("Not enough data to fit the persistence model.")

    best_p, best_bic = 1, np.inf
    for p in range(1, max_lag + 1):
        X = np.column_stack([np.ones(n), full[[f"lag{k}" for k in range(1, p + 1)]].to_numpy()])
        beta, *_ = np.linalg.lstsq(X, y_all, rcond=None)
        rss = float(np.sum((y_all - X @ beta) ** 2))
        bic = n * np.log(rss / n) + (p + 1) * np.log(n)
        if bic < best_bic:
            best_p, best_bic = p, bic

    # Refit the chosen order on as much data as it allows.
    lags = _lag_matrix(anomaly, best_p)
    data = pd.concat([target.rename("y"), lags], axis=1).dropna()
    X = np.column_stack([np.ones(len(data)), data.drop(columns="y").to_numpy()])
    beta, *_ = np.linalg.lstsq(X, data["y"].to_numpy(), rcond=None)

    pred = pd.Series(np.nan, index=anomaly.index, name="persistence")
    lag_vals = lags.to_numpy()
    usable = ~np.isnan(lag_vals).any(axis=1)
    pred[usable] = beta[0] + lag_vals[usable] @ beta[1:]
    return {"order": best_p, "intercept": beta[0], "coefs": beta[1:], "prediction": pred}


@dataclass
class Decomposition:
    city: str
    frame: pd.DataFrame          # temp, seasonal, anomaly, persistence, predictable, residual, split
    seasonal_coefs: dict
    ar_order: int
    ar_intercept: float
    ar_coefs: np.ndarray
    units: str = "C"
    train_end: pd.Timestamp | None = None
    summaries: dict = field(default_factory=dict)   # "train"/"test", or "all" with no split

    @property
    def has_split(self) -> bool:
        return self.train_end is not None

    @property
    def summary(self) -> dict:
        """Headline summary: the test period when there is one, else all days."""
        return self.summaries.get("test", self.summaries.get("all"))

    def mask(self, subset: str = "all") -> pd.Series:
        """Boolean mask over frame rows for 'train', 'test' or 'all'."""
        if subset == "all" or not self.has_split:
            return pd.Series(True, index=self.frame.index)
        return self.frame["split"] == subset


def decompose(
    series: pd.Series,
    city: str = "",
    units: str = "C",
    n_harmonics: int = 3,
    trend: bool = True,
    max_lag: int = 7,
    train_end: date | str | None = None,
) -> Decomposition:
    """Split a daily temperature series into predictable and unpredictable parts.

    With train_end set, every model coefficient (seasonal cycle, trend, AR order
    and AR coefficients) is estimated from days up to and including train_end
    only. Days after it are the test period: their residuals are genuine
    out-of-sample forecast errors. Without train_end, everything is in-sample.
    """
    series = series.asfreq("D")  # make gaps explicit so lags line up with real days
    if train_end is not None:
        train_end = pd.Timestamp(train_end)
        fit_mask = np.asarray(series.index <= train_end)
        n_test = int(series[~fit_mask].notna().sum())
        n_train = int(series[fit_mask].notna().sum())
        if n_train < 3 * 365 or n_test < 365:
            raise ValueError("Need at least 3 years of training data and 1 year of test data.")
    else:
        fit_mask = None

    seasonal, s_coefs = fit_seasonal(series, n_harmonics, trend, fit_mask)
    anomaly = (series - seasonal).rename("anomaly")
    ar = fit_ar(anomaly, max_lag, fit_mask)

    df = pd.DataFrame(
        {
            "temp": series,
            "seasonal": seasonal,
            "anomaly": anomaly,
            "persistence": ar["prediction"],
        }
    )
    df["predictable"] = df["seasonal"] + df["persistence"]
    df["residual"] = df["temp"] - df["predictable"]
    if fit_mask is not None:
        df["split"] = np.where(fit_mask, "train", "test")
    df = df.dropna(subset=["temp"])

    d = Decomposition(city, df, s_coefs, ar["order"], ar["intercept"], ar["coefs"], units, train_end)
    subsets = ["train", "test"] if d.has_split else ["all"]
    d.summaries = {s: summarize(d, s) for s in subsets}
    return d


# --------------------------------------------------------------------------
# Statistics on the unpredictable portion
# --------------------------------------------------------------------------
def acf(x: pd.Series, nlags: int = 30) -> np.ndarray:
    """Autocorrelation at lags 0..nlags, respecting gaps in a daily series."""
    x = x.asfreq("D")
    x = x - x.mean()
    var = float(np.nanmean(x.to_numpy() ** 2))
    out = [1.0]
    for k in range(1, nlags + 1):
        prod = (x * x.shift(k)).to_numpy()
        out.append(float(np.nanmean(prod)) / var)
    return np.array(out)


def ljung_box(resid: pd.Series, lags: int = 10, fitted_params: int = 0):
    """Ljung-Box test that the residuals are uncorrelated (white noise)."""
    n = int(resid.notna().sum())
    r = acf(resid, lags)[1:]
    q = n * (n + 2) * np.sum(r**2 / (n - np.arange(1, lags + 1)))
    dof = max(lags - fitted_params, 1)
    return float(q), float(stats.chi2.sf(q, dof))


def summarize(d: Decomposition, subset: str = "all") -> dict:
    """Statistics of the decomposition over one subset of days.

    Errors are measured as root-mean-square around zero (not around their own
    mean), so any systematic bias in the test period counts as error.
    """
    df = d.frame[d.mask(subset)].dropna(subset=["residual"])  # same days for every part
    temp, anom, resid = df["temp"], df["anomaly"], df["residual"]
    var_t = float(temp.var(ddof=0))
    ms_a = float(np.mean(anom**2))
    ms_r = float(np.mean(resid**2))
    rmse = np.sqrt(ms_r)
    # Naive baseline on the same days: "yesterday's anomaly continues".
    a_daily = d.frame["anomaly"].asfreq("D")
    naive_err = (a_daily - a_daily.shift(1)).reindex(df.index).dropna()
    q, p = ljung_box(resid, lags=10, fitted_params=d.ar_order)
    trend = d.seasonal_coefs.get("trend_per_year", np.nan)
    return {
        "city": d.city,
        "subset": subset,
        "start": df.index.min().date(),
        "end": df.index.max().date(),
        "days": len(df),
        "mean_temp": float(temp.mean()),
        "total_sd": float(np.sqrt(var_t)),
        "anomaly_sd": float(np.sqrt(ms_a)),
        "residual_sd": float(rmse),
        "residual_mean": float(resid.mean()),
        "anomaly_mean": float(anom.mean()),
        "residual_mad": float(np.mean(np.abs(resid))),
        "share_seasonal": 1 - ms_a / var_t,
        "share_persistence": (ms_a - ms_r) / var_t,
        "share_unpredictable": ms_r / var_t,
        "anomaly_explained_by_persistence": 1 - ms_r / ms_a,
        "skill_1day": 1 - ms_r / ms_a,
        "naive_skill_1day": 1 - float(np.mean(naive_err**2)) / ms_a,
        "ar_order": d.ar_order,
        "lag1_coef": float(d.ar_coefs[0]),
        "anomaly_lag1_autocorr": float(acf(anom, 1)[1]),
        "residual_skew": float(stats.skew(resid)),
        "residual_excess_kurtosis": float(stats.kurtosis(resid)),
        "pct_beyond_2sd": float(np.mean(np.abs(resid) > 2 * rmse) * 100),
        "pct_beyond_3sd": float(np.mean(np.abs(resid) > 3 * rmse) * 100),
        "trend_per_decade": float(trend * 10),
        "ljung_box_q10": q,
        "ljung_box_p": p,
    }


def forecast_horizon(d: Decomposition, max_lead: int = 14, subset: str = "test") -> pd.DataFrame:
    """How well can temperature be predicted 1..max_lead days ahead?

    From every forecast origin (day t) the fitted AR model is run forward
    recursively, feeding its own forecasts back in, to predict the anomaly on
    days t+1 ... t+max_lead. Only the information available on day t is used.
    Scored against target days in `subset` (default: the test period).

    Compared with two baselines:
      * climatology: forecast the seasonal normal (anomaly = 0)
      * naive persistence: today's anomaly stays the same for every lead

    Returns one row per lead day with RMSEs and skill scores, where
    skill = 1 - MSE(forecast) / MSE(climatology). Skill 1 is perfect, 0 is no
    better than the calendar, and below 0 is worse than the calendar.
    """
    a = d.frame["anomaly"].asfreq("D")
    p, phi, c = d.ar_order, d.ar_coefs, d.ar_intercept
    # State at each origin t: [a_t, a_{t-1}, ..., a_{t-p+1}]
    state = np.column_stack([a.shift(k).to_numpy() for k in range(p)])
    origin_ok = ~np.isnan(state).any(axis=1)
    a_vals = a.to_numpy()
    target_ok = d.mask(subset).reindex(a.index, fill_value=False).to_numpy()

    rows = []
    s = state.copy()
    for h in range(1, max_lead + 1):
        f = c + s @ phi                       # forecast for t+h made on day t
        s = np.column_stack([f, s[:, :-1]]) if p > 1 else f[:, None]
        target = np.full(len(a_vals), np.nan)
        target[:-h] = a_vals[h:]              # actual anomaly on day t+h
        tgt_in = np.zeros(len(a_vals), bool)
        tgt_in[:-h] = target_ok[h:]
        use = origin_ok & tgt_in & ~np.isnan(target)
        err_model = target[use] - f[use]
        err_clim = target[use]
        err_naive = target[use] - a_vals[use]
        mse_m, mse_c, mse_n = (float(np.mean(e**2)) for e in (err_model, err_clim, err_naive))
        rows.append({
            "lead_days": h,
            "rmse_model": np.sqrt(mse_m),
            "rmse_climatology": np.sqrt(mse_c),
            "rmse_naive": np.sqrt(mse_n),
            "skill_model": 1 - mse_m / mse_c,
            "skill_naive": 1 - mse_n / mse_c,
            "forecasts": int(use.sum()),
        })
    return pd.DataFrame(rows).set_index("lead_days")


def predictability_limit(horizon: pd.DataFrame, threshold: float = 0.10) -> int:
    """Last lead day (counting from 1) at which skill is still >= threshold; 0 if none."""
    ok = horizon["skill_model"] >= threshold
    if not ok.iloc[0]:
        return 0
    first_bad = (~ok).idxmax() if (~ok).any() else horizon.index[-1] + 1
    return int(first_bad - 1)


def monthly_residual_sd(d: Decomposition) -> pd.Series:
    r = d.frame["residual"].dropna()
    return r.groupby(r.index.month).std().rename(d.city)


def monthly_residual_skew(d: Decomposition) -> pd.Series:
    r = d.frame["residual"].dropna()
    return r.groupby(r.index.month).apply(lambda x: stats.skew(x)).rename(d.city)


def yearly_residual_sd(d: Decomposition, min_days: int = 300) -> pd.Series:
    r = d.frame["residual"].dropna()
    g = r.groupby(r.index.year)
    sd = g.std()
    return sd[g.size() >= min_days].rename(d.city)


def biggest_surprises(d: Decomposition, n: int = 10) -> pd.DataFrame:
    df = d.frame.dropna(subset=["residual"])
    top = df.loc[df["residual"].abs().nlargest(n).index]
    return pd.DataFrame(
        {
            "date": top.index.date,
            "actual": top["temp"].round(1),
            "predicted": top["predictable"].round(1),
            "surprise": top["residual"].round(1),
            **({"period": top["split"].to_numpy()} if "split" in top else {}),
        }
    )


def to_fahrenheit(series_c: pd.Series) -> pd.Series:
    return series_c * 9 / 5 + 32


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def main() -> None:
    import trends as tr  # local import: trends imports this module

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variable", default="T2M", choices=list(VARIABLES),
                    help="NASA POWER parameter (default T2M, daily mean temperature)")
    ap.add_argument("--start", default="1981-01-01")
    ap.add_argument("--end", default=(date.today() - timedelta(days=1)).isoformat())
    ap.add_argument("--units", default="US", choices=["US", "Metric"])
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--no-autocorr", action="store_true",
                    help="don't correct trend tests for year-to-year autocorrelation")
    ap.add_argument("--test-from", default="2016-01-01",
                    help="first day of the out-of-sample test period (model is fitted before it)")
    ap.add_argument("--max-lead", type=int, default=14, help="longest forecast lead, in days")
    ap.add_argument("--csv", help="write the yearly values for all cities to this CSV")
    args = ap.parse_args()

    var = VARIABLES[args.variable]
    u, au = unit(var, args.units), annual_unit(var, args.units)
    alpha = 1 - args.confidence
    pd.set_option("display.width", 200)
    series = {}
    for city, (lat, lon) in CITIES.items():
        print(f"Fetching {city} ...")
        series[city] = convert(fetch_daily_temps(lat, lon, args.start, args.end, var.code), var, args.units)

    # ---- long-term trends
    annual = {c: tr.annual_series(s, var.annual) for c, s in series.items()}
    years = np.unique(np.concatenate([a.index.to_numpy() for a in annual.values()]))
    baseline = tr.choose_baseline(years)
    rows = {}
    for city, a in annual.items():
        t = tr.trend_test(a, args.confidence, not args.no_autocorr, baseline)
        if t is None:
            continue
        rows[city] = {
            "Years": f"{t['first_year']}-{t['last_year']}",
            f"OLS slope ({au} per decade)": round(t["ols_per_decade"], 3),
            f"{args.confidence:.0%} CI": f"{t['ols_ci_low']:+.3f} to {t['ols_ci_high']:+.3f}",
            "t-test p": tr.format_p(t["ols_p"]),
            "Mann-Kendall tau": round(t["mk_tau"], 3),
            "Mann-Kendall p": tr.format_p(t["mk_p"]),
            f"Sen's slope ({au} per decade)": round(t["sen_per_decade"], 3),
            "Result": tr.verdict(t, alpha, trend_words(var)),
        }
    print(f"\nLONG-TERM TREND: yearly {'total' if var.annual == 'sum' else 'average'} "
          f"{var.short} ({au})\n")
    print(pd.DataFrame(rows).to_string())

    if args.csv:
        pd.DataFrame(annual).to_csv(args.csv, index_label="year")
        print(f"Wrote {args.csv}")

    if not var.continuous:
        return

    # ---- day-to-day predictability
    train_end = pd.Timestamp(args.test_from) - pd.Timedelta(days=1)
    decomps = [decompose(s, c, u, train_end=train_end) for c, s in series.items()]
    tr_ = pd.DataFrame([d.summaries["train"] for d in decomps]).set_index("city")
    te = pd.DataFrame([d.summaries["test"] for d in decomps]).set_index("city")
    print(f"\nDAY-TO-DAY PREDICTABILITY  (training {tr_['start'].iloc[0]} to {tr_['end'].iloc[0]}, "
          f"test {te['start'].iloc[0]} to {te['end'].iloc[0]})\n")
    show = pd.DataFrame(
        {
            f"Surprise RMSE, training ({u})": tr_["residual_sd"],
            f"Surprise RMSE, test ({u})": te["residual_sd"],
            "Change %": (te["residual_sd"] / tr_["residual_sd"] - 1) * 100,
            f"Test bias ({u})": te["residual_mean"],
            "Unpredictable % (test)": te["share_unpredictable"] * 100,
            "1-day skill, model (test)": te["skill_1day"],
            "1-day skill, naive (test)": te["naive_skill_1day"],
            "AR order": te["ar_order"],
        }
    )
    print(show.round(2).T.to_string())
    print("\nForecast skill vs. the seasonal normal, test period (1 = perfect, 0 = no better):")
    skill = pd.DataFrame({d.city: forecast_horizon(d, args.max_lead)["skill_model"] for d in decomps})
    print(skill.round(3).to_string())


if __name__ == "__main__":
    main()
