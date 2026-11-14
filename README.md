# Earth science trends for Dallas, NYC, and San Francisco

Artificial Intelligence (Claude Opus 5.5 High) was employed to assist the authors with the creation of this dashboard. The authors have reviewed the repo and claim responsbility for any errors thoughout. 

A Streamlit dashboard built on the [NASA POWER](https://power.larc.nasa.gov/) daily API. For **San Francisco, New York and Dallas** it asks two questions
about nine Earth science variables:

1. **Long-term trends.** Is the variable changing over the decades, how fast, and is the
   change statistically significant?
2. **Day-to-day predictability.** How much of each day's value can be predicted from the
   calendar and the last few days, and how much is pure surprise?

## Variables

| Group | Variable | NASA POWER code |
|---|---|---|
| Temperature | Daily mean, high, low (2 m) | `T2M`, `T2M_MAX`, `T2M_MIN` |
| Moisture | Dew point, relative humidity | `T2MDEW`, `RH2M` |
| Moisture | Precipitation (bias-corrected) | `PRECTOTCORR` |
| Wind & pressure | Wind speed (10 m), surface pressure | `WS10M`, `PS` |
| Sunlight | All-sky solar radiation at the surface (from 1984) | `ALLSKY_SFC_SW_DWN` |

All values come in metric units; the app can show US units (°F, inches, mph, inHg).

## Long-term trends view

- **Time series.** Each complete year's average (precipitation: total), shown as change from
  the 1991–2020 average, with the least-squares trend and its confidence band. A
  monthly series with a 12-month running mean is also included.
- **Rate of change.** The least-squares slope per decade with a confidence interval
  (90/95/99%), plus Sen's slope (median of pairwise slopes) as a robust check.
- **Significance tests.** A **t-test** on the slope and the **Mann-Kendall** test, both
  optionally corrected for year-to-year autocorrelation. A result is labeled *significant*
  only when both tests agree.
- **By season.** The same tests for winter, spring, summer and fall.
- **Extremes.** Unusually high or low days per year relative to the 1991–2020 baseline
  for that calendar month (for precipitation, heavy-rain days and dry days), with trend tests.
- **Variability.** Is the weather getting more erratic? A trend test on each year's
  typical day-to-day surprise.

Trend tests run on yearly values, not daily ones. Daily weather is strongly autocorrelated,
so daily-level tests are badly overconfident. In simulations on series with no real trend,
the corrected tests raised false alarms 5–9% of the time (target 5%). The uncorrected versions
reached 25% when neighboring years were correlated. The *Method* tab has the details.

## Day-to-day predictability view

Each day is split into a seasonal cycle and trend, persistence (an autoregressive model of
recent days) and an unpredictable residual. The model is fitted on earlier years only and
tested on later years it never saw. A forecast-horizon analysis shows how skill fades from
1 to 30 days ahead. (Not available for precipitation, which is mostly zeros.)

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first load downloads 1981–today for every variable, one request per city. That can take
a minute or two; results are then cached for a day.

There's also a command-line version:


## Caveats

- NASA POWER values are grid-cell averages (~50 km) from the MERRA-2 reanalysis and
  satellite products, not weather-station readings.
- Reanalysis records can shift when the observing system changes, and the solar record starts
  in 1984.
- With many cities, seasons and variables, some "significant" results will be due to chance.
- A significant linear trend doesn't mean the change is steady or will continue.

