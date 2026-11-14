"""
Streamlit dashboard: long-term trends and day-to-day predictability of Earth science
variables in San Francisco, New York and Dallas, from NASA POWER daily data.

Run locally:   streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

import weather_analysis as wa
from common import FIRST_YEAR, Context, get_series, load_city, yesterday

st.set_page_config(page_title="NASA POWER Trends", page_icon="🌎", layout="wide")

VIEWS = ["Long-term trends", "Day-to-day predictability"]
GROUP_ORDER = list(dict.fromkeys(v.group for v in wa.VARIABLES.values()))

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    view = st.radio("View", VIEWS, help="Trends: is it changing over decades, and is the change "
                                        "statistically significant? Predictability: how much of "
                                        "each day is a surprise?")
    st.divider()
    cities = st.multiselect("Cities", list(wa.CITIES), default=list(wa.CITIES))
    code = st.selectbox(
        "Variable", list(wa.VARIABLES),
        format_func=lambda c: wa.VARIABLES[c].label,
    )
    var = wa.VARIABLES[code]
    system = st.radio("Units", ["US", "Metric"], horizontal=True,
                      format_func=lambda s: "US (°F, in, mph)" if s == "US" else "Metric (°C, mm, m/s)")
    last_year = yesterday().year
    start_year = st.slider("Start year", FIRST_YEAR, last_year - 12, FIRST_YEAR,
                           help="Longer records give more reliable trends. NASA POWER daily data "
                                "starts in 1981 (solar: 1984).")

    if view == VIEWS[0]:
        st.subheader("Trend test")
        conf = st.select_slider("Confidence level", [0.90, 0.95, 0.99], value=0.95,
                                format_func=lambda c: f"{c:.0%}")
        adjust = st.checkbox("Correct for year-to-year autocorrelation", value=True,
                             help="Neighboring years are often alike, which makes ordinary "
                                  "tests overconfident. Recommended.")
    else:
        st.subheader("Model")
        test_lo, test_hi = start_year + 5, last_year - 1
        test_from = st.slider(
            "Test period starts", test_lo, test_hi, min(max(2016, test_lo), test_hi),
            help="The model is fitted only on years before this one. Later years are kept out "
                 "and used to test it on data it has never seen.",
        )
        with st.expander("Model options"):
            n_harmonics = st.slider("Seasonal harmonics", 1, 4, 3,
                                    help="How many sine/cosine pairs describe the yearly cycle.")
            max_lag = st.slider("Max persistence lag (days)", 1, 14, 7,
                                help="Longest lag the AR model may use; BIC picks the actual order.")
            use_trend = st.checkbox("Remove long-term linear trend", value=True)

    st.divider()
    st.caption("Data: [NASA POWER](https://power.larc.nasa.gov/) daily point API "
               "(MERRA-2 reanalysis and satellite solar data). No API key needed.")

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
st.title("Earth science trends for three US cities")

if not cities:
    st.info("Pick at least one city in the sidebar.")
    st.stop()

series = {}
with st.spinner("Downloading from NASA POWER (first load can take a minute per city)…"):
    for city in cities:
        try:
            load_city(city, yesterday().isoformat())
            series[city] = get_series(city, code, system, start_year)
        except Exception as exc:  # network errors, API errors, missing variable
            st.error(f"Couldn't load {city}: {exc}")
if not series:
    st.stop()

ctx = Context(cities=list(series), var=var, system=system, start_year=start_year, series=series)

if view == VIEWS[0]:
    import page_trends
    page_trends.render(ctx, conf, adjust)
else:
    import page_predictability
    page_predictability.render(ctx, test_from, n_harmonics, use_trend, max_lag)
