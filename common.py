"""Shared settings, colors and cached data loaders for the Streamlit app."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import streamlit as st

import weather_analysis as wa

# Fixed color per city (identity follows the city, never its position in a list).
CITY_COLORS = {"San Francisco": "#2a78d6", "New York": "#eb6834", "Dallas": "#1baf7a"}
NEUTRAL = "#8a8985"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
FIRST_YEAR = 1981  # NASA POWER daily meteorology starts in 1981


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def yesterday() -> date:
    return date.today() - timedelta(days=1)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_city(city: str, end: str) -> pd.DataFrame:
    """All variables for one city, 1981 to `end`, metric units. One download per city per day."""
    lat, lon = wa.CITIES[city]
    return wa.fetch_daily(lat, lon, f"{FIRST_YEAR}-01-01", end, list(wa.VARIABLES))


def get_series(city: str, code: str, system: str, start_year: int) -> pd.Series:
    df = load_city(city, yesterday().isoformat())
    if code not in df:
        err = df.attrs.get("errors", {}).get(code, "not returned by NASA POWER")
        raise RuntimeError(f"{wa.VARIABLES[code].short} unavailable: {err}")
    s = df[code].loc[f"{start_year}-01-01":]
    s = s.loc[s.first_valid_index():s.last_valid_index()]
    return wa.convert(s, wa.VARIABLES[code], system)


@st.cache_data(ttl=24 * 3600, show_spinner=False, max_entries=64)
def get_decomposition(city: str, code: str, system: str, start_year: int,
                      train_end: str | None, n_harmonics: int = 3,
                      trend: bool = True, max_lag: int = 7) -> wa.Decomposition:
    s = get_series(city, code, system, start_year)
    return wa.decompose(s, city, wa.unit(wa.VARIABLES[code], system), n_harmonics, trend,
                        max_lag, train_end=train_end)


@dataclass
class Context:
    """Everything a view needs from the sidebar."""
    cities: list[str]
    var: wa.Variable
    system: str            # "US" or "Metric"
    start_year: int
    series: dict[str, pd.Series]

    @property
    def U(self) -> str:      # daily unit
        return wa.unit(self.var, self.system)

    @property
    def AU(self) -> str:     # unit of a yearly summary
        return wa.annual_unit(self.var, self.system)
