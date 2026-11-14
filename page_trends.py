"""Long-term trends view: rate of change per decade and whether it is statistically significant."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import trends as tr
import weather_analysis as wa
from common import CITY_COLORS, NEUTRAL, Context, get_decomposition, rgba

PLOT_MARGIN = dict(l=10, r=10, t=50, b=10)


def _fmt(x: float, nd: int = 2, sign: bool = True) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _trend_traces(fig, x, y, t: dict, city: str, show_band: bool, show_sen: bool,
                  name_suffix: str = "", row=None, col=None, show_points=True, legend=True):
    """Add yearly points, OLS line, confidence band and optional Sen line for one city."""
    color = CITY_COLORS[city]
    kw = dict(row=row, col=col) if row else {}
    if show_points:
        fig.add_scatter(x=x, y=y, mode="lines+markers", name=f"{city}{name_suffix}",
                        legendgroup=city, showlegend=legend,
                        line=dict(color=rgba(color, 0.55), width=1.5),
                        marker=dict(size=8, color=color, line=dict(color="white", width=1)),
                        hovertemplate=f"{city} " + "%{x}: %{y:.2f}<extra></extra>", **kw)
    if t is None:
        return
    ols, xs = t["_ols"], t["_x"]
    if show_band:
        fig.add_scatter(x=np.r_[xs, xs[::-1]], y=np.r_[ols["band_high"], ols["band_low"][::-1]],
                        fill="toself", fillcolor=rgba(color, 0.15), line=dict(width=0),
                        hoverinfo="skip", showlegend=False, legendgroup=city, **kw)
    fig.add_scatter(x=xs, y=ols["fitted"], mode="lines", line=dict(color=color, width=2.5),
                    name=f"{city} trend", legendgroup=city, showlegend=False,
                    hovertemplate=f"{city} trend: {t['ols_per_decade']:+.2f} per decade<extra></extra>",
                    **kw)
    if show_sen:
        sen = t["_sen"]
        fig.add_scatter(x=xs, y=sen["intercept"] + sen["slope"] * xs, mode="lines",
                        line=dict(color=color, width=1.5, dash="dash"), legendgroup=city,
                        showlegend=False, name=f"{city} Sen's slope",
                        hovertemplate=f"{city} Sen's slope: {t['sen_per_decade']:+.2f} per decade"
                                      "<extra></extra>", **kw)


def _shifted(t: dict | None, offset: float) -> dict | None:
    """Copy of a trend result with fitted lines shifted down by `offset` (for anomaly plots)."""
    if t is None or offset == 0:
        return t
    tt = dict(t)
    tt["_ols"] = {k: (v - offset if k in ("fitted", "band_low", "band_high") else v)
                  for k, v in t["_ols"].items()}
    tt["_sen"] = dict(t["_sen"], intercept=t["_sen"]["intercept"] - offset)
    return tt


def _stats_table(tests: dict[str, dict | None], AU: str, conf: float, alpha: float,
                 words: tuple[str, str], pct: bool, adjust: bool) -> pd.DataFrame:
    rows = {}
    for city, t in tests.items():
        if t is None:
            rows[city] = {"Result": "Not enough complete years"}
            continue
        r = {
            "Result": tr.verdict(t, alpha, words),
            "Years analyzed": f"{t['first_year']}–{t['last_year']} ({t['years']})",
            f"Rate of change, OLS ({AU} per decade)": _fmt(t["ols_per_decade"], 3),
            f"{conf:.0%} confidence interval": f"{_fmt(t['ols_ci_low'], 3)} to {_fmt(t['ols_ci_high'], 3)}",
            "t-test p-value": tr.format_p(t["ols_p"]),
            "Mann-Kendall tau": _fmt(t["mk_tau"], 3),
            "Mann-Kendall p-value": tr.format_p(t["mk_p"]),
            f"Sen's slope ({AU} per decade)": _fmt(t["sen_per_decade"], 3),
            f"Sen's slope {conf:.0%} CI": f"{_fmt(t['sen_ci_low'], 3)} to {_fmt(t['sen_ci_high'], 3)}",
            f"Total change over the period ({AU})": _fmt(t["total_change"], 2),
        }
        if pct:
            r["Change per decade (% of baseline)"] = f"{t['pct_per_decade']:+.1f}%"
        if adjust:
            r["Year-to-year autocorrelation (r1)"] = _fmt(t["r1"], 2)
            r["Effective number of years"] = f"{t['n_eff']:.1f}"
            r["t-test p, unadjusted"] = tr.format_p(t["ols_p_naive"])
            r["Mann-Kendall p, unadjusted"] = tr.format_p(t["mk_p_naive"])
        rows[city] = r
    return pd.DataFrame(rows)


def _summary_frame(tests: dict, alpha: float, words) -> pd.DataFrame:
    rows = []
    for city, t in tests.items():
        if t is None:
            continue
        rows.append({"city": city, "years": t["years"], "first_year": t["first_year"],
                     "last_year": t["last_year"], "ols_per_decade": t["ols_per_decade"],
                     "ols_ci_low": t["ols_ci_low"], "ols_ci_high": t["ols_ci_high"],
                     "t_test_p": t["ols_p"], "mk_tau": t["mk_tau"], "mk_p": t["mk_p"],
                     "sen_per_decade": t["sen_per_decade"], "sen_ci_low": t["sen_ci_low"],
                     "sen_ci_high": t["sen_ci_high"], "r1": t["r1"], "n_eff": t["n_eff"],
                     "verdict": tr.verdict(t, alpha, words)})
    return pd.DataFrame(rows)


def render(ctx: Context, conf: float, adjust: bool) -> None:
    var, U, AU = ctx.var, ctx.U, ctx.AU
    RATE = f"{AU} per decade" if var.annual == "sum" else f"{AU}/decade"
    alpha = 1 - conf
    words = wa.trend_words(var)
    how = var.annual
    agg_word = "total" if how == "sum" else "average"

    annual = {c: tr.annual_series(s, how) for c, s in ctx.series.items()}
    annual = {c: a for c, a in annual.items() if len(a)}
    if not annual:
        st.warning("No complete years in the selected range.")
        return
    all_years = np.unique(np.concatenate([a.index.to_numpy() for a in annual.values()]))
    baseline = tr.choose_baseline(all_years)
    tests = {c: tr.trend_test(a, conf, adjust, baseline) for c, a in annual.items()}

    # ---------------------------------------------------------------- header
    st.header(f"Long-term trends: {var.label}")
    st.markdown(
        f"Each year's {agg_word} {var.short}, computed from NASA POWER daily data, "
        f"**{int(all_years.min())}–{int(all_years.max())}** (complete years only). "
        f"The rate of change is fitted by least squares and tested two ways: a **t-test** on "
        f"the slope and the non-parametric **Mann-Kendall** test, at {conf:.0%} confidence"
        + (", corrected for year-to-year autocorrelation." if adjust else ".")
    )
    if var.note:
        st.caption(var.note)

    cols = st.columns(len(tests))
    for col, (city, t) in zip(cols, tests.items()):
        if t is None:
            col.metric(f"{city}", "–", help="Fewer than 10 complete years.")
            continue
        col.metric(f"{city}: rate of change", f"{t['ols_per_decade']:+.2f} {RATE}",
                   help="Least-squares slope of the yearly values, per 10 years.")
        col.markdown(f"**{tr.verdict(t, alpha, words)}**")
        col.caption(f"{conf:.0%} CI {t['ols_ci_low']:+.2f} to {t['ols_ci_high']:+.2f} · "
                    f"t-test p {tr.format_p(t['ols_p'])} · Mann-Kendall p {tr.format_p(t['mk_p'])}")

    tab_names = ["Yearly trend", "By season", "Extremes", "Variability", "Method"]
    tab = dict(zip(tab_names, st.tabs(tab_names)))

    # ------------------------------------------------------------ yearly trend
    with tab["Yearly trend"]:
        c1, c2, c3 = st.columns([3, 2, 2])
        mode = c1.radio("Show", [f"Change from {baseline[0]}–{baseline[1]} average", "Actual values"],
                        horizontal=True, key="trend_mode")
        anomaly = mode.startswith("Change")
        show_band = c2.checkbox(f"{conf:.0%} confidence band", value=True)
        show_sen = c3.checkbox("Sen's slope line", value=False,
                               help="Robust median-of-slopes trend, drawn dashed.")
        fig = go.Figure()
        for city, a in annual.items():
            t = tests[city]
            offset = t["baseline_mean"] if (anomaly and t) else 0.0
            _trend_traces(fig, a.index, a.values - offset, _shifted(t, offset), city,
                          show_band, show_sen)
        if anomaly:
            fig.add_hline(y=0, line=dict(color=NEUTRAL, width=1))
        ytitle = (f"Change from {baseline[0]}–{baseline[1]} average ({AU})" if anomaly
                  else f"Yearly {agg_word} ({AU})")
        fig.update_layout(title=f"Yearly {agg_word} {var.short}, with least-squares trend",
                          height=480, yaxis_title=ytitle, xaxis_title="Year",
                          margin=PLOT_MARGIN, hovermode="closest",
                          legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, width="stretch")
        st.caption("Dots: each complete year. Solid line: least-squares trend"
                   + (f"; shaded: its {conf:.0%} confidence band" if show_band else "")
                   + ("; dashed: Sen's slope." if show_sen else "."))

        bullets = []
        for city, t in tests.items():
            if t is None:
                continue
            pct = (f" ({t['pct_per_decade']:+.1f}% of the {baseline[0]}–{baseline[1]} average "
                   "per decade)") if var.pct_change else ""
            bullets.append(
                f"- **{city}: {tr.verdict(t, alpha, words).lower()}.** Yearly {var.short} changed by "
                f"**{t['ols_per_decade']:+.2f} {AU} per decade**{pct} "
                f"({conf:.0%} CI {t['ols_ci_low']:+.2f} to {t['ols_ci_high']:+.2f}), about "
                f"{t['total_change']:+.2f} {AU} over {t['first_year']}–{t['last_year']}. "
                f"t-test p {tr.format_p(t['ols_p'])}, Mann-Kendall p {tr.format_p(t['mk_p'])}."
            )
        st.markdown("\n".join(bullets))

        st.subheader("Trend statistics")
        st.dataframe(_stats_table(tests, AU, conf, alpha, words, var.pct_change, adjust),
                     width="stretch")
        st.caption(
            "**Significant** means both tests reject 'no trend' at "
            f"α = {alpha:.2f}. **Possible** means only one does. The confidence interval "
            "covers the true rate of change with the stated probability; if it doesn't include "
            "zero, the t-test is significant."
        )

        with st.expander("Monthly time series (12-month running mean)"):
            n = len(ctx.series)
            fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                                subplot_titles=list(ctx.series))
            for i, (city, s) in enumerate(ctx.series.items(), start=1):
                monthly = (s.resample("MS").sum(min_count=25) if how == "sum"
                           else s.resample("MS").mean())
                base = monthly.loc[str(baseline[0]):str(baseline[1])]
                clim = base.groupby(base.index.month).mean()
                anom_m = monthly - clim.reindex(monthly.index.month).to_numpy()
                roll = anom_m.rolling(12, min_periods=10, center=True).mean()
                fig.add_bar(x=anom_m.index, y=anom_m.values, row=i, col=1, showlegend=False,
                            marker_color=np.where(anom_m.values >= 0, rgba("#e34948", 0.45),
                                                  rgba("#2a78d6", 0.45)),
                            hovertemplate="%{x|%b %Y}: %{y:+.2f}" + U + "<extra></extra>")
                fig.add_scatter(x=roll.index, y=roll.values, row=i, col=1, mode="lines",
                                line=dict(color=CITY_COLORS[city], width=2.5), showlegend=False,
                                hovertemplate="12-mo mean %{x|%b %Y}: %{y:+.2f}" + U + "<extra></extra>")
                fig.add_hline(y=0, row=i, col=1, line=dict(color=NEUTRAL, width=1))
            fig.update_layout(height=260 * n, margin=PLOT_MARGIN, bargap=0)
            st.plotly_chart(fig, width="stretch")
            st.caption(f"Bars: each month's departure from its {baseline[0]}–{baseline[1]} "
                       f"average for that calendar month ({'monthly totals' if how == 'sum' else 'monthly means'}, "
                       f"{U}). Line: 12-month running mean.")

        out = pd.DataFrame(annual)
        out.index.name = "year"
        c1, c2 = st.columns(2)
        c1.download_button("Download yearly values (CSV)", out.to_csv().encode(),
                           file_name=f"yearly_{var.code}_{ctx.system}.csv", mime="text/csv")
        c2.download_button("Download trend statistics (CSV)",
                           _summary_frame(tests, alpha, words).to_csv(index=False).encode(),
                           file_name=f"trend_stats_{var.code}_{ctx.system}.csv", mime="text/csv")

    # --------------------------------------------------------------- by season
    with tab["By season"]:
        season_series = {c: {name: tr.seasonal_series(s, months, how)
                             for name, months in tr.SEASONS.items()}
                         for c, s in ctx.series.items()}
        season_tests = {c: {name: tr.trend_test(ss, conf, adjust, baseline)
                            for name, ss in d.items()} for c, d in season_series.items()}
        fig = go.Figure()
        for city, d in season_tests.items():
            names = [n for n, t in d.items() if t is not None]
            ts = [d[n] for n in names]
            sig = [t["ols_p"] < alpha and t["mk_p"] < alpha for t in ts]
            color = CITY_COLORS[city]
            fig.add_scatter(x=[None], y=[None], mode="markers", name=city, legendgroup=city,
                            marker=dict(symbol="square", size=12, color=color))
            fig.add_bar(
                x=names, y=[t["ols_per_decade"] for t in ts], name=city, showlegend=False,
                legendgroup=city,
                marker=dict(color=[color if s else rgba(color, 0.3) for s in sig],
                            line=dict(color=color, width=1.5)),
                error_y=dict(type="data", symmetric=False,
                             array=[t["ols_ci_high"] - t["ols_per_decade"] for t in ts],
                             arrayminus=[t["ols_per_decade"] - t["ols_ci_low"] for t in ts],
                             color=NEUTRAL, thickness=1.5, width=4),
                customdata=[[tr.format_p(t["ols_p"]), tr.format_p(t["mk_p"]),
                             tr.verdict(t, alpha, words)] for t in ts],
                hovertemplate=f"{city}, " + "%{x}: %{y:+.2f} " + AU + "/decade<br>"
                              "t-test p %{customdata[0]} · MK p %{customdata[1]}<br>"
                              "%{customdata[2]}<extra></extra>",
            )
        fig.add_hline(y=0, line=dict(color=NEUTRAL, width=1))
        season_unit = f"{U}/season" if how == "sum" else U
        fig.update_xaxes(type="category", categoryorder="array", categoryarray=list(tr.SEASONS))
        fig.update_layout(title=f"Rate of change by season ({season_unit} per decade)", barmode="group",
                          height=440, margin=PLOT_MARGIN, yaxis_title=f"{season_unit} per decade",
                          legend=dict(orientation="h", y=-0.12))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Whiskers: {conf:.0%} confidence interval. Solid bars are significant by both "
                   "tests; faded bars are not. Winter includes December from the previous year.")

        rows = []
        for city, d in season_tests.items():
            for name, t in d.items():
                if t is None:
                    continue
                rows.append({"City": city, "Season": name,
                             f"Rate ({season_unit}/decade)": t["ols_per_decade"],
                             "CI low": t["ols_ci_low"], "CI high": t["ols_ci_high"],
                             "t-test p": tr.format_p(t["ols_p"]),
                             "Mann-Kendall p": tr.format_p(t["mk_p"]),
                             "Sen's slope": t["sen_per_decade"],
                             "Result": tr.verdict(t, alpha, words)})
        if rows:
            st.dataframe(pd.DataFrame(rows).set_index(["City", "Season"])
                         .style.format({f"Rate ({season_unit}/decade)": "{:+.3f}", "CI low": "{:+.3f}",
                                        "CI high": "{:+.3f}", "Sen's slope": "{:+.3f}"}),
                         width="stretch")
        st.caption(f"Testing 4 seasons × {len(season_tests)} cities means about "
                   f"{4 * len(season_tests) * alpha:.1f} 'significant' result(s) could appear by "
                   "chance alone. Weigh a single borderline season accordingly.")

        season = st.selectbox("See one season's yearly values", list(tr.SEASONS))
        fig = go.Figure()
        for city in season_series:
            ss, t = season_series[city][season], season_tests[city][season]
            offset = t["baseline_mean"] if t else 0.0
            _trend_traces(fig, ss.index, ss.values - offset, _shifted(t, offset), city,
                          show_band=True, show_sen=False)
        fig.add_hline(y=0, line=dict(color=NEUTRAL, width=1))
        fig.update_layout(title=f"{season}: seasonal {agg_word} {var.short}, change from "
                                f"{baseline[0]}–{baseline[1]} average ({season_unit})",
                          height=420, margin=PLOT_MARGIN, xaxis_title="Year",
                          yaxis_title=f"Change ({season_unit})",
                          legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, width="stretch")

    # ---------------------------------------------------------------- extremes
    with tab["Extremes"]:
        ext, info = {}, None
        for city, s in ctx.series.items():
            ext[city], info = tr.extremes_counts(s, var, ctx.system, baseline)
        kind = st.radio("Count", ["high", "low"], horizontal=True,
                        format_func=lambda k: info[f"{k}_label"], key="ext_kind")
        ext_tests = {c: tr.trend_test(e[kind], conf, adjust, baseline) for c, e in ext.items()}
        fig = go.Figure()
        for city, e in ext.items():
            _trend_traces(fig, e.index, e[kind].values, ext_tests[city], city,
                          show_band=True, show_sen=False)
        fig.update_layout(title=f"{info[f'{kind}_label']} per year", height=440,
                          yaxis_title="Days per year", xaxis_title="Year", margin=PLOT_MARGIN,
                          legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"{info[f'{kind}_label']}: {info[f'{kind}_def']} "
                   f"(baseline {info['baseline'][0]}–{info['baseline'][1]}).")

        rows = []
        for city, e in ext.items():
            for k in ("high", "low"):
                t = tr.trend_test(e[k], conf, adjust, baseline)
                if t is None:
                    continue
                base = e[k].loc[baseline[0]:baseline[1]]
                rows.append({"City": city, "Count": info[f"{k}_label"],
                             "Baseline average (days/yr)": base.mean(),
                             "Recent 10-yr average (days/yr)": e[k].iloc[-10:].mean(),
                             "Trend (days/decade)": t["ols_per_decade"],
                             "CI low": t["ols_ci_low"], "CI high": t["ols_ci_high"],
                             "t-test p": tr.format_p(t["ols_p"]),
                             "Mann-Kendall p": tr.format_p(t["mk_p"]),
                             "Result": tr.verdict(t, alpha, ("increase", "decrease"))})
        if rows:
            st.dataframe(pd.DataFrame(rows).set_index(["City", "Count"]).style.format(
                {"Baseline average (days/yr)": "{:.1f}", "Recent 10-yr average (days/yr)": "{:.1f}",
                 "Trend (days/decade)": "{:+.2f}", "CI low": "{:+.2f}", "CI high": "{:+.2f}"}),
                width="stretch")
        if var.kind != "precipitation":
            st.caption("Thresholds are set for each calendar month, so a hot day in January counts "
                       "even though it is colder than a normal day in July. By construction, about "
                       "10% of baseline days (≈ 36 per year) fall in each category; a trend means "
                       "that share is changing.")

    # ------------------------------------------------------------- variability
    with tab["Variability"]:
        st.markdown(
            f"**Is {var.short} getting more erratic?** For each year, this takes the size of the "
            "day-to-day surprises: what's left after removing the seasonal cycle, the long-term "
            "trend and persistence from the last few days (the model in the *Day-to-day "
            "predictability* view, fitted on all years). It then tests that yearly value for a trend."
        )
        if not var.continuous:
            st.info(f"This needs a continuous variable; daily {var.short} is mostly zeros. "
                    "Pick another variable to see it.")
        else:
            vol, vol_tests = {}, {}
            for city in ctx.series:
                try:
                    d = get_decomposition(city, var.code, ctx.system, ctx.start_year, None)
                except ValueError as exc:
                    st.warning(f"{city}: {exc}")
                    continue
                vol[city] = wa.yearly_residual_sd(d, min_days=347)
                vol_tests[city] = tr.trend_test(vol[city], conf, adjust, baseline)
            fig = go.Figure()
            for city, v in vol.items():
                _trend_traces(fig, v.index, v.values, vol_tests[city], city,
                              show_band=True, show_sen=False)
            fig.update_layout(title=f"Typical day-to-day surprise each year (SD, {U})", height=440,
                              yaxis_title=f"SD of surprises ({U})", xaxis_title="Year",
                              margin=PLOT_MARGIN, legend=dict(orientation="h", y=-0.15))
            st.plotly_chart(fig, width="stretch")
            st.dataframe(_stats_table(vol_tests, U, conf, alpha, ("more erratic", "steadier"),
                                      False, adjust), width="stretch")

    # ------------------------------------------------------------------ method
    with tab["Method"]:
        st.markdown(f"""
**Data.** NASA POWER daily point data (MERRA-2 reanalysis for meteorology, satellite products
for solar), downloaded for each city's coordinates. Values are averages over a ~50 km grid
cell, not station readings.

**Yearly values.** Daily values are averaged (precipitation: totaled) into one value per
calendar year. Years with less than 95% of their days are dropped, so a partial current year
can't bias the trend. Seasons need 90% of their days; winter is December–February, with December
counted toward the following year. Trends are tested on yearly values because daily weather is
strongly autocorrelated, so a test on raw daily values would be badly overconfident.

**Rate of change (OLS).** A least-squares line through the yearly values. The slope is reported
per decade with a {conf:.0%} confidence interval, and tested with a t-test (H₀: slope = 0).

**Mann-Kendall test.** A non-parametric test of a consistently rising or falling series. It
counts how often later years are higher than earlier ones and doesn't assume normally
distributed values or a straight-line trend. Ties are corrected for.

**Sen's slope.** The median of the slopes between every pair of years. A robust rate of change
that pairs naturally with Mann-Kendall and resists outlier years.

**Autocorrelation correction** ({'on' if adjust else 'off'}). Neighboring years are often alike
(r1 > 0), which makes ordinary tests overconfident. With the correction on, the t-test uses an
effective number of years, n·(1 − r1)/(1 + r1) (Santer et al. 2000). Mann-Kendall's variance
is inflated by the matching AR(1) factor (Hamed & Rao 1998; Yue & Wang 2004).

**Result labels.** *Significant* requires both tests to reject "no trend" at α = {alpha:.2f}.
*Possible* means only one does.

**Baseline.** Changes and extremes are measured against {baseline[0]}–{baseline[1]}
(the WMO 1991–2020 normal period when the data covers it).

**Extremes.** For each calendar month, the 90th and 10th percentiles of the baseline's
departures from the seasonal cycle define unusually high and low days. Precipitation uses
the baseline's 95th percentile of wet-day (≥ 1 mm) amounts for heavy days, and counts dry days.

**How well the tests behave.** Checked by simulation on 40-year series with no real trend:

| Year-to-year noise | t-test, unadjusted | t-test, adjusted | Mann-Kendall, unadjusted | Mann-Kendall, adjusted |
|---|---|---|---|---|
| Independent | 5.3% | 4.5% | 5.0% | 4.6% |
| Autocorrelated, r = 0.3 | 13.7% | 7.0% | 13.3% | 7.4% |
| Autocorrelated, r = 0.5 | 25.1% | 8.8% | 23.3% | 8.7% |

The target is 5%. The corrections remove most, though not all, of the excess false alarms.
With a real trend of 0.05 standard deviations per year, all versions detect it about 91–94% of
the time. Requiring both tests to agree adds a little more protection.

**Caveats.**
- Start year matters. Trends over short records swing a lot, so prefer the longest range.
- Reanalysis data can contain jumps when the observing system changes; NASA POWER's solar
  record starts in 1984.
- Many tests (cities × seasons × variables) mean some "significant" results will be chance.
- A significant linear trend doesn't mean the change is steady or will continue.
""")
