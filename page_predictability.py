"""Day-to-day predictability view: how much of each day's value is a surprise?"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

import weather_analysis as wa
from common import CITY_COLORS, MONTHS, NEUTRAL, Context, get_decomposition


def render(ctx: Context, test_from: int, n_harmonics: int, use_trend: bool, max_lag: int) -> None:
    var, U = ctx.var, ctx.U
    HI, LO = wa.change_words(var)
    IS_TEMP = var.kind == "temperature"
    if not var.continuous:
        st.info(f"The day-to-day predictability model needs a continuous variable. Daily "
                f"{var.short} is mostly zeros with occasional bursts, so it isn't modeled here. "
                "Pick another variable, or see its long-term trends in the other view.")
        return

    results: dict[str, wa.Decomposition] = {}
    with st.spinner("Fitting models…"):
        for city in ctx.series:
            try:
                results[city] = get_decomposition(city, var.code, ctx.system, ctx.start_year,
                                                  f"{test_from - 1}-12-31", n_harmonics,
                                                  use_trend, max_lag)
            except ValueError as exc:
                st.warning(f"{city}: {exc}")
    if not results:
        return

    # Headline numbers come from the test period: honest, out-of-sample results.
    summary = pd.DataFrame([d.summaries["test"] for d in results.values()]).set_index("city")
    train_summary = pd.DataFrame([d.summaries["train"] for d in results.values()]).set_index("city")
    first_tr = train_summary.iloc[0]
    first = summary.iloc[0]

    # --------------------------------------------------------------------------
    # Header
    # --------------------------------------------------------------------------
    st.header(f"How predictable is {var.short} from day to day?")
    st.markdown(
        f"Each city's daily {var.short} is split into what the "
        "**calendar** predicts (seasonal cycle and trend), what the **last few days** predict "
        "(persistence), and the **unpredictable** remainder: the day-to-day surprise.\n\n"
        f"The model is fitted on **{first_tr['start']:%Y}–{first_tr['end']:%Y}** only and tested on "
        f"**{first['start']:%b %Y} – {first['end']:%b %d, %Y}**, which it never saw. "
        "The numbers below are from that test period."
    )

    cols = st.columns(len(results))
    for col, (city, d) in zip(cols, results.items()):
        sm, tr = d.summaries["test"], d.summaries["train"]
        col.metric(
            f"{city}: typical daily surprise",
            f"±{sm['residual_sd']:.1f} {U}",
            delta=f"{sm['residual_sd'] - tr['residual_sd']:+.2f} {U} vs. training years",
            delta_color="inverse",
            help="Root-mean-square error of the next-day prediction over the test period, "
                 "from a model fitted only on the training years.",
        )
        col.caption(
            f"{sm['share_unpredictable']:.0%} of all day-to-day variance is "
            f"unpredictable · {sm['pct_beyond_2sd']:.1f}% of days miss by more than "
            f"±{2 * sm['residual_sd']:.1f} {U}"
        )

    # --------------------------------------------------------------------------
    # Variance breakdown
    # --------------------------------------------------------------------------
    st.subheader("Where does the variation come from? (test period)")
    parts = [
        ("share_seasonal", "Seasonal cycle and trend", "#b9b8b3"),
        ("share_persistence", "Persistence (recent days)", "#6f6e69"),
        ("share_unpredictable", "Unpredictable", "#e34948"),
    ]
    fig = go.Figure()
    for key, label, color in parts:
        fig.add_bar(
            y=summary.index, x=summary[key] * 100, name=label, orientation="h",
            marker=dict(color=color, line=dict(color="white", width=2)),
            text=[f"{v:.0%}" for v in summary[key]], textposition="inside",
            hovertemplate="%{y}<br>" + label + ": %{x:.1f}%<extra></extra>",
        )
    fig.update_layout(
        barmode="stack", height=90 + 60 * len(summary), margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="Share of total variance (%)", range=[0, 100]),
        yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=-0.35),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "A city with a small seasonal swing can have a big unpredictable *share* even when its "
        "surprises are small in absolute terms. Compare shares here with the ± values above."
    )

    with st.expander("Summary table (test period)"):
        tbl = pd.DataFrame(
            {
                f"Mean ({U})": summary["mean_temp"],
                f"Total SD ({U})": summary["total_sd"],
                f"Anomaly SD ({U})": summary["anomaly_sd"],
                f"Unpredictable RMSE ({U})": summary["residual_sd"],
                f"Average bias ({U})": summary["residual_mean"],
                f"Mean absolute surprise ({U})": summary["residual_mad"],
                "Seasonal %": summary["share_seasonal"] * 100,
                "Persistence %": summary["share_persistence"] * 100,
                "Unpredictable %": summary["share_unpredictable"] * 100,
                "1-day skill vs. seasonal normal %": summary["skill_1day"] * 100,
                "1-day skill, naive persistence %": summary["naive_skill_1day"] * 100,
                "AR order (days)": summary["ar_order"],
                "Lag-1 anomaly autocorrelation": summary["anomaly_lag1_autocorr"],
                "Residual skew": summary["residual_skew"],
                "Residual excess kurtosis": summary["residual_excess_kurtosis"],
                "% days beyond 2 SD (normal ≈ 4.6)": summary["pct_beyond_2sd"],
                "% days beyond 3 SD (normal ≈ 0.27)": summary["pct_beyond_3sd"],
                f"Trend ({U}/decade)": summary["trend_per_decade"],
                "Ljung-Box p (lag 10)": summary["ljung_box_p"],
                "Days analyzed": summary["days"],
            }
        ).T
        st.dataframe(tbl.style.format("{:.2f}"), width="stretch")

    # --------------------------------------------------------------------------
    # Tabs
    # --------------------------------------------------------------------------
    TAB_NAMES = ["Out-of-sample test", "Forecast horizon", "Daily breakdown", "Distribution",
                 "By month", "Memory", "Across cities", "Biggest surprises", "Method"]
    tab = dict(zip(TAB_NAMES, st.tabs(TAB_NAMES)))
    TEST_START = pd.Timestamp(f"{test_from}-01-01")

    # --- Out-of-sample test ---------------------------------------------------
    with tab["Out-of-sample test"]:
        st.markdown(
            f"Every part of the model (seasonal cycle, trend, persistence) was estimated from "
            f"**{first_tr['start']:%Y}–{first_tr['end']:%Y}** and then used, unchanged, to predict each "
            f"day of **{first['start']:%Y}–{first['end']:%Y}**. If the model had just memorized the "
            "training years, its errors would jump in the test period."
        )
        rows = []
        for city, d in results.items():
            tr, te = d.summaries["train"], d.summaries["test"]
            rows.append({
                "City": city,
                f"Training RMSE ({U})": tr["residual_sd"],
                f"Test RMSE ({U})": te["residual_sd"],
                "Change %": (te["residual_sd"] / tr["residual_sd"] - 1) * 100,
                f"Test bias ({U})": te["residual_mean"],
                "1-day skill, training %": tr["skill_1day"] * 100,
                "1-day skill, test %": te["skill_1day"] * 100,
                "Naive 1-day skill, test %": te["naive_skill_1day"] * 100,
                "Training days": tr["days"],
                "Test days": te["days"],
            })
        oos = pd.DataFrame(rows).set_index("City")

        c1, c2 = st.columns(2)
        fig = go.Figure()
        fig.add_bar(x=oos.index, y=oos[f"Training RMSE ({U})"], name="Training years (in-sample)",
                    marker=dict(color="#b9b8b3", line=dict(color="white", width=2)),
                    hovertemplate="%{x}: ±%{y:.2f}" + U + "<extra>Training</extra>")
        fig.add_bar(x=oos.index, y=oos[f"Test RMSE ({U})"], name="Test years (out-of-sample)",
                    marker=dict(color="#e34948", line=dict(color="white", width=2)),
                    hovertemplate="%{x}: ±%{y:.2f}" + U + "<extra>Test</extra>")
        fig.update_layout(title=f"Typical next-day surprise (RMSE, {U})", barmode="group", height=380,
                          margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.15))
        c1.plotly_chart(fig, width="stretch")

        fig = go.Figure()
        fig.add_bar(x=oos.index, y=oos["Naive 1-day skill, test %"],
                    name="Naive: yesterday's anomaly continues",
                    marker=dict(color="#b9b8b3", line=dict(color="white", width=2)),
                    hovertemplate="%{x}: %{y:.1f}%<extra>Naive</extra>")
        fig.add_bar(x=oos.index, y=oos["1-day skill, test %"], name="Model",
                    marker=dict(color="#52514e", line=dict(color="white", width=2)),
                    hovertemplate="%{x}: %{y:.1f}%<extra>Model</extra>")
        fig.update_layout(title="Next-day skill vs. the seasonal normal, test years (%)",
                          barmode="group", height=380, yaxis=dict(range=[0, 100]),
                          margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.15))
        c2.plotly_chart(fig, width="stretch")

        st.dataframe(oos.style.format("{:.2f}").format("{:,.0f}", subset=["Training days", "Test days"]),
                     width="stretch")
        lines = []
        for city, r in oos.iterrows():
            chg, bias = r["Change %"], r[f"Test bias ({U})"]
            verdict = ("about the same as" if abs(chg) < 3 else
                       "larger than" if chg > 0 else "smaller than")
            lines.append(
                f"- **{city}:** test-period errors are {verdict} in training ({chg:+.1f}%). "
                f"On average the test years ran {abs(bias):.1f} {U} "
                f"{HI if bias > 0 else LO} than the model expected."
            )
        st.markdown("\n".join(lines))
        st.caption(
            "Skill = 1 − (prediction error² ÷ error² of just using the seasonal normal). 100% is "
            "perfect; 0% is no better than the calendar. Bias is the average surprise: a persistent "
            "bias means the test years drifted away from the trend fitted on the training years."
        )

    # --- Forecast horizon -----------------------------------------------------
    with tab["Forecast horizon"]:
        c1, c2, c3 = st.columns([2, 2, 2])
        max_lead = c1.slider("Longest lead (days)", 3, 30, 14)
        show_naive = c2.checkbox("Show naive baseline", value=True,
                                 help="Assume today's departure from normal stays the same.")
        show_train = c3.checkbox("Show training-period curves", value=False,
                                 help="In-sample results, for comparison with the test period.")
        THRESHOLD = 0.10
        horizons = {c: wa.forecast_horizon(d, max_lead, "test") for c, d in results.items()}

        cols = st.columns(len(results))
        for col, (city, hz) in zip(cols, horizons.items()):
            lim = wa.predictability_limit(hz, THRESHOLD)
            col.metric(f"{city}: useful forecast range",
                       f"{lim} day{'s' if lim != 1 else ''}" if lim < max_lead else f"{max_lead}+ days",
                       help=f"Last lead day at which the model still has at least "
                            f"{THRESHOLD:.0%} skill over the seasonal normal (test period).")
            picks = [h for h in (1, 3, 7) if h <= max_lead]
            col.caption(" · ".join(f"{h}-day skill {hz.loc[h, 'skill_model']:.0%}" for h in picks))

        fig = go.Figure()
        for city, hz in horizons.items():
            color = CITY_COLORS[city]
            fig.add_scatter(x=hz.index, y=hz["skill_model"] * 100, name=city, legendgroup=city,
                            mode="lines+markers", line=dict(color=color, width=2), marker=dict(size=8),
                            hovertemplate=f"{city} model, " + "%{x}-day lead: %{y:.1f}%<extra></extra>")
            if show_naive:
                fig.add_scatter(x=hz.index, y=hz["skill_naive"] * 100, name=f"{city}: naive",
                                legendgroup=city, mode="lines", line=dict(color=color, width=2, dash="dot"),
                                hovertemplate=f"{city} naive, " + "%{x}-day lead: %{y:.1f}%<extra></extra>")
            if show_train:
                ht = wa.forecast_horizon(results[city], max_lead, "train")
                fig.add_scatter(x=ht.index, y=ht["skill_model"] * 100, name=f"{city}: training",
                                legendgroup=city, mode="lines",
                                line=dict(color=color, width=1, dash="dash"),
                                hovertemplate=f"{city} training, " + "%{x}-day lead: %{y:.1f}%<extra></extra>")
        fig.add_hline(y=THRESHOLD * 100, line=dict(color=NEUTRAL, width=1, dash="dash"),
                      annotation_text=f"{THRESHOLD:.0%} usefulness threshold",
                      annotation_position="top right")
        fig.add_hline(y=0, line=dict(color=NEUTRAL, width=1))
        lowest = min(h["skill_naive"].min() if show_naive else 0 for h in horizons.values()) * 100
        fig.update_layout(title="Forecast skill vs. the seasonal normal, by lead time (test period)",
                          height=460, xaxis=dict(title="Days ahead", dtick=1 if max_lead <= 14 else 2),
                          yaxis=dict(title="Skill (%)", range=[max(lowest, -50) - 5, 100]),
                          margin=dict(l=10, r=10, t=50, b=10), hovermode="x unified")
        st.plotly_chart(fig, width="stretch")
        st.markdown(
            "Each point answers: *standing on day t with only the values up to then, how well "
            "can you predict day t + lead?* The model runs itself forward, feeding its own forecasts "
            "back in. Skill fades toward 0% as predictions fall back to the seasonal normal. "
            "The **naive** baseline (dotted) usually drops below zero within a few days. Assuming "
            "an unusual spell continues unchanged becomes *worse* than the calendar, because spells fade."
        )

        fig = go.Figure()
        for city, hz in horizons.items():
            color = CITY_COLORS[city]
            fig.add_scatter(x=hz.index, y=hz["rmse_model"], name=city, legendgroup=city,
                            mode="lines+markers", line=dict(color=color, width=2), marker=dict(size=8),
                            hovertemplate=f"{city}, " + "%{x}-day lead: ±%{y:.2f}" + U + "<extra></extra>")
            fig.add_scatter(x=hz.index, y=hz["rmse_climatology"], name=f"{city}: seasonal normal only",
                            legendgroup=city, mode="lines", line=dict(color=color, width=1, dash="dash"),
                            hovertemplate=f"{city} seasonal normal: ±" + "%{y:.2f}" + U + "<extra></extra>")
        fig.update_layout(title=f"Typical forecast error by lead time (RMSE, {U}, test period)",
                          height=420, xaxis=dict(title="Days ahead", dtick=1 if max_lead <= 14 else 2),
                          yaxis=dict(title=f"RMSE ({U})", rangemode="tozero"),
                          margin=dict(l=10, r=10, t=50, b=10), hovermode="x unified")
        st.plotly_chart(fig, width="stretch")
        st.caption("Dashed lines are the error you'd get by forecasting the seasonal normal every day. "
                   "Where a solid line meets its dashed line, the record tells you nothing "
                   "more than the calendar does.")
        st.info(
            f"This is predictability from **each city's own {var.short} history alone**. National "
            "Weather Service forecasts model the whole atmosphere and typically stay useful for "
            "about a week or more. The gap between that and the range above shows how much "
            "information a single record leaves out."
        )
        with st.expander("Table: skill by lead day"):
            tbl = pd.DataFrame({c: hz["skill_model"] * 100 for c, hz in horizons.items()})
            tbl.index.name = "Days ahead"
            st.dataframe(tbl.style.format("{:.1f}%"), width="stretch")

    # --- Daily breakdown ------------------------------------------------------
    with tab["Daily breakdown"]:
        c1, c2 = st.columns([1, 3])
        city = c1.selectbox("City", list(results), key="ts_city")
        d = results[city]
        fr = d.frame
        end_default = fr.index.max().date()
        window = c2.slider(
            "Date range", min_value=fr.index.min().date(), max_value=end_default,
            value=(end_default - timedelta(days=365), end_default), format="MMM YYYY",
        )
        w = fr.loc[pd.Timestamp(window[0]):pd.Timestamp(window[1])]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35],
                            vertical_spacing=0.06,
                            subplot_titles=(f"Actual vs predictable ({U})", f"Unpredictable part ({U})"))
        fig.add_scatter(x=w.index, y=w["seasonal"], name="Seasonal normal", row=1, col=1,
                        line=dict(color=NEUTRAL, width=2, dash="dot"))
        fig.add_scatter(x=w.index, y=w["predictable"], name="Predictable (season + persistence)",
                        row=1, col=1, line=dict(color=CITY_COLORS[city], width=2))
        fig.add_scatter(x=w.index, y=w["temp"], name="Actual", row=1, col=1,
                        line=dict(color="#52514e", width=1.5))
        sd = d.summaries["test"]["residual_sd"]
        fig.add_bar(x=w.index, y=w["residual"], name="Surprise", row=2, col=1,
                    marker_color=np.where(w["residual"] >= 0, "#e34948", "#2a78d6"),
                    hovertemplate="%{x|%b %d %Y}: %{y:+.1f}" + U + "<extra></extra>")
        if pd.Timestamp(window[0]) < TEST_START < pd.Timestamp(window[1]):
            for r in (1, 2):
                fig.add_vline(x=TEST_START, row=r, col=1, line=dict(color=NEUTRAL, width=1, dash="dash"))
        for y in (2 * sd, -2 * sd):
            fig.add_hline(y=y, row=2, col=1, line=dict(color=NEUTRAL, width=1, dash="dash"))
        fig.update_layout(height=560, hovermode="x unified", margin=dict(l=10, r=10, t=40, b=10),
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Dashed lines mark ±2 SD (±{2 * sd:.1f} {U}). Red bars: {HI} than predicted; "
                   f"blue: {LO}.")

    # --- Distribution -------------------------------------------------------
    with tab["Distribution"]:
        fig = go.Figure()
        lim = max(4 * s["residual_sd"] for s in summary.to_dict("index").values())
        xs = np.linspace(-lim, lim, 300)
        for city, d in results.items():
            r = d.frame.loc[d.mask("test"), "residual"].dropna()
            fig.add_histogram(x=r, name=city, histnorm="probability density", opacity=0.45,
                              marker_color=CITY_COLORS[city], xbins=dict(size=lim / 40))
            fig.add_scatter(x=xs, y=stats.norm.pdf(xs, 0, r.std()), name=f"{city} (normal fit)",
                            line=dict(color=CITY_COLORS[city], width=2, dash="dot"))
        fig.update_layout(barmode="overlay", height=450, xaxis_title=f"Surprise ({U})",
                          yaxis_title="Density", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("Out-of-sample surprises from the test period.")
        st.markdown(
            "Dotted curves show a normal distribution with the same spread. **Negative skew** "
            f"means the big misses tend to be {LO} than expected"
            + (", such as a cold front arriving sooner than persistence expects. " if IS_TEMP else ". ")
            + "**Positive excess kurtosis** means fat tails: extreme days "
            "happen more often than a bell curve predicts."
        )
        st.dataframe(
            summary[["residual_skew", "residual_excess_kurtosis", "pct_beyond_2sd", "pct_beyond_3sd"]]
            .rename(columns={"residual_skew": "Skew", "residual_excess_kurtosis": "Excess kurtosis",
                             "pct_beyond_2sd": "% beyond 2 SD", "pct_beyond_3sd": "% beyond 3 SD"})
            .style.format("{:.2f}"),
            width="stretch",
        )

    # --- By month ------------------------------------------------------------
    with tab["By month"]:
        c1, c2 = st.columns(2)
        fig = go.Figure()
        for city, d in results.items():
            m = wa.monthly_residual_sd(d)
            fig.add_scatter(x=[MONTHS[i - 1] for i in m.index], y=m.values, name=city,
                            mode="lines+markers", line=dict(color=CITY_COLORS[city], width=2),
                            marker=dict(size=8))
        fig.update_layout(title=f"Typical surprise by month (SD, {U}, all years)", height=380,
                          margin=dict(l=10, r=10, t=50, b=10), hovermode="x unified")
        c1.plotly_chart(fig, width="stretch")
        fig = go.Figure()
        for city, d in results.items():
            m = wa.monthly_residual_skew(d)
            fig.add_scatter(x=[MONTHS[i - 1] for i in m.index], y=m.values, name=city,
                            mode="lines+markers", line=dict(color=CITY_COLORS[city], width=2),
                            marker=dict(size=8))
        fig.add_hline(y=0, line=dict(color=NEUTRAL, width=1))
        fig.update_layout(title="Skew of surprises by month", height=380,
                          margin=dict(l=10, r=10, t=50, b=10), hovermode="x unified")
        c2.plotly_chart(fig, width="stretch")
        st.caption(("Temperatures usually get harder to predict in winter, when storm systems move "
                    "through faster. " if IS_TEMP else "")
                   + f"Below zero, the big misses in that month were mostly {LO} than expected.")

    # --- Memory --------------------------------------------------------------
    with tab["Memory"]:
        nl = 20
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                            subplot_titles=("Anomaly (before removing persistence)",
                                            "Unpredictable residual"))
        for city, d in results.items():
            for i, colname in enumerate(["anomaly", "residual"], start=1):
                a = wa.acf(d.frame.loc[d.mask("test"), colname].dropna(), nl)
                fig.add_scatter(x=np.arange(1, nl + 1), y=a[1:], name=city, legendgroup=city,
                                showlegend=(i == 1), mode="lines+markers", row=1, col=i,
                                line=dict(color=CITY_COLORS[city], width=2), marker=dict(size=8))
        fig.add_hline(y=0, line=dict(color=NEUTRAL, width=1))
        fig.update_xaxes(title_text="Lag (days)")
        fig.update_yaxes(title_text="Autocorrelation", row=1, col=1)
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10), hovermode="x unified")
        st.plotly_chart(fig, width="stretch")
        st.markdown(
            "**Left:** how long an unusually high or low spell 'remembers' itself. The faster the curve drops, "
            "the less yesterday tells you about today. **Right:** after removing persistence, "
            "the residual should be close to zero at every lag. Anything left is memory the "
            "simple model missed."
        )
        st.dataframe(
            summary[["ar_order", "lag1_coef", "anomaly_lag1_autocorr", "ljung_box_p"]]
            .rename(columns={"ar_order": "AR order", "lag1_coef": "Lag-1 coefficient",
                             "anomaly_lag1_autocorr": "Lag-1 autocorrelation",
                             "ljung_box_p": "Ljung-Box p"})
            .style.format("{:.3f}"),
            width="stretch",
        )
        st.caption("With 10,000+ days, the Ljung-Box test flags even tiny leftover correlation. "
                   "Judge by the size of the bars on the right, not by the p-value alone.")

    # --- Across cities -------------------------------------------------------
    with tab["Across cities"]:
        if len(results) < 2:
            st.info("Select at least two cities to compare their surprises.")
        else:
            # Order cities west to east so each pair reads upstream -> downstream.
            names = sorted(results, key=lambda c: wa.CITIES[c][1])
            resid = pd.DataFrame({c: results[c].frame["residual"] for c in names}).dropna()
            corr = resid.corr().to_numpy(copy=True)
            np.fill_diagonal(corr, np.nan)  # self-correlation is always 1; hide it
            fig = go.Figure(go.Heatmap(
                z=corr, x=names, y=names, zmin=-1, zmax=1,
                colorscale=[[0, "#2a78d6"], [0.5, "#f0efec"], [1, "#e34948"]],
                text=[["" if np.isnan(v) else f"{round(v, 2) + 0.0:.2f}" for v in row] for row in corr],
                texttemplate="%{text}", hoverongaps=False,
                hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
            ))
            fig.update_layout(title="Same-day correlation of surprises", height=380,
                              yaxis=dict(autorange="reversed"),
                              margin=dict(l=10, r=10, t=50, b=10))
            c1, c2 = st.columns(2)
            c1.plotly_chart(fig, width="stretch")

            lags = np.arange(-5, 6)
            fig = go.Figure()
            pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
            pair_colors = ["#6250d6", "#eda100", "#e87ba4"]
            for (a, b), color in zip(pairs, pair_colors):
                cc = [resid[a].corr(resid[b].shift(-k)) for k in lags]
                fig.add_scatter(x=lags, y=cc, name=f"{a} → {b}", mode="lines+markers",
                                line=dict(color=color, width=2), marker=dict(size=8))
            fig.add_hline(y=0, line=dict(color=NEUTRAL, width=1))
            fig.update_layout(title="Lagged correlation of surprises", height=380,
                              xaxis_title="Lag k (days): first city today vs second city k days later",
                              margin=dict(l=10, r=10, t=50, b=10), hovermode="x unified")
            c2.plotly_chart(fig, width="stretch")
            st.caption(
                "Pairs read west → east. Weather in the US mostly moves west to east, so if Dallas's "
                "surprise today tends to show up in New York a day or two later, the Dallas → New York "
                "line peaks at a positive lag. That would mean part of New York's 'unpredictable' "
                "weather can be predicted from upstream cities."
            )

    # --- Biggest surprises ---------------------------------------------------
    with tab["Biggest surprises"]:
        n = st.slider("How many", 5, 50, 10)
        cols = st.columns(len(results))
        for col, (city, d) in zip(cols, results.items()):
            col.markdown(f"**{city}**")
            col.dataframe(wa.biggest_surprises(d, n).set_index("date"), width="stretch")
        st.caption(f"Actual vs predicted {var.short} in {U}."
                   + (" A big negative surprise is often a strong cold front; a big positive one, "
                      "a heat surge or offshore wind event." if IS_TEMP else ""))

    # --- Method --------------------------------------------------------------
    with tab["Method"]:
        st.markdown(f"""
    **Model**, fitted separately for each city:

    1. **Seasonal cycle + trend.** Least-squares fit of a constant, a linear trend
       ({'on' if use_trend else 'off'}) and {n_harmonics} annual harmonic pair(s)
       (sin/cos of 1…{n_harmonics} cycles per year). This is the "normal" for each date.
    2. **Anomaly** = actual − seasonal normal.
    3. **Persistence.** An autoregressive model AR(p) predicts today's anomaly from the previous
       p days' anomalies. BIC picks p, up to {max_lag}.
    4. **Unpredictable residual** = actual − (seasonal normal + persistence prediction).

    **Out-of-sample testing.** Every coefficient above (seasonal shape, trend, AR order and AR
    weights) is estimated from {first_tr['start']:%Y}–{test_from - 1} only. The model is then
    applied unchanged to {test_from} onward, so test-period residuals are honest prediction errors.
    The headline numbers, variance shares, *Distribution* and *Memory* tabs use the test period.
    *By month* and *Biggest surprises* use all years, so the pattern has more data behind it.

    Errors are measured as root-mean-square (RMSE) around zero, so a systematic bias in the test
    years counts against the model. Variance shares are mean-square anomaly and residual divided
    by the variance of the daily values, all on the same days.

    **Forecast horizon.** From each day t, the AR model is run forward recursively (its own
    forecasts become the inputs for the next step) to predict days t + 1 … t + N. Only data up to
    day t is used. Skill = 1 − MSE(forecast) ÷ MSE(seasonal normal), scored on test-period
    target days. The *useful forecast range* is the last lead with skill ≥ 10%, a rule-of-thumb
    cutoff. The *naive* baseline assumes today's anomaly stays the same.

    **Caveats**
    - The linear trend fitted on the training years is extended into the test years. If the
      trend speeds up or slows down, that shows up as test-period bias.
    - "Unpredictable" means unpredictable from this city's own {var.short} history. Real
      forecasts also use pressure, wind, humidity and upstream observations, so they do much
      better, especially 1–3 days ahead. The *Across cities* tab shows a little of that.
    - NASA POWER meteorology comes from the MERRA-2 reanalysis on a grid of about 0.5° × 0.625°
      (~50 km); solar values come from satellite products. Each value is an average over a grid
      cell, not a single weather station. That smooths things out, especially in San Francisco,
      where the cell mixes ocean, bay and inland air.
    - Most recent days can be missing while NASA processes them; they're dropped automatically.
    """)

    # --------------------------------------------------------------------------
    # Download
    # --------------------------------------------------------------------------
    out = pd.concat([d.frame.assign(city=c) for c, d in results.items()])
    st.download_button(
        "Download daily breakdown (CSV)",
        out.to_csv(index_label="date").encode(),
        file_name=f"predictability_{var.code}_{ctx.system}.csv",
        mime="text/csv",
    )
