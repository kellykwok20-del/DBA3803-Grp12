"""Streamlit presentation of the notebook's saved aggregate results."""

from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from notebook_data import NOTEBOOK, load_results


st.set_page_config(page_title="Healthcare Affordability Dashboard", page_icon="✚",
                   layout="wide", initial_sidebar_state="expanded")
st.html("""
<style>
.block-container{max-width:1400px;padding-top:2.3rem;padding-bottom:3rem}
h1{letter-spacing:-1.6px;font-weight:650!important}h2,h3{letter-spacing:-.5px}
[data-testid="stMetric"]{background:#fff;border:1px solid #dce4df!important;
border-radius:15px;padding:19px!important;min-height:130px}
[data-testid="stMetricValue"]{color:#087c70;font-weight:650;letter-spacing:-1px}
[data-testid="stMetricLabel"]{color:#5c7074;font-size:.8rem}
.eyebrow{color:#087c70;font-size:11px;font-weight:700;letter-spacing:1.8px;text-transform:uppercase;margin:0 0 10px}
.intro{color:#5c7074;max-width:760px;font-size:15px;line-height:1.65;margin:0 0 22px}
.context{background:#e8efe9;border:1px solid #dce6dd;border-radius:11px;padding:13px 18px;
display:flex;gap:25px;flex-wrap:wrap;color:#52685f;font-size:12px;margin-bottom:20px}
.context strong{color:#172d32}
.sidebar-brand{font-size:20px;line-height:1.35;letter-spacing:-.5px;font-weight:650;color:#edf6f2;margin:0}
.sidebar-kicker{font-size:10px;letter-spacing:1.8px;color:#a0d9bf;margin:5px 0 24px}
.profile-heading{background:#e5efe7;border-radius:12px;padding:19px;margin-bottom:18px;
display:flex;justify-content:space-between;align-items:center;gap:15px}
.profile-heading.blue{background:#e9eef5}
.profile-heading h3{font-size:21px;margin:3px 0;color:#172d32}
.profile-heading small{color:#426b58;font-size:10px;letter-spacing:1px;text-transform:uppercase}
.profile-share{font-size:29px;color:#172d32;text-align:right;line-height:1.3}
.profile-share span{display:block;font-size:10px;color:#5c7074}
.footer{border-top:1px solid #dce4df;padding-top:15px;color:#5c7074;font-size:11px;margin-top:30px}
@media(max-width:700px){.block-container{padding:1.3rem 1rem}.context{gap:10px 18px}}
</style>
""")

NAMES = {"rf": "Random Forest", "lr": "Logistic regression",
         "insurance": "Insurance only", "random": "Random selection"}
COLOURS = {"rf": "#087c70", "lr": "#668ca5", "insurance": "#ab7a37", "random": "#9caea2"}
PREDICTORS = {"HINOTCOVE": "Insurance coverage", "HEALTH": "Self-reported health",
              "AGE": "Age", "POVERTY": "Income / poverty category", "USUALPL": "Usual source of care",
              "SEX": "Sex", "EMPSTAT": "Employment", "HISPRACE": "Race / ethnicity",
              "EDUC": "Education", "REGION": "Region"}


@st.cache_data(show_spinner=False)
def read_saved_results(notebook_bytes):
    # Saving new notebook contents invalidates this cache, without retraining.
    return load_results(notebook_bytes)


def chart_style(fig, height=330):
    fig.update_layout(template="plotly_white", height=height,
                      font=dict(family="Arial, sans-serif", size=12, color="#52685f"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=5, r=15, t=15, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0, font_size=11),
                      hoverlabel=dict(bgcolor="white", font_color="#172d32"))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#e2e9e3", zeroline=False)
    return fig


def show_chart(fig, key):
    st.plotly_chart(fig, width="stretch", theme=None, key=key,
                    config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def capacity_frame(data, fraction):
    return pd.DataFrame([r for r in data["capacities"] if abs(r["capacity"] - fraction) < 1e-8])


def comparison_table(frame):
    columns = {"id": "Method", "contacts": "Contacts", "cases_found": "Cases found",
               "missed_cases": "Cases missed", "false_positives": "False positives",
               "precision": "Precision", "recall": "Recall", "lift": "Lift"}
    table = frame[list(columns)].rename(columns=columns).copy()
    table["Method"] = table["Method"].map(NAMES)
    table[["Precision", "Recall"]] *= 100
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        **{key: st.column_config.NumberColumn(format="localized") for key in
           ["Contacts", "Cases found", "Cases missed", "False positives"]},
        "Precision": st.column_config.NumberColumn(format="%.1f%%"),
        "Recall": st.column_config.NumberColumn(format="%.1f%%"),
        "Lift": st.column_config.NumberColumn(format="%.2f×"),
    })


def overview(data, frame, model, fraction):
    row = frame.loc[frame["id"].eq(model)].iloc[0]
    total = data["sample"]["reported_cases"]
    st.subheader("Make every outreach contact count")
    st.caption(f"{NAMES[model]} · Contacting {fraction:.0%} of adults = {int(row['contacts']):,} contacts")
    values = [("Cases found", f"{int(row['cases_found']):,}", "Contacted adults who reported this outcome."),
              ("Recall", f"{row['recall']:.1%}", "Share of all reported cases found."),
              ("Precision", f"{row['precision']:.1%}", "Share of contacted adults who reported this outcome."),
              ("Lift over random", f"{row['lift']:.2f}×", "Case yield divided by expected random-selection yield.")]
    for column, (label, value, help_text) in zip(st.columns(4), values):
        column.metric(label, value, help=help_text, border=True)
    st.write("")
    left, right = st.columns([1.25, 1])
    with left, st.container(border=True):
        st.markdown("#### More outreach, more cases reached")
        st.caption("Recall across the evaluated contact capacities")
        fig = go.Figure()
        for key in ["rf", "lr"]:
            rows = sorted([r for r in data["capacities"] if r["id"] == key], key=lambda r: r["capacity"])
            fig.add_trace(go.Scatter(x=[0] + [r["capacity"] for r in rows],
                                     y=[0] + [r["recall"] for r in rows], name=NAMES[key], mode="lines+markers",
                                     line=dict(color=COLOURS[key], width=3 if key == model else 2,
                                               dash="solid" if key == "rf" else "dot"),
                                     marker=dict(size=5),
                                     hovertemplate="%{x:.0%} capacity<br>%{y:.1%} recall<extra>%{fullData.name}</extra>"))
        maximum = max(data["fractions"])
        fig.add_trace(go.Scatter(x=[0, maximum], y=[0, maximum], name="Random expectation", mode="lines",
                                 line=dict(color="#9caea2", width=1.5, dash="dash"),
                                 hovertemplate="%{x:.0%} capacity<br>%{y:.1%} expected recall<extra></extra>"))
        fig.add_vline(x=fraction, line_width=1, line_dash="dash", line_color="#a2b7aa")
        fig.update_xaxes(title="Share of adults contacted", tickformat=".0%", range=[0, maximum + .01])
        fig.update_yaxes(title="Reported cases found", tickformat=".0%", range=[0, 1])
        show_chart(chart_style(fig), "capacity-gains")
    with right, st.container(border=True):
        st.markdown("#### Same contacts, different results")
        st.caption(f"Cases found with {int(row['contacts']):,} contacts")
        fig = go.Figure(go.Bar(x=frame["cases_found"], y=frame["id"].map(NAMES), orientation="h",
                               marker_color=frame["id"].map(COLOURS), text=frame["cases_found"],
                               textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}<br>%{x:,} cases found<extra></extra>"))
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_xaxes(title="Reported cases found", range=[0, frame["cases_found"].max() * 1.18])
        show_chart(chart_style(fig), "same-capacity-bars")
    insurance = frame.loc[frame["id"].eq("insurance")].iloc[0]
    extra = int(row["cases_found"] - insurance["cases_found"])
    st.info(f"**{NAMES[model]} finds {abs(extra):,} {'more' if extra >= 0 else 'fewer'} cases than insurance-only targeting** "
            f"with the same {int(row['contacts']):,} contacts. It misses {int(row['missed_cases']):,} of "
            f"{total:,} reported cases and selects {int(row['false_positives']):,} adults who did not report this barrier.")
    expected = int(row["contacts"]) * total / data["sample"]["adults"]
    st.caption(f"Random outreach would find about {expected:,.0f} cases on average. "
               "The random-selection bar uses the notebook’s fixed random tie order.")
    with st.expander("See the exact counts"):
        comparison_table(frame)


def models(data, frame, fraction):
    st.subheader("Compare models at the same outreach capacity")
    st.caption(f"{fraction:.0%} capacity · Every method makes {int(frame.iloc[0]['contacts']):,} contacts")
    comparison_table(frame)
    st.caption("Random selection uses a fixed random tie order. Small model differences are descriptive; statistical significance has not been established.")
    st.caption(f"Selected forest: {data['selected_forest']}.")
    st.write("")
    left, right = st.columns([1.25, 1])
    with left, st.container(border=True):
        st.markdown("#### A broader, recall-focused outreach rule")
        st.caption("Separate scenario: each model’s cutoff was selected on 2024 to maximise F2.")
        for column, row in zip(st.columns(2), data["fixed"]):
            with column:
                st.markdown(f"**{NAMES[row['id']]}**")
                st.caption(f"Fixed cutoff: {row['cutoff']:.6f}")
                details = {"Adults flagged": f"{row['flagged']:,}", "Cases found": f"{row['cases_found']:,}",
                           "Cases missed": f"{row['missed_cases']:,}", "False positives": f"{row['false_positives']:,}",
                           "Recall": f"{row['cases_found']/data['sample']['reported_cases']:.1%}",
                           "Precision": f"{row['cases_found']/row['flagged']:.1%}", "F2": f"{row['F2']:.4f}"}
                st.table(pd.DataFrame(details.items(), columns=["Measure", "Result"]).set_index("Measure"))
        st.caption("F2 gives recall greater weight than precision. These cutoffs produce different workloads; use the capacity table for an equal-workload comparison.")
    with right, st.container(border=True):
        st.markdown("#### What helps the forest rank people?")
        st.caption("2024 validation · decrease in average precision when a predictor is shuffled")
        imp = pd.DataFrame(data["importance"]).sort_values("validation_AP_drop", ascending=False).head(5)
        fig = go.Figure(go.Bar(x=imp["validation_AP_drop"], y=imp["predictor"].map(PREDICTORS),
                               orientation="h", marker_color="#6c9d87", text=imp["validation_AP_drop"],
                               texttemplate="%{text:.4f}", textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}<br>AP decrease: %{x:.4f}<extra></extra>"))
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_xaxes(title="Decrease in average precision", range=[0, imp["validation_AP_drop"].max() * 1.25])
        show_chart(chart_style(fig, 310), "importance-chart")
        st.caption("A larger decrease means greater predictive importance. This does not establish causation.")
    with st.expander("Overall ranking and probability metrics"):
        overall = pd.DataFrame(data["overall"])
        overall["Model"] = overall["id"].map(NAMES)
        overall = overall[["Model", "AP", "ROC_AUC", "Brier"]].rename(columns={
            "AP": "Average precision ↑", "ROC_AUC": "ROC AUC ↑", "Brier": "Brier score ↓"})
        st.dataframe(overall.style.format({c: "{:.4f}" for c in overall.columns[1:]}), hide_index=True, width="stretch")
        st.caption("Average precision summarises precision–recall performance. ROC AUC measures discrimination. Lower Brier score means less squared probability error. Good ranking does not guarantee calibrated probabilities.")


def profiles(data):
    st.subheader("Understand who is on the priority list")
    pool_size = sum(row["adults"] for row in data["profiles"])
    st.info(f"**Fixed profile scenario:** the highest-ranked 20% under Random Forest in 2025 ({pool_size:,} adults). "
            "The sidebar controls do not change these profiles.")
    traits = {"income_below_2x_poverty": "Income below 2× poverty", "fair_or_poor_health": "Fair or poor health",
              "no_usual_place": "No usual source of care", "not_employed": "Not employed"}
    for index, (column, row) in enumerate(zip(st.columns(2), data["profiles"])):
        with column, st.container(border=True):
            description = "Younger adults" if index == 0 else "Older adults"
            st.html(f'<div class="profile-heading {"blue" if index else ""}"><div>'
                    f'<small>{escape(row["profile"])}</small><h3>{description}</h3></div>'
                    f'<div class="profile-share">{row["adults"]/pool_size:.1%}<span>of the priority pool</span></div></div>')
            metrics = st.columns(3)
            metrics[0].metric("Adults", f"{row['adults']:,}")
            metrics[1].metric("Median age", f"{row['median_age']:.0f}")
            metrics[2].metric("Uninsured", f"{row['uninsured']:.1%}")
            fig = go.Figure(go.Bar(x=[row[key] for key in traits], y=list(traits.values()), orientation="h",
                                   marker_color="#829bb4" if index else "#71a28a",
                                   text=[row[key] for key in traits], texttemplate="%{text:.1%}",
                                   textposition="outside", cliponaxis=False,
                                   hovertemplate="%{y}<br>%{x:.1%} of profile<extra></extra>"))
            fig.update_yaxes(autorange="reversed", showgrid=False)
            fig.update_xaxes(range=[0, 1], tickformat=".0%", title="Share within profile")
            show_chart(chart_style(fig, 260), f"profile-{index}")
            st.markdown("**Possible outreach focus**")
            st.write("Ask about current cost barriers and assistance pathways. Explore help connecting to a regular source of care."
                     if index == 0 else "Review cost concerns alongside existing coverage. Coordinate care navigation and follow-up with the current care team.")
    st.caption("Profiles were developed on 2024 and applied unchanged to 2025. They describe group averages, not every individual. Suggested support pathways are hypotheses, not proven intervention effects.")
    st.caption("Characteristic percentages use all adults in each profile as the denominator. Missing responses remain in that denominator but do not count as having the characteristic; a missing answer does not establish its absence.")


def about(data):
    st.subheader("From prediction to a practical decision")
    left, right = st.columns([1.1, 1])
    with left, st.container(border=True):
        st.markdown("#### Learn first. Evaluate on a later year.")
        for year, role in [("2023", "Initial training and five-fold cross-validation"),
                           ("2024", "Model and cutoff selection; population profiling"),
                           ("2023–2024", "Refit the selected models before opening 2025"),
                           ("2025", "Final evaluation with models and cutoffs fixed")]:
            st.markdown(f"**{year}** · {role}")
        st.markdown("**Target:** medical care needed but not received because of cost during the past 12 months, as reported in the survey.")
        st.markdown("**Practical decision:** choose a feasible contact capacity, prioritise the ranked list and assess current needs before recommending support.")
        st.caption(f"Selected forest: {data['selected_forest']}. Recall-focused cutoff: {data['cutoff']:.6f}.")
        st.caption(f"Conservative forest cutoff: {data['conservative_cutoff']:.6f}.")
        if data["reference_retained"]:
            st.write("The original one-hot reference was retained after validation selection. Saved ‘revised forest’ labels refer to this same retained specification; their F2 results match the original reference.")
    with right, st.container(border=True):
        st.markdown("#### Read the results with care")
        st.markdown("""
- Model metrics and contact counts describe the **unweighted evaluation sample**.
- Weighted population rates are descriptive and are shown separately below.
- False positives did not report this outcome; they may have other support needs.
- Group performance differs. Read the notebook’s subgroup diagnostics too.
- The forest’s calibration plot shows scores above observed rates. Use its scores for ranking rather than literal individual probabilities.
- 2025 was examined in earlier project work. Further independent evaluation is needed.
- The analysis does not establish future outcomes, provider-population performance, intervention effects or financial returns.
""")
    with st.expander("Survey-weighted prevalence by year"):
        rates = [{"Year": str(r["year"]), "Weighted rate": r["weighted_rate"]} for r in data["prevalence"]]
        rates.append({"Year": "2025", "Weighted rate": data["sample"]["weighted_prevalence"]})
        fig = go.Figure(go.Bar(x=[r["Year"] for r in rates], y=[r["Weighted rate"] for r in rates],
                               marker_color="#6c9d87", texttemplate="%{y:.2%}", textposition="outside"))
        fig.update_yaxes(tickformat=".0%", title="Reported cost-related unmet care")
        fig.update_xaxes(type="category")
        show_chart(chart_style(fig, 280), "weighted-prevalence")
        st.caption("Survey-weighted prevalence is a population description, not model recall or precision.")
    with st.expander("Source and consistency checks"):
        st.write(f"Source: `{data['source']}` · Results read {data['loaded_at']}.")
        st.caption("Save the notebook and refresh the app to load updated outputs without retraining any model.")
        if not data["execution_in_order"]:
            st.info("Execution counters are out of notebook order. The saved result tables pass the dashboard’s consistency checks, but this does not verify a clean sequential run. The dashboard displays the saved outputs without executing notebook cells.")
            st.caption("Saved execution counters: " + ", ".join(map(str, data["execution_counts"])))
        st.dataframe(pd.DataFrame(data["checks"]).rename(columns={"check": "Consistency check", "passed": "Passed"}), hide_index=True)
        st.caption("Notebook SHA-256")
        st.code(data["source_sha256"], language=None)
        st.caption("Profiles and overall metrics use the precision displayed in the saved tables. Capacity recall, precision and lift are derived from saved integer counts.")


def main():
    try:
        data = read_saved_results(NOTEBOOK.read_bytes())
    except (OSError, ValueError, KeyError, IndexError, StopIteration, AttributeError) as error:
        st.error("The notebook’s saved results could not be loaded.")
        st.info("Review the saved notebook results for missing tables, errors or conflicting values, then reload. The dashboard does not execute notebook cells.")
        with st.expander("Details"):
            st.code(str(error))
        st.stop()
    with st.sidebar:
        st.html('<p class="sidebar-brand">✚ Healthcare<br>Affordability Dashboard</p><p class="sidebar-kicker">OUTREACH ANALYTICS</p>')
        page = st.radio("Explore", ["Outreach overview", "Model comparison", "Population profiles", "About the analysis"], key="page")
        st.divider()
        st.caption("OUTREACH SCENARIO")
        capacity = st.select_slider("Share of adults contacted", options=[round(q * 100) for q in data["fractions"]],
                                    value=10, format_func=lambda n: f"{n}%", key="capacity")
        model = st.selectbox("Model in focus", ["rf", "lr"], format_func=NAMES.get, key="focus_model")
        st.caption("Controls update outreach and equal-capacity comparisons. Profiles use a fixed 20% scenario.")
        csv = pd.DataFrame(data["capacities"]).drop(columns="id").to_csv(index=False).encode("utf-8-sig")
        st.download_button("Export all capacity results", csv, "healthcare-outreach-2025.csv", "text/csv", width="stretch")
        st.button("Reload saved notebook", width="stretch", help="Read the notebook saved on disk. This never trains or changes a model.")
        st.divider()
        st.caption("DBA3803 · Group 12\n\nPredictive Analytics for Business Applications")
    st.html('<p class="eyebrow">Healthcare affordability · United States</p>')
    st.title("Healthcare Affordability Dashboard")
    st.html('<p class="intro">Explore how predictive analytics can prioritise outreach to adults who report cost-related unmet medical care.</p>')
    sample = data["sample"]
    st.html(f'<div class="context"><span><strong>2025</strong> out-of-time evaluation</span>'
            f'<span><strong>{sample["adults"]:,}</strong> adults</span>'
            f'<span><strong>{sample["reported_cases"]:,}</strong> reported cases</span>'
            f'<span><strong>{sample["reported_cases"]/sample["adults"]:.1%}</strong> sample prevalence</span>'
            '<span>Source: <strong>IPUMS NHIS</strong></span></div>')
    fraction = capacity / 100
    frame = capacity_frame(data, fraction)
    if page == "Outreach overview":
        overview(data, frame, model, fraction)
    elif page == "Model comparison":
        models(data, frame, fraction)
    elif page == "Population profiles":
        profiles(data)
    else:
        about(data)
    st.html('<div class="footer">Healthcare Affordability Dashboard · DBA3803 Group 12 · Saved notebook results · No model retraining</div>')


if __name__ == "__main__":
    main()
