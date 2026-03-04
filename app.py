# ==========================================================
# Ontario Inpatient Hospital Demand Dashboard
# Historical Analysis + Predictive Forecast (2025+)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ----------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------

st.set_page_config(
    page_title="Ontario Inpatient Demand Forecast",
    layout="wide"
)

st.title("Ontario Inpatient Hospital Demand Forecasting Dashboard")
st.markdown(
    "Interactive planning tool combining historical analysis (1995–2024) and predictive forecasts (2025+)."
)
st.divider()

# ----------------------------------------------------------
# Load Data
# ----------------------------------------------------------

@st.cache_data
def load_data():
    hist = pd.read_csv("data/canada_hospital_inpatient_clean_1995_2024.csv")
    forecast = pd.read_csv("data/ontario_forecast_2025_2029.csv")
    return hist, forecast

df, forecast_df = load_data()

# ----------------------------------------------------------
# Prepare Ontario Historical Series
# ----------------------------------------------------------

ontario = df[
    (df["Province_Territory"] == "Ontario") &
    (df["Sex"] == "All")
]

ontario_yearly = (
    ontario.groupby("Fiscal_Year")["Number_of_Discharges"]
    .sum()
    .reset_index()
)

ontario_yearly["Year"] = (
    ontario_yearly["Fiscal_Year"]
    .astype(str)
    .str[:4]
    .astype(int)
)

ontario_yearly = ontario_yearly.sort_values("Year")

# ----------------------------------------------------------
# Tabs
# ----------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "Forecast Overview",
    "Scenario Analysis",
    "Demographic Analysis",
    "Structural Break Analysis"
])

# ==========================================================
# TAB 1 — Forecast Overview
# ==========================================================

with tab1:

    st.header("Historical Demand and Predictive Forecast")

    latest_actual = ontario_yearly.iloc[-1]["Number_of_Discharges"]
    final_forecast = forecast_df.iloc[-1]["Forecast"]

    growth_pct = ((final_forecast - latest_actual) / latest_actual) * 100

    col1, col2, col3 = st.columns(3)

    col1.metric("Latest Observed Discharges (2024)", f"{latest_actual:,.0f}")
    col2.metric("Projected Discharges (Final Forecast Year)", f"{final_forecast:,.0f}")
    col3.metric("Projected Growth (%)", f"{growth_pct:.2f}%")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=ontario_yearly["Year"],
        y=ontario_yearly["Number_of_Discharges"],
        mode="lines+markers",
        name="Historical"
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df["Year"],
        y=forecast_df["Forecast"],
        mode="lines+markers",
        line=dict(dash="dash"),
        name="Forecast (2025+)"
    ))

    fig.add_vline(
        x=2020,
        line_dash="dash",
        line_color="red",
        annotation_text="COVID Structural Break"
    )

    fig.update_layout(
        template="simple_white",
        xaxis_title="Year",
        yaxis_title="Number of Discharges"
    )

    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# TAB 2 — Scenario Analysis
# ==========================================================

with tab2:

    st.header("Alternative Growth Scenario")

    st.caption(
        "Scenario assumes a constant alternative annual growth rate applied to the forecast baseline."
    )

    adjustment = st.slider(
        "Alternative Annual Growth Rate (%)",
        -2.0, 2.0, 0.0, 0.1
    )

    scenario_df = forecast_df.copy()

    baseline_start = forecast_df["Forecast"].iloc[0]
    growth_rate = adjustment / 100

    scenario_values = []
    value = baseline_start

    for _ in range(len(forecast_df)):
        value = value * (1 + growth_rate)
        scenario_values.append(value)

    scenario_df["Scenario"] = scenario_values

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=forecast_df["Year"],
        y=forecast_df["Forecast"],
        mode="lines+markers",
        name="Baseline Forecast"
    ))

    fig2.add_trace(go.Scatter(
        x=scenario_df["Year"],
        y=scenario_df["Scenario"],
        mode="lines+markers",
        name="Alternative Scenario"
    ))

    fig2.update_layout(
        template="simple_white",
        xaxis_title="Year",
        yaxis_title="Projected Discharges"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.download_button(
        "Download Forecast Data",
        data=forecast_df.to_csv(index=False),
        file_name="ontario_forecast.csv",
        mime="text/csv"
    )

# ==========================================================
# TAB 3 — Demographic Analysis
# ==========================================================

with tab3:

    st.header("Ontario Demographic Breakdown")

    selected_age = st.selectbox(
        "Select Age Group",
        sorted(df["Age_Group"].unique())
    )

    selected_sex = st.selectbox(
        "Select Sex",
        sorted(df["Sex"].unique())
    )

    filtered = df[
        (df["Province_Territory"] == "Ontario") &
        (df["Age_Group"] == selected_age) &
        (df["Sex"] == selected_sex)
    ]

    demo_yearly = (
        filtered.groupby("Fiscal_Year")["Number_of_Discharges"]
        .sum()
        .reset_index()
    )

    fig3 = go.Figure()

    fig3.add_trace(go.Scatter(
        x=demo_yearly["Fiscal_Year"],
        y=demo_yearly["Number_of_Discharges"],
        mode="lines+markers"
    ))

    fig3.update_layout(
        template="simple_white",
        xaxis_title="Fiscal Year",
        yaxis_title="Discharges"
    )

    st.plotly_chart(fig3, use_container_width=True)

    st.download_button(
        "Download Demographic Data",
        data=demo_yearly.to_csv(index=False),
        file_name="ontario_demographics.csv",
        mime="text/csv"
    )

# ==========================================================
# TAB 4 — Structural Break Analysis
# ==========================================================

with tab4:

    st.header("COVID Structural Disruption")

    fig4 = go.Figure()

    fig4.add_trace(go.Scatter(
        x=ontario_yearly["Year"],
        y=ontario_yearly["Number_of_Discharges"],
        mode="lines+markers",
        name="Observed Demand"
    ))

    fig4.add_vline(
        x=2020,
        line_dash="dash",
        line_color="red",
        annotation_text="COVID Impact"
    )

    fig4.update_layout(
        template="simple_white",
        xaxis_title="Year",
        yaxis_title="Number of Discharges"
    )

    st.plotly_chart(fig4, use_container_width=True)

st.divider()

st.markdown(
"""
**Model Context**

- Historical inpatient discharge data (1995–2024)
- Predictive forecasting performed in modelling notebooks
- Dashboard visualises forecasts and planning scenarios
- Designed for medium-term healthcare capacity planning
"""
)

