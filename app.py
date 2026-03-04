# ==========================================================
# Ontario Inpatient Hospital Demand Dashboard
# Time Series Forecasting & Structural Break Analysis
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_percentage_error

# ----------------------------------------------------------
# Page Setup
# ----------------------------------------------------------

st.set_page_config(
    page_title="Ontario Inpatient Demand Dashboard",
    layout="wide"
)

st.title("Ontario Inpatient Hospital Demand Dashboard")
st.markdown("Forecasting and structural break analysis of inpatient hospital demand.")
st.divider()

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("data/canada_hospital_inpatient_clean_1995_2024.csv")
    return df

df = load_data()

# ----------------------------------------------------------
# Prepare Ontario Data
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
# Train / Test Split (Structural Break Handling)
# ----------------------------------------------------------

train = ontario_yearly[ontario_yearly["Year"] < 2020]
test = ontario_yearly[ontario_yearly["Year"] >= 2020]

# ----------------------------------------------------------
# Fit ETS Model
# ----------------------------------------------------------

model = ExponentialSmoothing(
    train["Number_of_Discharges"],
    trend="add",
    seasonal=None
).fit()

# ----------------------------------------------------------
# Backtest (COVID period)
# ----------------------------------------------------------

if not test.empty:

    test_forecast = model.forecast(len(test))

    mape = mean_absolute_percentage_error(
        test["Number_of_Discharges"],
        test_forecast
    ) * 100

else:
    mape = None

# ----------------------------------------------------------
# Forecast Future Demand
# ----------------------------------------------------------

forecast_horizon = 5

future_forecast = model.forecast(forecast_horizon)

forecast_start = ontario_yearly["Year"].max() + 1

future_years = np.arange(
    forecast_start,
    forecast_start + forecast_horizon
)

forecast_df = pd.DataFrame({
    "Year": future_years,
    "Forecast": future_forecast
})

# Confidence Interval

residual_std = np.std(model.resid)

forecast_df["Upper"] = forecast_df["Forecast"] + 1.96 * residual_std
forecast_df["Lower"] = forecast_df["Forecast"] - 1.96 * residual_std

# ----------------------------------------------------------
# Tabs
# ----------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "Forecast Overview",
    "Planning Scenarios",
    "Demographic Analysis",
    "Structural Break Analysis"
])

# ==========================================================
# TAB 1 – Forecast Overview
# ==========================================================

with tab1:

    st.header("Baseline Forecast")

    latest_actual = train.iloc[-1]["Number_of_Discharges"]
    final_forecast = forecast_df.iloc[-1]["Forecast"]

    growth_pct = ((final_forecast - latest_actual) / latest_actual) * 100

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Latest Pre-COVID Discharges",
        f"{latest_actual:,.0f}"
    )

    col2.metric(
        "5-Year Forecast",
        f"{final_forecast:,.0f}"
    )

    col3.metric(
        "Projected Growth",
        f"{growth_pct:.2f}%"
    )

    if mape:
        st.metric("Backtest MAPE", f"{mape:.2f}%")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=train["Year"],
        y=train["Number_of_Discharges"],
        mode="lines+markers",
        name="Pre-COVID Trend"
    ))

    fig.add_trace(go.Scatter(
        x=test["Year"],
        y=test["Number_of_Discharges"],
        mode="lines+markers",
        name="COVID Period"
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df["Year"],
        y=forecast_df["Forecast"],
        mode="lines",
        line=dict(dash="dash"),
        name="Forecast"
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df["Year"],
        y=forecast_df["Upper"],
        line=dict(color="lightgrey"),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df["Year"],
        y=forecast_df["Lower"],
        fill="tonexty",
        line=dict(color="lightgrey"),
        name="Confidence Interval"
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
        yaxis_title="Hospital Discharges"
    )

    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# TAB 2 – Planning Scenarios
# ==========================================================

with tab2:

    st.header("Demand Growth Scenario")

    adjustment = st.slider(
        "Alternative Annual Growth Rate (%)",
        -2.0, 2.0, 0.0, 0.1
    )

    scenario_df = forecast_df.copy()

    growth_rate = adjustment / 100

    baseline_start = forecast_df["Forecast"].iloc[0]

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
        name="Baseline"
    ))

    fig2.add_trace(go.Scatter(
        x=scenario_df["Year"],
        y=scenario_df["Scenario"],
        mode="lines+markers",
        name="Scenario"
    ))

    fig2.update_layout(
        template="simple_white",
        xaxis_title="Year",
        yaxis_title="Projected Discharges"
    )

    st.plotly_chart(fig2, use_container_width=True)

# ==========================================================
# TAB 3 – Demographic Analysis
# ==========================================================

with tab3:

    st.header("Demographic Trends")

    age = st.selectbox(
        "Age Group",
        sorted(df["Age_Group"].unique())
    )

    sex = st.selectbox(
        "Sex",
        sorted(df["Sex"].unique())
    )

    filtered = df[
        (df["Province_Territory"] == "Ontario") &
        (df["Age_Group"] == age) &
        (df["Sex"] == sex)
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
        xaxis_title="Year",
        yaxis_title="Discharges"
    )

    st.plotly_chart(fig3, use_container_width=True)

# ==========================================================
# TAB 4 – Structural Break
# ==========================================================

with tab4:

    st.header("COVID Structural Break")

    counterfactual = model.forecast(len(test))

    fig4 = go.Figure()

    fig4.add_trace(go.Scatter(
        x=ontario_yearly["Year"],
        y=ontario_yearly["Number_of_Discharges"],
        mode="lines+markers",
        name="Actual"
    ))

    fig4.add_trace(go.Scatter(
        x=test["Year"],
        y=counterfactual,
        mode="lines",
        line=dict(dash="dash"),
        name="Counterfactual"
    ))

    fig4.update_layout(
        template="simple_white",
        xaxis_title="Year",
        yaxis_title="Discharges"
    )

    st.plotly_chart(fig4, use_container_width=True)

st.divider()

st.markdown("""
Model Notes

• ETS model with additive trend  
• COVID years excluded from training  
• Counterfactual analysis used to quantify disruption  
• Designed for medium-term healthcare planning
""")
