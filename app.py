# ==========================================================
# Ontario Inpatient Hospital Demand Dashboard
# Analytical Forecasting & Structural Break Diagnostics
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_percentage_error

# ----------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------
st.set_page_config(
    page_title="Ontario Inpatient Planning Dashboard",
    layout="wide"
)

st.title("Ontario Inpatient Hospital Demand Dashboard")
st.markdown("Medium-term forecasting tool with structural break diagnostics.")
st.divider()

# ----------------------------------------------------------
# Load Data
# ----------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/canada_hospital_inpatient_clean_1995_2024.csv")
    return df

df = load_data()

# ----------------------------------------------------------
# Prepare Ontario Annual Series
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

ontario_yearly["Year"] = ontario_yearly["Fiscal_Year"].str[:4].astype(int)
ontario_yearly = ontario_yearly.sort_values("Year")

# ----------------------------------------------------------
# Train/Test Split (Structural Break Handling)
# ----------------------------------------------------------
train = ontario_yearly[ontario_yearly["Year"] < 2020]
test = ontario_yearly[ontario_yearly["Year"] >= 2020]

# ----------------------------------------------------------
# Fit ETS Model (Additive Trend)
# ----------------------------------------------------------
model = ExponentialSmoothing(
    train["Number_of_Discharges"],
    trend="add",
    seasonal=None
).fit()

# ----------------------------------------------------------
# Backtest on COVID Period
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
# 5-Year Forward Forecast
# ----------------------------------------------------------
forecast_horizon = 5
future_forecast = model.forecast(forecast_horizon)

forecast_start = train["Year"].max() + 1

future_years = np.arange(
    forecast_start,
    forecast_start + forecast_horizon
)

forecast_df = pd.DataFrame({
    "Year": future_years,
    "Forecast": future_forecast
})

# Approximate confidence interval
residual_std = np.std(model.resid)
forecast_df["Upper"] = forecast_df["Forecast"] + 1.96 * residual_std
forecast_df["Lower"] = forecast_df["Forecast"] - 1.96 * residual_std

# ----------------------------------------------------------
# Tabs
# ----------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Forecast Overview",
    "Scenario Analysis",
    "Demographic Analysis",
    "Structural Break & Diagnostics"
])

# ==========================================================
# TAB 1 — Forecast Overview
# ==========================================================
with tab1:

    st.header("Baseline ETS Forecast (Pre-COVID Training)")

    latest_actual = train.iloc[-1]["Number_of_Discharges"]
    final_forecast = forecast_df.iloc[-1]["Forecast"]
    growth_pct = ((final_forecast - latest_actual) / latest_actual) * 100

    col1, col2, col3 = st.columns(3)

    col1.metric("Latest Pre-COVID Discharges",
                f"{latest_actual:,.0f}")

    col2.metric("5-Year Forecast",
                f"{final_forecast:,.0f}")

    col3.metric("Projected Growth (%)",
                f"{growth_pct:.2f}%")

    if mape:
        st.metric("Backtest MAPE (COVID Period)",
                  f"{mape:.2f}%")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=train["Year"],
        y=train["Number_of_Discharges"],
        mode="lines+markers",
        name="Historical (Pre-COVID)"
    ))

    fig.add_trace(go.Scatter(
        x=test["Year"],
        y=test["Number_of_Discharges"],
        mode="lines+markers",
        name="Observed (COVID Period)"
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

    st.header("Growth Scenario Adjustment")

    adjustment = st.slider(
        "Adjust Annual Growth Rate (%)",
        -2.0, 2.0, 0.0, 0.1
    )

baseline_start = forecast_df["Forecast"].iloc[0]

growth_rate = adjustment / 100

scenario_values = []

value = baseline_start

for i in range(len(forecast_df)):
    value = value * (1 + growth_rate)
    scenario_values.append(value)

scenario_df["Scenario"] = scenario_values
    )

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=forecast_df["Year"],
        y=forecast_df["Forecast"],
        name="Baseline"
    ))

    fig2.add_trace(go.Scatter(
        x=scenario_df["Year"],
        y=scenario_df["Scenario"],
        name="Adjusted Scenario"
    ))

    fig2.update_layout(template="simple_white")

    st.plotly_chart(fig2, use_container_width=True)

    st.download_button(
        "Download Forecast Data (CSV)",
        data=forecast_df.to_csv(index=False),
        file_name="ontario_ets_forecast.csv",
        mime="text/csv"
    )

# ==========================================================
# TAB 3 — Demographic Breakdown
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

    fig3.update_layout(template="simple_white")

    st.plotly_chart(fig3, use_container_width=True)

    st.download_button(
        "Download Demographic Data (CSV)",
        data=demo_yearly.to_csv(index=False),
        file_name="ontario_demographic_data.csv",
        mime="text/csv"
    )

# ==========================================================
# TAB 4 — Structural Break & Diagnostics
# ==========================================================
with tab4:

    st.header("Structural Break Analysis & Diagnostics")

    # Counterfactual projection into COVID years
    if not test.empty:

        counterfactual = model.forecast(len(test))
        deviation = test["Number_of_Discharges"].values - counterfactual.values
        avg_deviation = deviation.mean()

        st.metric(
            "Average COVID Deviation from Trend",
            f"{avg_deviation:,.0f} discharges"
        )

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
            name="Counterfactual Trend"
        ))

        fig4.update_layout(template="simple_white")

        st.plotly_chart(fig4, use_container_width=True)

        fig5 = go.Figure()

        fig5.add_trace(go.Bar(
            x=test["Year"],
            y=deviation,
            name="Deviation"
        ))

        fig5.update_layout(template="simple_white")

        st.plotly_chart(fig5, use_container_width=True)

    # Residual diagnostics
    st.subheader("Training Residual Diagnostics")

    residuals = model.resid

    fig6 = go.Figure()

    fig6.add_trace(go.Scatter(
        x=train["Year"],
        y=residuals,
        mode="lines+markers"
    ))

    fig6.add_hline(y=0)

    fig6.update_layout(template="simple_white")

    st.plotly_chart(fig6, use_container_width=True)

st.divider()

st.markdown("""
Model Notes:
- ETS with additive trend
- COVID period excluded from training
- Counterfactual projection used to quantify structural break
- Designed for medium-term planning interpretation
""")

