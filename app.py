# ==========================================================
# Ontario Inpatient Hospital Demand Dashboard
# Structural Break • Demographics • Post-COVID Forecast
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ----------------------------------------------------------
# Page Setup
# ----------------------------------------------------------

st.set_page_config(
    page_title="Ontario Hospital Demand Dashboard",
    layout="wide"
)

st.title("Ontario Inpatient Hospital Demand Dashboard")
st.markdown("Exploring structural disruption, demographic drivers, and future demand.")
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
# Prepare Ontario Dataset
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
# Structural Break Split
# ----------------------------------------------------------

train = ontario_yearly[ontario_yearly["Year"] < 2020]
test = ontario_yearly[ontario_yearly["Year"] >= 2020]

# ----------------------------------------------------------
# ETS Model
# ----------------------------------------------------------

model = ExponentialSmoothing(
    train["Number_of_Discharges"],
    trend="add",
    seasonal=None
).fit()

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

# ----------------------------------------------------------
# Tabs
# ----------------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "Structural Break",
    "Demographic Analysis",
    "Future After COVID-19"
])

# ==========================================================
# TAB 1 – Structural Break
# ==========================================================

with tab1:

    st.header("COVID-19 Structural Break")

    counterfactual = model.forecast(len(test))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=ontario_yearly["Year"],
        y=ontario_yearly["Number_of_Discharges"],
        mode="lines+markers",
        name="Actual Discharges"
    ))

    fig.add_trace(go.Scatter(
        x=test["Year"],
        y=counterfactual,
        mode="lines",
        line=dict(dash="dash"),
        name="Expected Trend (No COVID)"
    ))

    fig.add_vline(
        x=2020,
        line_dash="dash",
        line_color="red",
        annotation_text="COVID Disruption"
    )

    fig.update_layout(
        template="simple_white",
        xaxis_title="Year",
        yaxis_title="Hospital Discharges"
    )

    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# TAB 2 – Demographic Analysis
# ==========================================================

with tab2:

    st.header("Demographic Demand Patterns")

    age_group = st.selectbox(
        "Age Group",
        sorted(df["Age_Group"].unique())
    )

    sex = st.selectbox(
        "Sex",
        sorted(df["Sex"].unique())
    )

    filtered = df[
        (df["Province_Territory"] == "Ontario") &
        (df["Age_Group"] == age_group) &
        (df["Sex"] == sex)
    ]

    demo_yearly = (
        filtered.groupby("Fiscal_Year")["Number_of_Discharges"]
        .sum()
        .reset_index()
    )

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=demo_yearly["Fiscal_Year"],
        y=demo_yearly["Number_of_Discharges"],
        mode="lines+markers"
    ))

    fig2.update_layout(
        template="simple_white",
        xaxis_title="Fiscal Year",
        yaxis_title="Discharges"
    )

    st.plotly_chart(fig2, use_container_width=True)

# ==========================================================
# TAB 3 – Future After COVID-19
# ==========================================================

with tab3:

    st.header("Post-COVID Demand Forecast")

    latest = ontario_yearly.iloc[-1]["Number_of_Discharges"]
    future = forecast_df.iloc[-1]["Forecast"]

    growth = ((future - latest) / latest) * 100

    col1, col2, col3 = st.columns(3)

    col1.metric("Latest Observed Discharges", f"{latest:,.0f}")
    col2.metric("Forecast (5 Years)", f"{future:,.0f}")
    col3.metric("Projected Growth", f"{growth:.2f}%")

    fig3 = go.Figure()

    fig3.add_trace(go.Scatter(
        x=ontario_yearly["Year"],
        y=ontario_yearly["Number_of_Discharges"],
        mode="lines+markers",
        name="Historical"
    ))

    fig3.add_trace(go.Scatter(
        x=forecast_df["Year"],
        y=forecast_df["Forecast"],
        mode="lines",
        line=dict(dash="dash"),
        name="Forecast"
    ))

    fig3.update_layout(
        template="simple_white",
        xaxis_title="Year",
        yaxis_title="Projected Discharges"
    )

    st.plotly_chart(fig3, use_container_width=True)

st.divider()

st.markdown("""
Model Notes

• ETS model trained on pre-COVID demand  
• Structural break analysis compares observed vs expected trend  
• Forecast represents post-COVID baseline demand projection  
""")
