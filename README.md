# Ontario Inpatient Hospital Demand Forecasting (Canada)

This project analyses long-term inpatient hospital discharge trends in Canada, with a focused case study on Ontario. The objective is to understand historical demand patterns, quantify the disruption caused by COVID-19, and produce a baseline forecast to support medium-term healthcare planning.

The analysis combines time series modelling, structural break analysis, and an interactive dashboard to explore hospital utilisation trends.

-----
## Project objective

The project aims to:

Understand historical inpatient demand patterns in Ontario

Identify structural disruptions caused by the COVID-19 pandemic

Develop a baseline demand forecast using time series modelling

Provide an interactive dashboard for exploratory analysis and planning discussion

-----

## Data

Source: Canadian hospital inpatient discharge data

Coverage: 1995 – 2024

Frequency: Annual

Geographic scope:

Canada (national context)

Ontario (primary case study)

The dataset includes demographic breakdowns such as age group and sex, enabling further analysis of demand drivers.

-----
## Analytical Approach

The project follows a structured analytical workflow.


## Data Cleaning

The raw dataset was cleaned and structured to ensure consistency in:

fiscal year representation

demographic categories

discharge counts

Missing values and formatting inconsistencies were resolved prior to analysis

-----

## Exploratory Data Analysis

Exploratory analysis was conducted to examine:

long-term demand trends

stability of discharge volumes

distribution across demographic groups

potential structural changes over time

This stage helped identify the significant disruption caused by the COVID-19 pandemic

------

## Structural Break Analysis

The COVID-19 pandemic represents a clear structural break in hospital utilisation patterns.

To quantify this disruption, a counterfactual approach was used:

A time series model was trained on pre-COVID data (1995–2019).

The model generated expected discharge volumes for the pandemic period.

Observed values were compared with the expected trend.

This allows the deviation introduced by COVID-19 to be visualised and measured.

-------

## Methodology

Data cleaning and validation

Exploratory Data Analysis to assess trends, concentration, and stability

ETS (Error–Trend–Seasonal) modelling with additive trend

Sensitivity analysis using a damped trend specification


##Forecasting Methodology

Future inpatient demand was projected using an Exponential Smoothing (ETS) model.

Model specification:

Additive trend

No seasonal component

Training period: 1995–2019

Excluding the COVID period prevents temporary disruptions from biasing the long-term trend estimate.

The model produces a five-year baseline forecast intended for planning interpretation rather than operational prediction.

-----
## Key findings

Ontario drives national inpatient demand trends

Pre-COVID demand follows a stable, gradual growth pattern

Post-COVID volumes deviate from baseline expectations

Forecast results are robust to alternative trend assumptions

---
## Output

Five-year baseline forecast for Ontario inpatient discharges

Planning-oriented interpretation, not operational prediction

------
## Intended use

This analysis is designed to support healthcare planning discussions, workforce considerations, and capacity assessment under normal operating conditions.
