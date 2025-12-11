# app.py
import streamlit as st
import numpy as np
import pandas as pd

from Categorization.categorizer import Categorization
import Forecasting.Model as md
from Forecasting.conversion import city

# Categorization(text)

# ---------- Sidebar controls ----------
st.sidebar.title("Settings")

# City & currency for forecast
city_name = st.sidebar.text_input("City", value="Boston")
country_name = st.sidebar.text_input("Country", value="United States")
output_currency = st.sidebar.selectbox("Output currency", ["INR", "USD", "AED"], index=2)
model_currency = "INR"

# ---------- Expense categorizer ----------
st.title("Expense Categorizer & Forecaster")

st.header("Single expense categorization")
user_text = st.text_input("Enter expense description", value="Uber ride last night")

if st.button("Predict category"):
    category, probs = predict_category(user_text, return_probs=True)
    st.success(f"Predicted category: {category}")
    st.write("Raw probabilities (per class index):")
    st.write(np.round(probs, 3))

st.header("Batch expense categorization")
st.caption("Enter one expense per line.")

batch_input = st.text_area(
    "Expenses",
    value="Dominos pizza\nElectricity bill\nZomato order, paneer tikka",
    height=150,
)

if st.button("Predict categories for batch"):
    lines = [x.strip() for x in batch_input.split("\n") if x.strip()]
    if lines:
        cats = predict_bulk(lines)
        df = pd.DataFrame({"expense": lines, "category": cats})
        st.dataframe(df)
    else:
        st.warning("Please enter at least one expense description.")

# ---------- Forecasting section ----------
st.header("Expense forecasting demo")

if st.button("Run forecast"):
    scale = city(city_name=city_name, country_name=country_name, output_currency=output_currency)

    # Your existing model API
    user_expenses, user_dates = md.Generate_sample()
    avg = md.Predict()
    forecast_mumbai_inr = np.array(avg)
    forecast_city_inr = forecast_mumbai_inr * scale

    # Simple table view
    df_forecast = pd.DataFrame(
        {
            "date": user_dates[: len(forecast_city_inr)],
            f"forecast_{city_name}_{model_currency}": forecast_mumbai_inr,
            f"forecast_{city_name}_{output_currency}": forecast_city_inr,
        }
    )
    st.subheader("Forecast table")
    st.dataframe(df_forecast)

    # Basic line chart in Streamlit
    st.subheader("Forecast plot")
    st.line_chart(df_forecast.set_index("date"))
