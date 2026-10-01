import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved models
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

# App title
st.title("Customer Segmentation App")

st.write("Enter customer details to predict the customer segment.")

# Customer inputs
age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

income = st.number_input(
    "Income",
    min_value=0,
    max_value=200000,
    value=50000
)

total_spending = st.number_input(
    "Total spending (sum of purchases)",
    min_value=0,
    max_value=5000,
    value=1000
)

num_web_purchases = st.number_input(
    "Number of web purchases",
    min_value=0,
    max_value=100,
    value=10
)

num_store_purchases = st.number_input(
    "Number of store purchases",
    min_value=0,
    max_value=100,
    value=10
)

num_web_visits = st.number_input(
    "Number of web visits",
    min_value=0,
    max_value=50,
    value=3
)

recency = st.number_input(
    "Recency (since last purchase)",
    min_value=0,
    max_value=365,
    value=30
)

# Create input DataFrame
input_data = pd.DataFrame({
    "Age": [age],
    "Income": [income],
    "Total_Spending": [total_spending],
    "NumWebPurchases": [num_web_purchases],
    "NumStorePurchases": [num_store_purchases],
    "NumWebVisitsMonth": [num_web_visits],
    "Recency": [recency]
})

# Scale the input
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict Segment"):

    cluster = kmeans.predict(input_scaled)[0]

    st.success(f"Predicted Segment: Cluster {cluster}")

    st.subheader("Customer Segments")

  