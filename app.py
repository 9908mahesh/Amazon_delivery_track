import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

# Load model
model = joblib.load("lightgbm_tuned.pkl")

st.set_page_config(page_title="Amazon Delivery Time Predictor", layout="centered")

# Title
st.title("📦 Amazon Delivery Time Prediction")
st.write("Predict delivery time (in minutes) based on order details.")

# Input form
with st.form("input_form"):
    st.subheader("Enter Order Details")

    distance = st.number_input("Distance (km)", min_value=0.0, step=0.1)
    pickup_delay = st.number_input("Pickup Delay (minutes)", min_value=0.0, step=1.0)
    order_hour = st.slider("Order Hour (0-23)", 0, 23, 12)
    order_day = st.slider("Order Day (1-31)", 1, 31, 15)
    order_weekday = st.selectbox("Order Weekday", [0,1,2,3,4,5,6], format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
    is_peak = st.checkbox("Peak Hour?")
    is_weekend = st.checkbox("Weekend?")
    weather = st.selectbox("Weather", [0,1,2,3])   # encode same as training
    traffic = st.selectbox("Traffic", [0,1,2,3])
    vehicle = st.selectbox("Vehicle", [0,1,2])
    area = st.selectbox("Area", [0,1,2])
    category = st.selectbox("Category", [0,1,2])

    submitted = st.form_submit_button("Predict")

if submitted:
    # Prepare input dataframe
    input_data = pd.DataFrame([[
        distance, pickup_delay, order_hour, order_day, order_weekday,
        is_peak, is_weekend, 
        np.digitize(distance, [0,2,5,10,20,50]),   # Distance_Bucket
        traffic*distance,                          # Traffic_Distance
        weather*pickup_delay,                      # Weather_Delay
        weather, traffic, vehicle, area, category
    ]], columns=[
        "Distance_km","Pickup_Delay","Order_Hour","Order_Day","Order_Weekday",
        "Is_Peak_Hour","Is_Weekend","Distance_Bucket","Traffic_Distance","Weather_Delay",
        "Weather","Traffic","Vehicle","Area","Category"
    ])

    # Prediction
    prediction = model.predict(input_data)[0]
    st.success(f"⏱️ Predicted Delivery Time: {prediction:.2f} minutes")
