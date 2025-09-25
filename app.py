import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and encoders
model = joblib.load("lightgbm_tuned.pkl")
feature_order = joblib.load("feature_order.pkl")

# Load categorical encoders
weather_enc = joblib.load("Weather_encoder.pkl")
traffic_enc = joblib.load("Traffic_encoder.pkl")
vehicle_enc = joblib.load("Vehicle_encoder.pkl")
area_enc = joblib.load("Area_encoder.pkl")
category_enc = joblib.load("Category_encoder.pkl")

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
    order_weekday = st.selectbox(
        "Order Weekday", [0,1,2,3,4,5,6],
        format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x]
    )
    is_peak = st.checkbox("Peak Hour?")
    is_weekend = st.checkbox("Weekend?")

    # NEW: Agent details
    agent_age = st.number_input("Agent Age", min_value=18, max_value=70, step=1)
    agent_rating = st.slider("Agent Rating", 1.0, 5.0, 4.5, step=0.1)

    # Raw category selections (match training dataset categories)
    weather = st.selectbox("Weather", weather_enc.classes_)
    traffic = st.selectbox("Traffic", traffic_enc.classes_)
    vehicle = st.selectbox("Vehicle", vehicle_enc.classes_)
    area = st.selectbox("Area", area_enc.classes_)
    category = st.selectbox("Category", category_enc.classes_)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Apply encoders
    weather_val = weather_enc.transform([weather])[0]
    traffic_val = traffic_enc.transform([traffic])[0]
    vehicle_val = vehicle_enc.transform([vehicle])[0]
    area_val = area_enc.transform([area])[0]
    category_val = category_enc.transform([category])[0]

    # Prepare input dict
    input_dict = {
        "Distance_km": distance,
        "Pickup_Delay": pickup_delay,
        "Order_Hour": order_hour,
        "Order_Day": order_day,
        "Order_Weekday": order_weekday,
        "Is_Peak_Hour": int(is_peak),
        "Is_Weekend": int(is_weekend),
        "Agent_Age": agent_age,
        "Agent_Rating": agent_rating,
        "Distance_Bucket": np.digitize(distance, [0,2,5,10,20,50]),
        "Traffic_Distance": traffic_val * distance,
        "Weather_Delay": weather_val * pickup_delay,
        "Weather": weather_val,
        "Traffic": traffic_val,
        "Vehicle": vehicle_val,
        "Area": area_val,
        "Category": category_val
    }

    # Convert to DataFrame and align feature order
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[feature_order]  # ensures same order as training

    # Prediction
    prediction = model.predict(input_df)[0]
    st.success(f"⏱️ Predicted Delivery Time: {prediction:.2f} minutes")
