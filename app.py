import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ============= Load Model & Preprocessing =============
model = joblib.load("lightgbm_tuned.pkl")
feature_order = joblib.load("feature_order.pkl")

# Load categorical encoders
weather_enc = joblib.load("Weather_encoder.pkl")
traffic_enc = joblib.load("Traffic_encoder.pkl")
vehicle_enc = joblib.load("Vehicle_encoder.pkl")
area_enc = joblib.load("Area_encoder.pkl")
category_enc = joblib.load("Category_encoder.pkl")

st.set_page_config(page_title="Amazon Delivery Time Predictor", layout="wide")

# Title
st.title("📦 Amazon Delivery Time Prediction")
st.write("Predict delivery time (in minutes) based on order details.")

# =========================================================
# Tabs
# =========================================================
tab1, tab2, tab3 = st.tabs(["🔮 Single Prediction", "📂 Bulk Upload", "📊 Feature Insights"])

# =========================================================
# 1. Single Prediction
# =========================================================
with tab1:
    st.subheader("Enter Order Details")

    with st.form("single_input"):
        distance = st.number_input("Distance (km)", min_value=0.0, step=0.1)
        pickup_delay = st.number_input("Pickup Delay (minutes)", min_value=0.0, step=1.0)
        order_hour = st.slider("Order Hour (0-23)", 0, 23, 12)
        order_day = st.slider("Order Day (1-31)", 1, 31, 15)
        order_weekday = st.selectbox("Order Weekday", [0,1,2,3,4,5,6],
                                     format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
        is_peak = st.checkbox("Peak Hour?")
        is_weekend = st.checkbox("Weekend?")
        agent_age = st.number_input("Agent Age", min_value=18, max_value=70, step=1)
        agent_rating = st.slider("Agent Rating", 1.0, 5.0, 4.5, step=0.1)

        weather = st.selectbox("Weather", weather_enc.classes_)
        traffic = st.selectbox("Traffic", traffic_enc.classes_)
        vehicle = st.selectbox("Vehicle", vehicle_enc.classes_)
        area = st.selectbox("Area", area_enc.classes_)
        category = st.selectbox("Category", category_enc.classes_)

        submitted = st.form_submit_button("Predict")

    if submitted:
        # Encode categorical features
        weather_val = weather_enc.transform([weather])[0]
        traffic_val = traffic_enc.transform([traffic])[0]
        vehicle_val = vehicle_enc.transform([vehicle])[0]
        area_val = area_enc.transform([area])[0]
        category_val = category_enc.transform([category])[0]

        # Build input dictionary
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

        input_df = pd.DataFrame([input_dict])
        input_df = input_df[feature_order]

        prediction = model.predict(input_df)[0]
        st.success(f"⏱️ Predicted Delivery Time: {prediction:.2f} minutes")

# =========================================================
# 2. Bulk Upload
# =========================================================
with tab2:
    st.subheader("Upload CSV File for Bulk Predictions")
    st.write("CSV must have the same columns as training data (Distance_km, Pickup_Delay, etc.).")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Encode categorical features using saved encoders
        for col, enc in zip(
            ["Weather", "Traffic", "Vehicle", "Area", "Category"],
            [weather_enc, traffic_enc, vehicle_enc, area_enc, category_enc]
        ):
            if df[col].dtype == "object":
                df[col] = enc.transform(df[col].astype(str))

        # Derived features (Distance_Bucket, Traffic_Distance, Weather_Delay)
        if "Distance_km" in df and "Pickup_Delay" in df and "Traffic" in df and "Weather" in df:
            df["Distance_Bucket"] = np.digitize(df["Distance_km"], [0,2,5,10,20,50])
            df["Traffic_Distance"] = df["Traffic"] * df["Distance_km"]
            df["Weather_Delay"] = df["Weather"] * df["Pickup_Delay"]

        df = df[feature_order]  # ensure correct column order
        preds = model.predict(df)

        df["Predicted_Delivery_Time"] = preds
        st.dataframe(df)

        # Option to download results
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")

# =========================================================
# 3. Feature Insights
# =========================================================
with tab3:
    st.subheader("Model Insights")

    # --- Feature Importance ---
    st.markdown("### 🔑 Feature Importance")
    importance = model.booster_.feature_importance()
    features = model.booster_.feature_name()

    fig, ax = plt.subplots(figsize=(8,5))
    pd.Series(importance, index=features).sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Feature Importance (LightGBM)")
    st.pyplot(fig)

    # --- Scenario Comparison ---
    st.markdown("### ⚖️ Scenario Comparison")
    st.write("Compare predicted delivery times under different conditions.")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Favourable Conditions**")
        dist = st.number_input("Distance (km)", min_value=0.0, step=0.1, key="fav_dist")
        delay = st.number_input("Pickup Delay (minutes)", min_value=0.0, step=1.0, key="fav_delay")
        fav = {
            "Distance_km": dist,
            "Pickup_Delay": delay,
            "Order_Hour": 14,
            "Order_Day": 12,
            "Order_Weekday": 2,  # Wednesday
            "Is_Peak_Hour": 0,
            "Is_Weekend": 0,
            "Agent_Age": 30,
            "Agent_Rating": 4.5,
            "Distance_Bucket": np.digitize(dist, [0,2,5,10,20,50]),
            "Traffic_Distance": 0 * dist,  # Low traffic
            "Weather_Delay": 0 * delay,   # Sunny
            "Weather": 0,
            "Traffic": 0,
            "Vehicle": 0,
            "Area": 0,
            "Category": 0
        }

    with col2:
        st.write("**Bad Conditions**")
        dist = st.number_input("Distance (km)", min_value=0.0, step=0.1, key="bad_dist")
        delay = st.number_input("Pickup Delay (minutes)", min_value=0.0, step=1.0, key="bad_delay")
        bad = {
            "Distance_km": dist,
            "Pickup_Delay": delay,
            "Order_Hour": 18,
            "Order_Day": 12,
            "Order_Weekday": 5,  # Saturday
            "Is_Peak_Hour": 1,
            "Is_Weekend": 1,
            "Agent_Age": 30,
            "Agent_Rating": 3.5,
            "Distance_Bucket": np.digitize(dist, [0,2,5,10,20,50]),
            "Traffic_Distance": 3 * dist,  # Jam
            "Weather_Delay": 2 * delay,   # Rainy
            "Weather": 2,
            "Traffic": 3,
            "Vehicle": 1,
            "Area": 1,
            "Category": 1
        }

    # Build DataFrames and predict
    fav_df = pd.DataFrame([fav])[feature_order]
    bad_df = pd.DataFrame([bad])[feature_order]

    fav_pred = model.predict(fav_df)[0]
    bad_pred = model.predict(bad_df)[0]

    st.write(f"🌞 Favourable Conditions Prediction: **{fav_pred:.2f} minutes**")
    st.write(f"🌧️ Bad Conditions Prediction: **{bad_pred:.2f} minutes**")
    st.write(f"⏳ Difference: **{bad_pred - fav_pred:.2f} minutes**")
