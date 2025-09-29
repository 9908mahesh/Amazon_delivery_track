import streamlit as st
import pandas as pd
import joblib
import numpy as np
import io
import mlflow
import dagshub
import traceback
from datetime import datetime

# --------------------------
# DagsHub / MLflow init
# --------------------------
import dagshub
import os

dagshub.init(
    repo_owner=os.getenv("DAGSHUB_USER"),
    repo_name="Amazon_delivery_track",
    mlflow=True,
    token=os.getenv("DAGSHUB_TOKEN")
)


# ==============================
# Load Model + Encoders + Feature Order
# ==============================
model = joblib.load("lightgbm_tuned.pkl")
feature_order = joblib.load("feature_order.pkl")

weather_enc = joblib.load("Weather_encoder.pkl")
traffic_enc = joblib.load("Traffic_encoder.pkl")
vehicle_enc = joblib.load("Vehicle_encoder.pkl")
area_enc = joblib.load("Area_encoder.pkl")
category_enc = joblib.load("Category_encoder.pkl")

st.set_page_config(page_title="Amazon Delivery Time Predictor", layout="wide")

st.title("📦 Amazon Delivery Time Prediction")
st.write("Predict delivery time (in minutes) based on order details or bulk data.")

# ==============================
# Helper functions
# ==============================
def sanitize_for_mlflow(d: dict):
    """Convert numpy types to native Python types and everything else to str."""
    out = {}
    for k, v in d.items():
        try:
            if isinstance(v, (np.integer,)):
                out[k] = int(v)
            elif isinstance(v, (np.floating,)):
                out[k] = float(v)
            elif isinstance(v, (int, float, str, bool)):
                out[k] = v
            else:
                out[k] = str(v)
        except Exception:
            out[k] = str(v)
    return out

def prepare_input(distance, pickup_delay, order_hour, order_day, order_weekday,
                  is_peak, is_weekend, agent_age, agent_rating,
                  weather, traffic, vehicle, area, category):
    """Prepare a single-row DataFrame in the feature order and return input_dict for logging."""
    def safe_single(encoder, value):
        return encoder.transform([value])[0] if value in encoder.classes_ else -1

    weather_val = safe_single(weather_enc, weather)
    traffic_val = safe_single(traffic_enc, traffic)
    vehicle_val = safe_single(vehicle_enc, vehicle)
    area_val = safe_single(area_enc, area)
    category_val = safe_single(category_enc, category)

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
        "Distance_Bucket": int(np.digitize(distance, [0,2,5,10,20,50])),
        "Traffic_Distance": traffic_val * distance,
        "Weather_Delay": weather_val * pickup_delay,
        "Weather": weather_val,
        "Traffic": traffic_val,
        "Vehicle": vehicle_val,
        "Area": area_val,
        "Category": category_val
    }

    input_df = pd.DataFrame([input_dict])
    # ensure columns exist in the same order as training (if any missing this will be checked by caller)
    return input_df, input_dict

def safe_transform(encoder, series):
    """Safely transform categories. Unseen labels -> -1"""
    # Normalize strings a bit (strip). Keep membership check case-sensitive to match encoders.
    s = series.astype(str).str.strip()
    return s.apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)

def ensure_and_align_features(df, feature_order, fill_value=0):
    """Make sure df contains all features in feature_order. Fill missing with fill_value and align order."""
    missing = [c for c in feature_order if c not in df.columns]
    if missing:
        st.warning(f"Uploaded data missing features {missing}. Filling with default={fill_value}.")
        for c in missing:
            df[c] = fill_value
    # reorder
    return df[feature_order]

# ==============================
# Sidebar Mode Selection
# ==============================
mode = st.sidebar.radio("Choose Mode:", ["Single Prediction", "Bulk CSV Upload", "Scenario Comparison"])

# ==============================
# Mode 1: Single Prediction
# ==============================
if mode == "Single Prediction":
    with st.form("single_form"):
        st.subheader("Enter Order Details")

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
        input_df, input_dict = prepare_input(distance, pickup_delay, order_hour, order_day, order_weekday,
                                             is_peak, is_weekend, agent_age, agent_rating,
                                             weather, traffic, vehicle, area, category)
        # align columns — safe guard
        try:
            input_df = ensure_and_align_features(input_df, feature_order)
        except Exception as e:
            st.error(f"Feature alignment error: {e}")
            raise

        prediction = model.predict(input_df)[0]
        st.success(f"⏱️ Predicted Delivery Time: {prediction:.2f} minutes")

        # Log to MLflow safely
        try:
            run_name = "App_SinglePrediction"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params(sanitize_for_mlflow(input_dict))
                mlflow.log_metric("predicted_delivery_time", float(prediction))
            st.info("📊 Prediction logged to MLflow ✅")
        except Exception as e:
            st.warning("MLflow logging failed (prediction saved locally only).")
            st.write(str(e))

# ==============================
# Mode 2: Bulk CSV Upload
# ==============================
elif mode == "Bulk CSV Upload":
    st.subheader("Upload a CSV file with order details")

    # Generate and offer sample CSV for download
    sample_csv = pd.DataFrame([{
        "Distance_km": 5,
        "Pickup_Delay": 15,
        "Order_Hour": 10,
        "Order_Day": 12,
        "Order_Weekday": 2,
        "Is_Peak_Hour": 0,
        "Is_Weekend": 0,
        "Agent_Age": 28,
        "Agent_Rating": 4.5,
        "Weather": "Sunny",
        "Traffic": "Low",
        "Vehicle": "Motorcycle",
        "Area": "Urban",
        "Category": "Food"
    }])
    csv_buffer = io.StringIO()
    sample_csv.to_csv(csv_buffer, index=False)
    st.download_button("📥 Download Sample CSV", csv_buffer.getvalue(), "sample_input.csv", "text/csv")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:", df.head())

        # Encode categorical columns safely
        try:
            df["Weather"] = safe_transform(weather_enc, df["Weather"])
            df["Traffic"] = safe_transform(traffic_enc, df["Traffic"])
            df["Vehicle"] = safe_transform(vehicle_enc, df["Vehicle"])
            df["Area"] = safe_transform(area_enc, df["Area"])
            df["Category"] = safe_transform(category_enc, df["Category"])
        except KeyError as e:
            st.error(f"Uploaded CSV missing expected column: {e}")
            st.stop()

        # Add engineered features
        df["Distance_Bucket"] = np.digitize(df["Distance_km"], [0,2,5,10,20,50]).astype(int)
        df["Traffic_Distance"] = df["Traffic"] * df["Distance_km"]
        df["Weather_Delay"] = df["Weather"] * df["Pickup_Delay"]

        # align features and handle missing columns
        try:
            df = ensure_and_align_features(df, feature_order, fill_value=0)
        except Exception as e:
            st.error(f"Feature alignment error: {e}")
            st.stop()

        preds = model.predict(df)
        df["Predicted_Delivery_Time"] = preds

        st.success("✅ Predictions Completed")
        st.write(df.head())

        st.download_button("📥 Download Predictions", df.to_csv(index=False), "predictions.csv", "text/csv")

        # Log to MLflow (summary)
        try:
            run_name = f"App_BulkPrediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("num_records", int(len(df)))
                mlflow.log_metric("bulk_avg_prediction", float(df["Predicted_Delivery_Time"].mean()))
                mlflow.log_metric("bulk_max_prediction", float(df["Predicted_Delivery_Time"].max()))
            st.info("📊 Bulk predictions logged to MLflow ✅")
        except Exception as e:
            st.warning("MLflow logging failed for bulk prediction.")
            st.write(str(e))

# ==============================
# Mode 3: Scenario Comparison
# ==============================
elif mode == "Scenario Comparison":
    st.subheader("Compare Predictions in Different Conditions")

    distance = st.slider("Distance (km)", 1, 50, 10)
    pickup_delay = st.slider("Pickup Delay (minutes)", 0, 120, 20)
    order_hour = st.slider("Order Hour (0-23)", 0, 23, 12)

    agent_age = st.slider("Agent Age", 18, 60, 30)
    agent_rating = st.slider("Agent Rating", 1.0, 5.0, 4.5, step=0.1)

    base_df, base_dict = prepare_input(distance, pickup_delay, order_hour, 15, 2, 0, 0,
                                       agent_age, agent_rating,
                                       weather_enc.classes_[0], traffic_enc.classes_[0],
                                       vehicle_enc.classes_[0], area_enc.classes_[0], category_enc.classes_[0])

    adverse_df, adverse_dict = prepare_input(distance, pickup_delay, order_hour, 15, 2, 1, 0,
                                             agent_age, agent_rating,
                                             weather_enc.classes_[-1], traffic_enc.classes_[-1],
                                             vehicle_enc.classes_[-1], area_enc.classes_[0], category_enc.classes_[0])

    # Align (should be fine)
    base_df = ensure_and_align_features(base_df, feature_order)
    adverse_df = ensure_and_align_features(adverse_df, feature_order)

    base_pred = model.predict(base_df)[0]
    adverse_pred = model.predict(adverse_df)[0]

    st.write(f"✅ **Favourable Conditions:** {base_pred:.2f} mins")
    st.write(f"⚠️ **Adverse Conditions:** {adverse_pred:.2f} mins")
    st.write(f"📊 **Difference:** {adverse_pred - base_pred:.2f} mins")

    # Log to MLflow
    try:
        run_name = "App_ScenarioComparison"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({"scenario": "comparison"})
            mlflow.log_metric("favourable_time", float(base_pred))
            mlflow.log_metric("adverse_time", float(adverse_pred))
            mlflow.log_metric("difference", float(adverse_pred - base_pred))
        st.info("📊 Scenario comparison logged to MLflow ✅")
    except Exception as e:
        st.warning("MLflow logging failed for scenario.")
        st.write(str(e))
