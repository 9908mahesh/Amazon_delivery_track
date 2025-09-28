import streamlit as st
import pandas as pd
import joblib
import numpy as np
import io
import mlflow
import dagshub

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

# ==============================
# MLflow Setup
# ==============================
dagshub.init(repo_owner="9908mahesh", repo_name="Amazon_delivery_track", mlflow=True)
mlflow.set_experiment("amazon_delivery_time_prediction")

st.set_page_config(page_title="Amazon Delivery Time Predictor", layout="wide")

st.title("📦 Amazon Delivery Time Prediction")
st.write("Predict delivery time (in minutes) based on order details or bulk data.")

# ==============================
# Sidebar Mode Selection
# ==============================
mode = st.sidebar.radio("Choose Mode:", ["Single Prediction", "Bulk CSV Upload", "Scenario Comparison"])

# ==============================
# Helper: Prepare input features (safe encoding)
# ==============================
def prepare_input(distance, pickup_delay, order_hour, order_day, order_weekday,
                  is_peak, is_weekend, agent_age, agent_rating,
                  weather, traffic, vehicle, area, category):

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
    return input_df[feature_order], input_dict

# ==============================
# Helper: Safe categorical transform
# ==============================
def safe_transform(encoder, series):
    """Safely transform categories. Unseen labels -> -1"""
    return series.apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)

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

        prediction = model.predict(input_df)[0]

        # ✅ Log prediction to MLflow
        with mlflow.start_run(run_name="App_SinglePrediction"):
            mlflow.log_params(input_dict)
            mlflow.log_metrics({"predicted_delivery_time": float(prediction)})

        st.success(f"⏱️ Predicted Delivery Time: {prediction:.2f} minutes")

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
        df["Weather"] = safe_transform(weather_enc, df["Weather"])
        df["Traffic"] = safe_transform(traffic_enc, df["Traffic"])
        df["Vehicle"] = safe_transform(vehicle_enc, df["Vehicle"])
        df["Area"] = safe_transform(area_enc, df["Area"])
        df["Category"] = safe_transform(category_enc, df["Category"])

        # Add engineered features
        df["Distance_Bucket"] = np.digitize(df["Distance_km"], [0,2,5,10,20,50])
        df["Traffic_Distance"] = df["Traffic"] * df["Distance_km"]
        df["Weather_Delay"] = df["Weather"] * df["Pickup_Delay"]

        df = df[feature_order]
        preds = model.predict(df)
        df["Predicted_Delivery_Time"] = preds

        # ✅ Log bulk predictions
        with mlflow.start_run(run_name="App_BulkPrediction"):
            mlflow.log_param("rows", len(df))
            mlflow.log_metrics({
                "mean_predicted_time": float(df["Predicted_Delivery_Time"].mean()),
                "max_predicted_time": float(df["Predicted_Delivery_Time"].max())
            })

        st.success("✅ Predictions Completed")
        st.write(df.head())

        st.download_button("📥 Download Predictions", df.to_csv(index=False), "predictions.csv", "text/csv")

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

    base, _ = prepare_input(distance, pickup_delay, order_hour, 15, 2, 0, 0,
                            agent_age, agent_rating,
                            "Sunny", "Low", "Motorcycle", "Urban", "Food")

    adverse, _ = prepare_input(distance, pickup_delay, order_hour, 15, 2, 1, 0,
                               agent_age, agent_rating,
                               "Rainy", "High", "Bicycle", "Urban", "Food")

    base_pred = model.predict(base)[0]
    adverse_pred = model.predict(adverse)[0]

    # ✅ Log scenario comparison
    with mlflow.start_run(run_name="App_ScenarioComparison"):
        mlflow.log_params({"scenario": "comparison"})
        mlflow.log_metrics({
            "favourable_time": float(base_pred),
            "adverse_time": float(adverse_pred),
            "difference": float(adverse_pred - base_pred)
        })

    st.write(f"✅ **Favourable Conditions:** {base_pred:.2f} mins")
    st.write(f"⚠️ **Adverse Conditions:** {adverse_pred:.2f} mins")
    st.write(f"📊 **Difference:** {adverse_pred - base_pred:.2f} mins")
