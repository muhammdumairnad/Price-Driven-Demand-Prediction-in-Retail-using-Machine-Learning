import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import datetime

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Demand Prediction App", layout="wide")

# ----------------------------
# LOAD MODEL (SAFE)
# ----------------------------
try:
    model = joblib.load("model.pkl")
except Exception as e:
    st.error("❌ Model failed to load. Check model.pkl file.")
    st.stop()

# ----------------------------
# HEADER
# ----------------------------
st.markdown("# 🛒 Retail Demand Prediction Dashboard")
st.markdown("Predict demand and optimize pricing using Machine Learning")

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.header("⚙️ Settings")
st.sidebar.info("Adjust parameters and explore predictions")

# ----------------------------
# INPUT SECTION (COLUMNS)
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    store_id = st.number_input("Store ID", 1, 100, 1)
    sku_id = st.number_input("SKU ID", 1, 5000, 1)
    is_featured = st.selectbox("Featured?", [0, 1])

with col2:
    total_price = st.number_input("Price (£)", 1.0, 500.0, 50.0)
    base_price = st.number_input("Base Price (£)", 1.0, 500.0, 60.0)
    is_display = st.selectbox("Display?", [0, 1])

# ----------------------------
# FEATURE ENGINEERING
# ----------------------------
discount = base_price - total_price

today = datetime.datetime.now()
month = today.month
day = today.day

month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)
day_sin = np.sin(2 * np.pi * day / 31)
day_cos = np.cos(2 * np.pi * day / 31)

# ----------------------------
# CREATE INPUT DATA
# ----------------------------
input_data = pd.DataFrame({
    'store_id': [store_id],
    'sku_id': [sku_id],
    'total_price': [total_price],
    'is_featured_sku': [is_featured],
    'is_display_sku': [is_display],
    'discount': [discount],
    'month_sin': [month_sin],
    'month_cos': [month_cos],
    'day_sin': [day_sin],
    'day_cos': [day_cos]
})

# Ensure feature order matches model
if hasattr(model, "feature_names_in_"):
    input_data = input_data[model.feature_names_in_]

# ----------------------------
# PREDICTION SECTION
# ----------------------------
st.markdown("## 📊 Prediction")

if st.button("🔍 Predict Demand"):
    try:
        demand = model.predict(input_data)[0]
        revenue = demand * total_price

        col1, col2 = st.columns(2)
        col1.metric("📦 Predicted Demand", int(demand))
        col2.metric("💰 Expected Revenue", f"£{revenue:.2f}")

    except Exception as e:
        st.error("Prediction failed. Check input format.")

# ----------------------------
# OPTIMIZATION FUNCTION
# ----------------------------
def optimize_price(sample_row):
    prices = np.linspace(base_price * 0.5, base_price * 1.5, 50)

    best_price = None
    max_revenue = -np.inf
    revenues = []

    for p in prices:
        temp = sample_row.copy()

        temp['total_price'] = p
        temp['discount'] = base_price - p

        temp_df = pd.DataFrame(temp)

        if hasattr(model, "feature_names_in_"):
            temp_df = temp_df[model.feature_names_in_]

        pred = model.predict(temp_df)[0]
        rev = p * pred

        revenues.append(rev)

        if rev > max_revenue:
            max_revenue = rev
            best_price = p

    return prices, revenues, best_price, max_revenue

# ----------------------------
# OPTIMIZATION SECTION
# ----------------------------
st.markdown("## 🎯 Price Optimization")

if st.button("📈 Find Optimal Price"):
    try:
        prices, revenues, best_price, max_rev = optimize_price(input_data)

        col1, col2 = st.columns(2)
        col1.metric("🎯 Optimal Price", f"£{best_price:.2f}")
        col2.metric("💰 Max Revenue", f"£{max_rev:.2f}")

        # Plot
        fig, ax = plt.subplots()
        ax.plot(prices, revenues, label="Revenue Curve")
        ax.axvline(best_price, linestyle='--', label="Optimal Price")

        ax.set_xlabel("Price (£)")
        ax.set_ylabel("Revenue")
        ax.set_title("Revenue Optimization Curve")
        ax.legend()

        st.pyplot(fig)

    except Exception as e:
        st.error("Optimization failed.")

# ----------------------------
# ABOUT SECTION
# ----------------------------
st.markdown("## 📘 About This App")
st.info("""
This app predicts product demand based on pricing strategy using a Machine Learning model.

Features:
- 📦 Demand prediction
- 💰 Revenue estimation
- 🎯 Price optimization

Built using Streamlit and deployed on Streamlit Cloud.
""")
