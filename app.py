import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import datetime

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Retail AI Dashboard", layout="wide")

# ----------------------------
# CUSTOM CSS (PREMIUM UI)
# ----------------------------
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
}
.card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    text-align: center;
}
.header {
    background: linear-gradient(90deg, #4CAF50, #2E7D32);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# LOAD MODEL
# ----------------------------
try:
    model = joblib.load("model.pkl")
except:
    st.error("❌ Model not found. Upload model.pkl to GitHub.")
    st.stop()

# ----------------------------
# HEADER
# ----------------------------
st.markdown("""
<div class="header">
<h1>🛒 Retail Demand Prediction AI</h1>
<p>Smart Pricing • Demand Forecasting • Revenue Optimization</p>
</div>
""", unsafe_allow_html=True)

st.markdown("")

# ----------------------------
# SIDEBAR (PREMIUM)
# ----------------------------
st.sidebar.markdown("## ⚙️ Control Panel")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏬 Store Info")

store_id = st.sidebar.number_input("Store ID", 1, 100, 1)
sku_id = st.sidebar.number_input("SKU ID", 1, 5000, 1)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💰 Pricing")

total_price = st.sidebar.slider("Selling Price (£)", 1.0, 500.0, 50.0)
base_price = st.sidebar.slider("Base Price (£)", 1.0, 500.0, 60.0)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📢 Promotion")

is_featured = st.sidebar.selectbox("Featured Product", ["No", "Yes"])
is_display = st.sidebar.selectbox("On Display", ["No", "Yes"])

is_featured = 1 if is_featured == "Yes" else 0
is_display = 1 if is_display == "Yes" else 0

st.sidebar.markdown("---")
st.sidebar.info("💡 Adjust inputs to predict demand & optimize pricing")

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
# INPUT DATA
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

if hasattr(model, "feature_names_in_"):
    input_data = input_data[model.feature_names_in_]

# ----------------------------
# BUTTONS
# ----------------------------
col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    predict_clicked = st.button("🔍 Predict Demand")

with col_btn2:
    optimize_clicked = st.button("📈 Optimize Price")

st.markdown("---")

# ----------------------------
# PREDICTION
# ----------------------------
if predict_clicked:
    try:
        demand = model.predict(input_data)[0]
        revenue = demand * total_price

        col1, col2 = st.columns(2)

        col1.markdown(f"""
        <div class="card">
        <h4>📦 Predicted Demand</h4>
        <h2>{int(demand)}</h2>
        </div>
        """, unsafe_allow_html=True)

        col2.markdown(f"""
        <div class="card">
        <h4>💰 Expected Revenue</h4>
        <h2>£{revenue:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

    except:
        st.error("Prediction failed.")

# ----------------------------
# OPTIMIZATION
# ----------------------------
def optimize_price(sample_row):
    prices = np.linspace(base_price * 0.5, base_price * 1.5, 50)
    revenues = []

    best_price = None
    max_revenue = -np.inf

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

if optimize_clicked:
    try:
        prices, revenues, best_price, max_rev = optimize_price(input_data)

        col1, col2 = st.columns(2)

        col1.markdown(f"""
        <div class="card">
        <h4>🎯 Optimal Price</h4>
        <h2>£{best_price:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

        col2.markdown(f"""
        <div class="card">
        <h4>💰 Max Revenue</h4>
        <h2>£{max_rev:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

        # Plot
        fig, ax = plt.subplots()
        ax.plot(prices, revenues)
        ax.axvline(best_price, linestyle='--')

        ax.set_xlabel("Price (£)")
        ax.set_ylabel("Revenue")
        ax.set_title("Revenue Optimization Curve")
        ax.grid(True)

        st.pyplot(fig)

    except:
        st.error("Optimization failed.")

# ----------------------------
# ABOUT
# ----------------------------
st.markdown("---")
st.markdown("## 📘 About This App")

st.info("""
This AI-powered dashboard helps retailers:

✔ Predict product demand  
✔ Estimate revenue  
✔ Optimize pricing strategy  

Built with Machine Learning + Streamlit Cloud.
""")
