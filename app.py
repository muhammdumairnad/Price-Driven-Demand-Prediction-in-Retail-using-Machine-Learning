%%writefile app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import datetime

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Demand Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# ----------------------------
# LOAD MODEL (SAFE)
# ----------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("model.pkl")
    except Exception as e:
        st.error("❌ Model not found! Please upload model.pkl to GitHub.")
        st.stop()

model = load_model()

# ----------------------------
# CUSTOM CSS (PREMIUM UI)
# ----------------------------
st.markdown("""
<style>
.main {background-color: #0e1117;}
h1, h2, h3 {color: #ffffff;}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
.metric-box {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# TITLE
# ----------------------------
st.title("📊 Price-Driven Demand Prediction Dashboard")
st.markdown("### 💡 Predict demand & optimize pricing using Machine Learning")

# ----------------------------
# SIDEBAR (IMPROVED)
# ----------------------------
with st.sidebar:
    st.header("⚙️ Control Panel")

    store_id = st.number_input(
        "Store ID",
        min_value=1,
        max_value=100,
        value=1,
        step=1
    )

    sku_id = st.number_input(
        "SKU ID",
        min_value=1,
        max_value=5000,
        value=1,
        step=1
    )

    total_price = st.number_input(
        "Selling Price (£)",
        min_value=1.0,
        max_value=1000.0,
        value=50.0,
        step=1.0,
        format="%.2f"
    )

    base_price = st.number_input(
        "Base Price (£)",
        min_value=1.0,
        max_value=1000.0,
        value=60.0,
        step=1.0,
        format="%.2f"
    )

    is_featured = st.selectbox("Featured Product?", ["No", "Yes"])
    is_display = st.selectbox("On Display?", ["No", "Yes"])

    st.markdown("---")
    st.info("💡 Tip: Keep base price ≥ selling price for realistic results")

# Convert Yes/No to 0/1
is_featured = 1 if is_featured == "Yes" else 0
is_display = 1 if is_display == "Yes" else 0

# ----------------------------
# VALIDATION
# ----------------------------
if base_price < total_price:
    st.warning("⚠️ Selling price is higher than base price (unusual case)")

discount = base_price - total_price

# ----------------------------
# TIME FEATURES
# ----------------------------
today = datetime.datetime.now()

month_sin = np.sin(2 * np.pi * today.month / 12)
month_cos = np.cos(2 * np.pi * today.month / 12)
day_sin = np.sin(2 * np.pi * today.day / 31)
day_cos = np.cos(2 * np.pi * today.day / 31)

# ----------------------------
# DATAFRAME
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
# MAIN LAYOUT (2 COLUMNS)
# ----------------------------
col1, col2 = st.columns(2)

# ----------------------------
# PREDICTION
# ----------------------------
with col1:
    st.subheader("📦 Demand Prediction")

    if st.button("Predict Demand"):
        demand = model.predict(input_data)[0]
        revenue = demand * total_price

        st.markdown(f"""
        <div class="metric-box">
            <h3>📦 Demand</h3>
            <h2>{int(demand)}</h2>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-box">
            <h3>💰 Revenue</h3>
            <h2>£{revenue:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

# ----------------------------
# OPTIMIZATION
# ----------------------------
def optimize_price(sample_row):
    prices = np.linspace(base_price * 0.5, base_price * 1.5, 50)

    best_price = 0
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
# OPTIMIZATION UI
# ----------------------------
with col2:
    st.subheader("🎯 Price Optimization")

    if st.button("Find Optimal Price"):
        prices, revenues, best_price, max_rev = optimize_price(input_data)

        st.markdown(f"""
        <div class="metric-box">
            <h3>🎯 Best Price</h3>
            <h2>£{best_price:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-box">
            <h3>💰 Max Revenue</h3>
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

        st.pyplot(fig)
