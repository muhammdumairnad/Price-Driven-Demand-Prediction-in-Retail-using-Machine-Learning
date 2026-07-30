import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import datetime

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Retail AI Dashboard", layout="wide")

# ----------------------------
# LOGIN SYSTEM
# ----------------------------
def login():
    st.title("🔐 Login to Dashboard")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state["logged_in"] = True
            st.success("Login successful ✅")
            st.rerun()
        else:
            st.error("Invalid credentials ❌")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
    st.stop()

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# ----------------------------
# PREMIUM UI
# ----------------------------
st.markdown("""
<style>
.main {background-color: #0e1117;}
h1, h2, h3 {color: white;}
.metric {
    background: #1c1f26;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Retail Demand Prediction AI")

# ----------------------------
# SIDEBAR (NO LIMIT INPUTS)
# ----------------------------
with st.sidebar:
    st.header("⚙️ Inputs")

    store_id = st.text_input("Store ID", "1")
    sku_id = st.text_input("SKU ID", "1")

    total_price = st.text_input("Selling Price (£)", "50")
    base_price = st.text_input("Base Price (£)", "60")

    is_featured = st.selectbox("Featured?", ["No", "Yes"])
    is_display = st.selectbox("Display?", ["No", "Yes"])

# Convert inputs safely
try:
    store_id = int(store_id)
    sku_id = int(sku_id)
    total_price = float(total_price)
    base_price = float(base_price)
except:
    st.error("❌ Please enter valid numeric values")
    st.stop()

is_featured = 1 if is_featured == "Yes" else 0
is_display = 1 if is_display == "Yes" else 0

# ----------------------------
# FEATURES
# ----------------------------
discount = base_price - total_price

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
# BUTTONS
# ----------------------------
col1, col2 = st.columns(2)

predict_btn = col1.button("🔍 Predict Demand")
optimize_btn = col2.button("📈 Optimize Price")

# ----------------------------
# PREDICTION
# ----------------------------
if predict_btn:
    demand = model.predict(input_data)[0]
    revenue = demand * total_price

    c1, c2 = st.columns(2)

    c1.markdown(f"""
    <div class="metric">
    <h3>📦 Demand</h3>
    <h2>{int(demand)}</h2>
    </div>
    """, unsafe_allow_html=True)

    c2.markdown(f"""
    <div class="metric">
    <h3>💰 Revenue</h3>
    <h2>£{revenue:.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------
# OPTIMIZATION
# ----------------------------
def optimize_price(sample_row):
    prices = np.linspace(base_price * 0.5, base_price * 1.5, 50)
    revenues = []

    best_price = 0
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

# ----------------------------
# OPTIMIZATION OUTPUT
# ----------------------------
if optimize_btn:
    prices, revenues, best_price, max_rev = optimize_price(input_data)

    c1, c2 = st.columns(2)

    c1.markdown(f"""
    <div class="metric">
    <h3>🎯 Best Price</h3>
    <h2>£{best_price:.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

    c2.markdown(f"""
    <div class="metric">
    <h3>💰 Max Revenue</h3>
    <h2>£{max_rev:.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

    # ----------------------------
    # INTERACTIVE PLOTLY CHART
    # ----------------------------
    df_plot = pd.DataFrame({
        "Price": prices,
        "Revenue": revenues
    })

    fig = px.line(df_plot, x="Price", y="Revenue",
                  title="Revenue vs Price Optimization")

    st.plotly_chart(fig, use_container_width=True)
