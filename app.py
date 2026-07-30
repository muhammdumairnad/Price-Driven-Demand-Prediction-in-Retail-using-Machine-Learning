import streamlit as st
import json
import os
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
# USER DATABASE (JSON FILE)
# ----------------------------
USER_DB = "users.json"

def load_users():
    if not os.path.exists(USER_DB):
        return {}
    with open(USER_DB, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USER_DB, "w") as f:
        json.dump(users, f)

# ----------------------------
# AUTH SYSTEM
# ----------------------------
def login_page():
    st.title("🔐 Login")

    users = load_users()

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users and users[username] == password:
            st.session_state["user"] = username
            st.success("Login successful ✅")
            st.rerun()
        else:
            st.error("Invalid credentials ❌")

    if st.button("Go to Signup"):
        st.session_state["page"] = "signup"
        st.rerun()

def signup_page():
    st.title("📝 Signup")

    users = load_users()

    new_user = st.text_input("Create Username")
    new_pass = st.text_input("Create Password", type="password")

    if st.button("Signup"):
        if new_user in users:
            st.error("User already exists ❌")
        elif new_user == "" or new_pass == "":
            st.error("Fields cannot be empty ❌")
        else:
            users[new_user] = new_pass
            save_users(users)
            st.success("Account created! Go to login ✅")

    if st.button("Back to Login"):
        st.session_state["page"] = "login"
        st.rerun()

# ----------------------------
# SESSION STATE INIT
# ----------------------------
if "user" not in st.session_state:
    st.session_state["user"] = None

if "page" not in st.session_state:
    st.session_state["page"] = "login"

# ----------------------------
# ROUTING
# ----------------------------
if st.session_state["user"] is None:
    if st.session_state["page"] == "login":
        login_page()
    else:
        signup_page()
    st.stop()

# ----------------------------
# LOGOUT
# ----------------------------
with st.sidebar:
    st.write(f"👤 Logged in as: {st.session_state['user']}")
    if st.button("Logout"):
        st.session_state["user"] = None
        st.session_state["page"] = "login"
        st.rerun()

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# ----------------------------
# UI
# ----------------------------
st.title("📊 Retail Demand Prediction Dashboard")

# ----------------------------
# INPUTS (NO LIMIT)
# ----------------------------
col1, col2 = st.columns(2)

store_id = col1.text_input("Store ID", "1")
sku_id = col1.text_input("SKU ID", "1")

total_price = col2.text_input("Selling Price (£)", "50")
base_price = col2.text_input("Base Price (£)", "60")

is_featured = st.selectbox("Featured?", ["No", "Yes"])
is_display = st.selectbox("Display?", ["No", "Yes"])

# Convert safely
try:
    store_id = int(store_id)
    sku_id = int(sku_id)
    total_price = float(total_price)
    base_price = float(base_price)
except:
    st.error("Enter valid numeric values ❌")
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
# DATA
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
predict_btn = col1.button("🔍 Predict")
optimize_btn = col2.button("📈 Optimize")

# ----------------------------
# PREDICT
# ----------------------------
if predict_btn:
    demand = model.predict(input_data)[0]
    revenue = demand * total_price

    st.success(f"📦 Demand: {int(demand)}")
    st.success(f"💰 Revenue: £{revenue:.2f}")

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
# OPTIMIZE OUTPUT
# ----------------------------
if optimize_btn:
    prices, revenues, best_price, max_rev = optimize_price(input_data)

    st.success(f"🎯 Best Price: £{best_price:.2f}")
    st.success(f"💰 Max Revenue: £{max_rev:.2f}")

    df_plot = pd.DataFrame({
        "Price": prices,
        "Revenue": revenues
    })

    fig = px.line(df_plot, x="Price", y="Revenue", title="Revenue vs Price")
    st.plotly_chart(fig, use_container_width=True)
