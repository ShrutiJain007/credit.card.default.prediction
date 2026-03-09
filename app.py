import streamlit as st
import joblib
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Credit Card Default Predictor",
    page_icon="💳",
    layout="wide"
)

# Load model and preprocessing pipeline
model = joblib.load("credit_default_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Title
st.title("💳 Credit Card Default Prediction Dashboard")

st.markdown(
"""
Predict whether a **credit card customer is likely to default** based on demographic
information and payment behavior.

Enter the customer information in the sidebar and click **Predict**.
"""
)

# Sidebar Inputs
st.sidebar.header("Customer Information")

limit_bal = st.sidebar.number_input("Credit Limit", min_value=0)
sex = st.sidebar.selectbox("Sex", [1, 2])
education = st.sidebar.selectbox("Education", [1, 2, 3, 4])
marriage = st.sidebar.selectbox("Marriage", [1, 2, 3])
age = st.sidebar.number_input("Age", min_value=18, max_value=100)
pay_0 = st.sidebar.number_input("Recent Payment Status")

# Prediction button
if st.sidebar.button("Predict Default Risk"):

    input_data = pd.DataFrame({
        "LIMIT_BAL":[limit_bal],
        "SEX":[sex],
        "EDUCATION":[education],
        "MARRIAGE":[marriage],
        "AGE":[age],
        "PAY_0":[pay_0],

        "PAY_2":[0],
        "PAY_3":[0],
        "PAY_4":[0],
        "PAY_5":[0],
        "PAY_6":[0],

        "BILL_AMT1":[0],
        "BILL_AMT2":[0],
        "BILL_AMT3":[0],
        "BILL_AMT4":[0],
        "BILL_AMT5":[0],
        "BILL_AMT6":[0],

        "PAY_AMT1":[0],
        "PAY_AMT2":[0],
        "PAY_AMT3":[0],
        "PAY_AMT4":[0],
        "PAY_AMT5":[0],
        "PAY_AMT6":[0]
    })

    processed_input = preprocessor.transform(input_data)

    # Predict probability
    probability = model.predict_proba(processed_input)[0][1]

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    # Risk message based on probability
    with col1:
        if probability < 0.3:
            st.success("✅ Low Default Risk")
        elif probability < 0.6:
            st.warning("⚠ Medium Default Risk")
        else:
            st.error("🚨 High Default Risk")

    # Probability display
    with col2:
        st.metric("Default Probability", f"{probability:.2f}")

    # Visual progress bar
    st.progress(probability)