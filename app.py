import streamlit as st
import joblib
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Credit Card Default Predictor",
    layout="centered"
)

# Load model and preprocessing pipeline
model = joblib.load("credit_default_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Title
st.title("💳 Credit Card Default Prediction")

st.markdown(
"""
This application predicts whether a credit card customer is **likely to default**
based on demographic information and payment history.
"""
)

st.subheader("Enter Customer Details")

# Input fields
limit_bal = st.number_input("Credit Limit", min_value=0)

col1, col2 = st.columns(2)

with col1:
    sex = st.selectbox("Sex", [1, 2])
    education = st.selectbox("Education", [1, 2, 3, 4])

with col2:
    marriage = st.selectbox("Marriage", [1, 2, 3])
    age = st.number_input("Age", min_value=18, max_value=100)

pay_0 = st.number_input("Recent Payment Status")

# Predict button
if st.button("Predict Default Risk"):

    # Create dataframe with required columns
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

    # Apply preprocessing
    processed_input = preprocessor.transform(input_data)

    # Model prediction
    prediction = model.predict(processed_input)

    # Probability of default
    probability = model.predict_proba(processed_input)[0][1]

    st.divider()
    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠ Customer likely to default")
    else:
        st.success("✅ Customer unlikely to default")

    # Show probability
    st.metric("Default Probability", f"{probability:.2f}")

    # Risk interpretation
    if probability < 0.3:
        st.success("Low Risk Customer")
    elif probability < 0.6:
        st.warning("Medium Risk Customer")
    else:
        st.error("High Risk Customer")

    # Visual progress bar
    st.progress(probability)