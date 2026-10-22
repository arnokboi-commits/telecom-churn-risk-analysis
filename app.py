
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Customer Churn Risk Predictor",
    page_icon="📊",
    layout="centered"
)

# Load the saved model pipeline
MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "churn_random_forest_pipeline.joblib"
)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(
            "Model file not found. Check that the models folder "
            "contains churn_random_forest_pipeline.joblib."
        )
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

# Application heading
st.title("Customer Churn Risk Predictor")
st.write(
    "Enter customer details to estimate their likelihood of churning."
)

st.info(
    "This tool provides an estimate to support business decisions; "
    "it does not guarantee that a customer will churn."
)

# Customer input form
with st.form("customer_form"):
    st.subheader("Customer Information")

    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents?", ["Yes", "No"])

    tenure = st.number_input(
        "Tenure (months)", min_value=0, max_value=72, value=12
    )

    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox(
        "Multiple Lines", ["No", "Yes", "No phone service"]
    )
    internet_service = st.selectbox(
        "Internet Service", ["DSL", "Fiber optic", "No"]
    )
    online_security = st.selectbox(
        "Online Security", ["Yes", "No", "No internet service"]
    )
    online_backup = st.selectbox(
        "Online Backup", ["Yes", "No", "No internet service"]
    )
    device_protection = st.selectbox(
        "Device Protection", ["Yes", "No", "No internet service"]
    )
    tech_support = st.selectbox(
        "Tech Support", ["Yes", "No", "No internet service"]
    )
    streaming_tv = st.selectbox(
        "Streaming TV", ["Yes", "No", "No internet service"]
    )
    streaming_movies = st.selectbox(
        "Streaming Movies", ["Yes", "No", "No internet service"]
    )
    contract = st.selectbox(
        "Contract", ["Month-to-month", "One year", "Two year"]
    )
    paperless_billing = st.selectbox(
        "Paperless Billing", ["Yes", "No"]
    )
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges", min_value=0.0, value=70.0
    )
    total_charges = st.number_input(
        "Total Charges", min_value=0.0, value=840.0
    )

    submitted = st.form_submit_button("Predict Churn Risk")

# Prediction
if submitted:
    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    prediction = model.predict(customer)[0]
    probabilities = model.predict_proba(customer)[0]
    classes = list(model.named_steps["model"].classes_)
    churn_probability = probabilities[classes.index("Yes")]

    st.subheader("Prediction Result")
    st.metric(
        "Estimated Churn Probability",
        f"{churn_probability:.1%}"
    )

    if prediction == "Yes":
        st.warning(
            "The model classifies this customer as at risk of churn."
        )
    else:
        st.success(
            "The model does not classify this customer as at risk "
            "at the current classification threshold."
        )

    st.caption(
        "The probability is a model estimate and may not be calibrated. "
        "Use it to support, not replace, business judgment."
    )
