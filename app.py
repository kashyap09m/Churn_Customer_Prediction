import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Churn Prediction App",
    page_icon="👋",
    layout="centered"
)

# --- LOAD MODEL AND SCALER ---
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

# --- SESSION STATE INITIALIZATION ---
# This helps the app remember which button was last clicked
if 'page' not in st.session_state:
    st.session_state.page = "predict"

# --- UI RENDERING ---
st.title("Customer Churn Prediction App")

# --- NAVIGATION BUTTONS ---
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("👨‍💻 Predict Churn", use_container_width=True):
        st.session_state.page = "predict"

with col2:
    if st.button("📊 Show Visualizations", use_container_width=True):
        st.session_state.page = "visualize"


# --- PAGE CONTENT ---

# --- PREDICTION PAGE ---
if st.session_state.page == "predict":
    st.header("Predict Customer Churn")
    st.write("Enter the customer's details below to predict whether they will churn.")
    
    with st.form("prediction_form"):
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=12)
        monthlycharge = st.number_input("Monthly Charge ($)", min_value=10.0, max_value=200.0, value=70.0, format="%.2f")
        gender = st.selectbox("Gender", ["Male", "Female"])
        
        # The button to trigger the prediction inside the form
        predict_button = st.form_submit_button("Predict", use_container_width=True)

    if predict_button:
        # Prepare the input for the model
        gender_selected = 1 if gender == "Female" else 0
        user_input = np.array([[age, gender_selected, tenure, monthlycharge]])
        
        # Scale the input
        scaled_input = scaler.transform(user_input)
        
        # Make a prediction
        prediction = model.predict(scaled_input)[0]
        
        # Show a popup message based on the prediction
        if prediction == 1:
            st.error("Prediction: Yes, the customer is likely to churn. 🎉")
        else:
            st.success("Prediction: No, the customer is likely to stay! 😞")


# --- VISUALIZATION PAGE ---
elif st.session_state.page == "visualize":
    st.header("Data Visualizations")
    
    # Load data
    df = pd.read_csv("customer_churn_data.csv")
    
    # Set plot style
    sns.set_style("whitegrid")

    st.subheader("Churn Rate by Gender")
    fig, ax = plt.subplots()
    sns.countplot(x='Gender', hue='Churn', data=df, ax=ax, palette="pastel")
    st.pyplot(fig)

    st.subheader("Age Distribution of Customers")
    fig, ax = plt.subplots()
    sns.histplot(df['Age'], kde=True, bins=30, ax=ax, color="skyblue")
    st.pyplot(fig)

    st.subheader("Churn Rate by Contract Type")
    fig, ax = plt.subplots()
    sns.countplot(x='ContractType', hue='Churn', data=df, ax=ax, palette="viridis")
    st.pyplot(fig)

    st.subheader("Internet Service Distribution")
    fig, ax = plt.subplots()
    df['InternetService'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, cmap='YlGnBu', ax=ax)
    ax.set_ylabel('') # Hide the y-label for pie charts
    st.pyplot(fig)