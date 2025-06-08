import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load models and artifacts
try:
    log_model = joblib.load('models/logistic_regression_model.pkl')
    rf_model = joblib.load('models/random_forest_model.pkl')
    svm_model = joblib.load('models/svm_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    accuracies = joblib.load('models/accuracies.pkl')
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Title
st.title("Diabetes Prediction System")
st.write("Select a model, enter patient details, and get prediction.")

# Sidebar: model selection
model_choice = st.selectbox("Select ML Model", list(accuracies.keys()))

# Input form
with st.form("prediction_form"):
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    smoking_history = st.selectbox("Smoking History", ["never", "No Info", "current", "former", "ever", "not current"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5)
    glucose = st.number_input("Blood Glucose Level", min_value=50, max_value=300, value=120)

    submit = st.form_submit_button("Predict")

if submit:
    try:
        # Encode gender (assuming model was trained with Male=1, Female=0, Other=2)
        gender_mapping = {"Female": 0, "Male": 1, "Other": 2}
        gender_encoded = gender_mapping.get(gender, 0)

        # One-hot encode smoking history (must match training data columns)
        smoking_mapping = {
            "never": [1, 0, 0, 0, 0],
            "No Info": [0, 1, 0, 0, 0],
            "current": [0, 0, 1, 0, 0],
            "former": [0, 0, 0, 1, 0],
            "ever": [0, 0, 0, 0, 1],
            "not current": [0, 0, 0, 0, 0]
        }
        smoking_encoded = smoking_mapping.get(smoking_history, [0, 0, 0, 0, 0])

        # Form final input vector (must match training data feature order)
        input_data = np.array([
            gender_encoded, age, hypertension, heart_disease,
            bmi, hba1c, glucose, *smoking_encoded
        ]).reshape(1, -1)

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Select model
        selected_model = {
            "Logistic Regression": log_model,
            "Random Forest": rf_model,
            "SVM": svm_model
        }[model_choice]

        # Prediction
        prediction = selected_model.predict(input_scaled)[0]
        probability = selected_model.predict_proba(input_scaled)[0][1]
        
        result = "Diabetic" if prediction == 1 else "Not Diabetic"
        confidence = probability if prediction == 1 else (1 - probability)

        st.subheader("Prediction Result")
        st.success(f"The person is: {result} (Confidence: {confidence:.2%})")

        st.subheader("Model Performance")
        st.dataframe(pd.DataFrame.from_dict(accuracies, orient='index', columns=['Accuracy']).style.format("{:.2%}"))

    except Exception as e:
        st.error(f"Error during prediction: {e}")