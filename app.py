import streamlit as st
import joblib
import numpy as np
import pandas as pd
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .header {
        color: #2c3e50;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .subheader {
        color: #3498db;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .stSelectbox, .stNumberInput, .stButton {
        margin-bottom: 1rem;
    }
    .prediction-card {
        background-color: white;
        border-radius: 10px;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    .result-positive {
        color: #e74c3c;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .result-negative {
        color: #2ecc71;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .confidence-meter {
        height: 20px;
        background: linear-gradient(90deg, #2ecc71 0%, #f39c12 50%, #e74c3c 100%);
        border-radius: 10px;
        margin: 1rem 0;
    }
    .model-card {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .accuracy-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 12px;
        background-color: #3498db;
        color: white;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Load models and artifacts
@st.cache_resource
def load_models():
    try:
        models = {
            "Logistic Regression": joblib.load('models/logistic_regression_model.pkl'),
            "Random Forest": joblib.load('models/random_forest_model.pkl'),
            "SVM": joblib.load('models/svm_model.pkl'),
            "Naive Bayes": joblib.load('models/naive_bayes_model.pkl')
        }
        scaler = joblib.load('models/scaler.pkl')
        accuracies = joblib.load('models/accuracies.pkl')
        return models, scaler, accuracies
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

models, scaler, accuracies = load_models()

# Header with logo
col1, col2 = st.columns([1, 4])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/2965/2965300.png", width=80)
with col2:
    st.markdown('<div class="header">Diabetes Prediction System</div>', unsafe_allow_html=True)
    st.caption("Predict the likelihood of diabetes based on patient health metrics")

# Main content
tab1, tab2 = st.tabs(["Prediction", "Model Information"])

with tab1:
    # Two-column layout for input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="subheader">Patient Information</div>', unsafe_allow_html=True)
        
        with st.form("prediction_form"):
            gender = st.selectbox("Gender", ["Female", "Male", "Other"])
            age = st.number_input("Age", min_value=1, max_value=120, value=30, 
                                help="Patient's age in years")
            hypertension = st.selectbox("Hypertension", [0, 1], 
                                      format_func=lambda x: "No" if x == 0 else "Yes",
                                      help="Whether the patient has hypertension")
            heart_disease = st.selectbox("Heart Disease", [0, 1], 
                                       format_func=lambda x: "No" if x == 0 else "Yes",
                                       help="Whether the patient has heart disease")
            
            submit = st.form_submit_button("Predict Diabetes Risk", 
                                         use_container_width=True,
                                         type="primary")
    
    with col2:
        st.markdown('<div class="subheader">Health Metrics</div>', unsafe_allow_html=True)
        
        smoking_history = st.selectbox("Smoking History", 
                                      ["never", "No Info", "current", "former", "ever", "not current"],
                                      help="Patient's smoking history")
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0,
                            help="Body Mass Index (weight in kg/(height in m)^2)")
        hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5,
                               help="Glycated hemoglobin level (3-month average blood sugar)")
        glucose = st.number_input("Blood Glucose Level", min_value=50, max_value=300, value=120,
                                 help="Current blood glucose level (mg/dL)")
    
    if submit:
        try:
            # Encode inputs
            gender_mapping = {"Female": 0, "Male": 1, "Other": 2}
            gender_encoded = gender_mapping.get(gender, 0)

            smoking_mapping = {
                "never": [1, 0, 0, 0, 0],
                "No Info": [0, 1, 0, 0, 0],
                "current": [0, 0, 1, 0, 0],
                "former": [0, 0, 0, 1, 0],
                "ever": [0, 0, 0, 0, 1],
                "not current": [0, 0, 0, 0, 0]
            }
            smoking_encoded = smoking_mapping.get(smoking_history, [0, 0, 0, 0, 0])

            # Create input vector
            input_data = np.array([
                gender_encoded, age, hypertension, heart_disease,
                bmi, hba1c, glucose, *smoking_encoded
            ]).reshape(1, -1)

            # Scale input
            input_scaled = scaler.transform(input_data)

            # Get selected model
            selected_model = models[st.session_state.get("selected_model", "Random Forest")]

            # Prediction
            prediction = selected_model.predict(input_scaled)[0]
            probability = selected_model.predict_proba(input_scaled)[0][1]
            
            # Display results in a card
            with st.container():
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                
                col_res1, col_res2 = st.columns([1, 2])
                with col_res1:
                    st.markdown("### Prediction Result")
                    if prediction == 1:
                        st.markdown(f'<div class="result-positive">Diabetic</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="result-negative">Not Diabetic</div>', unsafe_allow_html=True)
                    
                    st.metric("Probability", f"{probability:.1%}")
                
                with col_res2:
                    st.markdown("### Confidence Level")
                    st.markdown(f'<div class="confidence-meter" style="width: {probability*100}%"></div>', unsafe_allow_html=True)
                    
                    if probability > 0.7:
                        st.warning("High risk of diabetes - recommend further testing")
                    elif probability > 0.3:
                        st.info("Moderate risk - suggest lifestyle changes and monitoring")
                    else:
                        st.success("Low risk - maintain healthy habits")
                
                st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error during prediction: {e}")

with tab2:
    st.markdown('<div class="subheader">Model Performance</div>', unsafe_allow_html=True)
    
    # Sort models by accuracy
    sorted_accuracies = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)
    
    for model_name, accuracy in sorted_accuracies:
        with st.container():
            st.markdown('<div class="model-card">', unsafe_allow_html=True)
            
            col_mod1, col_mod2 = st.columns([4, 1])
            with col_mod1:
                st.markdown(f"**{model_name}**")
                st.progress(accuracy)
            with col_mod2:
                st.markdown(f'<div class="accuracy-badge">{accuracy:.1%}</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Model Selection")
    model_choice = st.selectbox("Select default model for predictions", 
                               list(accuracies.keys()),
                               key="selected_model")
    
    st.markdown("""
    **Model Descriptions:**
    - **Logistic Regression:** Linear model for binary classification
    - **Random Forest:** Ensemble of decision trees, robust to outliers
    - **SVM:** Finds optimal boundary between classes
    - **Naive Bayes:** Probabilistic classifier based on Bayes' theorem
    """)

# Sidebar with additional info
with st.sidebar:
    st.markdown("## About")
    st.markdown("""
    This system predicts diabetes risk using machine learning models trained on clinical data.
    
    **Input Features:**
    - Demographic information
    - Medical history
    - Blood test results
    - Lifestyle factors
    
    **Note:** This tool is for informational purposes only and should not replace professional medical advice.
    """)
    
    st.markdown("---")
    st.markdown("**Developed by:** [Your Name]")
    st.markdown("**Version:** 1.0.0")