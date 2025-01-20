import streamlit as st
import pandas as pd
import numpy as np
import joblib
from ml_models import categorical, continuous

# Set page config
st.set_page_config(
    page_title="Pleural TB Prediction",
    page_icon="🏥",
    layout="wide"
)

# Load models and preprocessors
@st.cache_resource
def load_models():
    models = {}
    model_names = ['Random Forest', 'XGBoost', 'Logistic Regression', 'SVM', 'MLP']
    for model_name in model_names:
        file_name = model_name.lower().replace(' ', '_')
        models[model_name] = joblib.load(f'models/{file_name}.joblib')
    
    preprocessors = {
        'cat_imputer': joblib.load('models/cat_imputer.joblib'),
        'cont_imputer': joblib.load('models/cont_imputer.joblib'),
        'scaler': joblib.load('models/scaler.joblib')
    }
    return models, preprocessors

# Preprocess input data
def preprocess_input(data, preprocessors):
    # Convert categorical variables
    data['sex'] = data['sex'].map({'male': 1, 'female': 0})
    for col in ['cough', 'fever', 'sweating', 'weight_loss', 'hiv', 'cancer']:
        data[col] = data[col].map({'yes': 1, 'no': 0})
    
    # Convert duration to ordinal
    duration_map = {'<= 1 month': 0, ']1, 2 month]': 1, ']2, 3 month]': 2, '> 3 month': 3}
    data['duration'] = data['duration'].map(duration_map)
    
    # Split categorical and continuous features
    cat_data = data[categorical]
    cont_data = data[continuous]
    
    # Impute missing values
    cat_imputed = preprocessors['cat_imputer'].transform(cat_data)
    cont_imputed = preprocessors['cont_imputer'].transform(cont_data)
    
    # Scale continuous features
    cont_scaled = preprocessors['scaler'].transform(cont_imputed)
    
    # Combine features
    return np.hstack((cat_imputed, cont_scaled))

# Main app
def main():
    st.title("Pleural TB Prediction App")
    st.write("Enter patient information to predict the likelihood of Pleural TB")
    
    # Load models and preprocessors
    models, preprocessors = load_models()
    
    # Create form for input
    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Patient Demographics")
            sex = st.selectbox("Sex", options=['male', 'female'])
            age = st.number_input("Age", min_value=0, max_value=120)
            imc = st.number_input("BMI (IMC)", min_value=10.0, max_value=50.0)
            
            st.subheader("Clinical Symptoms")
            cough = st.selectbox("Cough", options=['yes', 'no'])
            fever = st.selectbox("Fever", options=['yes', 'no'])
            sweating = st.selectbox("Night Sweats", options=['yes', 'no'])
            weight_loss = st.selectbox("Weight Loss", options=['yes', 'no'])
        
        with col2:
            st.subheader("Medical History")
            duration = st.selectbox("Duration of Symptoms", 
                                  options=['<= 1 month', ']1, 2 month]', ']2, 3 month]', '> 3 month'])
            hiv = st.selectbox("HIV Positive", options=['yes', 'no'])
            cancer = st.selectbox("Cancer History", options=['yes', 'no'])
            
            st.subheader("Laboratory Values")
            proteins = st.number_input("PleuralProteins Level", min_value=0.0)
            crp_pleu = st.number_input("Pleural CRP Level", min_value=0.0)
            ldh = st.number_input("Pleural LDH Level", min_value=0.0)
            pnn = st.number_input("Pleural Polymorphonuclear Neutrophil Count(%)", min_value=0.0)
        
        submitted = st.form_submit_button("Predict")
    
    if submitted:
        # Prepare input data
        input_data = pd.DataFrame({
            'sex': [sex],
            'age': [age],
            'IMC': [imc],
            'cough': [cough],
            'fever': [fever],
            'sweating': [sweating],
            'weight_loss': [weight_loss],
            'duration': [duration],
            'hiv': [hiv],
            'cancer': [cancer],
            'proteins': [proteins],
            'crp_pleu': [crp_pleu],
            'ldh': [ldh],
            'PNN': [pnn]
        })
        
        # Preprocess input
        processed_input = preprocess_input(input_data, preprocessors)
        
        # Make predictions with all models
        st.subheader("Prediction Results")
        col1, col2 = st.columns(2)
        
        with col1:
            for name, model in models.items():
                prob = model.predict_proba(processed_input)[0][1]
                st.write(f"{name}: {prob:.2%} probability of Pleural TB")
        
        # Calculate ensemble prediction
        ensemble_prob = np.mean([model.predict_proba(processed_input)[0][1] 
                               for model in models.values()])
        
        with col2:
            st.metric(
                label="Ensemble Prediction",
                value=f"{ensemble_prob:.2%}",
                delta="probability of Pleural TB"
            )

if __name__ == "__main__":
    main()
