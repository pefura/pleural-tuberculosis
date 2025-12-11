import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.error(f"DEBUG INFO: Scikit-learn Version: {sklearn.__version__}")
st.error(f"DEBUG INFO: Numpy Version: {numpy.__version__}")

# On importe les listes de variables depuis votre fichier models.py
# Assurez-vous que models.py contient bien :
# continuous = ['age', 'IMC', 'proteins', 'crp_pleu', 'ldh', 'PNN']
# categorical = ['sex', 'cough', 'fever', 'sweating', 'weight_loss', 'duration', 'hiv', 'cancer']
from models import categorical, continuous

# Configuration de la page
st.set_page_config(
    page_title="Pleural TB Prediction App",
    page_icon="🏥",
    layout="wide"
)

# Chargement des modèles et préprocesseurs
@st.cache_resource
def load_models():
    models = {}
    model_names = ['Random Forest', 'XGBoost', 'Logistic Regression', 'SVM', 'MLP']
    for model_name in model_names:
        # Construit le nom de fichier (ex: random_forest.joblib)
        file_name = model_name.lower().replace(' ', '_')
        try:
            models[model_name] = joblib.load(f'models/{file_name}.joblib')
        except FileNotFoundError:
            st.error(f"Modèle non trouvé : models/{file_name}.joblib")
    
    preprocessors = {}
    try:
        preprocessors['cat_imputer'] = joblib.load('models/cat_imputer.joblib')
        preprocessors['cont_imputer'] = joblib.load('models/cont_imputer.joblib')
        preprocessors['scaler'] = joblib.load('models/scaler.joblib')
    except FileNotFoundError:
        st.error("Fichiers de préprocesseurs (imputer/scaler) non trouvés dans le dossier 'models/'.")
        
    return models, preprocessors

# Fonction de prétraitement
def preprocess_input(data, preprocessors):
    # 1. Séparation des variables
    # On utilise les listes importées de models.py pour être sûr de l'ordre
    cat_data = data[categorical]
    cont_data = data[continuous]
    
    # 2. Imputation des variables catégorielles (Chaînes de caractères)
    # L'imputer renvoie un tableau numpy, on le remet en DataFrame pour pouvoir mapper les valeurs
    cat_imputed_array = preprocessors['cat_imputer'].transform(cat_data)
    cat_imputed_df = pd.DataFrame(cat_imputed_array, columns=categorical)
    
    # 3. Encodage (Mapping) : Conversion Texte -> Chiffres
    # Note : Ces mappings doivent correspondre exactement à ceux utilisés lors de l'entraînement
    
    # Sexe
    cat_imputed_df['sex'] = cat_imputed_df['sex'].map({'male': 1, 'female': 0})
    
    # Variables binaires (Oui/Non)
    binary_cols = ['cough', 'fever', 'sweating', 'weight_loss', 'hiv', 'cancer']
    for col in binary_cols:
        cat_imputed_df[col] = cat_imputed_df[col].map({'yes': 1, 'no': 0})
    
    # Durée (Ordinale)
    duration_map = {
        '<= 1 month': 0, 
        ']1, 2 month]': 1, 
        ']2, 3 month]': 2, 
        '> 3 month': 3
    }
    cat_imputed_df['duration'] = cat_imputed_df['duration'].map(duration_map)
    
    # Conversion finale en numpy array pour la concaténation
    cat_final = cat_imputed_df.to_numpy()

    # 4. Traitement des variables continues (Imputation + Scaling)
    cont_imputed = preprocessors['cont_imputer'].transform(cont_data)
    cont_scaled = preprocessors['scaler'].transform(cont_imputed)
    
    # 5. Combinaison des deux tableaux
    return np.hstack((cat_final, cont_scaled))

# Application principale
def main():
    st.title("Pleural TB Prediction App")
    st.write("Enter patient information to predict the likelihood of Pleural TB")
    
    # Chargement
    models, preprocessors = load_models()
    
    if not models or not preprocessors:
        st.stop() # Arrête l'app si les fichiers manquent
    
    # Formulaire
    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Patient Demographics")
            sex = st.selectbox("Sex", options=['male', 'female'])
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
            # Utilisation de IMC en majuscule pour l'affichage, sera mappé dans le DF plus bas
            imc = st.number_input("BMI (IMC) kg/m2", min_value=10.0, max_value=50.0, value=22.0)
            
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
            proteins = st.number_input("Pleural Proteins Levels", min_value=0.0, value=40.0)
            crp_pleu = st.number_input("Pleural CRP Levels", min_value=0.0, value=10.0)
            ldh = st.number_input("Pleural LDH Levels", min_value=0.0, value=200.0)
            pnn = st.number_input("Pleural PNN (%)", min_value=0.0, max_value=100.0, value=50.0)
        
        submitted = st.form_submit_button("Predict")
    
    if submitted:
        # Création du DataFrame
        # ATTENTION : Les clés ici doivent correspondre EXACTEMENT à names(df)
        input_data = pd.DataFrame({
            'sex': [sex],
            'age': [age],
            'IMC': [imc],       # MAJUSCULE
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
            'PNN': [pnn]        # MAJUSCULE
        })
        
        try:
            # Prétraitement
            processed_input = preprocess_input(input_data, preprocessors)
            
            # Prédictions
            st.subheader("Prediction Results")
            col1, col2 = st.columns(2)
            
            probs = []
            
            with col1:
                for name, model in models.items():
                    # On récupère la probabilité de la classe positive (1)
                    prob = model.predict_proba(processed_input)[0][1]
                    probs.append(prob)
                    
                    # Code couleur pour l'affichage
                    color = "red" if prob > 0.5 else "green"
                    st.markdown(f"**{name}**: <span style='color:{color}'>{prob:.2%}</span>", unsafe_allow_html=True)
            
            # Calcul de la moyenne (Ensemble)
            ensemble_prob = np.mean(probs)
            
            with col2:
                st.metric(
                    label="Ensemble Prediction (Average)",
                    value=f"{ensemble_prob:.2%}",
                    delta="Risk Probability"
                )
                
                if ensemble_prob > 0.5:
                    st.error("High risk of Pleural Tuberculosis")
                else:
                    st.success("Low risk of Pleural Tuberculosis")
                    
        except Exception as e:
            st.error(f"Une erreur est survenue lors de la prédiction : {e}")
            st.write("Détails pour le débogage :")
            st.write(input_data)

if __name__ == "__main__":
    main()
