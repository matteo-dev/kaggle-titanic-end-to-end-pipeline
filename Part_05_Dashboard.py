# Fichier python pour la partie "Dashboard" du projet "Titanic" de Kaggle

# Importation des bibliothèques nécessaires
import streamlit as st # Bibliothèque pour créer des dashboards interactifs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb # Bibliothèque pour XGBoost
import os

# --- Configuration de la page ---
st.set_page_config(page_title="Titanic End-to-End Pipeline", page_icon="🚢", layout="wide")

# --- Chargement des données en cache pour optimiser les performances ---
@st.cache_data

# Fonction pour charger les données depuis les fichiers CSV
def load_data():
    raw_df = pd.read_csv('train.csv')
    clean_df = pd.read_csv('train_clean.csv')

    # Chargement des résultats si le fichier a bien été généré à l'étape 3
    try:
        results_df = pd.read_csv('model_results.csv')
    except FileNotFoundError:
        results_df = None
    return raw_df, clean_df, results_df

# --- Entraînement rapide du modèle pour alimenter le simulateur ---
@st.cache_resource

# Fonction pour entraîner un modèle XGBoost sur l'ensemble de données nettoyé
def train_simulator_model(clean_df):
    X = clean_df.drop(columns=['Survived'])
    y = clean_df['Survived']
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X, y)
    return model, X.columns.tolist()

raw_df, clean_df, results_df = load_data()
model, feature_cols = train_simulator_model(clean_df)

# --- Titre principal ---
st.title("🚢 Titanic Kaggle Challenge : End-to-End ML Pipeline")
st.markdown("Navigate through the tabs to explore the complete lifecycle of this machine learning project.")

# --- Création des 4 onglets demandés ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 1. Exploratory Data Analysis", 
    "🛠️ 2. Feature Engineering", 
    "📈 3. Models Comparison", 
    "🎮 4. Interactive Simulator"
])

# ONGLET 1 : EXPLORATORY DATA ANALYSIS (EDA)
with tab1:
    st.header("Exploratory Data Analysis (EDA)")
    st.write("Initial assessment of the passenger manifest to identify survival patterns.")
    
    st.subheader("Raw Dataset Preview")
    st.dataframe(raw_df.head(10))
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Survival by Passenger Class")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.countplot(data=raw_df, x='Pclass', hue='Survived', palette='Set1', ax=ax1)
        ax1.set_ylabel("Passenger Count")
        st.pyplot(fig1)
        
    with col2:
        st.subheader("Survival by Gender")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.countplot(data=raw_df, x='Sex', hue='Survived', palette='Set2', ax=ax2)
        ax2.set_ylabel("Passenger Count")
        st.pyplot(fig2)

# ONGLET 2 : FEATURE ENGINEERING
with tab2:
    st.header("Feature Engineering & Preprocessing")
    st.write("Transforming raw data into meaningful predictive signals for the algorithm.")
    
    st.markdown("""
    **Key Transformations:**
    * **Title Extraction:** Parsed passenger names to group social titles (Mr, Mrs, Miss, Rare).
    * **Smart Imputation:** Filled missing `Age` values using the median age grouped by `Pclass` and `Title`.
    * **Family Metrics:** Created `FamilySize` and `IsAlone` binary flags to capture group dynamics.
    * **Ticket & Deck:** Extracted cabin deck levels and grouped passengers traveling on the same ticket (`Ticket_Freq`).
    """)
    
    st.subheader("Cleaned Dataset Ready for Modeling")
    st.dataframe(clean_df.head(10))
    st.info(f"Final feature count: {len(clean_df.columns) - 1} predictive variables fed into the models.")

# ONGLET 3 : MODEL COMPARISON
with tab3:
    st.header("Model Evaluation & Comparison")
    st.write("Performance metrics across different algorithms evaluated on the validation set.")
    
    # Affichage du classement des modèles
    if results_df is not None:
        st.subheader("Algorithm Leaderboard")
        st.dataframe(results_df.style.highlight_max(subset=['Accuracy'], color='lightgreen'))
    
    # Affichage des graphiques générés à l'étape 3
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Accuracy Comparison")
        if os.path.exists("model_accuracies.png"):
            st.image("model_accuracies.png", use_container_width=True)
        else:
            st.warning("Image 'model_accuracies.png' not found in directory.")
            
    with col2:
        st.subheader("Confusion Matrices")
        if os.path.exists("confusion_matrices.png"):
            st.image("confusion_matrices.png", use_container_width=True)
        else:
            st.warning("Image 'confusion_matrices.png' not found in directory.")
            
    # Graphique d'importance des variables généré à la volée
    st.subheader("XGBoost Feature Importance")
    importance = model.feature_importances_
    imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importance}).sort_values(by='Importance', ascending=False)
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    sns.barplot(data=imp_df.head(10), x='Importance', y='Feature', palette='magma', ax=ax3)
    ax3.set_xlabel("Relative Importance")
    st.pyplot(fig3)

# ONGLET 4 : INTERACTIVE SIMULATOR
with tab4:
    st.header("Interactive Survival Simulator")
    st.write("Adjust the parameters below to predict the survival probability of a custom passenger based on our optimized XGBoost engine.")
    
    # Création du formulaire de saisie
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pclass = st.selectbox("Passenger Class", [1, 2, 3], index=2)
            sex = st.selectbox("Gender", ["male", "female"])
            age = st.slider("Age", 0, 80, 25)
            
        with col2:
            sibsp = st.number_input("Siblings/Spouses Aboard", 0, 8, 0)
            parch = st.number_input("Parents/Children Aboard", 0, 6, 0)
            fare = st.slider("Ticket Fare (£)", 0.0, 500.0, 15.0)
            
        with col3:
            embarked = st.selectbox("Port of Embarkation", ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"])
            title = st.selectbox("Passenger Title", ["Mr", "Mrs", "Miss", "Rare"])
            deck = st.selectbox("Cabin Deck", ["M (Missing)", "A", "B", "C", "D", "E", "F", "G"])
            
        submit = st.form_submit_button("Predict Survival")
        
    # Logique de prédiction exécutée lors de la soumission du formulaire
    if submit:
        # 1. Initialisation du dictionnaire avec toutes les colonnes à 0
        input_dict = {col: 0 for col in feature_cols}
        
        # 2. Assignation des variables numériques
        input_dict['Pclass'] = pclass
        input_dict['Age'] = age
        input_dict['SibSp'] = sibsp
        input_dict['Parch'] = parch
        input_dict['Fare'] = fare
        input_dict['FamilySize'] = sibsp + parch + 1
        input_dict['IsAlone'] = 1 if input_dict['FamilySize'] == 1 else 0
        input_dict['Ticket_Freq'] = 1 
        
        # 3. Encodage One-Hot des variables catégorielles
        if sex == "male": input_dict['Sex_male'] = 1
        
        if "Q" in embarked: input_dict['Embarked_Q'] = 1
        elif "S" in embarked: input_dict['Embarked_S'] = 1
        
        if title == "Miss": input_dict['Title_Miss'] = 1
        elif title == "Mr": input_dict['Title_Mr'] = 1
        elif title == "Mrs": input_dict['Title_Mrs'] = 1
        elif title == "Rare": input_dict['Title_Rare'] = 1
        
        deck_initial = deck[0]
        if f"Deck_{deck_initial}" in input_dict:
            input_dict[f"Deck_{deck_initial}"] = 1
            
        # 4. Conversion en DataFrame (l'ordre des colonnes est garanti par feature_cols)
        input_df = pd.DataFrame([input_dict])[feature_cols]
        
        # 5. Calcul des prédictions
        prob = model.predict_proba(input_df)[0][1]
        pred = model.predict(input_df)[0]
        
        # 6. Affichage des résultats
        st.markdown("---")
        if pred == 1:
            st.success(f"🎉 **Survival Predicted!** Based on the model, this passenger has a **{prob*100:.1f}%** chance of surviving the disaster.")
            st.balloons()
        else:
            st.error(f"💀 **Perished.** Based on the model, this passenger has only a **{prob*100:.1f}%** chance of survival.")