# kaggle-titanic-end-to-end-pipeline

## 🇫🇷 Version Française

### Vue d'ensemble
Ce dépôt contient un projet complet de Machine Learning pour la célèbre compétition Kaggle : **Titanic - Machine Learning from Disaster**.
Le projet illustre un flux de travail rigoureux en Data Science, de l'exploration des données brutes jusqu'au déploiement d'une application web interactive. Il s'appuie sur **XGBoost** optimisé avec **Optuna** pour prédire la survie des passagers grâce à un Feature Engineering avancé.

### Structure du Projet
Le pipeline est divisé en 5 scripts Python distincts pour garantir la modularité et la lisibilité du code :

*   **`eda.py` (Analyse Exploratoire des Données) :** Analyse les données brutes, génère des graphiques de répartition (par classe, sexe, âge) et identifie les valeurs manquantes.
*   **`preprocessing.py` (Feature Engineering) :** Nettoie les données et crée des variables prédictives puissantes (Extraction des Titres, extraction du Pont de la Cabine, taille de la famille, fréquence des billets, et imputation intelligente de l'âge).
*   **`model.py` (Modélisation et Évaluation) :** Compare plusieurs algorithmes (Régression Logistique, Random Forest, Réseaux de Neurones) et optimise les hyperparamètres de XGBoost via Optuna. Génère les matrices de confusion et les graphiques d'accuracy.
*   **`submission.py` (Soumission Kaggle) :** Ré-entraîne le meilleur modèle sur 100% des données avec une validation croisée (Stratified K-Fold) et génère le fichier `submission.csv` final.
*   **`dashboard.py` (Application Streamlit) :** Un tableau de bord interactif regroupant l'EDA, l'importance des variables, et un "Simulateur de Survie" permettant de tester les probabilités de survie d'un profil passager personnalisé.

### Performances
*   **Accuracy en Validation (XGBoost + Optuna) :** ~84.9%
*   **Score Public Kaggle :** ~0.761

### Comment lancer le projet

1. **Installer les dépendances :**
   ```bash
   pip install pandas numpy scikit-learn xgboost optuna matplotlib seaborn streamlit
   ```

2. **Exécuter le pipeline étape par étape :**
   ```bash
   python eda.py
   python preprocessing.py
   python model.py
   python submission.py
   ```

3. **Lancer le dashboard web :**
   ```bash
   streamlit run dashboard.py
   ```
   
**Project Description:** An end-to-end machine learning pipeline predicting Titanic passenger survival, featuring advanced feature engineering, an Optuna-optimized XGBoost model, and an interactive Streamlit web dashboard.

## 🇬🇧 English Version

### Overview
This repository contains a complete, end-to-end machine learning project for the classic Kaggle competition: **Titanic - Machine Learning from Disaster**. 
The project demonstrates a rigorous data science workflow, from raw data exploration to the deployment of an interactive web application. It leverages **XGBoost** optimized with **Optuna** to predict passenger survival based on advanced feature engineering.

### Project Structure
The pipeline is divided into 5 distinct Python scripts to ensure modularity and clean code:

*   **`eda.py` (Exploratory Data Analysis):** Analyzes raw data, visualizes survival distributions (by class, gender, age), and identifies missing values.
*   **`preprocessing.py` (Feature Engineering):** Cleans data and creates highly predictive features (Title extraction, Deck extraction from Cabin, Family Size calculation, Ticket Frequency, and smart Age imputation).
*   **`model.py` (Modeling & Evaluation):** Compares multiple baseline models (Logistic Regression, Random Forest, Neural Networks) and performs Bayesian hyperparameter tuning on XGBoost using Optuna. Outputs confusion matrices and accuracy metrics.
*   **`submission.py` (Kaggle Submission):** Re-trains the optimal XGBoost model on 100% of the training data with Stratified K-Fold cross-validation and generates the final `submission.csv` for Kaggle.
*   **`dashboard.py` (Streamlit App):** A fully interactive dashboard featuring the EDA insights, model feature importance, and a "Survival Simulator" where users can input custom passenger traits to get live survival probabilities.

### Performance
*   **Validation Accuracy (XGBoost + Optuna):** ~84.9%
*   **Kaggle Public Score:** ~0.761

### How to Run

1. **Install dependencies:**
   ```bash
   pip install pandas numpy scikit-learn xgboost optuna matplotlib seaborn streamlit
   ```

2. **Execute the pipeline sequentially:**
   ```bash
   python eda.py
   python preprocessing.py
   python model.py
   python submission.py
   ```

3. **Launch the interactive dashboard:**
   ```bash
   streamlit run dashboard.py
   ```
