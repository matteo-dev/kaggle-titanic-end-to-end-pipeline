# Fichier python pour la partie "Model" du projet "Titanic" de Kaggle

# Importation des bibliothèques necessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb # Bibliothèque pour XGBoost
import optuna # Bibliothèque pour l'optimisation des hyperparamètres
from sklearn.model_selection import train_test_split # Pour separer les donnees en train/validation
from sklearn.linear_model import LogisticRegression # Modèle de regression logistique
from sklearn.ensemble import RandomForestClassifier # Modèle de forêt aleatoire
from sklearn.neural_network import MLPClassifier # Modèle de reseau de neurones (MLP)
from sklearn.metrics import accuracy_score, confusion_matrix # Pour evaluer les modèles
import warnings

# Desactivation des warnings pour une console propre
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Fonction pour charger les donnees et les separer en ensembles d'entraînement et de validation
def load_and_split_data(filepath):
    """Charge les donnees nettoyees et separe en train/validation."""

    df = pd.read_csv(filepath)
    X = df.drop(columns=['Survived'])
    y = df['Survived']

    # stratify=y garantit la même proportion de survivants dans train et val
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fonction pour optimiser les hyperparamètres de XGBoost via Optuna
def optimize_xgboost(X_train, y_train, X_val, y_val, n_trials=30):
    """Fonction d'optimisation des hyperparamètres XGBoost via Optuna."""

    print(f"\n--- Lancement d'Optuna pour XGBoost ({n_trials} trials) ---")
    
    # Definition de la fonction objective pour Optuna
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 400),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'random_state': 42
        }
        
        # Entraînement du modèle XGBoost avec les hyperparamètres proposes
        model = xgb.XGBClassifier(**param)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return accuracy_score(y_val, preds)

    # Creation de l'etude Optuna et optimisation
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"Meilleure Accuracy Optuna (Validation): {study.best_value:.4f}")
    print("Meilleurs hyperparametres :", study.best_params)
    
    # Entraînement du meilleur modèle trouve
    best_xgb = xgb.XGBClassifier(**study.best_params, use_label_encoder=False, eval_metric='logloss', random_state=42)
    best_xgb.fit(X_train, y_train)
    return best_xgb

# Fonction pour generer et sauvegarder les graphiques des resultats
def plot_and_save_results(results_df, conf_matrices):
    """Genère et sauvegarde les graphiques des resultats."""
    sns.set_theme(style="whitegrid")
    
    # 1. Graphique des Accuracies
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Accuracy', y='Modele', data=results_df.sort_values(by='Accuracy', ascending=False), palette='viridis')
    plt.title('Comparaison de l\'Accuracy des differents modèles (Validation Set)', fontsize=14)
    plt.xlim(0.7, 1.0)
    for i in ax.containers:
        ax.bar_label(i, fmt='%.4f', padding=5)
    plt.tight_layout()
    plt.savefig('model_accuracies.png')
    plt.close()

    # 2. Grille des Matrices de Confusion
    num_models = len(conf_matrices)
    fig, axes = plt.subplots(int(np.ceil(num_models/2)), 2, figsize=(12, 4 * int(np.ceil(num_models/2))))
    fig.suptitle('Matrices de Confusion par Modèle', fontsize=16)
    axes = axes.flatten()

    for i, (model_name, cm) in enumerate(conf_matrices.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
        axes[i].set_title(model_name)
        axes[i].set_xlabel('Predictions')
        axes[i].set_ylabel('Valeurs Reelles')

    # Masquer les subplots vides si le nombre de modèles est impair
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('confusion_matrices.png')
    plt.show()

# Fonction principale pour orchestrer le processus de modelisation
def main():
    print("--- DEBUT DE LA MODELISATION ---")
    
    # 1. Chargement des donnees
    X_train, X_val, y_train, y_val = load_and_split_data('train_clean.csv')
    
    # 2. Definition du catalogue de modèles de base
    models = {
        "Regression Logistique": LogisticRegression(max_iter=2000, random_state=42),
        "Random Forest (Base)": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost (Base)": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "Reseau de Neurones (MLP)": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    }
    
    # 3. Entraînement et evaluation des modèles de base
    results = []
    conf_matrices = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        results.append({'Modele': name, 'Accuracy': acc})
        conf_matrices[name] = confusion_matrix(y_val, preds)
        print(f"{name} entraine. Accuracy : {acc:.4f}")
    
    # 4. Optimisation du meilleur candidat (XGBoost)
    best_xgb_model = optimize_xgboost(X_train, y_train, X_val, y_val, n_trials=40)
    best_xgb_preds = best_xgb_model.predict(X_val)
    best_xgb_acc = accuracy_score(y_val, best_xgb_preds)
    
    # Ajout du modèle optimise aux resultats
    results.append({'Modele': 'XGBoost (Optimise Optuna)', 'Accuracy': best_xgb_acc})
    conf_matrices['XGBoost (Optimise Optuna)'] = confusion_matrix(y_val, best_xgb_preds)
    
    # 5. Sauvegarde et Visualisation
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='Accuracy', ascending=False)
    results_df.to_csv('model_results.csv', index=False)
    
    print("\n--- RESULTATS FINAUX ---")
    print(results_df.to_string(index=False))
    print("\nGeneration des graphiques...")
    plot_and_save_results(results_df, conf_matrices)
    print("Fichiers 'model_results.csv', 'model_accuracies.png' et 'confusion_matrices.png' generes avec succes")

if __name__ == "__main__":
    main()