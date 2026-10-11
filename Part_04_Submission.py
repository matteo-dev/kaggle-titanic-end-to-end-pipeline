# Fichier pour générer le fichier de soumission pour Kaggle après avoir trouvé les meilleurs hyperparamètres avec Optuna.

# Importation des bibliothèques nécessaires
import pandas as pd
import xgboost as xgb # Bibliothèque pour XGBoost
import optuna # Bibliothèque pour l'optimisation des hyperparamètres
from sklearn.model_selection import cross_val_score, StratifiedKFold # Pour la validation croisée
import warnings

# Désactivation des warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

#  Fonction pour trouver les meilleurs hyperparamètres avec validation croisée
def find_best_params(X, y, n_trials=50):
    """Trouve les meilleurs hyperparamètres avec validation croisée (Stratified K-Fold)."""

    print(f"Recherche des meilleurs hyperparametres ({n_trials} essais) sur 100% des donnees d'entrainement...")
    
    # Définition de la fonction objective pour Optuna
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 400),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 6),
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'random_state': 42
        }

        # Entraînement du modèle XGBoost avec les hyperparamètres proposés
        model = xgb.XGBClassifier(**param)

        # Validation croisée sur 5 plis pour être très robuste
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        return scores.mean()

    # Création de l'étude Optuna et optimisation
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"Meilleure Accuracy moyenne en Cross-Validation : {study.best_value:.4f}")
    return study.best_params

# Fonction principale pour générer le fichier de soumission
def main():
    print("--- DEBUT DE LA GENERATION DE LA SOUMISSION KAGGLE ---")
    
    # 1. Chargement des données nettoyées
    train_df = pd.read_csv('train_clean.csv')
    test_df = pd.read_csv('test_clean.csv')
    
    # Séparation Features / Target pour le train
    X_train = train_df.drop(columns=['Survived'])
    y_train = train_df['Survived']
    
    # Séparation Features / ID pour le test
    X_test = test_df.drop(columns=['PassengerId'])
    passenger_ids = test_df['PassengerId']
    
    # 2. Recherche des hyperparamètres optimaux
    best_params = find_best_params(X_train, y_train, n_trials=20)
    
    # 3. Entraînement du modèle final sur TOUTES les données d'entraînement
    print("\nEntrainement du modele XGBoost final sur 100% du dataset...")
    final_model = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss', random_state=42)
    final_model.fit(X_train, y_train)
    
    # 4. Prédictions sur le jeu de test
    print("Generation des predictions pour les passagers du test set...")
    predictions = final_model.predict(X_test)
    
    # 5. Création du fichier de soumission
    submission = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': predictions
    })
    
    # Vérification de la taille (Kaggle attend exactement 418 lignes)
    if len(submission) == 418:
        print("Verification OK : Le fichier contient bien 418 predictions.")
    else:
        print(f"ATTENTION : Le fichier contient {len(submission)} lignes au lieu de 418")
        
    submission.to_csv('submission.csv', index=False)
    print("\n Fichier 'submission.csv' genere avec succes")

if __name__ == "__main__":
    main()