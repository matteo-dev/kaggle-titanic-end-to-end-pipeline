# Fichier python pour la partie "Preprocessing" du projet "Titanic" de Kaggle

# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np

# Fonction pour extraire les titres depuis la colonne Name
def extract_titles(df):
    """Extrait les titres (Mr, Mrs, etc.) depuis la colonne Name."""

    # Le 'r' devant la chaîne corrige le SyntaxWarning
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    
    # Regroupement des titres rares
    rare_titles = ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 
                   'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    
    # Standardisation
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    return df

# Fonction principale de preprocessing
def preprocess_data(df, is_train=True):
    """Applique le nettoyage et le feature engineering avancé."""

    df_clean = df.copy()
    
    # 1. Feature Engineering : Titres
    df_clean = extract_titles(df_clean)
    
    # 2. Imputations
    df_clean['Age'] = df_clean.groupby(['Title', 'Pclass'])['Age'].transform(
        lambda x: x.fillna(x.median())
    )
    df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median())
    df_clean['Embarked'] = df_clean['Embarked'].fillna('S')
    df_clean['Fare'] = df_clean['Fare'].fillna(df_clean['Fare'].median())
    
    # 3. Extraction du Pont (Deck) de la Cabine
    # On prend la 1ère lettre. Si NaN, on met 'M' pour Missing
    df_clean['Deck'] = df_clean['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
    
    # 4. Feature Engineering : Famille et Groupes
    df_clean['FamilySize'] = df_clean['SibSp'] + df_clean['Parch'] + 1
    df_clean['IsAlone'] = (df_clean['FamilySize'] == 1).astype(int)
    
    # Fréquence du Billet (Groupes cachés)
    # Compte combien de passagers partagent le même numéro de billet
    df_clean['Ticket_Freq'] = df_clean.groupby('Ticket')['Ticket'].transform('count')
    
    # 5. Suppression des colonnes devenues inutiles
    cols_to_drop = ['Name', 'Ticket', 'Cabin']
    if is_train:
        cols_to_drop.append('PassengerId')
        
    df_clean = df_clean.drop(columns=cols_to_drop)
    
    # 6. Encodage des variables catégorielles (On ajoute 'Deck')
    df_clean = pd.get_dummies(df_clean, columns=['Sex', 'Embarked', 'Title', 'Deck'], drop_first=True)
    
    return df_clean

# Fonction principale pour orchestrer le preprocessing
def main():
    print("--- DEBUT DU PREPROCESSING AVANCE ---")
    
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    # Preprocessing
    train_clean = preprocess_data(train_df, is_train=True)
    test_clean = preprocess_data(test_df, is_train=False)
    
    # Alignement des colonnes (Train vs Test)
    train_features = train_clean.drop(columns=['Survived'])
    test_features = test_clean.drop(columns=['PassengerId'])
    
    for col in train_features.columns:
        if col not in test_features.columns:
            test_features[col] = 0
            
    test_features = test_features[train_features.columns]
    
    # Reconstruction
    train_final = pd.concat([train_clean[['Survived']], train_features], axis=1)
    test_final = pd.concat([test_clean[['PassengerId']], test_features], axis=1)
    
    print(f"Nouvelle taille de train_final : {train_final.shape}")
    
    # Sauvegarde
    train_final.to_csv('train_clean.csv', index=False)
    test_final.to_csv('test_clean.csv', index=False)
    
    print("Fichiers generes / Voici les variables qui nourriront le modele :")
    print(list(train_features.columns))

if __name__ == "__main__":
    main()