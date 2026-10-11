# Fichier python pour l'analyse exploratoire des données (EDA) du projet "Titanic" de Kaggle

# Importation des bibliothèques nécessaires
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Fonction pour charger les données et gérer les erreurs
def load_data(file_path):
    """Charge le dataset et gère les erreurs potentielles."""

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} est introuvable. Vérifiez le chemin.")
    return pd.read_csv(file_path)

# Fonction pour afficher les statistiques de base et les valeurs manquantes
def basic_statistics(df):

    """Affiche les statistiques de base et les valeurs manquantes."""
    print("--- APERCU DES DONNEES ---")
    print(df.head(), "\n")
    
    print("--- INFORMATIONS SUR LE DATASET ---")
    print(df.info(), "\n")
    
    print("--- STATISTIQUES DESCRIPTIVES ---")
    print(df.describe(include='all'), "\n")
    
    print("--- VALEURS MANQUANTES ---")
    missing_data = df.isnull().sum()
    print(missing_data[missing_data > 0].sort_values(ascending=False), "\n")

# Fonction pour générer des graphiques exploratoires
def plot_exploratory_analysis(df):
    """Génère et affiche des graphiques pour analyser les variables clés."""
    sns.set_theme(style="whitegrid")
    
    # Création d'une figure avec plusieurs sous-graphiques
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Analyse Exploratoire des Données (Titanic)", fontsize=16)

    # 1. Survie par Classe (Pclass)
    sns.countplot(data=df, x='Pclass', hue='Survived', ax=axes[0, 0], palette='Set1')
    axes[0, 0].set_title('Survie selon la Classe du Billet')
    axes[0, 0].set_ylabel('Nombre de passagers')

    # 2. Survie par Sexe (Sex)
    sns.countplot(data=df, x='Sex', hue='Survived', ax=axes[0, 1], palette='Set2')
    axes[0, 1].set_title('Survie selon le Sexe')
    axes[0, 1].set_ylabel('Nombre de passagers')

    # 3. Distribution de l'Âge (Age) par Survie
    sns.histplot(data=df, x='Age', hue='Survived', multiple="stack", bins=30, ax=axes[1, 0], palette='viridis')
    axes[1, 0].set_title('Distribution de l\'Âge selon la Survie')
    axes[1, 0].set_ylabel('Nombre de passagers')

    # 4. Survie par Port d'embarquement (Embarked)
    sns.countplot(data=df, x='Embarked', hue='Survived', ax=axes[1, 1], palette='pastel')
    axes[1, 1].set_title('Survie selon le Port d\'Embarquement')
    axes[1, 1].set_ylabel('Nombre de passagers')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # 5. Correlation entre les variables numériques et la survie
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = df[numeric_features].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title('Matrice de Corrélation entre les Variables Numériques')
    plt.show()

    # 6. Sauvegarde des graphiques dans un fichier PNG
    output_dir = 'eda_plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig.savefig(os.path.join(output_dir, 'eda_analysis.png'))

    # 7. Sauvegarde de la matrice de corrélation
    correlation_matrix.to_csv(os.path.join(output_dir, 'correlation_matrix.csv'))




# Fonction principale pour orchestrer l'analyse exploratoire
def main():

    # 1. Chargement des données d'entraînement
    print("Chargement des donnees train.csv...")
    train_df = load_data('train.csv')
    
    # 2. Statistiques et valeurs manquantes
    basic_statistics(train_df)
    
    # 3. Visualisation des corrélations avec 'Survived'
    print("Generation des graphiques exploratoires...")
    plot_exploratory_analysis(train_df)

if __name__ == "__main__":
    main()