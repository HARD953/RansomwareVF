import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Charger le modèle, le scaler, et le label encoder
best_model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Liste complète des valeurs possibles pour 'Target Audience'
target_audience_values = [
    "Particuliers", "Utilisateurs de périphériques USB", "Amis", "Clients", "Collègues", 
    "Famille", "Professionnels", "Internautes", "Utilisateurs de sites de téléchargement", 
    "Employes d'entreprise", "Utilisateurs de messagerie électronique", 
    "Partenaires professionnels", "Utilisateurs d'applications mobiles", 
    "Utilisateurs de messagerie électronique"
]

# Initialiser le LabelEncoder et ajuster avec toutes les valeurs possibles
label_encoder = LabelEncoder()
label_encoder.fit(target_audience_values)

# Fonction de prédiction
def predire(data_simulation):
    if 'Target Audience' in data_simulation:
        try:
            data_simulation['Target Audience'] = label_encoder.transform([data_simulation['Target Audience']])[0]
        except ValueError as e:
            print(f"Erreur lors de l'encodage du 'Target Audience': {e}")
            return None

    df_simulation = pd.DataFrame([data_simulation])

    expected_columns = scaler.feature_names_in_  
    missing_columns = set(expected_columns) - set(df_simulation.columns)
    extra_columns = set(df_simulation.columns) - set(expected_columns)

    for col in missing_columns:
        df_simulation[col] = 0

    df_simulation = df_simulation[expected_columns]

    features = scaler.transform(df_simulation.values)

    prediction = best_model.predict(features)[0]
    return prediction

# Fonction pour enregistrer les données dans un fichier CSV
def enregistrer_donnees_csv(data_simulation, filename='predictions.csv'):
    df_simulation = pd.DataFrame([data_simulation])
    df_simulation.to_csv(filename, mode='a', header=not pd.io.common.file_exists(filename), index=False)

# Fonction pour calculer la matrice de confusion à partir d'un fichier CSV
def calculer_matrice_confusion(filename='predictions.csv'):
    df = pd.read_csv(filename)
    y_true = df['Hacker']
    y_pred = df['Prediction']
    matrix = confusion_matrix(y_true, y_pred)
    return matrix

# Fonction pour afficher et sauvegarder la matrice de confusion
def sauvegarder_matrice_confusion(matrix, filename='matrice_confusion.png'):
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non Hacker', 'Hacker'], yticklabels=['Non Hacker', 'Hacker'])
    plt.title('Matrice de Confusion')
    plt.xlabel('Prédictions')
    plt.ylabel('Réel')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Exemple de simulation de données pour prédire
data_simulation = {
    "Delivery_usb": 0, "Delivery_Web": 0, "Delivery_email": 1,
    "fileType_docx": 1, "fileType_xlsx": 1, "fileType_pdf": 0, "fileType_exe": 1, "fileType_zip": 1, "fileType_rar": 1,
    "FileEncryptionMethod_AES": 1, "FileEncryptionMethod_DES": 1, "FileEncryptionMethod_RSA": 0,
    "Deleting Backup": 0, "Communication C&C status": 0, "Payment Information": 1,
    "Target Audience": "Utilisateurs de messagerie électronique", "Normal": 1, "Hacker": 0
}

# Enregistrer les données dans le fichier CSV
prediction = predire(data_simulation)
data_simulation['Prediction'] = prediction
enregistrer_donnees_csv(data_simulation)

# Calculer la matrice de confusion
matrix = calculer_matrice_confusion()

# Afficher la prédiction et la matrice de confusion
print(f"Prédiction : {'Hacker' if prediction == 1 else 'Non Hacker'}")
print("Matrice de confusion :\n", matrix)

# Sauvegarder l'image de la matrice de confusion
sauvegarder_matrice_confusion(matrix)