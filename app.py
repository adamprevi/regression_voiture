# ==============================
# IMPORTATION
# ==============================
import gradio as gr
import joblib as jb
import numpy as np


# ==============================
# CHARGEMENT DES ENCODEURS
# ==============================
encoders = jb.load('encoders.joblib')


# ==============================
# CHARGEMENT DU PIPELINE XGBOOST
# ==============================
rf_model = jb.load('pipe_from_grid_rf.joblib')


# ==============================
# FONCTION DE PREDICTION
# ==============================
def Pred_func(kms_driven, present_price, fuel_type,
              seller_type, transmission, age):

    # Encoder les variables catégorielles
    fuel_type = encoders[0].transform([fuel_type])[0]
    seller_type = encoders[1].transform([seller_type])[0]
    transmission = encoders[2].transform([transmission])[0]

    # Créer le vecteur des variables
    x_new = np.array([
        kms_driven,
        present_price,
        fuel_type,
        seller_type,
        transmission,
        age
    ])

    # Transformer en tableau 2D
    x_new = x_new.reshape(1, -1)

    # Prédiction
    y_pred = rf_model.predict(x_new)

    # Retourner le prix prédit
    return round(float(y_pred[0]), 2)
# Fonction de prédiction multiple
def Pred_func_csv(file):

    # Lire le fichier CSV
    df = pd.read_csv(file)

    predictions = []

    # Parcourir les lignes du dataframe
    for _, row in df.iterrows():

        # Prédiction
        y_pred = Pred_func(
            row['Kms_Driven'],
            row['Present_Price'],
            row['Fuel_Type'],
            row['Seller_Type'],
            row['Transmission'],
            row['Age']
        )

        predictions.append(y_pred)

    # Ajouter les prédictions
    df['Selling_Price_Predite'] = predictions

    # Sauvegarder le fichier
    df.to_csv('predictions.csv', index=False)

    # Retourner le fichier
    return 'predictions.csv'
  # Définir les blocks
demo = gr.Blocks(theme='shivi/calm_seafoam')


# ==============================
# INTERFACE 1 : PRÉDICTION SIMPLE
# ==============================

inputs = [
    gr.Number(label='Kilométrage (Kms_Driven)'),
    
    gr.Number(label='Prix actuel (Present_Price)'),
    
    gr.Dropdown(
        choices=uniques[0],
        label='Type de carburant (Fuel_Type)'
    ),
    
    gr.Dropdown(
        choices=uniques[1],
        label='Type de vendeur (Seller_Type)'
    ),
    
    gr.Dropdown(
        choices=uniques[2],
        label='Transmission'
    ),
    
    gr.Number(label='Âge du véhicule (Age)')
]


# Sortie
outputs = gr.Number(label='Prix de vente prédit')


# Interface 1
interface1 = gr.Interface(
    fn=Pred_func,
    inputs=inputs,
    outputs=outputs,
    title="Prédiction du prix d'une voiture",
    description="""
    Ce modèle permet de prédire le prix de vente d'une voiture
    à partir du kilométrage, du prix actuel, du type de carburant,
    du type de vendeur, de la transmission et de l'âge du véhicule.
    """
)


# ==============================
# INTERFACE 2 : PRÉDICTION MULTIPLE
# ==============================

interface2 = gr.Interface(
    fn=Pred_func_csv,
    
    inputs=gr.File(
        label='Importer un fichier CSV',
        file_types=['.csv']
    ),
    
    outputs=gr.File(
        label='Télécharger les prédictions'
    ),
    
    title="Prédiction multiple du prix des voitures",
    
    description="""
    Importez un fichier CSV contenant les caractéristiques des voitures.
    Le modèle prédira automatiquement le prix de vente de chaque voiture.
    """
)


# ==============================
# TABBING DES INTERFACES
# ==============================

with demo:
    gr.TabbedInterface(
        [interface1, interface2],
        ['Prédiction simple', 'Prédiction multiple']
    )


# ==============================
# LANCEMENT
# ==============================

demo.launch(share=True)

