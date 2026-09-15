# ==============================
# IMPORTATION
# ==============================
import streamlit as st
import joblib as jb
import numpy as np
import pandas as pd


# ==============================
# CHARGEMENT DES FICHIERS
# ==============================
encoders = jb.load('encoders.joblib')
rf_model = jb.load('pipe_from_grid_rf.joblib')
uniques = jb.load('uniques.joblib')


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


# ==============================
# FONCTION DE PREDICTION MULTIPLE
# ==============================
def Pred_func_csv(file):

    # Lire le fichier CSV
    df = pd.read_csv(file)

    predictions = []

    # Parcourir les lignes du dataframe
    for _, row in df.iterrows():

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

    return df


# ==============================
# CONFIGURATION DE LA PAGE
# ==============================
st.set_page_config(
    page_title="Prédiction du prix d'une voiture",
    page_icon="🚗",
    layout="centered"
)


# ==============================
# TITRE
# ==============================
st.title("🚗 Prédiction du prix d'une voiture")

st.write(
    """
    Cette application permet de prédire le prix de vente d'une voiture
    à partir de ses caractéristiques.
    """
)


# ==============================
# ONGLETS
# ==============================
tab1, tab2 = st.tabs([
    "🔮 Prédiction simple",
    "📂 Prédiction multiple"
])


# ==========================================================
# INTERFACE 1 : PRÉDICTION SIMPLE
# ==========================================================
with tab1:

    st.subheader("Prédiction du prix d'une voiture")

    kms_driven = st.number_input(
        "Kilométrage (Kms_Driven)",
        min_value=0.0,
        value=50000.0
    )

    present_price = st.number_input(
        "Prix actuel (Present_Price)",
        min_value=0.0,
        value=5.0
    )

    fuel_type = st.selectbox(
        "Type de carburant (Fuel_Type)",
        uniques[0]
    )

    seller_type = st.selectbox(
        "Type de vendeur (Seller_Type)",
        uniques[1]
    )

    transmission = st.selectbox(
        "Transmission",
        uniques[2]
    )

    age = st.number_input(
        "Âge du véhicule (Age)",
        min_value=0.0,
        value=5.0
    )

    if st.button("🚀 Prédire le prix"):

        prediction = Pred_func(
            kms_driven,
            present_price,
            fuel_type,
            seller_type,
            transmission,
            age
        )

        st.success(
            f"💰 Prix de vente prédit : {prediction:.2f} k$"
        )


# ==========================================================
# INTERFACE 2 : PRÉDICTION MULTIPLE
# ==========================================================
with tab2:

    st.subheader("Prédiction multiple")

    st.write(
        """
        Importez un fichier CSV contenant les caractéristiques
        des voitures.
        """
    )

    uploaded_file = st.file_uploader(
        "Choisir un fichier CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        if st.button("🚀 Lancer les prédictions"):

            result_df = Pred_func_csv(uploaded_file)

            st.success("Les prédictions ont été effectuées.")

            # Afficher les résultats
            st.dataframe(result_df)

            # Créer le fichier CSV
            csv = result_df.to_csv(index=False).encode("utf-8")

            # Bouton de téléchargement
            st.download_button(
                label="⬇️ Télécharger les prédictions",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )
