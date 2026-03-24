import pandas as pd



def charger_donnees():
    df = pd.read_excel("sorted_data.xlsx", names=["date", "titre", "lien"], header=0)

    # Création d'un ID unique pour chaque article basé sur l'index
    df['id_article'] = df.index

    # Extraction des titres sous forme de liste pour les modèles NLP
    titres = df["titre"].tolist()
    return df, titres



