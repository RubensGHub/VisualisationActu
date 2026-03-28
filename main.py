from clustering import charger_donnees, clusteriser_bertopic


if __name__ == "__main__":
    # Charger les données
    print("Chargement des données...")
    df, titres = charger_donnees("data/sorted_data.xlsx")

    # Clusterisation avec BERTopic
    print("Clusterisation avec BERTopic...")
    df_bertopic, resume_bertopic = clusteriser_bertopic(df, titres)
