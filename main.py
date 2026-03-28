from clustering import charger_donnees, clusteriser_bertopic
from visu_front import visualisation_chart


if __name__ == "__main__":
    # Charger les données
    print("Chargement des données...")
    df, titres = charger_donnees("data/sorted_data.xlsx")

    # Clusterisation avec BERTopic
    print("Clusterisation avec BERTopic...")
    df_bertopic, resume_bertopic = clusteriser_bertopic(df, titres)

    # Front : 
    print("Génération du front...")
    id_to_titre = df_bertopic.set_index("id_article")["titre"].to_dict()
    resume_bertopic["liste_titres"] = resume_bertopic["liste_ids_articles"].apply(
        lambda ids: [id_to_titre[i] for i in ids if i in id_to_titre]
    )
    visualisation_chart(df_bertopic, resume_bertopic)


