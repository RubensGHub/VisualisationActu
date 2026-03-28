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
    visualisation_chart(df_bertopic, resume_bertopic)


