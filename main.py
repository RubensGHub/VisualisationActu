from clustering import charger_donnees, clusteriser_bertopic, clusteriser_pipelineClassique





if __name__ == "__main__":
    # Charger les données
    df, titres = charger_donnees("data/sorted_data.xlsx")

    # Clusterisation avec BERTopic
    df_bertopic, resume_bertopic = clusteriser_bertopic(df.copy(), titres)
    
    # Clusterisation avec pipeline classique (Embeddings + UMAP + HDBSCAN + TF-IDF)   
    df_classique, resume_classique = clusteriser_pipelineClassique(df.copy(), titres) 
   