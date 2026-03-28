import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN



def charger_donnees(path):
    df = pd.read_excel(path, names=["date", "titre", "lien"], header=0)

    # Création d'un ID unique pour chaque article basé sur l'index
    df['id_article'] = df.index

    # Extraction des titres sous forme de liste pour les modèles NLP
    titres = df["titre"].tolist()
    return df, titres


def clusteriser_bertopic(df, titres):
    """"
    Clusterisation avec BERTopic :
    - Utilisation d'un modèle d'embedding multilingue pour mieux gérer les titres en français
    - UMAP et HDBSCAN configurés pour réduire le bruit (articles classés -1)
    - Réduction des outliers post-hoc avec reduce_outliers()
    - Assignation des IDs et noms de sujets à chaque article
    - Agrégation du nombre d'articles et des IDs par sujet
    Résultat : un DataFrame avec les sujets identifiés et un résumé du nombre d'articles par sujet
    Retourne les résultats et les sauvegarde dans des fichiers Excel
    """
    # Choix du modèle d'embedding multilingue pour mieux gérer les titres en français
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    # UMAP personnalisé : n_neighbors bas pour capter les petits groupes
    umap_model = UMAP(
        n_neighbors=10,
        n_components=5,
        metric='cosine',
        random_state=42
    )

    # HDBSCAN personnalisé : min_cluster_size bas pour permettre de petits clusters
    hdbscan_model = HDBSCAN(
        min_cluster_size=5,
        min_samples=2,
        metric='euclidean',
        prediction_data=True
    )

    # Initialisation et entraînement de BERTopic
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        min_topic_size=3,
        language="multilingual"
    )
    topics, probs = topic_model.fit_transform(titres)

    topics = topic_model.reduce_outliers(titres, topics, strategy="probabilities")
    topic_model.update_topics(titres, topics=topics)

    nb_bruit = sum(1 for t in topics if t == -1)
    print(f"Articles restant en bruit après reduce_outliers : {nb_bruit}/{len(topics)} ({100*nb_bruit/len(topics):.1f}%)")

    # Assigner l'ID du sujet à chaque article dans le DataFrame
    df['id_sujet'] = topics

    # Récupérer le nom du sujet généré par BERTopic
    topic_info = topic_model.get_topic_info() #renvoie un DataFrame avec les ids des sujets Topic leurs noms et autres infos
    dict_IdNoms = dict(zip(topic_info['Topic'], topic_info['Name'])) #crée un dictionnaire {id : nom du sujet}
    df['nom_sujet'] = df['id_sujet'].map(dict_IdNoms)

    # Nombre d'articles et liste des IDs par sujet
    resume_bertopic = df.groupby(['id_sujet', 'nom_sujet']).agg(
        nombre_articles=('id_article', 'count'),
        liste_ids_articles=('id_article', lambda x: list(x))
    ).reset_index()

    # Sauvegarder le petit tableau résumé du compte des articles
    resume_bertopic.to_excel("resume_bertopic.xlsx", index=False)
    # sauvgarder les données avec les nouvelles colonnes 'id_sujet' et 'nom_sujet'
    df.to_excel("clustering_bertopic.xlsx", index=False)

    print("Les fichiers du modèle BERTopic ont été sauvegardés !")

    #Affichage du résultat final
    print("Résultat BERTopic")
    print(resume_bertopic.head())

    return df, resume_bertopic
