import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer



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
    - Entraînement de BERTopic pour identifier les sujets
    - Assignation des IDs et noms de sujets à chaque article
    - Agrégation du nombre d'articles et des IDs par sujet
    Résultat : un DataFrame avec les sujets identifiés et un résumé du nombre d'articles par sujet
    Retourne les résultats et les sauvegarde dans des fichiers Excel
    """
    # Choix du modèle d'embedding multilingue pour mieux gérer les titres en français
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    # Initialisation et entraînement de BERTopic
    topic_model = BERTopic(embedding_model=embedding_model, language="multilingual")
    topics, probs = topic_model.fit_transform(titres) #renvoie -1 pour les titres pas classés

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




def extraire_noms_clusters_tfidf(df):
    """
    Génère des noms de sujets basés sur les 3 mots-clés les plus fréquents (TF-IDF) pour chaque cluster identifié.
    """
    # Regrouper tous les titres d'un même cluster
    docs_par_cluster = df.groupby('id_sujet')['titre'].apply(lambda x: ' '.join(x)).reset_index()
    
    # Préparer l'extracteur TF-IDF
    mots_vides_fr = ['le', 'la', 'les', 'de', 'des', 'du', 'un', 'une', 'en', 'et', 'pour', 'sur', 'dans', 'au', 'aux', 'avec', 'qui', 'que', 'est', 'sont', 'a', 'pas', 'plus', 'par', 'ou', 'ce', 'se']
    vectorizer = TfidfVectorizer(stop_words=mots_vides_fr)
    
    mapping_noms = {}
    
    # Calculer les mots-clés pour chaque cluster
    for index, row in docs_par_cluster.iterrows():
        cluster_id = row['id_sujet']
        
        # Traitement spécial pour le bruit (outliers)
        if cluster_id == -1:
            mapping_noms[cluster_id] = "-1_bruit_outliers"
            continue
            
        try:
            tfidf_matrix = vectorizer.fit_transform([row['titre']])
            mots = vectorizer.get_feature_names_out()
            # Récupérer les 3 mots avec le score le plus haut
            top_indices = np.argsort(tfidf_matrix.toarray()[0])[::-1][:3]
            top_mots = [mots[i] for i in top_indices]
            
            # Formater le nom final
            mapping_noms[cluster_id] = f"{cluster_id}_" + "_".join(top_mots)
        except ValueError:
            # Sécurité au cas où tous les mots du cluster seraient des mots vides
            mapping_noms[cluster_id] = f"{cluster_id}_sans_nom"
            
    return mapping_noms




def clusteriser_pipelineClassique(df, titres):
    """
    Clusterisation avec pipeline classique :
    - Génération d'embeddings avec un modèle multilingue
    - Réduction de dimension avec UMAP
    - Clustering avec HDBSCAN
    - Génération de noms de sujets avec TF-IDF
    - Assignation des IDs et noms de sujets à chaque article
    - Agrégation du nombre d'articles et des IDs par sujet
    Résultat : un DataFrame avec les sujets identifiés et un résumé du nombre d'articles par sujet
    Retourne les résultats et les sauvegarde dans des fichiers Excel
    """
    # Embeddings
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode(titres, show_progress_bar=True)

    # On réduit à 5 dimensions pour aider l'algorithme de clustering
    umap_embeddings = umap.UMAP(
        n_neighbors=15, 
        n_components=5, 
        metric='cosine',
        random_state=42 # Pour avoir les mêmes résultats à chaque exécution
    ).fit_transform(embeddings)

    # Clustering avec HDBSCAN    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5, #un sujet doit avoir au moins 5 articles pour exister
        metric='euclidean', 
        cluster_selection_method='eom'
    )
    clusters = clusterer.fit_predict(umap_embeddings)
    df['id_sujet'] = clusters # Assigner l'ID du sujet à chaque article dans le DataFrame

    #Appel de la fonction TF-IDF pour générer les noms de sujets
    mapping_noms = extraire_noms_clusters_tfidf(df) 
    df['nom_sujet'] = df['id_sujet'].map(mapping_noms)    # Assigner les noms générés au DataFrame


    # Nombre d'articles et liste des IDs par sujet
    resume_classique = df.groupby(['id_sujet', 'nom_sujet']).agg(
        nombre_articles=('id_article', 'count'),
        liste_ids_articles=('id_article', lambda x: list(x))
    ).reset_index()

    # Sauvegarder le petit tableau résumé du compte des articles
    resume_classique.to_excel("resume_pipelineClassique.xlsx", index=False)
    # Sauvgarder les données avec les nouvelles colonnes 'id_sujet' et 'nom_sujet'
    df.to_excel("clustering_pipelineClassique.xlsx", index=False)      

    print("Les fichiers du pipeline classique ont été sauvegardés !")

    # Affichage du résultat final
    print("Résultat Pipeline Classique")
    print(resume_classique.head())
    return df, resume_classique