import os
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP
from hdbscan import HDBSCAN
from tqdm import tqdm
import hashlib
import json
import numpy as np
import os
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer


nltk.download('stopwords', quiet=True)
mots_vides_fr = stopwords.words('french')
mots_vides_fr.extend(['euros', 'euro', 'dollars', 'dollar', 'milliards', 'millions', 'mds', 'plus', 'très', 'cette', 'cet', 'comme', 'tout', 'faire', 'ça', 'ont', 'être'])

MACRO_THEMES = ["nucléaire", "politique", "énergie renouvelable", "aviation","gaz", "pétrole", "loi", "carbone", "véhicule", "électrique","bâtiment", "CO2", "climat", "eau", "frugal", "IA",]


OUTPUT_DIR = "data/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_umap_hash(params):
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(param_str.encode()).hexdigest()

def get_embedding_hash(titres, model_name):
    data = {
        "model": model_name,
        "titres": titres  
    }
    data_str = json.dumps(data, sort_keys=True)
    return hashlib.md5(data_str.encode()).hexdigest()


def charger_donnees(path):
    df = pd.read_excel(path, names=["date", "titre", "lien"], header=0)

    # Création d'un ID unique pour chaque article basé sur l'index
    df['id_article'] = df.index

    # Extraction des titres sous forme de liste pour les modèles NLP
    titres = df["titre"].tolist()
    return df, titres





def compute_or_load_embeddings(titres, model, cache_dir="cache"):
    os.makedirs(cache_dir, exist_ok=True)

    model_name = model.get_sentence_embedding_dimension()  
    hash_id = get_embedding_hash(titres, str(model_name))

    cache_file = f"{cache_dir}/embeddings_{hash_id}.npy"

    if os.path.exists(cache_file):
        print("Chargement des embeddings depuis le cache...")
        return np.load(cache_file)

    print("Calcul des embeddings...")
    embeddings = model.encode(titres, show_progress_bar=True)

    np.save(cache_file, embeddings)
    print("Embeddings sauvegardés.")

    return embeddings


    
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

     # Création du Vectorizer avec le filtre anti-chiffres et les mots vides
    vectorizer_model = CountVectorizer(
        stop_words=mots_vides_fr,
        token_pattern=r"(?u)\b[a-zA-ZÀ-ÿ]{3,}\b" # Ignore les nombres et garde les mots >= 3 lettres
    )
    steps = [
        "Chargement du modèle d'embedding",
        "Création UMAP",
        "Création HDBSCAN",
        "Initialisation BERTopic",
        "Entraînement BERTopic",
        "Réduction des outliers",
        "Mise à jour des topics",
        "Post-traitement DataFrame",
        "Sauvegarde"
    ]

    with tqdm(total=len(steps), desc="Pipeline BERTopic") as pbar:

        embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        embeddings = compute_or_load_embeddings(titres, embedding_model)
        pbar.update(1)

        umap_model = UMAP(
            n_neighbors=20,
            n_components=5,
            metric="cosine",
            random_state=42
        )
        pbar.update(1)

        hdbscan_model = HDBSCAN(
            min_cluster_size=50,
            min_samples=5,
            metric='euclidean',
            prediction_data=True
        )
        pbar.update(1)


        seed_topic_list = [[theme] for theme in MACRO_THEMES]

        # Initialisation et entraînement de BERTopic
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            min_topic_size=50,
            language="multilingual",
            seed_topic_list=seed_topic_list
        )
        pbar.update(1)

        topics, _ = topic_model.fit_transform(titres, embeddings)
        nb_bruit_avant = sum(1 for t in topics if t == -1)
        print(f"Articles en bruit avant reduce_outliers : {nb_bruit_avant}/{len(topics)} ({100*nb_bruit_avant/len(topics):.1f}%)")
        pbar.update(1)
        topics = topic_model.reduce_outliers(titres, topics, strategy="c-tf-idf", threshold=0.1)
        topic_model.update_topics(titres, topics=topics)
        pbar.update(1)
        nb_bruit_apres = sum(1 for t in topics if t == -1)
        print(f"Articles en bruit après reduce_outliers  : {nb_bruit_apres}/{len(topics)} ({100*nb_bruit_apres/len(topics):.1f}%)")
        pbar.update(1)
        # Assigner l'ID du sujet à chaque article dans le DataFrame
        df['id_sujet'] = topics

        topic_info = topic_model.get_topic_info()
        dict_IdNoms = dict(zip(topic_info['Topic'], topic_info['Name']))
        df['nom_sujet'] = df['id_sujet'].map(dict_IdNoms)
        
        resume_bertopic = df.groupby(['id_sujet', 'nom_sujet']).agg(
            nombre_articles=('id_article', 'count'),
            liste_ids_articles=('id_article', lambda x: list(x))
        ).reset_index()

        pbar.update(1)
        # Sauvegarder le petit tableau résumé du compte des articles
        resume_bertopic.to_excel(f"{OUTPUT_DIR}/resume_bertopic.xlsx", index=False)
        # Sauvegarder les données avec les nouvelles colonnes 'id_sujet' et 'nom_sujet'
        df.to_excel(f"{OUTPUT_DIR}/clustering_bertopic.xlsx", index=False)

        pbar.update(1)

        print("Les fichiers du modèle BERTopic ont été sauvegardés !")
        return df, resume_bertopic, topic_model, embedding_model

def macro_clustering(df, topic_model, embedding_model, seuil_similarite=0.3):
    """
    Associe les sous-clusters générés par BERTopic à la liste prédéfinie de macro-thèmes.
    """
     
    # Calculer les embeddings des grandes catégories
    print("Calcul des vecteurs pour les macro-thèmes...")
    macro_embeddings = embedding_model.encode(MACRO_THEMES)
    
    # Récupérer les infos des sujets
    topic_info = topic_model.get_topic_info()
    dict_macro_mapping = {}
    
    # Associer chaque sous-cluster
    for index, row in topic_info.iterrows():
        topic_id = row['Topic']
        
        if topic_id == -1:
            dict_macro_mapping[topic_id] = "Bruit"
            continue
            
        # Extraire les mots clés de ce sous-cluster
        words = [word for word, score in topic_model.get_topic(topic_id)]
        topic_text = " ".join(words[:5]) 
        
        # Calculer la similarité cosinus
        topic_emb = embedding_model.encode([topic_text])
        similarities = cosine_similarity(topic_emb, macro_embeddings)[0]
        
        # Trouver le meilleur match
        best_match_idx = np.argmax(similarities)
        best_match_score = similarities[best_match_idx]
        
        if best_match_score > seuil_similarite:
            dict_macro_mapping[topic_id] = MACRO_THEMES[best_match_idx]
        else:
            dict_macro_mapping[topic_id] = "Autre"
            
    # Appliquer au DataFrame
    df['macro_sujet'] = df['id_sujet'].map(dict_macro_mapping)
    
    # Créer le résumé 
    print("Génération du tableau résumé...")
    resume_macro = df.groupby(['macro_sujet', 'id_sujet', 'nom_sujet']).agg(
        nombre_articles=('id_article', 'count'),
        liste_ids_articles=('id_article', lambda x: list(x))
    ).reset_index()

    resume_macro.to_excel(f"{OUTPUT_DIR}/resume_clustering_macro.xlsx", index=False)
    
    print("Fichier sauvegardé avec succès !")
    print("Clustering terminé avec succès !")
    return df, resume_macro
