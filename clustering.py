import os
import os
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from tqdm import tqdm
import hashlib
import json
import numpy as np
import os



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


def compute_or_load_umap(embeddings, umap_params, cache_dir="cache"):
    os.makedirs(cache_dir, exist_ok=True)

    # Hash basé sur les paramètres
    hash_id = get_umap_hash(umap_params)
    cache_file = f"{cache_dir}/umap_{hash_id}.npy"

    if os.path.exists(cache_file):
        print("Chargement UMAP depuis le cache...")
        return np.load(cache_file)

    print("Calcul UMAP...")
    umap_model = UMAP(**umap_params)
    reduced = umap_model.fit_transform(embeddings)

    np.save(cache_file, reduced)
    print("UMAP sauvegardé.")

    return reduced


def compute_or_load_embeddings(titres, model, cache_dir="cache"):
    os.makedirs(cache_dir, exist_ok=True)

    model_name = model.get_sentence_embedding_dimension()  # ou nom manuel
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
    - UMAP et HDBSCAN configurés pour réduire le bruit (articles classés -1)
    - Réduction des outliers post-hoc avec reduce_outliers()
    - Assignation des IDs et noms de sujets à chaque article
    - Agrégation du nombre d'articles et des IDs par sujet
    Résultat : un DataFrame avec les sujets identifiés et un résumé du nombre d'articles par sujet
    Retourne les résultats et les sauvegarde dans des fichiers Excel
    """
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

        umap_params = {
            "n_neighbors": 10,
            "n_components": 5,
            "metric": "cosine",
            "random_state": 42
        }

        umap_embeddings = compute_or_load_umap(embeddings, umap_params)
        pbar.update(1)

        hdbscan_model = HDBSCAN(
            min_cluster_size=50,
            min_samples=5,
            metric='euclidean',
            prediction_data=True
        )
        pbar.update(1)

        topic_model = BERTopic(
            hdbscan_model=hdbscan_model,
            min_topic_size=3,
            language="multilingual"
        )
        pbar.update(1)

        topics, probs = topic_model.fit_transform(titres, umap_embeddings)
        pbar.update(1)


        topics = topic_model.reduce_outliers(
            titres, topics, probabilities=probs, strategy="probabilities"
        )
        pbar.update(1)

        topic_model.update_topics(titres, topics=topics)
        pbar.update(1)

        nb_bruit = sum(1 for t in topics if t == -1)

        df['id_sujet'] = topics

        topic_info = topic_model.get_topic_info()
        dict_IdNoms = dict(zip(topic_info['Topic'], topic_info['Name']))
        df['nom_sujet'] = df['id_sujet'].map(dict_IdNoms)

        resume_bertopic = df.groupby(['id_sujet', 'nom_sujet']).agg(
            nombre_articles=('id_article', 'count'),
            liste_ids_articles=('id_article', lambda x: list(x))
        ).reset_index()

        pbar.update(1)

        resume_bertopic.to_excel(f"{OUTPUT_DIR}/resume_bertopic.xlsx", index=False)
        df.to_excel(f"{OUTPUT_DIR}/clustering_bertopic.xlsx", index=False)

        pbar.update(1)

    print(f"Articles restant en bruit : {nb_bruit}/{len(topics)}")
    return df, resume_bertopic