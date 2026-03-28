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
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from bokeh.plotting import figure, output_file, save
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    TapTool,
    CustomJS,
    Div,
    Text,
)
from bokeh.layouts import row
from bokeh.palettes import Turbo256


nltk.download('stopwords', quiet=True)
mots_vides_fr = stopwords.words('french')
mots_vides_fr.extend(['euros', 'euro', 'dollars', 'dollar', 'milliards', 'millions', 'mds', 'plus', 'très', 'cette', 'cet', 'comme', 'tout', 'faire', 'ça', 'ont', 'être'])



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
    print(df)
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

        umap_params = {
            "n_neighbors": 20,
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


        seed_topic_list = [
            ["nucléaire"],
            ["politique"],
            ["éolien"],
            ["solaire"],
            ["énergie renouvelable"],
            ["aviation"],
            ["gaz"],
            ["pétrole"],
            ["loi"],
            ["carbone"],
            ["véhicule"],
            ["électrique"],
            ["tarif"],
            ["bâtiment"],
            ["CO2"],
            ["climat"],
            ["eau"],
            ["frugal"],
            ["IA"],
        ]

        # Initialisation et entraînement de BERTopic
        topic_model = BERTopic(
            embedding_model=embedding_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            min_topic_size=50,
            language="multilingual",
        seed_topic_list=seed_topic_list
        )
        pbar.update(1)
        topics, _ = topic_model.fit_transform(titres, umap_embeddings)

        nb_bruit_avant = sum(1 for t in topics if t == -1)
        print(f"Articles en bruit avant reduce_outliers : {nb_bruit_avant}/{len(topics)} ({100*nb_bruit_avant/len(topics):.1f}%)")

        topics = topic_model.reduce_outliers(titres, topics, strategy="c-tf-idf", threshold=0.1)
        topic_model.update_topics(titres, topics=topics)

        nb_bruit_apres = sum(1 for t in topics if t == -1)
        print(f"Articles en bruit après reduce_outliers  : {nb_bruit_apres}/{len(topics)} ({100*nb_bruit_apres/len(topics):.1f}%)")

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
        return df, resume_bertopic


