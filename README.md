# VisualisationActu

Projet de visualisation de l'actualité de la transition énergétique

## Description

Ce projet permet de visualiser et analyser l'actualité liée à la transition énergétique. Il utilise BERTopic pour la clusterisation thématique des articles, suivi d'un clustering macro pour avoir le point de vu dans les gros cluster, et génère des visualisations interactives avec Bokeh que vous pouvez retrouver dans data/output/visualisation.html .

## Installation

1. Clonez ce dépôt :
   ```bash
   git clone https://github.com/RubensGHub/VisualisationActu.git
   cd VisualisationActu
   ```

2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## Utilisation

1. Placez vos données dans le dossier `data/` (par exemple, `sorted_data.xlsx`).

2. Exécutez le script principal :
   ```bash
   python main.py
   ```

   Cela chargera les données, effectuera la clusterisation et générera la visualisation dans `data/output/visualisation.html`.

## Structure du projet

- `main.py` : Script principal pour exécuter le pipeline complet.
- `clustering.py` : Fonctions de clusterisation (BERTopic et macro) et de cahrgement des données.
- `visu_front.py` : Génération des visualisations avec Bokeh.
- `visu_timeline.py` : Génération de time chart avec Bokeh.
- `data/` : Dossier pour les données d'entrée et sorties.
- `cache/` : Cache pour les embeddings.
- `requirements.txt` : Dépendances Python.

## Dépendances

Voir `requirements.txt` pour la liste complète des bibliothèques utilisées, incluant BERTopic, sentence-transformers, pandas, bokeh, etc.
