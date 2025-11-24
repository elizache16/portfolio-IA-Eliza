Projet Récapitulatif : Analyse Multimodale de Produits E-Commerce
🧠 1. Contexte du projet

Vous êtes Data Scientist dans une plateforme de vente en ligne.
Votre manager souhaite savoir s’il est possible de prédire automatiquement la note moyenne (rating) d’un produit avant sa mise en ligne, en utilisant :

ses données structurées (prix, catégorie, nombre de ventes)

sa description marketing (texte)

l’image du produit

Ce projet constitue une révision complète de toutes les compétences apprises durant le semestre : preprocessing, Machine Learning, NLP, Computer Vision, Deep Learning.

🗂️ 2. Données utilisées
🔹 Données Structurées

prix

categorie

nb_ventes

🔹 Données Non-Structurées — Texte

description marketing du produit

🔹 Données Non-Structurées — Images

image_url (liens Unsplash)

🎯 Cible du modèle

note (entre 0 et 5)

🧪 3. Objectifs pédagogiques
1. Préprocessing

Nettoyage

Gestion des valeurs manquantes

Encodage (OneHot)

Mise à l’échelle

2. Machine Learning

Régression Linéaire

Ridge Regression

Validation croisée

RMSE, R²

3. NLP

Analyse de sentiment avec Transformers

Extraction d’un score numérique pour ajouter comme feature

4. Computer Vision

Extraction de vecteurs d’images via ResNet50 (Transfer Learning)

5. Deep Learning

Construction d’un MLP multimodal fusionnant tabulaire + sentiment (+ image optionnel)

🏗️ 4. Workflow du Projet
🧩 Étape 0 — Configuration

Importation des librairies : Scikit-Learn, TensorFlow/Keras, Transformers, Requests, PIL, etc.

📊 Étape 1 — Création du Dataset

Simulation de 100 produits réalistes + ajout de bruit (noise) + valeurs manquantes pour l’exercice.

⚙️ Étape 2 — Preprocessing

Création d’un ColumnTransformer :

Variables numériques (prix, nb_ventes)

Imputation : médiane

StandardScaler

Variable catégorielle (categorie)

Imputation : mode

OneHotEncoder

📈 Étape 3 — Baseline ML

Modèle simple :

Ridge Regression

Split Train/Test

Validation croisée

RMSE & R²

Servira de comparaison pour le modèle final.

📝 Étape 4 — NLP : Sentiment

Utilisation d’un modèle multilingue HuggingFace pour obtenir un sentiment_score.

🖼️ Étape 5 — Computer Vision

Pipeline d’extraction via ResNet50 pré-entraîné :

Téléchargement de l’image

Redimension 224x224

Extraction du vecteur 2048 dimensions

🧬 Étape 6 — Modèle Multimodal (MLP)

Fusion des données :

Features tabulaires preprocessées

Score NLP

(Optionnel) Vecteurs d’images

Architecture du MLP :Entrée tabulaire → Dense → BatchNorm → ReLU  
Entrée sentiment → Dense  
-----------------------------------------
Fusion → Dense → Dropout → Dense → Sortie
Évaluation :

RMSE

Courbes d’apprentissage

Comparaison avec la baseline

📌 5. Questions de réflexion

Le MLP multimodal dépasse-t-il la baseline Ridge ? Pourquoi ?

Comment intégrer les vecteurs d’images dans le MLP ?

Si l’on change l’objectif en “Populaire vs Non Populaire”, que changerait-on dans la dernière couche ?

🛠️ 6. Stack Technique

Python 3

Pandas, NumPy

Scikit-Learn

TensorFlow / Keras

Transformers (HuggingFace)

PIL, Requests

Matplotlib, Seaborn

👩‍💻 7. Auteure

Eliza Ecaterina Marinica
Étudiante en Intelligence Artificielle – Montréal
Projet préparé pour mon portfolio IA et ma demande de stage en Data Science / Machine Learning.
  
