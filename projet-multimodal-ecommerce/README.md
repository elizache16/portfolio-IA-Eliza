📦 Projet Récapitulatif : Analyse Multimodale de Produits E-Commerce
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

Extraction d’un score numérique à ajouter comme feature

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

RMSE & R²

📝 Étape 4 — NLP : Sentiment

Utilisation d’un modèle multilingue HuggingFace pour obtenir un sentiment_score.

🖼️ Étape 5 — Computer Vision

Pipeline d’extraction via ResNet50 pré-entraîné :

Redimension 224x224

Extraction du vecteur 2048 dimensions

🧬 Étape 6 — Modèle Final Multimodal (MLP)

Fusion des données :
🟦 features tabulaires preprocessées
🟩 score NLP
🟧 vecteurs d’images

Architecture du MLP :

Entrée tabulaire → Dense → BatchNorm → ReLU  
Entrée sentiment → Dense  
-----------------------------------------
Fusion → Dense → Dropout → Dense → Sortie

📌 5. Questions de réflexion
1️⃣ Le modèle multimodal dépasse-t-il la baseline Ridge ? Pourquoi ?

Oui — car il capture des relations non linéaires et combine texte + images + tabulaire.

2️⃣ Comment intégrer les vecteurs d’images ?

Avec une simple concaténation :
np.hstack([X_tab, sentiment, image_features])

3️⃣ Si l’objectif était « Populaire vs Pas populaire » ?

Changer la couche finale → sigmoid
Changer la loss → binary_crossentropy

🛠️ 6. Stack Technique

Python 3

Pandas, NumPy

Scikit-Learn

TensorFlow / Keras

Transformers

PIL, Requests

Matplotlib, Seaborn

👩‍💻 7. Auteure

Eliza Ecaterina Marinica
Étudiante en Intelligence Artificielle – Montréal
Projet réalisé dans le cadre de ma formation et de mon portfolio pour le stage en Data Science / Machine Learning.
