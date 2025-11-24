# Projet Multimodal E-Commerce

Projet de Data Science et Deep Learning appliqué au domaine de l'e-commerce.  
L'objectif est de combiner **plusieurs modalités de données** – images de produits et texte descriptif – pour construire un modèle de classification performant.

---

## 🎯 Objectif du projet

Dans ce projet, nous cherchons à prédire la **catégorie d’un produit e-commerce** en utilisant :

- 🖼️ **L'image du produit** (modèle CNN)
- 📝 **Le texte** : titre et/ou description (modèle NLP)
- 🔗 **Fusion multimodale** des deux représentations (image + texte)

Ce projet illustre comment les techniques de **Computer Vision** et de **Traitement Automatique du Langage (NLP)** peuvent être combinées pour améliorer les performances d’un système de recommandation ou de recherche produit.

---

## 🧩 Contexte pédagogique

Ce projet a été réalisé dans le cadre de ma formation en **Intelligence Artificielle** (CDI – Collège, Montréal), comme projet avancé pour :

- Mettre en pratique un **pipeline complet de machine learning / deep learning**
- Manipuler des données **hétérogènes** (images + texte)
- Concevoir un modèle **multimodal** en Python

---

## 📂 Jeu de données

Le projet repose sur un jeu de données e-commerce contenant :

- Des **images de produits** (fichiers `.jpg` ou `.png`)
- Un fichier `CSV` avec :
  - un identifiant produit
  - un titre
  - une description courte
  - une catégorie (label de classification)

> Pour des raisons de taille/licence, les données complètes ne sont pas stockées dans ce dépôt.  
> Un échantillon ou un lien vers la source (Kaggle, etc.) peut être ajouté dans le dossier `data/`.

---

## 🛠️ Approche & Méthodologie

Le projet suit les étapes classiques d’un pipeline de Data Science, adaptées au contexte multimodal :

### 1. Exploration & Préparation des données

- Analyse des distributions (catégories, longueur des textes, etc.)
- Nettoyage du texte :  
  - mise en minuscules  
  - suppression de la ponctuation / caractères spéciaux  
  - éventuellement stopwords, lemmatisation
- Vérification des images : taille, format, présence de valeurs manquantes
- Création d’un **dataset aligné** image + texte + label

### 2. Modèle Images – CNN

- Prétraitement des images : redimensionnement, normalisation
- Construction d’un modèle **Convolutional Neural Network (CNN)** avec Keras/TensorFlow  
  (ou utilisation d’un modèle pré-entraîné type *Transfer Learning*)
- Entraînement pour extraire un **vecteur de caractéristiques** pour chaque image

### 3. Modèle Texte – NLP

- Tokenisation et vectorisation du texte (par ex. `Tokenizer` Keras, `Embedding`, ou TF-IDF + couche dense)
- Construction d’un petit réseau de neurones pour apprendre une représentation dense du texte

### 4. Fusion multimodale

- **Fusion tardive (late fusion)** :  
  - concaténation des embeddings image et texte
  - passage dans des couches denses (Fully Connected)
- Application d’une couche de sortie (Softmax) pour prédire la **catégorie du produit**

### 5. Évaluation

- Séparation train / validation / test
- Métriques principales :
  - `accuracy`
  - matrice de confusion
  - éventuellement `precision`, `recall`, `f1-score` par classe
- Comparaison :
  - modèle **Image seule**
  - modèle **Texte seul**
  - modèle **Multimodal (image + texte)**

---

## 🧪 Stack technique

Les principaux outils et bibliothèques utilisés :

- **Python 3**
- **NumPy, Pandas** – manipulation des données
- **Matplotlib / Seaborn** – visualisation
- **TensorFlow / Keras** – modèles CNN et réseau multimodal
- **Scikit-learn** – métriques, split train/test
- Éventuellement : `nltk` ou `spaCy` pour le NLP

Les dépendances sont listées dans `requirements.txt`.

---

## 📁 Structure du dépôt

```text
projet-multimodal-ecommerce/
│
├── data/                       # (optionnel) échantillons ou lien vers le dataset
├── notebooks/
│   └── 01_multimodal_ecommerce.ipynb   # Notebook principal (expériences et visualisations)
├── src/
│   ├── data_loader.py          # Fonctions de chargement et préparation des données
│   ├── image_model.py          # Définition du modèle CNN pour les images
│   ├── text_model.py           # Modèle texte (embedding + réseau dense ou LSTM)
│   └── multimodal_fusion.py    # Fusion des deux modalités et modèle final
├── requirements.txt            # Dépendances Python
└── README.md                   # Présent fichier de présentation du projet
