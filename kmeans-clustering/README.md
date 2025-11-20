# 🌸 Projet de Clustering K-Means — Iris Dataset

## 📝 Description du projet
Ce projet applique l’algorithme de **clustering K-Means** sur le célèbre **jeu de données Iris**, afin de regrouper automatiquement les fleurs en plusieurs clusters selon leurs caractéristiques mesurées.

Ce travail fait partie de ma formation en **Intelligence Artificielle (Collège CDI, 2024-2025)** et démontre ma capacité à :
- analyser des données,
- appliquer un algorithme d’apprentissage non supervisé,
- visualiser et interpréter les résultats,
- structurer un projet complet en Python.

---

## 🎯 Objectifs
- Comprendre la logique de l’apprentissage non supervisé  
- Appliquer l’algorithme **K-Means**  
- Visualiser les clusters formés  
- Comparer les clusters avec les vraies classes (Iris Setosa, Versicolor, Virginica)

---

## 📚 Jeu de données
Le dataset Iris contient **150 échantillons**, chacun décrit par :
- longueur du sépale (sepal length)  
- largeur du sépale  
- longueur du pétale  
- largeur du pétale  
- l’espèce (pour comparaison)

---

## 🧠 Méthodologie

### 1️⃣ Importation des librairies
- pandas  
- numpy  
- matplotlib / seaborn  
- sklearn (KMeans, StandardScaler, train_test_split)

### 2️⃣ Préparation des données
- Chargement du dataset depuis `sklearn.datasets`
- Sélection des features (colonnes numériques)
- Mise à l’échelle avec **StandardScaler**

### 3️⃣ Modèle K-Means
- Choix du nombre de clusters **K = 3**
- Entraînement du modèle
- Prédiction des clusters
- Ajout des labels prédits au DataFrame

### 4️⃣ Visualisation
- Graphique scatter 2D  
- Couleurs selon les clusters  
- Comparaison avec les classes réelles

---

## 📊 Résultats
Le modèle parvient à regrouper correctement :

- Iris Setosa : parfaitement séparée  
- Iris Versicolor / Virginica : partiellement séparées (comportement attendu du dataset)

---

## 📦 Fichiers dans ce repository
- `KMeans_Iris.ipynb` — Notebook complet du projet  
- `README.md` — Documentation du projet  

---

## 🧑‍💻 Technologies utilisées
- Python  
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib  
- Seaborn  

---

## ✨ Auteur
**Eliza Ecaterina Marinica**  
Candidate — Stage en Intelligence Artificielle / Python  
