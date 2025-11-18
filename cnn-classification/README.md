# CNN Classification – Réseaux de Neurones Convolutifs

Projet réalisé dans le cadre de ma formation en Intelligence Artificielle au Collège CDI.

## 🎯 Objectif

Construire un modèle de **classification d’images** avec un **Réseau de Neurones Convolutifs (CNN)** afin de reconnaître automatiquement des classes d’objets à partir d’images.

Je montre dans ce projet :
- la préparation des données d’images,
- la construction d’un CNN avec Keras / TensorFlow,
- l’entraînement, l’évaluation et la visualisation des résultats.

## 📊 Jeu de données

- **Dataset** : images de vêtements (Fashion-MNIST) / objets (CIFAR-10)  
- **Taille** : 60 000 images d’entraînement, 10 000 images de test  
- **Format** : images en niveaux de gris ou RGB, dimension normalisée.

## 🛠️ Compétences développées

- Prétraitement d’images (normalisation, reshape, one-hot encoding)
- Conception d’un CNN (Conv2D, MaxPooling2D, Dropout, Dense)
- Entraînement avec optimisation (Adam) et fonction de perte adaptée
- Visualisation des courbes `loss` / `accuracy`
- Matrice de confusion et analyse des erreurs
- Sauvegarde et chargement de modèles entraînés

## 📁 Structure du projet

- `cnn_classification_project.ipynb` — Notebook complet du projet (code + explications)
- *(Optionnel)* `cnn_model.h5` — Modèle CNN entraîné
- *(Optionnel)* `images_exemples/` — Quelques exemples d’images pour la démonstration

## ✅ Résultats

- Précision finale : **XX–YY %** sur le jeu de test (selon le dataset utilisé)
- Le modèle est capable de distinguer correctement les différentes classes d’images.
- Le projet illustre ma capacité à utiliser **Keras / TensorFlow** pour résoudre une tâche de **Computer Vision** de bout en bout.
