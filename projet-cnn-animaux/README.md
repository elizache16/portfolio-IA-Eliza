Projet CNN – Classification d’Animaux
Un modèle de Deep Learning pour reconnaître 6 classes d’animaux

🎯 Objectif

Construire et entraîner un réseau de neurones convolutifs (CNN) capable de classifier des images d’animaux en 6 catégories.

📁 Jeu de données

Les images étaient déjà séparées en :

Train : 24 000 images

Test : 6 000 images

Conformément aux consignes du professeur, un troisième ensemble a été créé :

Validation : 10–15% des images d’entraînement → 2 400 images

🧠 Architecture du modèle

CNN simple mais efficace

Optimiseur : Adam

Fonction de perte : categorical_crossentropy

Data augmentation appliquée pour améliorer la généralisation

📈 Résultats

Ensemble	  Accuracy	  Loss

Validation	~80 %	      ~0.58
Test	     79.5 %	       0.59

Les performances sont stables et cohérentes entre validation et test.

🔍 Analyse d’erreurs

Certains animaux sont confondus lorsque :

Les poses sont similaires

Le fond est complexe

L’éclairage est variable

Cela montre la nécessité d’un dataset encore plus riche.

💾 Modèle entraîné

Le fichier du modèle sauvegardé :
➡️ modele_cnn_animaux.h5

📝 Conclusion

Le modèle obtient une bonne performance générale et constitue une base solide pour améliorer :

la data augmentation

l’architecture (CNN plus profond)

le temps d’entraînement (GPU)
