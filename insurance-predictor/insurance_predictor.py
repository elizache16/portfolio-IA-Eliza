"""InsurancePredictor – Modèle de Régression (Python & Programmation Orientée Objet)

Ce module contient la classe InsurancePredictor qui :
- charge le fichier insurance.csv
- prépare les données (encodage, scaling, split train/test)
- entraîne plusieurs modèles de régression
- évalue les modèles (RMSE, R²)
- affiche le meilleur modèle
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score


class InsurancePredictor:
    def __init__(self, csv_path: str):
        """
        Paramètres
        ----------
        csv_path : str
            Chemin vers le fichier insurance.csv
        """
        self.csv_path = csv_path
        self.df = None

        # Objets qui seront créés dans le pipeline
        self.preprocessor = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.models = {}
        self.evaluation_results = None

    # -------------------------------------------------
    # 1. Chargement & EDA
    # -------------------------------------------------
    def explore_data(self):
        print("\n--- Chargement des données ---")
        self.df = pd.read_csv(self.csv_path)
        print("Données chargées avec succès.")
        print(f"Le jeu de données contient {self.df.shape[0]} lignes et {self.df.shape[1]} colonnes.\n")

        print("--- Aperçu des données ---")
        print(self.df.head(), "\n")

        print("--- Statistiques descriptives (numériques) ---")
        print(self.df.describe(), "\n")

        # Exemple simple de visualisation (facultatif pour l'exécution)
        try:
            plt.figure(figsize=(6, 4))
            sns.histplot(self.df["charges"], kde=True)
            plt.title("Distribution des frais d'assurance (charges)")
            plt.xlabel("charges")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"(Info) Impossible d'afficher le graphique: {e}")

    # -------------------------------------------------
    # 2. Prétraitement
    # -------------------------------------------------
    def preprocess_data(self, test_size: float = 0.2, random_state: int = 42):
        print("\n--- Prétraitement des Données ---")

        # Variable cible
        y = self.df["charges"]
        # Features (toutes les autres colonnes)
        X = self.df.drop(columns=["charges"])

        # Colonnes numériques & catégorielles
        numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
        categorical_features = X.select_dtypes(include=["object"]).columns

        # Transformers
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")

        # ColumnTransformer : applique le bon traitement à chaque type de colonne
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        # Split train / test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        print("Données divisées et prêtes pour l'entraînement.")
        print(f"Train : {self.X_train.shape[0]} lignes")
        print(f"Test  : {self.X_test.shape[0]} lignes\n")

    # -------------------------------------------------
    # 3. Entraînement des modèles
    # -------------------------------------------------
    def train_models(self):
        print("\n--- Entraînement des Modèles ---")

        # Définition des modèles de régression
        candidate_models = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.001),
        }

        for name, model in candidate_models.items():
            # Chaque modèle est un Pipeline : preprocessing + modèle
            clf = Pipeline(
                steps=[
                    ("preprocess", self.preprocessor),
                    ("model", model),
                ]
            )
            clf.fit(self.X_train, self.y_train)
            self.models[name] = clf
            print(f"Modèle '{name}' entraîné.")

    # -------------------------------------------------
    # 4. Évaluation des modèles
    # -------------------------------------------------
    def evaluate_models(self):
        print("\n--- Évaluation des Modèles ---")
        results = []

        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            r2 = r2_score(self.y_test, y_pred)

            results.append({"Modèle": name, "RMSE": rmse, "R2": r2})
            print(f"{name:>18} | RMSE = {rmse:8.2f} | R² = {r2:6.3f}")

        # DataFrame récapitulatif
        self.evaluation_results = pd.DataFrame(results).set_index("Modèle")
        print("\nRésultats récapitulatifs :")
        print(self.evaluation_results)

    # -------------------------------------------------
    # 5. Visualisation du meilleur modèle
    # -------------------------------------------------
    def visualize_best_model(self):
        print("\n--- Visualisation du Meilleur Modèle ---")

        # Vérifier que l'évaluation a été faite
        if self.evaluation_results is None or self.evaluation_results.empty:
            print("Aucun résultat d'évaluation trouvé. Appelez d'abord evaluate_models().")
            return

        # Meilleur modèle : celui qui a le plus grand R²
        best_model_name = self.evaluation_results["R2"].idxmax()
        best_model = self.models[best_model_name]

        print(f"Le meilleur modèle est : {best_model_name}")
        print(self.evaluation_results.loc[best_model_name], "\n")

        # Courbe prédictions vs vraies valeurs
        try:
            y_pred = best_model.predict(self.X_test)

            plt.figure(figsize=(6, 6))
            plt.scatter(self.y_test, y_pred, alpha=0.5)
            plt.xlabel("Vraies valeurs (y_test)")
            plt.ylabel("Prédictions")
            plt.title(f"Meilleur modèle : {best_model_name}")
            # diagonale
            lims = [
                min(self.y_test.min(), y_pred.min()),
                max(self.y_test.max(), y_pred.max()),
            ]
            plt.plot(lims, lims, "r--")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"(Info) Impossible d'afficher le graphique: {e}")

    # -------------------------------------------------
    # 6. Pipeline complet
    # -------------------------------------------------
    def run(self):
        """
        Exécute le pipeline complet :
        1) EDA
        2) Prétraitement
        3) Entraînement des modèles
        4) Évaluation
        5) Visualisation du meilleur modèle
        """
        self.explore_data()
        self.preprocess_data()
        self.train_models()
        self.evaluate_models()
        self.visualize_best_model()


# Exemple d'utilisation (facultatif)
if __name__ == "__main__":
    DATA_PATH = "insurance.csv"
    predictor = InsurancePredictor(DATA_PATH)
    predictor.run()
