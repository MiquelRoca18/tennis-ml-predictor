"""
Selección de features más importantes
Fase 3 - Optimización
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Selección de features más importantes
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.feature_scores = {}

    def feature_importance_tree_based(self, modelo=None):
        """
        Feature importance usando modelo basado en árboles
        """

        logger.info("=" * 60)
        logger.info("🌲 FEATURE IMPORTANCE - TREE-BASED")
        logger.info("=" * 60)

        if modelo is None:
            modelo = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            logger.info("\n🔄 Entrenando Random Forest...")
            modelo.fit(self.X, self.y)

        # Feature importance
        importances = modelo.feature_importances_

        df_importance = pd.DataFrame(
            {"feature": self.X.columns, "importance": importances}
        ).sort_values("importance", ascending=False)

        self.feature_scores["tree_based"] = df_importance

        logger.info("\n🏆 Top 20 features:")
        logger.info("\n" + df_importance.head(20).to_string(index=False))

        # Visualizar
        plt.figure(figsize=(12, 8))
        top_20 = df_importance.head(20)
        plt.barh(top_20["feature"], top_20["importance"])
        plt.xlabel("Importance", fontsize=12)
        plt.title("Top 20 Features - Tree-Based Importance", fontsize=14, fontweight="bold")
        plt.gca().invert_yaxis()
        plt.tight_layout()

        Path("resultados").mkdir(exist_ok=True)
        plt.savefig("resultados/feature_importance_tree.png", dpi=150, bbox_inches="tight")
        logger.info("\n📊 Gráfico guardado: resultados/feature_importance_tree.png")
        plt.close()

        return df_importance

    def feature_importance_statistical(self, method="f_classif"):
        """
        Feature importance usando métodos estadísticos

        Args:
            method: 'f_classif' o 'mutual_info'
        """

        logger.info("\n" + "=" * 60)
        logger.info(f"📊 FEATURE IMPORTANCE - {method.upper()}")
        logger.info("=" * 60)

        if method == "f_classif":
            scores, pvalues = f_classif(self.X, self.y)
            score_col = "f_score"
        else:  # mutual_info
            scores = mutual_info_classif(self.X, self.y, random_state=42)
            pvalues = np.zeros(len(scores))
            score_col = "mi_score"

        df_scores = pd.DataFrame(
            {"feature": self.X.columns, score_col: scores, "p_value": pvalues}
        ).sort_values(score_col, ascending=False)

        self.feature_scores[method] = df_scores

        logger.info(f"\n🏆 Top 20 features ({method}):")
        logger.info("\n" + df_scores.head(20).to_string(index=False))

        return df_scores

    def seleccionar_mejores_k(self, k=30, method="tree_based"):
        """
        Selecciona las mejores K features
        """

        logger.info("\n" + "=" * 60)
        logger.info(f"🎯 SELECCIONANDO TOP {k} FEATURES")
        logger.info("=" * 60)

        if method not in self.feature_scores:
            logger.warning(f"⚠️  Primero ejecuta el método '{method}'")
            return None, None

        df_scores = self.feature_scores[method]
        top_features = df_scores.head(k)["feature"].tolist()

        logger.info(f"\n✅ Top {k} features seleccionadas:")
        for i, feat in enumerate(top_features, 1):
            logger.info(f"   {i:2d}. {feat}")

        # Crear dataset reducido
        X_reducido = self.X[top_features]

        logger.info(f"\n📊 Dataset original: {self.X.shape[1]} features")
        logger.info(f"📊 Dataset reducido: {X_reducido.shape[1]} features")
        logger.info(f"📉 Reducción: {(1 - k/self.X.shape[1])*100:.1f}%")

        return X_reducido, top_features

    def comparar_metodos(self):
        """
        Compara diferentes métodos de feature selection
        """

        logger.info("\n" + "=" * 60)
        logger.info("🔄 COMPARANDO MÉTODOS")
        logger.info("=" * 60)

        # Obtener top 20 de cada método
        metodos = ["tree_based", "f_classif", "mutual_info"]

        for method in metodos:
            if method not in self.feature_scores:
                if method == "tree_based":
                    self.feature_importance_tree_based()
                else:
                    self.feature_importance_statistical(method)

        # Features en común
        top_20_sets = []
        for method in metodos:
            if method in self.feature_scores:
                top_20 = set(self.feature_scores[method].head(20)["feature"])
                top_20_sets.append(top_20)

        # Intersección
        if len(top_20_sets) > 0:
            features_comunes = set.intersection(*top_20_sets)

            logger.info(f"\n🎯 Features en TOP 20 de TODOS los métodos: {len(features_comunes)}")
            for feat in sorted(features_comunes):
                logger.info(f"   - {feat}")

            return list(features_comunes)
        else:
            return []

    def eliminar_features_correlacionadas(self, threshold=0.9):
        """
        Elimina features altamente correlacionadas
        """

        logger.info("\n" + "=" * 60)
        logger.info(f"🔗 ELIMINANDO FEATURES CORRELACIONADAS (threshold={threshold})")
        logger.info("=" * 60)

        # Calcular correlaciones
        corr_matrix = self.X.corr().abs()

        # Encontrar pares altamente correlacionados
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Features a eliminar
        to_drop = [
            column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)
        ]

        logger.info(f"\n📊 Features correlacionadas (>{threshold}):")
        if len(to_drop) > 0:
            for feat in to_drop:
                logger.info(f"   - {feat}")

            X_sin_correlacion = self.X.drop(columns=to_drop)
            logger.info(f"\n✅ Features eliminadas: {len(to_drop)}")
            logger.info(f"📊 Features restantes: {X_sin_correlacion.shape[1]}")
        else:
            logger.info("   ✅ No hay features altamente correlacionadas")
            X_sin_correlacion = self.X

        return X_sin_correlacion, to_drop


if __name__ == "__main__":
    # Cargar datos
    logger.info("📂 Cargando dataset...")
    df = pd.read_csv("datos/processed/dataset_features_fase3_completas.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)

    # Features
    feature_cols = [col for col in df.columns if col not in ["resultado", "fecha"]]

    # Usar solo datos de entrenamiento para feature selection
    n = len(df)
    train_end = int(n * 0.6)

    X = df.iloc[:train_end][feature_cols]
    y = df.iloc[:train_end]["resultado"]

    logger.info(f"\n📊 Datos: {X.shape[0]} partidos, {X.shape[1]} features")

    # Crear selector
    selector = FeatureSelector(X, y)

    # Método 1: Tree-based
    imp_tree = selector.feature_importance_tree_based()

    # Método 2: F-statistic
    imp_f = selector.feature_importance_statistical("f_classif")

    # Método 3: Mutual Information
    imp_mi = selector.feature_importance_statistical("mutual_info")

    # Comparar
    features_comunes = selector.comparar_metodos()

    # Eliminar correlacionadas
    X_sin_corr, dropped = selector.eliminar_features_correlacionadas(threshold=0.9)

    # Seleccionar top 30
    X_reducido, top_features = selector.seleccionar_mejores_k(k=30, method="tree_based")

    # Guardar lista de features seleccionadas
    if top_features:
        Path("resultados").mkdir(exist_ok=True)
        pd.Series(top_features).to_csv(
            "resultados/selected_features.txt", index=False, header=False
        )
        logger.info("\n💾 Features seleccionadas guardadas: resultados/selected_features.txt")

    logger.info("\n✅ Feature selection completado!")
