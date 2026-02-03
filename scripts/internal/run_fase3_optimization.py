#!/usr/bin/env python3
"""
Fase 3 Optimization - Feature Selection + Entrenamiento + Calibración
=====================================================================

1. Carga dataset con features completas
2. Selección de top 30 features
3. Entrena y compara modelos (RF, XGBoost, GB, LogReg)
4. Calibra y guarda modelos
5. Guarda selected_features.txt para producción

Requisitos:
  - datos/processed/dataset_features_fase3_completas.csv

Uso:
    python scripts/internal/run_fase3_optimization.py
"""

import sys
from pathlib import Path

# Añadir raíz del proyecto al path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    input_path = project_root / "datos" / "processed" / "dataset_features_fase3_completas.csv"

    if not input_path.exists():
        logger.error(f"❌ No existe {input_path}")
        logger.info("   Ejecuta primero: python scripts/internal/run_feature_engineering_fase3.py")
        sys.exit(1)

    logger.info("📂 Cargando dataset...")
    df = pd.read_csv(input_path)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)

    # Compatibilidad: feature_engineer usa ganador_j1, comparacion/feature_selection usan resultado
    if "ganador_j1" in df.columns and "resultado" not in df.columns:
        df["resultado"] = df["ganador_j1"]

    feature_cols = [c for c in df.columns if c not in ["resultado", "fecha", "ganador_j1"]]
    logger.info(f"   Dataset: {len(df)} partidos, {len(feature_cols)} features")

    # 1. Feature Selection
    logger.info("\n" + "=" * 60)
    logger.info("🎯 FEATURE SELECTION")
    logger.info("=" * 60)

    n = len(df)
    train_end = int(n * 0.6)
    X_train_fs = df.iloc[:train_end][feature_cols]
    y_train_fs = df.iloc[:train_end]["resultado"]

    from src.models.feature_selection import FeatureSelector

    selector = FeatureSelector(X_train_fs, y_train_fs)
    selector.feature_importance_tree_based()
    X_reducido, top_features = selector.seleccionar_mejores_k(k=30, method="tree_based")

    if not top_features:
        logger.error("❌ No se pudieron seleccionar features")
        sys.exit(1)

    # Guardar selected_features.txt
    resultados_dir = project_root / "resultados"
    resultados_dir.mkdir(exist_ok=True)
    with open(resultados_dir / "selected_features.txt", "w") as f:
        for feat in top_features:
            f.write(feat + "\n")
    logger.info(f"💾 Guardado: resultados/selected_features.txt ({len(top_features)} features)")

    # 2. Entrenar y evaluar modelos (solo con features seleccionadas)
    logger.info("\n" + "=" * 60)
    logger.info("🤖 ENTRENAMIENTO Y CALIBRACIÓN")
    logger.info("=" * 60)

    val_end = int(n * 0.8)
    X_train = df.iloc[:train_end][top_features]
    y_train = df.iloc[:train_end]["resultado"]
    X_val = df.iloc[train_end:val_end][top_features]
    y_val = df.iloc[train_end:val_end]["resultado"]
    X_test = df.iloc[val_end:][top_features]
    y_test = df.iloc[val_end:]["resultado"]

    logger.info(f"   Splits: Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)}")

    from src.models.comparacion_modelos import ModelComparator

    comparador = ModelComparator()
    comparador.inicializar_modelos_default()
    comparador.entrenar_y_evaluar(X_train, y_train, X_val, y_val, X_test, y_test)
    mejor_modelo, df_res = comparador.comparar_resultados()

    logger.info("\n" + "=" * 60)
    logger.info("✅ OPTIMIZACIÓN FASE 3 COMPLETADA")
    logger.info("=" * 60)
    logger.info(f"🥇 Mejor modelo: {mejor_modelo}")
    logger.info(f"   Modelo producción: modelos/random_forest_calibrado.pkl")


if __name__ == "__main__":
    main()
