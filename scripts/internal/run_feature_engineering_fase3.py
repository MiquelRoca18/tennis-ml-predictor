#!/usr/bin/env python3
"""
Feature Engineering Fase 3 - Genera 149 features avanzadas
=========================================================

Ejecuta el pipeline completo de feature engineering usando CompleteFeatureEngineer.
Genera dataset_features_fase3_completas.csv con 1 fila por partido.

Requisitos:
  - datos/processed/atp_matches_clean.csv (generado por data_processor.py)

Uso:
    python scripts/internal/run_feature_engineering_fase3.py
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
    input_path = project_root / "datos" / "processed" / "atp_matches_clean.csv"
    output_path = project_root / "datos" / "processed" / "dataset_features_fase3_completas.csv"

    if not input_path.exists():
        logger.error(f"❌ No existe {input_path}")
        logger.info("   Ejecuta primero: python src/data/data_processor.py")
        sys.exit(1)

    logger.info("Cargando datos limpios...")
    df = pd.read_csv(input_path)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"])

    logger.info(f"📊 Partidos cargados: {len(df):,}")
    logger.info(f"📅 Rango: {df['tourney_date'].min()} - {df['tourney_date'].max()}")

    from src.features.feature_engineer_completo import CompleteFeatureEngineer

    logger.info("\n" + "=" * 60)
    logger.info("🚀 INICIANDO FEATURE ENGINEERING FASE 3")
    logger.info("=" * 60)

    engineer = CompleteFeatureEngineer(df)
    df_features = engineer.procesar_dataset_completo(save_path=str(output_path))

    logger.info("\n" + "=" * 60)
    logger.info("✅ FEATURE ENGINEERING COMPLETADO")
    logger.info("=" * 60)
    logger.info(f"📊 Total features: {len(df_features.columns) - 2}")
    logger.info(f"📊 Total filas: {len(df_features):,}")
    logger.info(f"💾 Guardado: {output_path}")


if __name__ == "__main__":
    main()
