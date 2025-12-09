"""
Script principal para ejecutar todo el pipeline de Fase 1
"""
import sys
from pathlib import Path
import logging

# Añadir src al path
sys.path.append(str(Path(__file__).parent / 'src'))

from data.descargar_datos import descargar_datos_atp
from data.data_processor import explorar_datos, limpiar_datos
from features.feature_engineer import crear_dataset_jugadores, crear_features_basicas, preparar_dataset_final
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ejecutar_pipeline_completo():
    """
    Ejecuta todo el pipeline de Fase 1
    """
    
    logger.info("=" * 70)
    logger.info("🚀 INICIANDO PIPELINE COMPLETO - FASE 1")
    logger.info("=" * 70)
    
    # Paso 1: Descargar datos
    logger.info("\n📥 PASO 1: DESCARGANDO DATOS...")
    df_raw = descargar_datos_atp(years=[2020, 2021, 2022, 2023, 2024])
    
    # Paso 2: Explorar datos
    logger.info("\n📊 PASO 2: EXPLORANDO DATOS...")
    explorar_datos(df_raw)
    
    # Paso 3: Limpiar datos
    logger.info("\n🧹 PASO 3: LIMPIANDO DATOS...")
    df_clean = limpiar_datos(df_raw)
    
    # Paso 4: Crear dataset de jugadores
    logger.info("\n🔧 PASO 4: CREANDO DATASET DE JUGADORES...")
    df_jugadores = crear_dataset_jugadores(df_clean)
    
    # Paso 5: Crear features
    logger.info("\n🎯 PASO 5: CREANDO FEATURES...")
    df_con_features = crear_features_basicas(df_jugadores)
    
    # Paso 6: Preparar dataset final
    logger.info("\n📦 PASO 6: PREPARANDO DATASET FINAL...")
    X, y, df_final = preparar_dataset_final(df_con_features)
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ PIPELINE DE PREPARACIÓN DE DATOS COMPLETADO")
    logger.info("=" * 70)
    logger.info(f"\n📊 Dataset final:")
    logger.info(f"   - Total de filas: {len(df_final)}")
    logger.info(f"   - Features: {X.shape[1]}")
    logger.info(f"   - Balance de clases: {y.value_counts().to_dict()}")
    
    logger.info("\n🎯 SIGUIENTE PASO:")
    logger.info("   Ejecutar: cd /Users/miquelroca/Desktop/proyecto/tennis-ml-predictor && python src/models/trainer.py")
    
    return df_final


if __name__ == "__main__":
    try:
        df_final = ejecutar_pipeline_completo()
        logger.info("\n🎉 ¡Pipeline ejecutado exitosamente!")
    except Exception as e:
        logger.error(f"\n❌ Error en el pipeline: {e}", exc_info=True)
        sys.exit(1)
