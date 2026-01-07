"""
Script para ejecutar el feature engineering completo de Fase 3
"""
import sys
from pathlib import Path

# Añadir src al path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from features.feature_engineer_completo import CompleteFeatureEngineer
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Ejecuta el feature engineering completo de Fase 3
    """
    
    logger.info("=" * 80)
    logger.info("🚀 FASE 3: FEATURE ENGINEERING COMPLETO")
    logger.info("=" * 80)
    
    # 1. Cargar datos limpios
    logger.info("\n📂 Cargando datos limpios...")
    df = pd.read_csv("datos/processed/atp_matches_clean.csv")
    df['tourney_date'] = pd.to_datetime(df['tourney_date'])
    
    logger.info(f"   ✅ Partidos cargados: {len(df):,}")
    logger.info(f"   📅 Rango: {df['tourney_date'].min().date()} - {df['tourney_date'].max().date()}")
    logger.info(f"   🎾 Jugadores únicos: {pd.concat([df['winner_name'], df['loser_name']]).nunique():,}")
    
    # 2. Crear feature engineer completo
    logger.info("\n🔧 Inicializando Feature Engineer Completo...")
    logger.info("   Esto incluye:")
    logger.info("   ⭐⭐⭐ ELO Rating System (general + por superficie)")
    logger.info("   ⭐⭐⭐ Estadísticas de Servicio y Resto")
    logger.info("   ⭐⭐ Métricas de Fatiga")
    logger.info("   ⭐⭐ Forma Reciente")
    logger.info("   ⭐⭐ Head-to-Head Mejorado")
    logger.info("   ⭐ Especialización por Superficie")
    logger.info("   ➕ Features de Interacción")
    
    engineer = CompleteFeatureEngineer(df)
    
    # 3. Procesar dataset completo
    logger.info("\n🔄 Procesando dataset completo...")
    logger.info("   (Esto puede tomar varios minutos...)")
    
    df_features = engineer.procesar_dataset_completo(
        save_path="datos/processed/dataset_features_fase3_completas.csv"
    )
    
    # 4. Resumen final
    logger.info("\n" + "=" * 80)
    logger.info("✅ FEATURE ENGINEERING COMPLETADO")
    logger.info("=" * 80)
    
    logger.info(f"\n📊 RESUMEN:")
    logger.info(f"   Total de features: {len(df_features.columns) - 2}")
    logger.info(f"   Total de filas: {len(df_features):,}")
    logger.info(f"   Archivo guardado: datos/processed/dataset_features_fase3_completas.csv")
    
    # Estadísticas básicas
    logger.info(f"\n📈 ESTADÍSTICAS:")
    logger.info(f"   Columnas totales: {len(df_features.columns)}")
    
    # Verificar valores faltantes
    missing = df_features.isnull().sum().sum()
    logger.info(f"\n   Valores faltantes: {missing}")
    
    if missing > 0:
        logger.warning(f"   ⚠️  Hay {missing} valores faltantes que deben ser manejados")
        logger.info("\n   Columnas con valores faltantes:")
        for col in df_features.columns:
            missing_col = df_features[col].isnull().sum()
            if missing_col > 0:
                logger.info(f"      {col}: {missing_col} ({missing_col/len(df_features)*100:.2f}%)")
    
    logger.info("\n🎯 SIGUIENTE PASO:")
    logger.info("   Ejecutar entrenamiento del modelo con:")
    logger.info("   python src/models/trainer.py")
    
    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    main()
