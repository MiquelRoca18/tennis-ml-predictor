"""
Procesamiento y limpieza de datos ATP
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def limpiar_datos(df):
    """
    Limpia y prepara los datos para el modelo
    
    Args:
        df: DataFrame con datos crudos
        
    Returns:
        DataFrame limpio
    """
    
    logger.info("=" * 60)
    logger.info("🧹 LIMPIEZA DE DATOS")
    logger.info("=" * 60)
    
    logger.info(f"\n📊 Partidos iniciales: {len(df)}")
    
    # 1. Eliminar partidos sin ranking (son raros pero existen)
    df = df.dropna(subset=['winner_rank', 'loser_rank'])
    logger.info(f"✅ Después de eliminar sin ranking: {len(df)}")
    
    # 2. Eliminar rankings mayores a 500 (jugadores muy bajos)
    df = df[(df['winner_rank'] <= 500) & (df['loser_rank'] <= 500)]
    logger.info(f"✅ Después de filtrar ranking <= 500: {len(df)}")
    
    # 3. Eliminar partidos sin información de superficie
    df = df.dropna(subset=['surface'])
    logger.info(f"✅ Después de eliminar sin superficie: {len(df)}")
    
    # 4. Quedarnos solo con superficies principales
    superficies_principales = ['Hard', 'Clay', 'Grass']
    df = df[df['surface'].isin(superficies_principales)]
    logger.info(f"✅ Después de filtrar superficies: {len(df)}")
    
    # 5. Eliminar partidos walkover (retiros antes de empezar)
    if 'score' in df.columns:
        df = df[~df['score'].str.contains('W/O', na=False)]
        df = df[~df['score'].str.contains('DEF', na=False)]
        logger.info(f"✅ Después de eliminar walkovers: {len(df)}")
    
    # 6. Convertir fecha a datetime
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['tourney_date'])
    
    # 7. Ordenar por fecha (IMPORTANTE para evitar data leakage)
    df = df.sort_values('tourney_date').reset_index(drop=True)
    
    logger.info(f"\n📊 Partidos finales: {len(df)}")
    logger.info(f"📅 Rango de fechas: {df['tourney_date'].min()} - {df['tourney_date'].max()}")
    
    # Guardar datos limpios
    output_dir = Path("datos/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "atp_matches_clean.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"\n💾 Datos limpios guardados en: {output_file}")
    
    return df


def explorar_datos(df):
    """
    Exploración inicial de datos
    
    Args:
        df: DataFrame con datos crudos
    """
    
    logger.info("=" * 60)
    logger.info("📊 EXPLORACIÓN INICIAL DE DATOS")
    logger.info("=" * 60)
    
    # Información básica
    logger.info(f"\n📈 Total de partidos: {len(df)}")
    logger.info(f"📅 Rango de fechas: {df['tourney_date'].min()} - {df['tourney_date'].max()}")
    
    # Valores faltantes
    logger.info("\n❓ Valores faltantes principales:")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Columna': missing.index,
        'Valores faltantes': missing.values,
        'Porcentaje': missing_percent.values
    })
    top_missing = missing_df[missing_df['Valores faltantes'] > 0].sort_values('Valores faltantes', ascending=False).head(10)
    for _, row in top_missing.iterrows():
        logger.info(f"   {row['Columna']}: {row['Valores faltantes']} ({row['Porcentaje']:.1f}%)")
    
    # Distribución de superficies
    logger.info("\n🎾 Distribución por superficie:")
    for superficie, count in df['surface'].value_counts().items():
        logger.info(f"   {superficie}: {count}")
    
    # Estadísticas de rankings
    logger.info("\n🏆 Estadísticas de rankings:")
    logger.info(f"   Ranking ganador - Media: {df['winner_rank'].mean():.1f}, Mediana: {df['winner_rank'].median():.1f}")
    logger.info(f"   Ranking perdedor - Media: {df['loser_rank'].mean():.1f}, Mediana: {df['loser_rank'].median():.1f}")


if __name__ == "__main__":
    # Cargar datos crudos (actualizados con 2018-2025)
    df_raw = pd.read_csv("datos/raw/atp_matches_raw_updated.csv")
    
    # Explorar
    explorar_datos(df_raw)
    
    # Limpiar
    df_clean = limpiar_datos(df_raw)
    
    logger.info("\n✅ Procesamiento completado!")
