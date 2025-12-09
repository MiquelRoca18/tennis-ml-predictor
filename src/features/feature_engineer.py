"""
Feature Engineering para predicción de partidos de tenis
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def crear_dataset_jugadores(df):
    """
    Convierte dataset de partidos a formato:
    [jugador_1, jugador_2, resultado]
    
    Donde resultado = 1 si jugador_1 ganó, 0 si perdió
    
    Args:
        df: DataFrame con partidos limpios
        
    Returns:
        DataFrame con formato de jugadores
    """
    
    logger.info("=" * 60)
    logger.info("🔧 CREANDO DATASET DE JUGADORES")
    logger.info("=" * 60)
    
    # Lista para almacenar filas del nuevo dataset
    datos = []
    
    for idx, row in df.iterrows():
        # Partido original: Ganador vs Perdedor (resultado = 1, ganador gana)
        partido_1 = {
            'fecha': row['tourney_date'],
            'jugador_nombre': row['winner_name'],
            'oponente_nombre': row['loser_name'],
            'jugador_rank': row['winner_rank'],
            'oponente_rank': row['loser_rank'],
            'superficie': row['surface'],
            'resultado': 1  # Ganador
        }
        
        # Partido invertido: Perdedor vs Ganador (resultado = 0, perdedor pierde)
        partido_2 = {
            'fecha': row['tourney_date'],
            'jugador_nombre': row['loser_name'],
            'oponente_nombre': row['winner_name'],
            'jugador_rank': row['loser_rank'],
            'oponente_rank': row['winner_rank'],
            'superficie': row['surface'],
            'resultado': 0  # Perdedor
        }
        
        datos.append(partido_1)
        datos.append(partido_2)
    
    df_jugadores = pd.DataFrame(datos)
    
    logger.info(f"📊 Partidos originales: {len(df)}")
    logger.info(f"📊 Filas en nuevo dataset: {len(df_jugadores)} (2x cada partido)")
    logger.info(f"✅ Balance de clases:")
    logger.info(f"   Ganadores (1): {(df_jugadores['resultado'] == 1).sum()}")
    logger.info(f"   Perdedores (0): {(df_jugadores['resultado'] == 0).sum()}")
    
    return df_jugadores


def crear_features_basicas(df):
    """
    Crea features básicas para el modelo
    
    Args:
        df: DataFrame con formato de jugadores
        
    Returns:
        DataFrame con features añadidas
    """
    
    logger.info("\n" + "=" * 60)
    logger.info("🎯 CREANDO FEATURES BÁSICAS")
    logger.info("=" * 60)
    
    df = df.copy()
    
    # Feature 1: Diferencia de ranking
    df['rank_diff'] = df['oponente_rank'] - df['jugador_rank']
    # Positivo = jugador mejor rankeado, Negativo = oponente mejor rankeado
    
    # Feature 2: Ratio de ranking
    df['rank_ratio'] = df['jugador_rank'] / df['oponente_rank']
    # < 1 = jugador mejor rankeado, > 1 = oponente mejor rankeado
    
    # Feature 3: Jugador en top 10
    df['jugador_top10'] = (df['jugador_rank'] <= 10).astype(int)
    
    # Feature 4: Oponente en top 10
    df['oponente_top10'] = (df['oponente_rank'] <= 10).astype(int)
    
    # Feature 5: Jugador en top 50
    df['jugador_top50'] = (df['jugador_rank'] <= 50).astype(int)
    
    # Feature 6: Oponente en top 50
    df['oponente_top50'] = (df['oponente_rank'] <= 50).astype(int)
    
    # Feature 7-9: One-hot encoding de superficie
    df['surface_hard'] = (df['superficie'] == 'Hard').astype(int)
    df['surface_clay'] = (df['superficie'] == 'Clay').astype(int)
    df['surface_grass'] = (df['superficie'] == 'Grass').astype(int)
    
    logger.info("✅ Features creadas:")
    logger.info("   - rank_diff (diferencia de ranking)")
    logger.info("   - rank_ratio (ratio de ranking)")
    logger.info("   - jugador_top10 / oponente_top10")
    logger.info("   - jugador_top50 / oponente_top50")
    logger.info("   - surface_hard / surface_clay / surface_grass")
    
    # Ver estadísticas
    logger.info("\n📊 Estadísticas de features:")
    logger.info(f"   rank_diff - Media: {df['rank_diff'].mean():.2f}, Std: {df['rank_diff'].std():.2f}")
    logger.info(f"   rank_ratio - Media: {df['rank_ratio'].mean():.2f}, Std: {df['rank_ratio'].std():.2f}")
    logger.info(f"   jugador_top10 - {(df['jugador_top10'] == 1).sum()} jugadores top 10")
    logger.info(f"   jugador_top50 - {(df['jugador_top50'] == 1).sum()} jugadores top 50")
    
    return df


def preparar_dataset_final(df):
    """
    Prepara el dataset final para entrenamiento
    
    Args:
        df: DataFrame con features
        
    Returns:
        Tuple (X, y, df) con features, target y dataframe completo
    """
    
    # Features que usaremos
    feature_columns = [
        'jugador_rank',
        'oponente_rank',
        'rank_diff',
        'rank_ratio',
        'jugador_top10',
        'oponente_top10',
        'jugador_top50',
        'oponente_top50',
        'surface_hard',
        'surface_clay',
        'surface_grass'
    ]
    
    # Seleccionar features y target
    X = df[feature_columns]
    y = df['resultado']
    
    logger.info("\n" + "=" * 60)
    logger.info("📦 DATASET FINAL")
    logger.info("=" * 60)
    logger.info(f"Features (X): {X.shape}")
    logger.info(f"Target (y): {y.shape}")
    logger.info(f"\nColumnas features:")
    for col in feature_columns:
        logger.info(f"   - {col}")
    
    # Guardar
    output_dir = Path("datos/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "dataset_con_features.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"\n💾 Dataset guardado en: {output_file}")
    
    return X, y, df


if __name__ == "__main__":
    # Cargar datos limpios
    df_clean = pd.read_csv("datos/processed/atp_matches_clean.csv")
    df_clean['tourney_date'] = pd.to_datetime(df_clean['tourney_date'])
    
    # Crear dataset de jugadores
    df_jugadores = crear_dataset_jugadores(df_clean)
    
    # Crear features
    df_con_features = crear_features_basicas(df_jugadores)
    
    # Preparar dataset final
    X, y, df_final = preparar_dataset_final(df_con_features)
    
    logger.info("\n✅ Feature engineering completado!")
