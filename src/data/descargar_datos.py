"""
Script para descargar datos de ATP desde GitHub de Jeff Sackmann
"""
import pandas as pd
import requests
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def descargar_datos_atp(years=[2020, 2021, 2022, 2023, 2024]):
    """
    Descarga datos de ATP desde GitHub de Jeff Sackmann
    
    Args:
        years: Lista de años a descargar
        
    Returns:
        DataFrame con todos los partidos
    """
    
    base_url = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/"
    
    datos = []
    
    for year in years:
        url = f"{base_url}atp_matches_{year}.csv"
        logger.info(f"📥 Descargando datos de {year}...")
        
        try:
            df = pd.read_csv(url)
            datos.append(df)
            logger.info(f"   ✅ {len(df)} partidos descargados")
        except Exception as e:
            logger.error(f"   ❌ Error: {e}")
    
    # Combinar todos los años
    df_completo = pd.concat(datos, ignore_index=True)
    
    logger.info(f"\n📊 Total de partidos: {len(df_completo)}")
    
    # Guardar
    output_dir = Path("datos/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "atp_matches_raw.csv"
    df_completo.to_csv(output_file, index=False)
    logger.info(f"💾 Datos guardados en: {output_file}")
    
    return df_completo


if __name__ == "__main__":
    df = descargar_datos_atp(years=[2020, 2021, 2022, 2023, 2024])
    logger.info("\n✅ Datos descargados exitosamente!")
    logger.info(f"📊 Total de partidos: {len(df)}")
    logger.info(f"📅 Rango de fechas: {df['tourney_date'].min()} - {df['tourney_date'].max()}")
