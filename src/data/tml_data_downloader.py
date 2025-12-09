"""
Descargador de datos desde TML-Database (actualizado diariamente)
Fuente: https://github.com/Tennismylife/TML-Database
"""
import pandas as pd
import requests
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def descargar_tml_data(year=2025):
    """
    Descarga datos de TML-Database (actualizado diariamente)
    
    Args:
        year: Año a descargar
        
    Returns:
        DataFrame con partidos
    """
    
    base_url = "https://raw.githubusercontent.com/Tennismylife/TML-Database/master/"
    
    # URL correcta: archivos nombrados como year.csv en rama master
    url = f"{base_url}{year}.csv"
    
    logger.info(f"📥 Descargando datos TML de {year}...")
    logger.info(f"   URL: {url}")
    
    try:
        df = pd.read_csv(url)
        logger.info(f"   ✅ {len(df)} partidos descargados")
        
        # Guardar
        output_dir = Path("datos/raw")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"atp_matches_{year}_tml.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"💾 Guardado en: {output_file}")
        
        return df
        
    except Exception as e:
        logger.error(f"   ❌ Error: {e}")
        return None


def actualizar_datos_completos():
    """
    Actualiza con datos de 2018-2025 desde TML
    """
    
    logger.info("=" * 60)
    logger.info("🔄 ACTUALIZACIÓN COMPLETA DE DATOS - TML DATABASE")
    logger.info("=" * 60)
    
    # Descargar 2020-2025 (datos más recientes = mejor modelo)
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    datos = []
    
    for year in years:
        df = descargar_tml_data(year)
        if df is not None:
            datos.append(df)
    
    if datos:
        df_completo = pd.concat(datos, ignore_index=True)
        
        # Guardar combinado
        output_file = Path("datos/raw/atp_matches_raw_updated.csv")
        df_completo.to_csv(output_file, index=False)
        
        logger.info(f"\n✅ TOTAL: {len(df_completo)} partidos")
        logger.info(f"📁 Archivo: {output_file}")
        
        # Estadísticas por año
        df_completo['tourney_date'] = pd.to_datetime(df_completo['tourney_date'], format='%Y%m%d', errors='coerce')
        logger.info("\n📊 Partidos por año:")
        for year in years:
            count = len(df_completo[df_completo['tourney_date'].dt.year == year])
            logger.info(f"   {year}: {count} partidos")
        
        # Mostrar última fecha
        ultima_fecha = df_completo['tourney_date'].max()
        logger.info(f"\n📅 Última fecha de datos: {ultima_fecha.date()}")
        
        # Verificar datos recientes (últimos 30 días)
        from datetime import timedelta
        hace_30_dias = datetime.now() - timedelta(days=30)
        recientes = df_completo[df_completo['tourney_date'] >= hace_30_dias]
        logger.info(f"📊 Partidos últimos 30 días: {len(recientes)}")
        
        return df_completo
    
    return None


def verificar_compatibilidad():
    """
    Verifica que el formato TML es compatible con el código existente
    """
    
    logger.info("\n" + "=" * 60)
    logger.info("🔍 VERIFICANDO COMPATIBILIDAD DE FORMATO")
    logger.info("=" * 60)
    
    # Descargar muestra de 2024 para verificar
    df_tml = descargar_tml_data(2024)
    
    if df_tml is None:
        logger.error("❌ No se pudo descargar datos de TML")
        return False
    
    # Columnas requeridas por nuestro código
    columnas_requeridas = [
        'tourney_date', 'surface', 'winner_name', 'loser_name',
        'winner_rank', 'loser_rank', 'score'
    ]
    
    # Verificar columnas
    columnas_faltantes = [col for col in columnas_requeridas if col not in df_tml.columns]
    
    if columnas_faltantes:
        logger.error(f"❌ Columnas faltantes: {columnas_faltantes}")
        logger.info(f"📋 Columnas disponibles: {df_tml.columns.tolist()}")
        return False
    
    logger.info("✅ Todas las columnas requeridas están presentes")
    
    # Verificar tipos de datos
    logger.info("\n📊 Verificando tipos de datos:")
    logger.info(f"   tourney_date: {df_tml['tourney_date'].dtype}")
    logger.info(f"   winner_rank: {df_tml['winner_rank'].dtype}")
    logger.info(f"   loser_rank: {df_tml['loser_rank'].dtype}")
    
    # Verificar valores
    logger.info("\n📊 Muestra de datos:")
    logger.info(f"   Superficies: {df_tml['surface'].value_counts().to_dict()}")
    logger.info(f"   Rankings nulos: winner={df_tml['winner_rank'].isna().sum()}, loser={df_tml['loser_rank'].isna().sum()}")
    
    logger.info("\n✅ Formato TML es compatible con el código existente")
    return True


if __name__ == "__main__":
    # Verificar compatibilidad primero
    if verificar_compatibilidad():
        logger.info("\n" + "=" * 60)
        logger.info("✅ COMPATIBILIDAD VERIFICADA - Procediendo con descarga completa")
        logger.info("=" * 60)
        
        # Actualizar todo
        df = actualizar_datos_completos()
        
        if df is not None:
            logger.info("\n🎉 ¡Datos actualizados exitosamente!")
            logger.info(f"📊 Total de partidos: {len(df)}")
            logger.info(f"📅 Último partido: {df['tourney_date'].max()}")
            logger.info("\n🎯 SIGUIENTE PASO:")
            logger.info("   python run_pipeline.py")
        else:
            logger.error("\n❌ Error al actualizar datos")
    else:
        logger.error("\n❌ Formato incompatible - revisar código")
