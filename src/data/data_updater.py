"""
Sistema de actualización automática de datos ATP
"""
import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataUpdater:
    """
    Sistema de actualización automática de datos
    """
    
    def __init__(self, data_dir="datos/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.current_year = datetime.now().year
    
    def actualizar_datos_github(self):
        """
        Actualiza datos desde Jeff Sackmann (se actualiza semanalmente)
        """
        
        logger.info("🔄 Actualizando datos desde GitHub...")
        
        # URL base
        base_url = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/"
        
        # Años a descargar (últimos 3 años + año actual)
        years = [self.current_year - 2, self.current_year - 1, self.current_year]
        
        datos_actualizados = []
        
        for year in years:
            url = f"{base_url}atp_matches_{year}.csv"
            
            try:
                logger.info(f"   Descargando {year}...")
                df = pd.read_csv(url)
                datos_actualizados.append(df)
                logger.info(f"   ✅ {len(df)} partidos de {year}")
            except Exception as e:
                logger.error(f"   ❌ Error en {year}: {e}")
        
        if datos_actualizados:
            df_completo = pd.concat(datos_actualizados, ignore_index=True)
            
            # Guardar con timestamp
            timestamp = datetime.now().strftime('%Y%m%d')
            output_file = self.data_dir / f"atp_matches_updated_{timestamp}.csv"
            df_completo.to_csv(output_file, index=False)
            
            # Guardar también como "latest"
            latest_file = self.data_dir / "atp_matches_latest.csv"
            df_completo.to_csv(latest_file, index=False)
            
            logger.info(f"\n✅ {len(df_completo)} partidos totales guardados")
            logger.info(f"📁 Archivo: {latest_file}")
            
            # Mostrar estadísticas
            self._mostrar_estadisticas(df_completo)
            
            return df_completo
        
        return None
    
    def verificar_actualizacion_necesaria(self):
        """
        Verifica si es necesario actualizar los datos
        """
        
        latest_file = self.data_dir / "atp_matches_latest.csv"
        
        if not latest_file.exists():
            logger.info("📥 No hay datos previos - descarga necesaria")
            return True
        
        # Verificar antigüedad del archivo
        import os
        file_time = os.path.getmtime(latest_file)
        file_date = datetime.fromtimestamp(file_time)
        days_old = (datetime.now() - file_date).days
        
        logger.info(f"📅 Datos actuales tienen {days_old} día(s) de antigüedad")
        
        if days_old >= 7:
            logger.info("🔄 Actualización recomendada (>7 días)")
            return True
        else:
            logger.info("✅ Datos recientes, no es necesario actualizar")
            return False
    
    def actualizar_si_necesario(self, force=False):
        """
        Actualiza solo si es necesario o si se fuerza
        """
        
        if force or self.verificar_actualizacion_necesaria():
            df = self.actualizar_datos_github()
            return df
        else:
            # Cargar datos existentes
            latest_file = self.data_dir / "atp_matches_latest.csv"
            return pd.read_csv(latest_file)
    
    def _mostrar_estadisticas(self, df):
        """
        Muestra estadísticas de los datos actualizados
        """
        
        df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')
        
        fecha_min = df['tourney_date'].min()
        fecha_max = df['tourney_date'].max()
        
        logger.info(f"\n📊 ESTADÍSTICAS:")
        logger.info(f"   Rango de fechas: {fecha_min.date()} - {fecha_max.date()}")
        logger.info(f"   Partidos totales: {len(df)}")
        
        # Partidos recientes (últimos 30 días)
        hace_30_dias = datetime.now() - timedelta(days=30)
        recientes = df[df['tourney_date'] >= hace_30_dias]
        logger.info(f"   Partidos últimos 30 días: {len(recientes)}")
        
        # Partidos por superficie
        logger.info(f"\n   Por superficie:")
        for superficie, count in df['surface'].value_counts().items():
            logger.info(f"      {superficie}: {count}")


if __name__ == "__main__":
    updater = DataUpdater()
    
    # Actualizar datos
    df = updater.actualizar_si_necesario(force=True)
    
    if df is not None:
        logger.info("\n🎉 Actualización completada!")
    else:
        logger.error("\n❌ Error en la actualización")
