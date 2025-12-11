"""
Sistema de actualización automática de datos
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path
import logging
import shutil

logger = logging.getLogger(__name__)


class DataUpdater:
    """
    Actualiza automáticamente los datos de partidos desde TML GitHub
    """
    
    def __init__(self, data_path="datos/processed/dataset_features_completas.csv"):
        self.data_path = Path(data_path)
        self.tml_base_url = "https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master"
        
    def obtener_ultima_fecha_local(self):
        """
        Obtiene la fecha del partido más reciente en datos locales
        """
        try:
            if not self.data_path.exists():
                logger.warning(f"⚠️  Dataset no encontrado: {self.data_path}")
                return None
            
            df = pd.read_csv(self.data_path)
            
            if 'fecha' in df.columns:
                df['fecha'] = pd.to_datetime(df['fecha'])
                ultima_fecha = df['fecha'].max()
                logger.info(f"📅 Última fecha en datos locales: {ultima_fecha.date()}")
                return ultima_fecha
            else:
                logger.warning("⚠️  Columna 'fecha' no encontrada en dataset")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error obteniendo última fecha: {e}")
            return None
    
    def hay_datos_nuevos(self):
        """
        Verifica si hay datos más recientes disponibles
        """
        ultima_fecha = self.obtener_ultima_fecha_local()
        
        if ultima_fecha is None:
            return True  # Si no hay datos, necesitamos actualizar
        
        # Verificar si han pasado más de 7 días desde última actualización
        dias_desde_actualizacion = (datetime.now() - ultima_fecha).days
        
        logger.info(f"📊 Días desde última actualización: {dias_desde_actualizacion}")
        
        if dias_desde_actualizacion >= 7:
            logger.info("✅ Han pasado más de 7 días - actualización necesaria")
            return True
        else:
            logger.info("ℹ️  Datos relativamente recientes - no es necesario actualizar")
            return False
    
    def descargar_datos_tml(self, year=None):
        """
        Descarga datos del repositorio TML
        
        Args:
            year: Año específico a descargar (None = año actual)
        """
        if year is None:
            year = datetime.now().year
        
        try:
            # URL del archivo de datos del año
            url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"
            
            logger.info(f"📥 Descargando datos de {year} desde TML...")
            
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                # Guardar temporalmente
                temp_path = Path(f"datos/raw/atp_matches_{year}_temp.csv")
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(temp_path, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"✅ Datos de {year} descargados correctamente")
                return temp_path
            else:
                logger.warning(f"⚠️  No se pudieron descargar datos de {year} (Status: {response.status_code})")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error descargando datos: {e}")
            return None
    
    def actualizar_dataset(self, nuevos_datos_path):
        """
        Integra nuevos datos al dataset existente
        """
        try:
            # Cargar datos actuales
            if self.data_path.exists():
                df_actual = pd.read_csv(self.data_path)
                logger.info(f"📊 Dataset actual: {len(df_actual)} partidos")
            else:
                df_actual = pd.DataFrame()
                logger.info("📊 No hay dataset previo - creando nuevo")
            
            # Cargar nuevos datos
            df_nuevos = pd.read_csv(nuevos_datos_path)
            logger.info(f"📊 Datos nuevos: {len(df_nuevos)} partidos")
            
            # Combinar (evitar duplicados por fecha y jugadores)
            if not df_actual.empty:
                # Identificar duplicados
                df_combinado = pd.concat([df_actual, df_nuevos], ignore_index=True)
                
                # Eliminar duplicados (si existen columnas clave)
                if 'fecha' in df_combinado.columns and 'jugador_nombre' in df_combinado.columns:
                    antes = len(df_combinado)
                    df_combinado = df_combinado.drop_duplicates(
                        subset=['fecha', 'jugador_nombre', 'oponente_nombre'],
                        keep='first'
                    )
                    despues = len(df_combinado)
                    logger.info(f"🔄 Eliminados {antes - despues} duplicados")
                
                df_final = df_combinado
            else:
                df_final = df_nuevos
            
            # Backup del dataset actual
            if self.data_path.exists():
                backup_path = self.data_path.parent / f"dataset_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                shutil.copy2(self.data_path, backup_path)
                logger.info(f"💾 Backup creado: {backup_path}")
            
            # Guardar dataset actualizado
            df_final.to_csv(self.data_path, index=False)
            logger.info(f"✅ Dataset actualizado: {len(df_final)} partidos totales")
            
            return df_final
            
        except Exception as e:
            logger.error(f"❌ Error actualizando dataset: {e}")
            return None
    
    def actualizar_si_necesario(self, force=False):
        """
        Actualiza datos solo si es necesario
        
        Args:
            force: Forzar actualización aunque no sea necesario
        
        Returns:
            DataFrame actualizado o None si no se actualizó
        """
        logger.info("=" * 60)
        logger.info("🔄 VERIFICANDO ACTUALIZACIÓN DE DATOS")
        logger.info("=" * 60)
        
        if not force and not self.hay_datos_nuevos():
            logger.info("✅ No es necesario actualizar datos")
            return None
        
        logger.info("📥 Iniciando actualización de datos...")
        
        # Descargar datos del año actual
        year = datetime.now().year
        nuevos_datos = self.descargar_datos_tml(year)
        
        if nuevos_datos is None:
            logger.error("❌ No se pudieron descargar datos nuevos")
            return None
        
        # Actualizar dataset
        df_actualizado = self.actualizar_dataset(nuevos_datos)
        
        if df_actualizado is not None:
            logger.info("✅ Actualización completada exitosamente")
        else:
            logger.error("❌ Error en la actualización")
        
        return df_actualizado
    
    def obtener_estadisticas(self):
        """
        Obtiene estadísticas del dataset actual
        """
        try:
            if not self.data_path.exists():
                return None
            
            df = pd.read_csv(self.data_path)
            
            stats = {
                'total_partidos': len(df),
                'fecha_mas_antigua': None,
                'fecha_mas_reciente': None,
                'años_cubiertos': None
            }
            
            if 'fecha' in df.columns:
                df['fecha'] = pd.to_datetime(df['fecha'])
                stats['fecha_mas_antigua'] = df['fecha'].min().date()
                stats['fecha_mas_reciente'] = df['fecha'].max().date()
                stats['años_cubiertos'] = df['fecha'].dt.year.nunique()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return None


# Script principal
def main():
    """
    Script para ejecutar actualización de datos
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/data_updater.log'),
            logging.StreamHandler()
        ]
    )
    
    updater = DataUpdater()
    
    # Mostrar estadísticas actuales
    stats = updater.obtener_estadisticas()
    if stats:
        logger.info("\n📊 ESTADÍSTICAS ACTUALES:")
        logger.info(f"   Total partidos: {stats['total_partidos']}")
        logger.info(f"   Fecha más antigua: {stats['fecha_mas_antigua']}")
        logger.info(f"   Fecha más reciente: {stats['fecha_mas_reciente']}")
        logger.info(f"   Años cubiertos: {stats['años_cubiertos']}")
    
    # Actualizar si es necesario
    df = updater.actualizar_si_necesario()
    
    if df is not None:
        logger.info("\n✅ Datos actualizados correctamente")
    else:
        logger.info("\nℹ️  No se realizaron actualizaciones")


if __name__ == "__main__":
    main()
