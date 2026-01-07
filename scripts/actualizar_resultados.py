"""
Script de Actualización de Resultados
======================================

Script que actualiza los resultados de partidos finalizados:
1. Obtiene resultados de partidos del día anterior
2. Actualiza base de datos con resultados reales
3. Calcula ROI real vs predicho
4. Genera reporte de rendimiento

Uso:
    # Actualizar resultados del día anterior
    python scripts/actualizar_resultados.py
    
    # Actualizar resultados de una fecha específica
    python scripts/actualizar_resultados.py --date 2026-01-06
    
    # Dry run (no actualiza BD)
    python scripts/actualizar_resultados.py --dry-run

Cron job (ejecutar cada día a las 8 AM):
    0 8 * * * cd /path/to/tennis-ml-predictor && python scripts/actualizar_resultados.py
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime, date, timedelta
import logging
import pandas as pd

# Añadir src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import Config
from src.data.tml_data_downloader import TMLDataDownloader
from src.tracking.database_setup import TennisDatabase

# Configurar logging
log_file = Config.LOG_DIR / f"actualizar_resultados_{date.today()}.log"
Config.LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ActualizadorResultados:
    """Actualizador de resultados de partidos"""
    
    def __init__(self, db_path: str, dry_run: bool = False):
        """
        Args:
            db_path: Path a la base de datos
            dry_run: Si True, no actualiza base de datos
        """
        self.db_path = db_path
        self.dry_run = dry_run
        
        # Inicializar componentes
        logger.info("Inicializando componentes...")
        self.db = TennisDatabase(db_path)
        self.downloader = TMLDataDownloader()
        
        logger.info("✅ Componentes inicializados")
    
    
    def obtener_resultados(self, fecha: date) -> pd.DataFrame:
        """
        Obtiene resultados de partidos de una fecha
        
        Args:
            fecha: Fecha a consultar
        
        Returns:
            DataFrame con resultados
        """
        logger.info(f"📅 Obteniendo resultados para {fecha}")
        
        try:
            # Descargar datos de la fecha
            fecha_str = fecha.strftime('%Y-%m-%d')
            fecha_siguiente = (fecha + timedelta(days=1)).strftime('%Y-%m-%d')
            
            df = self.downloader.download_matches(
                start_date=fecha_str,
                end_date=fecha_siguiente
            )
            
            if df.empty:
                logger.warning(f"⚠️  No se encontraron resultados para {fecha}")
                return pd.DataFrame()
            
            # Filtrar solo partidos finalizados
            df = df[df['winner'].notna()].copy()
            
            logger.info(f"✅ Encontrados {len(df)} resultados para {fecha}")
            return df
        
        except Exception as e:
            logger.error(f"❌ Error obteniendo resultados: {e}", exc_info=True)
            return pd.DataFrame()
    
    
    def actualizar_predicciones(self, df_resultados: pd.DataFrame, fecha: date) -> dict:
        """
        Actualiza predicciones con resultados reales
        
        Args:
            df_resultados: DataFrame con resultados
            fecha: Fecha de los partidos
        
        Returns:
            Estadísticas de actualización
        """
        if df_resultados.empty:
            return {'actualizadas': 0, 'no_encontradas': 0}
        
        # Obtener predicciones de la fecha
        predicciones = self.db.obtener_predicciones()
        
        if predicciones.empty:
            logger.warning("⚠️  No hay predicciones en la base de datos")
            return {'actualizadas': 0, 'no_encontradas': 0}
        
        # Filtrar predicciones de la fecha
        predicciones['fecha_partido'] = pd.to_datetime(predicciones['fecha_partido']).dt.date
        predicciones_fecha = predicciones[predicciones['fecha_partido'] == fecha].copy()
        
        logger.info(f"🔍 Encontradas {len(predicciones_fecha)} predicciones para {fecha}")
        
        stats = {
            'actualizadas': 0,
            'no_encontradas': 0,
            'ganadas': 0,
            'perdidas': 0
        }
        
        for idx, pred in predicciones_fecha.iterrows():
            try:
                jugador1 = pred['jugador1']
                jugador2 = pred['jugador2']
                
                # Buscar resultado
                resultado = self._buscar_resultado(df_resultados, jugador1, jugador2)
                
                if resultado is None:
                    logger.warning(f"⚠️  No se encontró resultado para {jugador1} vs {jugador2}")
                    stats['no_encontradas'] += 1
                    continue
                
                # Actualizar en base de datos
                if not self.dry_run:
                    self.db.actualizar_resultado(
                        prediccion_id=pred['id'],
                        resultado_real=resultado
                    )
                
                stats['actualizadas'] += 1
                
                if resultado == 1:
                    stats['ganadas'] += 1
                    logger.info(f"  ✅ {jugador1} vs {jugador2}: GANADA")
                else:
                    stats['perdidas'] += 1
                    logger.info(f"  ❌ {jugador1} vs {jugador2}: PERDIDA")
            
            except Exception as e:
                logger.error(f"❌ Error actualizando {jugador1} vs {jugador2}: {e}")
                continue
        
        logger.info(f"✅ Actualizadas {stats['actualizadas']} predicciones")
        return stats
    
    
    def _buscar_resultado(self, df_resultados: pd.DataFrame, jugador1: str, jugador2: str) -> int:
        """
        Busca el resultado de un partido
        
        Args:
            df_resultados: DataFrame con resultados
            jugador1: Nombre del jugador 1
            jugador2: Nombre del jugador 2
        
        Returns:
            1 si ganó jugador1, 0 si ganó jugador2, None si no se encuentra
        """
        # Normalizar nombres (quitar acentos, mayúsculas, etc.)
        j1_norm = self._normalizar_nombre(jugador1)
        j2_norm = self._normalizar_nombre(jugador2)
        
        for idx, partido in df_resultados.iterrows():
            p1 = self._normalizar_nombre(partido.get('player1_name', ''))
            p2 = self._normalizar_nombre(partido.get('player2_name', ''))
            ganador = self._normalizar_nombre(partido.get('winner', ''))
            
            # Verificar si es el partido correcto
            if (j1_norm in p1 and j2_norm in p2) or (j1_norm in p2 and j2_norm in p1):
                # Determinar ganador
                if j1_norm in ganador:
                    return 1
                elif j2_norm in ganador:
                    return 0
        
        return None
    
    
    def _normalizar_nombre(self, nombre: str) -> str:
        """
        Normaliza un nombre para comparación
        
        Args:
            nombre: Nombre a normalizar
        
        Returns:
            Nombre normalizado
        """
        import unicodedata
        
        # Quitar acentos
        nombre = unicodedata.normalize('NFKD', nombre)
        nombre = nombre.encode('ASCII', 'ignore').decode('ASCII')
        
        # Minúsculas y quitar espacios extra
        nombre = nombre.lower().strip()
        
        return nombre
    
    
    def generar_reporte(self, stats: dict, fecha: date):
        """
        Genera reporte de rendimiento
        
        Args:
            stats: Estadísticas de actualización
            fecha: Fecha procesada
        """
        logger.info("=" * 70)
        logger.info("📊 REPORTE DE RENDIMIENTO")
        logger.info("=" * 70)
        logger.info(f"📅 Fecha: {fecha}")
        logger.info(f"  Predicciones actualizadas: {stats['actualizadas']}")
        logger.info(f"  Predicciones no encontradas: {stats['no_encontradas']}")
        
        if stats['actualizadas'] > 0:
            win_rate = stats['ganadas'] / stats['actualizadas']
            logger.info(f"  Ganadas: {stats['ganadas']}")
            logger.info(f"  Perdidas: {stats['perdidas']}")
            logger.info(f"  Win Rate: {win_rate:.2%}")
        
        logger.info("=" * 70)
    
    
    def ejecutar(self, fecha: date = None):
        """
        Ejecuta el proceso completo de actualización
        
        Args:
            fecha: Fecha a procesar (default: ayer)
        """
        logger.info("=" * 70)
        logger.info("🎾 ACTUALIZACIÓN DE RESULTADOS - INICIO")
        logger.info("=" * 70)
        
        if fecha is None:
            fecha = date.today() - timedelta(days=1)
        
        logger.info(f"📅 Fecha: {fecha}")
        logger.info(f"🔍 Dry Run: {self.dry_run}")
        
        # 1. Obtener resultados
        df_resultados = self.obtener_resultados(fecha)
        
        if df_resultados.empty:
            logger.warning("⚠️  No hay resultados para procesar")
            return
        
        # 2. Actualizar predicciones
        stats = self.actualizar_predicciones(df_resultados, fecha)
        
        # 3. Generar reporte
        self.generar_reporte(stats, fecha)
        
        logger.info("=" * 70)
        logger.info("✅ ACTUALIZACIÓN DE RESULTADOS - COMPLETADO")
        logger.info("=" * 70)


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Actualizador de resultados')
    parser.add_argument('--dry-run', action='store_true',
                       help='No actualizar base de datos (solo simular)')
    parser.add_argument('--date', type=str,
                       help='Fecha a procesar (formato: YYYY-MM-DD, default: ayer)')
    parser.add_argument('--db', type=str, default=Config.DB_PATH,
                       help='Path a la base de datos')
    
    args = parser.parse_args()
    
    # Parsear fecha
    fecha = None
    if args.date:
        try:
            fecha = datetime.strptime(args.date, '%Y-%m-%d').date()
        except ValueError:
            logger.error(f"❌ Formato de fecha inválido: {args.date}")
            return
    
    # Ejecutar
    try:
        actualizador = ActualizadorResultados(
            db_path=args.db,
            dry_run=args.dry_run
        )
        
        actualizador.ejecutar(fecha)
    
    except Exception as e:
        logger.error(f"❌ Error fatal: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
