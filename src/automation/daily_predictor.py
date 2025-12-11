#!/usr/bin/env python3
"""
Script que corre diariamente de forma automática
"""

import sys
import os
from datetime import datetime, date
import logging
from pathlib import Path
import pandas as pd
import shutil

# Añadir src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.automation.config import Config
from src.bookmakers.odds_fetcher import OddsFetcher
from src.bookmakers.odds_comparator import OddsComparator
from src.tracking.tracking_system import TrackingSystem
from src.bookmakers.alert_system import AlertSystem

# Configurar logging
Config.create_directories()

logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{Config.LOG_DIR}/daily_predictor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def run_daily_predictions():
    """
    Flujo completo de predicciones diarias
    """
    logger.info("=" * 60)
    logger.info("🤖 INICIANDO PREDICCIONES DIARIAS")
    logger.info("=" * 60)
    
    try:
        # 1. Obtener cuotas
        logger.info("📊 Paso 1: Obteniendo cuotas...")
        
        if not Config.ODDS_API_KEY:
            logger.error("❌ ODDS_API_KEY no configurada")
            return False
        
        fetcher = OddsFetcher(Config.ODDS_API_KEY)
        df_cuotas = fetcher.obtener_todas_cuotas()
        
        if len(df_cuotas) == 0:
            logger.warning("⚠️  No hay partidos disponibles hoy")
            return True  # No es un error, simplemente no hay partidos
        
        logger.info(f"✅ {len(df_cuotas)} cuotas obtenidas")
        
        # 2. Preparar datos de partidos
        logger.info("📋 Paso 2: Preparando partidos...")
        partidos = preparar_partidos_para_prediccion(df_cuotas)
        
        if len(partidos) == 0:
            logger.warning("⚠️  No se pudieron preparar partidos")
            return False
        
        logger.info(f"✅ {len(partidos)} partidos preparados")
        
        # 3. Generar predicciones
        logger.info("🎯 Paso 3: Generando predicciones...")
        
        sistema = TrackingSystem(
            modelo_path=Config.MODEL_PATH,
            db_path=Config.DB_PATH
        )
        
        resultados = sistema.procesar_jornada(partidos, umbral_ev=Config.EV_THRESHOLD)
        logger.info(f"✅ {len(resultados)} predicciones generadas")
        
        # 4. Identificar oportunidades
        logger.info("🔍 Paso 4: Identificando oportunidades...")
        
        alert = AlertSystem(
            smtp_server=Config.EMAIL_SMTP_SERVER,
            smtp_port=Config.EMAIL_SMTP_PORT,
            email_address=Config.EMAIL_ADDRESS,
            email_password=Config.EMAIL_PASSWORD
        )
        
        oportunidades = alert.verificar_oportunidades(
            resultados.to_dict('records'),
            umbral_ev=0.05
        )
        
        if len(oportunidades) > 0:
            logger.info(f"🚨 {len(oportunidades)} oportunidades detectadas!")
            
            # Enviar alertas por email
            try:
                alert.enviar_alerta_oportunidades(oportunidades)
                logger.info("✅ Alertas enviadas por email")
            except Exception as e:
                logger.error(f"❌ Error enviando alertas: {e}")
        else:
            logger.info("✅ No hay oportunidades con EV suficiente")
        
        # 5. Generar reporte diario
        logger.info("📊 Paso 5: Generando reporte diario...")
        generar_reporte_diario(resultados, oportunidades)
        logger.info("✅ Reporte generado")
        
        # 6. Backup de base de datos
        logger.info("💾 Paso 6: Backup de datos...")
        hacer_backup_db()
        logger.info("✅ Backup completado")
        
        logger.info("\n🎉 PROCESO COMPLETADO EXITOSAMENTE")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ ERROR: {str(e)}", exc_info=True)
        enviar_alerta_error(str(e))
        return False


def preparar_partidos_para_prediccion(df_cuotas):
    """
    Convierte cuotas a formato para predicción
    """
    try:
        comparador = OddsComparator(df_cuotas)
        
        # Obtener partidos únicos
        partidos_unicos = df_cuotas.groupby(['jugador1', 'jugador2']).first().reset_index()
        
        partidos = []
        
        for _, row in partidos_unicos.iterrows():
            # Obtener mejor cuota
            mejor = comparador.encontrar_mejor_cuota(row['jugador1'], row['jugador2'])
            
            if mejor:
                # Obtener rankings (desde fuente externa o estimación)
                rank_j1, rank_j2 = obtener_rankings(row['jugador1'], row['jugador2'])
                
                partidos.append({
                    'fecha_partido': row['fecha'].date() if hasattr(row['fecha'], 'date') else row['fecha'],
                    'jugador_nombre': row['jugador1'],
                    'jugador_rank': rank_j1,
                    'oponente_nombre': row['jugador2'],
                    'oponente_rank': rank_j2,
                    'superficie': 'Hard',  # TODO: Obtener de fuente externa
                    'torneo': row.get('torneo', 'Unknown'),
                    'cuota': mejor['mejor_cuota_j1'],
                    'bookmaker': mejor['bookmaker_j1']
                })
        
        return pd.DataFrame(partidos)
    
    except Exception as e:
        logger.error(f"Error preparando partidos: {e}")
        return pd.DataFrame()


def obtener_rankings(jugador1, jugador2):
    """
    Obtiene rankings actuales
    
    TODO: Implementar con API o scraping de ATP
    Por ahora, estimación básica
    """
    # Estimación básica basada en nombres conocidos
    # En producción, esto debería consultar una API de rankings ATP
    return 50, 60


def generar_reporte_diario(resultados, oportunidades):
    """
    Genera reporte HTML del día
    """
    try:
        fecha_hoy = date.today().strftime('%Y-%m-%d')
        reporte_path = Path(f'resultados/reportes_diarios/reporte_{fecha_hoy}.html')
        reporte_path.parent.mkdir(parents=True, exist_ok=True)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reporte Diario - {fecha_hoy}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                .oportunidad {{ background-color: #2ecc71; color: white; font-weight: bold; }}
                .no-apostar {{ background-color: #e74c3c; color: white; }}
            </style>
        </head>
        <body>
            <h1>📊 Reporte Diario - {fecha_hoy}</h1>
            
            <h2>📈 Resumen</h2>
            <p>Total predicciones: {len(resultados)}</p>
            <p>Oportunidades detectadas: {len(oportunidades)}</p>
            
            <h2>🎯 Predicciones del Día</h2>
            {resultados.to_html(index=False, classes='table')}
            
            <h2>🚨 Oportunidades de Apuesta</h2>
            {pd.DataFrame(oportunidades).to_html(index=False, classes='table') if len(oportunidades) > 0 else '<p>No hay oportunidades hoy</p>'}
        </body>
        </html>
        """
        
        with open(reporte_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"   Reporte guardado: {reporte_path}")
        
    except Exception as e:
        logger.error(f"Error generando reporte: {e}")


def hacer_backup_db():
    """
    Crea backup de la base de datos
    """
    try:
        db_path = Path(Config.DB_PATH)
        
        if not db_path.exists():
            logger.warning(f"⚠️  Base de datos no encontrada: {db_path}")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = Path(Config.DATA_BACKUP_DIR) / f'apuestas_tracker_{timestamp}.db'
        
        shutil.copy2(db_path, backup_path)
        logger.info(f"   Backup guardado: {backup_path}")
        
        # Limpiar backups antiguos (mantener solo últimos 30 días)
        limpiar_backups_antiguos()
        
    except Exception as e:
        logger.error(f"Error creando backup: {e}")


def limpiar_backups_antiguos(dias=30):
    """
    Elimina backups más antiguos de N días
    """
    try:
        backup_dir = Path(Config.DATA_BACKUP_DIR)
        limite = datetime.now() - pd.Timedelta(days=dias)
        
        for backup in backup_dir.glob('apuestas_tracker_*.db'):
            if datetime.fromtimestamp(backup.stat().st_mtime) < limite:
                backup.unlink()
                logger.info(f"   Backup antiguo eliminado: {backup.name}")
    
    except Exception as e:
        logger.error(f"Error limpiando backups: {e}")


def enviar_alerta_error(error_msg):
    """
    Envía alerta si hay error crítico
    """
    logger.critical(f"🚨 ERROR CRÍTICO: {error_msg}")
    
    # Intentar enviar email de alerta
    try:
        if Config.EMAIL_ADDRESS and Config.EMAIL_PASSWORD:
            alert = AlertSystem(
                smtp_server=Config.EMAIL_SMTP_SERVER,
                smtp_port=Config.EMAIL_SMTP_PORT,
                email_address=Config.EMAIL_ADDRESS,
                email_password=Config.EMAIL_PASSWORD
            )
            
            alert.enviar_alerta_error(error_msg)
    except Exception as e:
        logger.error(f"No se pudo enviar alerta de error: {e}")


if __name__ == "__main__":
    # Crear directorios necesarios
    Config.create_directories()
    Path("resultados/reportes_diarios").mkdir(parents=True, exist_ok=True)
    
    # Ejecutar
    success = run_daily_predictions()
    
    sys.exit(0 if success else 1)
