"""
Sistema de monitoreo y alertas
"""

import logging
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import sys
import os

# Añadir src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.automation.config import Config
from src.tracking.database_setup import TennisDatabase

logger = logging.getLogger(__name__)


class SystemMonitor:
    """
    Monitorea el sistema y detecta problemas
    """
    
    def __init__(self, db_path=None):
        self.db = TennisDatabase(db_path or Config.DB_PATH)
        self.logger = logging.getLogger(__name__)
    
    def check_daily_execution(self):
        """
        Verifica que el script diario se haya ejecutado
        """
        try:
            # Verificar si hay predicciones de hoy
            df = self.db.obtener_predicciones()
            
            if df.empty:
                self.logger.warning("⚠️  No hay predicciones en la base de datos")
                return False
            
            df['fecha_prediccion'] = pd.to_datetime(df['fecha_prediccion'])
            
            hoy = datetime.now().date()
            df_hoy = df[df['fecha_prediccion'].dt.date == hoy]
            
            if len(df_hoy) == 0:
                self.logger.warning("⚠️  No hay predicciones de hoy")
                return False
            
            self.logger.info(f"✅ {len(df_hoy)} predicciones generadas hoy")
            return True
        
        except Exception as e:
            self.logger.error(f"Error verificando ejecución diaria: {e}")
            return False
    
    def check_model_performance(self, lookback_days=7):
        """
        Verifica performance del modelo en últimos N días
        """
        try:
            fecha_limite = datetime.now() - timedelta(days=lookback_days)
            
            df = self.db.obtener_predicciones({'decision': 'APOSTAR'})
            
            if df.empty:
                self.logger.warning("⚠️  No hay apuestas registradas")
                return True
            
            df['fecha_prediccion'] = pd.to_datetime(df['fecha_prediccion'])
            df_recientes = df[df['fecha_prediccion'] >= fecha_limite]
            
            if len(df_recientes) == 0:
                self.logger.warning("⚠️  No hay apuestas recientes para evaluar")
                return True
            
            # Calcular win rate solo de apuestas completadas
            completadas = df_recientes[df_recientes['resultado_real'].notna()]
            
            if len(completadas) > 0:
                win_rate = (completadas['resultado_real'] == 1).sum() / len(completadas)
                
                self.logger.info(f"📊 Últimos {lookback_days} días:")
                self.logger.info(f"   Apuestas completadas: {len(completadas)}")
                self.logger.info(f"   Win rate: {win_rate*100:.1f}%")
                
                if win_rate < 0.45:
                    self.logger.error(f"🚨 Win rate bajo: {win_rate*100:.1f}%")
                    return False
                
                self.logger.info(f"✅ Win rate aceptable: {win_rate*100:.1f}%")
            else:
                self.logger.info("ℹ️  No hay apuestas completadas aún")
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error verificando performance: {e}")
            return False
    
    def check_data_freshness(self, max_days=14):
        """
        Verifica que los datos no estén muy desactualizados
        """
        try:
            data_path = Path(Config.DATA_PATH)
            
            if not data_path.exists():
                self.logger.error(f"❌ Dataset no encontrado: {data_path}")
                return False
            
            df = pd.read_csv(data_path)
            
            if 'fecha' not in df.columns:
                self.logger.warning("⚠️  No se puede verificar frescura de datos (sin columna fecha)")
                return True
            
            df['fecha'] = pd.to_datetime(df['fecha'])
            ultima_fecha = df['fecha'].max()
            dias_desde_actualizacion = (datetime.now() - ultima_fecha).days
            
            self.logger.info(f"📅 Última fecha en datos: {ultima_fecha.date()}")
            self.logger.info(f"📊 Días desde última actualización: {dias_desde_actualizacion}")
            
            if dias_desde_actualizacion > max_days:
                self.logger.warning(f"⚠️  Datos desactualizados ({dias_desde_actualizacion} días)")
                return False
            
            self.logger.info("✅ Datos actualizados")
            return True
        
        except Exception as e:
            self.logger.error(f"Error verificando frescura de datos: {e}")
            return False
    
    def check_model_exists(self):
        """
        Verifica que el modelo de producción exista
        """
        try:
            model_path = Path(Config.MODEL_PATH)
            
            if not model_path.exists():
                self.logger.error(f"❌ Modelo de producción no encontrado: {model_path}")
                return False
            
            # Verificar fecha del modelo
            model_time = datetime.fromtimestamp(model_path.stat().st_mtime)
            dias_desde_entrenamiento = (datetime.now() - model_time).days
            
            self.logger.info(f"🤖 Modelo encontrado: {model_path}")
            self.logger.info(f"📅 Días desde último entrenamiento: {dias_desde_entrenamiento}")
            
            if dias_desde_entrenamiento > 30:
                self.logger.warning(f"⚠️  Modelo antiguo ({dias_desde_entrenamiento} días)")
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error verificando modelo: {e}")
            return False
    
    def check_logs(self):
        """
        Verifica que los logs se estén generando correctamente
        """
        try:
            log_dir = Path(Config.LOG_DIR)
            
            if not log_dir.exists():
                self.logger.warning(f"⚠️  Directorio de logs no encontrado: {log_dir}")
                return False
            
            # Verificar logs recientes
            log_files = list(log_dir.glob('*.log'))
            
            if not log_files:
                self.logger.warning("⚠️  No se encontraron archivos de log")
                return False
            
            # Verificar log más reciente
            log_mas_reciente = max(log_files, key=lambda p: p.stat().st_mtime)
            ultima_modificacion = datetime.fromtimestamp(log_mas_reciente.stat().st_mtime)
            horas_desde_log = (datetime.now() - ultima_modificacion).total_seconds() / 3600
            
            self.logger.info(f"📝 Último log: {log_mas_reciente.name}")
            self.logger.info(f"⏰ Horas desde última escritura: {horas_desde_log:.1f}")
            
            if horas_desde_log > 48:
                self.logger.warning(f"⚠️  Logs no actualizados en {horas_desde_log:.1f} horas")
                return False
            
            self.logger.info("✅ Logs actualizados")
            return True
        
        except Exception as e:
            self.logger.error(f"Error verificando logs: {e}")
            return False
    
    def run_all_checks(self):
        """
        Ejecuta todos los checks
        """
        self.logger.info("=" * 60)
        self.logger.info("🔍 EJECUTANDO CHECKS DEL SISTEMA")
        self.logger.info("=" * 60)
        
        checks = {
            'Modelo existe': self.check_model_exists(),
            'Datos actualizados': self.check_data_freshness(),
            'Ejecución diaria': self.check_daily_execution(),
            'Performance del modelo': self.check_model_performance(),
            'Logs actualizados': self.check_logs()
        }
        
        self.logger.info("\n📊 RESUMEN DE CHECKS:")
        for check_name, resultado in checks.items():
            status = "✅" if resultado else "❌"
            self.logger.info(f"   {status} {check_name}")
        
        all_ok = all(checks.values())
        
        if all_ok:
            self.logger.info("\n✅ TODOS LOS CHECKS PASARON")
        else:
            self.logger.error("\n❌ ALGUNOS CHECKS FALLARON")
            # Enviar alerta
            self._enviar_alerta_checks(checks)
        
        return all_ok
    
    def _enviar_alerta_checks(self, checks):
        """
        Envía alerta si hay checks fallidos
        """
        try:
            from src.bookmakers.alert_system import AlertSystem
            
            if not Config.EMAIL_ADDRESS or not Config.EMAIL_PASSWORD:
                self.logger.warning("⚠️  No se pueden enviar alertas (email no configurado)")
                return
            
            alert = AlertSystem(
                smtp_server=Config.EMAIL_SMTP_SERVER,
                smtp_port=Config.EMAIL_SMTP_PORT,
                email_address=Config.EMAIL_ADDRESS,
                email_password=Config.EMAIL_PASSWORD
            )
            
            checks_fallidos = [name for name, result in checks.items() if not result]
            mensaje = f"Los siguientes checks fallaron:\n" + "\n".join(f"  - {c}" for c in checks_fallidos)
            
            alert.enviar_alerta_error(mensaje)
            self.logger.info("📧 Alerta de checks enviada por email")
        
        except Exception as e:
            self.logger.error(f"Error enviando alerta: {e}")


# Script principal
def main():
    """
    Script para ejecutar monitoring
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/monitoring.log'),
            logging.StreamHandler()
        ]
    )
    
    monitor = SystemMonitor()
    success = monitor.run_all_checks()
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
