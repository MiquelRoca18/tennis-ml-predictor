"""
Validación de la Fase 7: Automatización y Producción
"""

import sys
import os
from pathlib import Path
import logging

# Añadir src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.automation.config import Config

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validar_estructura_directorios():
    """
    Valida que existan todos los directorios necesarios
    """
    logger.info("\n" + "=" * 60)
    logger.info("📁 VALIDANDO ESTRUCTURA DE DIRECTORIOS")
    logger.info("=" * 60)
    
    directorios = [
        'src/automation',
        'src/api',
        'logs',
        'backups',
        'modelos/backups',
        'resultados/reportes_diarios'
    ]
    
    todos_ok = True
    
    for directorio in directorios:
        path = Path(directorio)
        if path.exists():
            logger.info(f"✅ {directorio}")
        else:
            logger.error(f"❌ {directorio} - NO EXISTE")
            todos_ok = False
    
    return todos_ok


def validar_archivos_codigo():
    """
    Valida que existan todos los archivos de código necesarios
    """
    logger.info("\n" + "=" * 60)
    logger.info("📄 VALIDANDO ARCHIVOS DE CÓDIGO")
    logger.info("=" * 60)
    
    archivos = [
        'src/automation/config.py',
        'src/automation/data_updater.py',
        'src/automation/model_retrainer.py',
        'src/automation/daily_predictor.py',
        'src/automation/monitoring.py',
        'src/api/api_server.py',
        'setup_cron.sh',
        'setup_scheduler.ps1'
    ]
    
    todos_ok = True
    
    for archivo in archivos:
        path = Path(archivo)
        if path.exists():
            logger.info(f"✅ {archivo}")
        else:
            logger.error(f"❌ {archivo} - NO EXISTE")
            todos_ok = False
    
    return todos_ok


def validar_configuracion():
    """
    Valida la configuración del sistema
    """
    logger.info("\n" + "=" * 60)
    logger.info("⚙️  VALIDANDO CONFIGURACIÓN")
    logger.info("=" * 60)
    
    try:
        # Verificar que .env existe
        if not Path('.env').exists():
            logger.warning("⚠️  Archivo .env no encontrado")
            logger.info("   Usa .env.template como referencia")
            return False
        
        logger.info("✅ Archivo .env encontrado")
        
        # Validar configuración
        logger.info("\n📋 Configuración actual:")
        logger.info(f"   ODDS_API_KEY: {'✅ Configurada' if Config.ODDS_API_KEY else '❌ NO configurada'}")
        logger.info(f"   EMAIL_ADDRESS: {'✅ Configurada' if Config.EMAIL_ADDRESS else '❌ NO configurada'}")
        logger.info(f"   EMAIL_PASSWORD: {'✅ Configurada' if Config.EMAIL_PASSWORD else '❌ NO configurada'}")
        logger.info(f"   MODEL_PATH: {Config.MODEL_PATH}")
        logger.info(f"   DB_PATH: {Config.DB_PATH}")
        logger.info(f"   EV_THRESHOLD: {Config.EV_THRESHOLD}")
        logger.info(f"   RETRAIN_STRATEGY: {Config.RETRAIN_STRATEGY}")
        
        # Verificar que existen archivos críticos
        model_exists = Path(Config.MODEL_PATH).exists()
        logger.info(f"   Modelo existe: {'✅' if model_exists else '❌'}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error validando configuración: {e}")
        return False


def test_data_updater():
    """
    Prueba el módulo de actualización de datos
    """
    logger.info("\n" + "=" * 60)
    logger.info("🔄 PROBANDO DATA UPDATER")
    logger.info("=" * 60)
    
    try:
        from src.automation.data_updater import DataUpdater
        
        updater = DataUpdater()
        
        # Obtener estadísticas actuales
        stats = updater.obtener_estadisticas()
        
        if stats:
            logger.info("✅ DataUpdater funciona correctamente")
            logger.info(f"   Total partidos: {stats['total_partidos']}")
            logger.info(f"   Fecha más reciente: {stats['fecha_mas_reciente']}")
            return True
        else:
            logger.warning("⚠️  No se pudieron obtener estadísticas")
            return False
        
    except Exception as e:
        logger.error(f"❌ Error probando DataUpdater: {e}")
        return False


def test_model_retrainer():
    """
    Prueba el módulo de reentrenamiento
    """
    logger.info("\n" + "=" * 60)
    logger.info("🤖 PROBANDO MODEL RETRAINER")
    logger.info("=" * 60)
    
    try:
        from src.automation.model_retrainer import ModelRetrainer
        
        retrainer = ModelRetrainer()
        
        # Verificar si debería reentrenar (sin ejecutar)
        deberia = retrainer.deberia_reentrenar()
        
        logger.info(f"✅ ModelRetrainer funciona correctamente")
        logger.info(f"   ¿Debería reentrenar?: {'Sí' if deberia else 'No'}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error probando ModelRetrainer: {e}")
        return False


def test_monitoring():
    """
    Prueba el sistema de monitoreo
    """
    logger.info("\n" + "=" * 60)
    logger.info("📊 PROBANDO MONITORING SYSTEM")
    logger.info("=" * 60)
    
    try:
        from src.automation.monitoring import SystemMonitor
        
        monitor = SystemMonitor()
        
        # Ejecutar checks (sin enviar alertas)
        logger.info("\nEjecutando checks del sistema...")
        monitor.run_all_checks()
        
        logger.info("✅ Monitoring funciona correctamente")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error probando Monitoring: {e}")
        return False


def test_api_imports():
    """
    Prueba que la API se puede importar correctamente
    """
    logger.info("\n" + "=" * 60)
    logger.info("🌐 PROBANDO API SERVER (imports)")
    logger.info("=" * 60)
    
    try:
        # Solo importar, no ejecutar
        from src.api import api_server
        
        logger.info("✅ API Server se puede importar correctamente")
        logger.info("   Para ejecutar la API: python src/api/api_server.py")
        logger.info("   O en producción: gunicorn -w 4 -b 0.0.0.0:5000 src.api.api_server:app")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error importando API Server: {e}")
        return False


def validar_dependencias():
    """
    Valida que estén instaladas todas las dependencias
    """
    logger.info("\n" + "=" * 60)
    logger.info("📦 VALIDANDO DEPENDENCIAS")
    logger.info("=" * 60)
    
    dependencias = {
        'flask': 'Flask',
        'flask_cors': 'Flask-CORS',
        'dotenv': 'python-dotenv',
        'requests': 'requests',
        'pandas': 'pandas',
        'sklearn': 'scikit-learn',
        'xgboost': 'xgboost'
    }
    
    todas_ok = True
    
    for modulo, nombre in dependencias.items():
        try:
            __import__(modulo)
            logger.info(f"✅ {nombre}")
        except ImportError:
            logger.error(f"❌ {nombre} - NO INSTALADA")
            todas_ok = False
    
    if not todas_ok:
        logger.error("\n❌ Faltan dependencias. Instalar con:")
        logger.error("   pip install -r requirements.txt")
    
    return todas_ok


def main():
    """
    Ejecuta todas las validaciones
    """
    logger.info("=" * 60)
    logger.info("🚀 VALIDACIÓN FASE 7: AUTOMATIZACIÓN Y PRODUCCIÓN")
    logger.info("=" * 60)
    
    resultados = {
        'Estructura de directorios': validar_estructura_directorios(),
        'Archivos de código': validar_archivos_codigo(),
        'Dependencias': validar_dependencias(),
        'Configuración': validar_configuracion(),
        'DataUpdater': test_data_updater(),
        'ModelRetrainer': test_model_retrainer(),
        'Monitoring': test_monitoring(),
        'API Server': test_api_imports()
    }
    
    logger.info("\n" + "=" * 60)
    logger.info("📊 RESUMEN DE VALIDACIÓN")
    logger.info("=" * 60)
    
    for nombre, resultado in resultados.items():
        status = "✅ PASS" if resultado else "❌ FAIL"
        logger.info(f"{status} - {nombre}")
    
    total_ok = sum(resultados.values())
    total = len(resultados)
    
    logger.info("\n" + "=" * 60)
    logger.info(f"📈 RESULTADO: {total_ok}/{total} validaciones pasadas")
    logger.info("=" * 60)
    
    if total_ok == total:
        logger.info("\n🎉 ¡FASE 7 VALIDADA EXITOSAMENTE!")
        logger.info("\n📋 Próximos pasos:")
        logger.info("   1. Configurar .env con tus credenciales")
        logger.info("   2. Ejecutar setup_cron.sh (Linux/Mac) o setup_scheduler.ps1 (Windows)")
        logger.info("   3. Probar ejecución manual: python src/automation/daily_predictor.py")
        logger.info("   4. Iniciar API: python src/api/api_server.py")
        logger.info("   5. Dejar correr automáticamente durante 7 días")
        return True
    else:
        logger.error("\n❌ Algunas validaciones fallaron. Revisa los errores arriba.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
