#!/usr/bin/env python3
"""
Script Maestro - Fase 2 Completa
=================================

Ejecuta todo el proceso de validación de calibración y backtesting:
1. Validación de calibración con reliability diagrams
2. Backtesting completo con múltiples umbrales
3. Generación de reporte consolidado
4. Documentación de resultados
"""

import subprocess
import sys
import logging
from pathlib import Path
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/fase2_ejecucion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def ejecutar_script(script_name, descripcion):
    """
    Ejecuta un script de Python y maneja errores
    
    Args:
        script_name: Nombre del script a ejecutar
        descripcion: Descripción del paso
        
    Returns:
        bool: True si exitoso, False si error
    """
    logger.info("\n" + "="*70)
    logger.info(f"🚀 EJECUTANDO: {descripcion}")
    logger.info("="*70)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=True,
            text=True
        )
        
        # Mostrar output
        if result.stdout:
            print(result.stdout)
        
        logger.info(f"✅ {descripcion} completado exitosamente")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error en {descripcion}")
        logger.error(f"Código de salida: {e.returncode}")
        if e.stdout:
            logger.error(f"Output:\n{e.stdout}")
        if e.stderr:
            logger.error(f"Error:\n{e.stderr}")
        return False
    except Exception as e:
        logger.error(f"❌ Error inesperado: {e}")
        return False


def verificar_archivos_necesarios():
    """
    Verifica que existan los archivos necesarios
    
    Returns:
        bool: True si todo OK, False si falta algo
    """
    logger.info("🔍 Verificando archivos necesarios...")
    
    archivos_necesarios = [
        "datos/processed/dataset_features_fase3_completas.csv",
        "modelos/random_forest_calibrado.pkl",
        "modelos/xgboost_calibrado.pkl",
        "modelos/gradient_boosting_calibrado.pkl",
        "modelos/logistic_regression_calibrado.pkl"
    ]
    
    faltantes = []
    for archivo in archivos_necesarios:
        if not Path(archivo).exists():
            faltantes.append(archivo)
    
    if faltantes:
        logger.error("❌ Faltan archivos necesarios:")
        for archivo in faltantes:
            logger.error(f"   - {archivo}")
        return False
    
    logger.info("✅ Todos los archivos necesarios están presentes")
    return True


def crear_directorios():
    """Crea directorios necesarios para resultados"""
    logger.info("📁 Creando directorios de resultados...")
    
    directorios = [
        "resultados/calibracion",
        "resultados/calibracion/reliability_diagrams",
        "resultados/backtesting",
        "logs"
    ]
    
    for directorio in directorios:
        Path(directorio).mkdir(parents=True, exist_ok=True)
    
    logger.info("✅ Directorios creados")


def main():
    """
    Función principal que ejecuta todo el proceso
    """
    inicio = datetime.now()
    
    logger.info("\n" + "="*70)
    logger.info("🎯 FASE 2 - EJECUCIÓN COMPLETA")
    logger.info("="*70)
    logger.info(f"Inicio: {inicio.strftime('%d/%m/%Y %H:%M:%S')}")
    
    # Verificar archivos
    if not verificar_archivos_necesarios():
        logger.error("❌ Proceso abortado: faltan archivos necesarios")
        sys.exit(1)
    
    # Crear directorios
    crear_directorios()
    
    # Lista de pasos a ejecutar
    pasos = [
        ("validacion_calibracion.py", "Validación de Calibración"),
        ("backtesting_fase2.py", "Backtesting Completo"),
        ("generar_reporte_fase2.py", "Generación de Reporte HTML")
    ]
    
    # Ejecutar cada paso
    resultados = []
    for script, descripcion in pasos:
        exito = ejecutar_script(script, descripcion)
        resultados.append((descripcion, exito))
        
        if not exito:
            logger.warning(f"⚠️  {descripcion} falló, pero continuando...")
    
    # Resumen final
    fin = datetime.now()
    duracion = fin - inicio
    
    logger.info("\n" + "="*70)
    logger.info("📊 RESUMEN DE EJECUCIÓN")
    logger.info("="*70)
    
    for descripcion, exito in resultados:
        estado = "✅ ÉXITO" if exito else "❌ ERROR"
        logger.info(f"{estado}: {descripcion}")
    
    logger.info(f"\n⏱️  Duración total: {duracion}")
    logger.info(f"🏁 Finalizado: {fin.strftime('%d/%m/%Y %H:%M:%S')}")
    
    # Verificar si todo fue exitoso
    todos_exitosos = all(exito for _, exito in resultados)
    
    if todos_exitosos:
        logger.info("\n" + "="*70)
        logger.info("🎉 ¡FASE 2 COMPLETADA AL 100%!")
        logger.info("="*70)
        logger.info("\n📁 Resultados disponibles en:")
        logger.info("   - resultados/calibracion/")
        logger.info("   - resultados/backtesting/")
        logger.info("   - resultados/REPORTE_FASE_2.html")
        logger.info("\n🚀 Listo para avanzar a Fase 3!")
        return 0
    else:
        logger.warning("\n⚠️  Algunos pasos fallaron. Revisa los logs.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
