"""
Módulo de Demos - Tracking System
=================================

Demo refactorizada del sistema de tracking.
"""

from src.utils import print_header, print_metric
from src.tracking import TrackingSystem
from src.config import Config


def demo_tracking():
    """Demo del sistema de tracking"""
    print_header("DEMO - SISTEMA DE TRACKING", "📊")
    
    try:
        # Crear sistema de tracking
        sistema = TrackingSystem(
            modelo_path=Config.MODEL_PATH,
            db_path=Config.DB_PATH
        )
        
        print("\n✅ Sistema de tracking inicializado")
        print(f"   Modelo: {Config.MODEL_PATH}")
        print(f"   Database: {Config.DB_PATH}")
        
        # Generar reporte
        print_header("GENERANDO REPORTE", "📊")
        sistema.generar_reporte()
        
        print("\n✅ Demo de tracking completada!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
