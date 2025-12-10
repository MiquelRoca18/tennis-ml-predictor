"""
Script de Actualización Automática
Ejecuta todas las actualizaciones del sistema de tracking automáticamente
"""

import sys
from pathlib import Path

# Añadir path
sys.path.append(str(Path(__file__).parent))

from src.tracking.database_setup import TennisDatabase
from src.tracking.dashboard_generator import DashboardGenerator
from src.tracking.analisis_categorias import AnalisisCategorias
from src.tracking.reporte_periodico import ReportePeriodico
from datetime import datetime


def actualizar_sistema_completo(db_path="apuestas_tracker.db"):
    """
    Actualiza todo el sistema de tracking automáticamente
    """
    
    print("\n" + "=" * 70)
    print("🔄 ACTUALIZACIÓN AUTOMÁTICA DEL SISTEMA DE TRACKING")
    print("=" * 70)
    print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. Verificar base de datos
    print("1️⃣  Verificando base de datos...")
    db = TennisDatabase(db_path)
    metricas_generales = db.calcular_metricas()
    
    if metricas_generales['total_apuestas'] == 0:
        print("⚠️  No hay apuestas registradas aún")
        print("💡 Usa el sistema de tracking para registrar predicciones primero")
        return
    
    print(f"   ✅ {metricas_generales['total_apuestas']} apuestas encontradas")
    
    # 2. Generar dashboard principal
    print("\n2️⃣  Generando dashboard principal...")
    dashboard = DashboardGenerator(db_path)
    dashboard.generar_dashboard_completo("resultados/dashboard.html")
    
    # 3. Generar análisis por categorías
    print("\n3️⃣  Generando análisis por categorías...")
    analisis = AnalisisCategorias(db_path)
    analisis.generar_reporte_completo()
    
    # 4. Generar reportes periódicos
    print("\n4️⃣  Generando reportes periódicos...")
    reporte = ReportePeriodico(db_path)
    
    # Reporte semanal
    metricas_semanal = reporte.generar_reporte_semanal()
    
    # Reporte mensual
    metricas_mensual = reporte.generar_reporte_mensual()
    
    # Comparar periodos
    print("\n5️⃣  Comparando periodos...")
    reporte.comparar_periodos(dias_actual=7, dias_anterior=7)
    
    # 6. Resumen final
    print("\n" + "=" * 70)
    print("✅ ACTUALIZACIÓN COMPLETADA")
    print("=" * 70)
    print("\n📊 ARCHIVOS GENERADOS:")
    print("   - resultados/dashboard.html (Dashboard principal)")
    print("   - resultados/reportes/reporte_semanal_*.html")
    print("   - resultados/reportes/reporte_mensual_*.html")
    print("\n💡 PRÓXIMOS PASOS:")
    print("   1. Abre resultados/dashboard.html en tu navegador")
    print("   2. Revisa los reportes periódicos en resultados/reportes/")
    print("   3. Analiza las tendencias y ajusta tu estrategia")
    
    # Exportar métricas
    print("\n6️⃣  Exportando datos...")
    db.exportar_a_csv("resultados/export_predicciones.csv")
    
    print("\n" + "=" * 70)
    print("🎉 ¡Sistema actualizado exitosamente!")
    print("=" * 70)


if __name__ == "__main__":
    # Verificar si se pasó un path de DB personalizado
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        db_path = "apuestas_tracker.db"
    
    actualizar_sistema_completo(db_path)
