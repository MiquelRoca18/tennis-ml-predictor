"""
Script de Demostración - Sistema de Tracking Fase 4
Genera datos de ejemplo y muestra todas las funcionalidades
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Añadir path
sys.path.append(str(Path(__file__).parent))

from src.tracking.database_setup import TennisDatabase
from src.tracking.dashboard_generator import DashboardGenerator
from src.tracking.analisis_categorias import AnalisisCategorias


def generar_datos_ejemplo():
    """
    Genera datos de ejemplo para demostrar el sistema
    """
    
    print("=" * 60)
    print("🎾 GENERANDO DATOS DE EJEMPLO")
    print("=" * 60)
    
    # Crear base de datos
    db = TennisDatabase("apuestas_tracker_demo.db")
    
    # Jugadores de ejemplo
    jugadores = [
        ('Carlos Alcaraz', 3),
        ('Jannik Sinner', 1),
        ('Novak Djokovic', 7),
        ('Daniil Medvedev', 5),
        ('Alexander Zverev', 4),
        ('Andrey Rublev', 8),
        ('Stefanos Tsitsipas', 11),
        ('Taylor Fritz', 6),
        ('Casper Ruud', 9),
        ('Alex de Minaur', 10)
    ]
    
    superficies = ['Hard', 'Clay', 'Grass']
    torneos = ['ATP Finals', 'Masters 1000', 'ATP 500', 'ATP 250']
    
    # Generar 50 predicciones de ejemplo
    np.random.seed(42)
    
    for i in range(50):
        # Seleccionar jugadores aleatorios
        jugador = jugadores[np.random.randint(0, len(jugadores))]
        oponente = jugadores[np.random.randint(0, len(jugadores))]
        
        while oponente == jugador:
            oponente = jugadores[np.random.randint(0, len(jugadores))]
        
        # Generar probabilidad basada en rankings
        rank_diff = jugador[1] - oponente[1]
        prob_base = 0.5 + (rank_diff * 0.03)
        prob_modelo = np.clip(prob_base + np.random.normal(0, 0.1), 0.2, 0.8)
        
        # Generar cuota (inversa de probabilidad con margen de bookmaker)
        cuota = (1 / prob_modelo) * np.random.uniform(0.95, 1.05)
        cuota = round(cuota, 2)
        
        # Calcular EV
        ev = (prob_modelo * cuota) - 1
        
        # Decisión de apostar
        decision = 'APOSTAR' if ev > 0.03 else 'NO_APOSTAR'
        apuesta_cantidad = 10.0 if decision == 'APOSTAR' else None
        
        # Simular resultado (con sesgo hacia la probabilidad del modelo)
        resultado_real = 1 if np.random.random() < prob_modelo else 0
        
        # Calcular ganancia
        if decision == 'APOSTAR':
            if resultado_real == 1:
                ganancia = apuesta_cantidad * (cuota - 1)
            else:
                ganancia = -apuesta_cantidad
        else:
            ganancia = None
        
        # Fecha
        fecha_partido = (datetime.now() - timedelta(days=50-i)).strftime('%Y-%m-%d')
        
        # Crear predicción
        prediccion = {
            'fecha_partido': fecha_partido,
            'jugador_nombre': jugador[0],
            'jugador_rank': jugador[1],
            'oponente_nombre': oponente[0],
            'oponente_rank': oponente[1],
            'superficie': np.random.choice(superficies),
            'torneo': np.random.choice(torneos),
            'prob_modelo': prob_modelo,
            'prob_modelo_calibrada': prob_modelo,
            'cuota': cuota,
            'bookmaker': 'Bet365',
            'ev': ev,
            'decision': decision,
            'apuesta_cantidad': apuesta_cantidad,
            'resultado_real': resultado_real,
            'ganancia': ganancia,
            'modelo_usado': 'XGBoost Optimizado Demo',
            'version_modelo': 'v3.0'
        }
        
        # Insertar en DB
        pred_id = db.insertar_prediccion(prediccion)
        
        # Actualizar resultado
        if ganancia is not None:
            db.actualizar_resultado(pred_id, resultado_real, ganancia)
    
    print(f"\n✅ {50} predicciones de ejemplo generadas")
    
    db.desconectar()
    
    return "apuestas_tracker_demo.db"


def mostrar_metricas(db_path):
    """
    Muestra métricas del sistema
    """
    
    db = TennisDatabase(db_path)
    metricas = db.calcular_metricas()
    
    print("\n" + "=" * 60)
    print("📊 MÉTRICAS DEL SISTEMA")
    print("=" * 60)
    
    print(f"\n💰 FINANCIERO:")
    print(f"   Total apostado:   {metricas['total_apostado']:.2f}€")
    print(f"   Total retornado:  {metricas['total_ganado']:.2f}€")
    print(f"   Ganancia neta:    {metricas['ganancia_neta']:+.2f}€")
    print(f"   ROI:              {metricas['roi']:+.2f}%")
    
    print(f"\n🎯 PERFORMANCE:")
    print(f"   Apuestas totales: {metricas['total_apuestas']}")
    print(f"   Ganadas:          {metricas['apuestas_ganadas']}")
    print(f"   Perdidas:         {metricas['apuestas_perdidas']}")
    print(f"   Win Rate:         {metricas['win_rate']:.1f}%")
    
    print(f"\n📈 MODELO:")
    print(f"   EV promedio:      +{metricas['ev_promedio']:.2f}%")
    
    db.desconectar()


def generar_dashboard_demo(db_path):
    """
    Genera dashboard de demostración
    """
    
    print("\n" + "=" * 60)
    print("📊 GENERANDO DASHBOARD")
    print("=" * 60)
    
    generator = DashboardGenerator(db_path)
    generator.generar_dashboard_completo("resultados/dashboard_demo.html")
    
    print("\n✅ Dashboard generado en: resultados/dashboard_demo.html")


def generar_analisis_demo(db_path):
    """
    Genera análisis por categorías
    """
    
    analisis = AnalisisCategorias(db_path)
    analisis.generar_reporte_completo()


def main():
    """
    Función principal de demostración
    """
    
    print("\n" + "=" * 60)
    print("🎾 DEMOSTRACIÓN SISTEMA DE TRACKING - FASE 4")
    print("=" * 60)
    
    # 1. Generar datos de ejemplo
    db_path = generar_datos_ejemplo()
    
    # 2. Mostrar métricas
    mostrar_metricas(db_path)
    
    # 3. Generar análisis por categorías
    generar_analisis_demo(db_path)
    
    # 4. Generar dashboard
    generar_dashboard_demo(db_path)
    
    print("\n" + "=" * 60)
    print("✅ DEMOSTRACIÓN COMPLETADA")
    print("=" * 60)
    print("\n📋 Archivos generados:")
    print("   - apuestas_tracker_demo.db (Base de datos SQLite)")
    print("   - resultados/dashboard_demo.html (Dashboard interactivo)")
    print("\n💡 Abre resultados/dashboard_demo.html en tu navegador para ver el dashboard")


if __name__ == "__main__":
    main()
