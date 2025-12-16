"""
Módulo de Validación de Kelly Criterion
=======================================

Lógica refactorizada para validar Kelly Criterion.
Extraído y optimizado desde validacion_kelly_fase5.py.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.utils import print_header, print_metric, print_section
from src.betting import KellyCalculator, BankrollSimulator


def cargar_datos_backtesting():
    """Carga datos del backtesting de Fase 2"""
    print_header("CARGANDO DATOS DE BACKTESTING", "📂")
    
    resultados_path = Path('resultados/backtesting_results.csv')
    
    if not resultados_path.exists():
        print("ℹ️  Generando datos de ejemplo...")
        return generar_datos_ejemplo()
    
    df = pd.read_csv(resultados_path)
    print(f"✅ Cargados {len(df)} registros")
    
    # Filtrar apuestas con valor
    df_con_valor = df[df['prob_modelo'] * df['cuota'] > 1.0].copy()
    print(f"📊 Apuestas con valor: {len(df_con_valor)} ({len(df_con_valor)/len(df)*100:.1f}%)")
    
    return df_con_valor


def generar_datos_ejemplo():
    """Genera datos de ejemplo"""
    np.random.seed(42)
    n_apuestas = 200
    
    df = pd.DataFrame({
        'prob_modelo': np.random.uniform(0.52, 0.68, n_apuestas),
        'cuota': np.random.uniform(1.80, 2.30, n_apuestas),
        'resultado': np.random.binomial(1, 0.55, n_apuestas)
    })
    
    df = df[df['prob_modelo'] * df['cuota'] > 1.0].copy()
    print(f"✅ Generadas {len(df)} apuestas de ejemplo")
    
    return df


def analizar_distribucion(df):
    """Analiza distribución de apuestas"""
    print_header("ANÁLISIS DE DISTRIBUCIÓN", "📊")
    
    df['ev'] = df['prob_modelo'] * df['cuota'] - 1
    
    print_section("Probabilidades del modelo")
    print_metric("Media", df['prob_modelo'].mean(), "", 3)
    print_metric("Mediana", df['prob_modelo'].median(), "", 3)
    
    print_section("Expected Value (EV)")
    print_metric("Media", df['ev'].mean() * 100, "%", 2)
    print_metric("Mediana", df['ev'].median() * 100, "%", 2)
    
    print_section("Resultados")
    win_rate = df['resultado'].mean()
    print_metric("Win Rate", win_rate * 100, "%", 2)
    print_metric("Ganadas", df['resultado'].sum(), "", 0)


def ejecutar_simulaciones(df, bankroll_inicial=1000):
    """Ejecuta simulaciones de bankroll"""
    print_header("EJECUTANDO SIMULACIONES", "🎲")
    
    sim = BankrollSimulator(bankroll_inicial=bankroll_inicial)
    resultados = sim.comparar_estrategias(df, apuesta_flat=10)
    
    return resultados


def validar_kelly():
    """Función principal de validación de Kelly"""
    print_header("VALIDACIÓN KELLY CRITERION - FASE 5", "💎")
    
    # Cargar datos
    df = cargar_datos_backtesting()
    
    if len(df) == 0:
        print("❌ No hay datos para simular")
        return False
    
    # Analizar
    analizar_distribucion(df)
    
    # Simular
    resultados = ejecutar_simulaciones(df, bankroll_inicial=1000)
    
    # Conclusiones
    print_header("CONCLUSIONES", "✅")
    
    if resultados['kelly']['roi'] > resultados['flat']['roi']:
        mejora = resultados['kelly']['roi'] - resultados['flat']['roi']
        print(f"🏆 Kelly Criterion supera a Flat Betting")
        print_metric("Mejora en ROI", mejora, "%", 2)
        print(f"\n💡 Recomendación: Usar Kelly Criterion (25%)")
    else:
        print(f"⚠️  En esta muestra, Flat Betting tuvo mejor resultado")
        print(f"💡 Recomendación: Probar con más datos")
    
    print("\n✅ Validación completada!")
    
    return True
