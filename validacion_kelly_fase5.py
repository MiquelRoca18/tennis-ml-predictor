"""
Validación de Kelly Criterion con datos históricos de backtesting

Este script ejecuta simulaciones usando los resultados reales del backtesting
de Fase 2 para comparar Flat Betting vs Kelly Criterion.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Añadir src al path
sys.path.append(str(Path(__file__).parent / 'src'))

from kelly_calculator import KellyCalculator
from bankroll_simulator import BankrollSimulator


def cargar_datos_backtesting():
    """Carga los datos del backtesting de Fase 2"""
    
    print("\n" + "="*60)
    print("📂 CARGANDO DATOS DE BACKTESTING")
    print("="*60)
    
    # Intentar cargar el archivo de backtesting
    resultados_path = Path(__file__).parent / 'resultados' / 'backtesting_results.csv'
    
    if not resultados_path.exists():
        print(f"⚠️  No se encontró: {resultados_path}")
        print("ℹ️  Generando datos de ejemplo...")
        return generar_datos_ejemplo()
    
    df = pd.read_csv(resultados_path)
    print(f"✅ Cargados {len(df)} registros de backtesting")
    
    # Verificar columnas necesarias
    columnas_necesarias = ['prob_modelo', 'cuota', 'resultado']
    columnas_faltantes = [col for col in columnas_necesarias if col not in df.columns]
    
    if columnas_faltantes:
        print(f"⚠️  Columnas faltantes: {columnas_faltantes}")
        print("ℹ️  Intentando mapear columnas...")
        
        # Mapear nombres alternativos
        if 'prob_p1' in df.columns:
            df['prob_modelo'] = df['prob_p1']
        if 'cuota_p1' in df.columns:
            df['cuota'] = df['cuota_p1']
        if 'ganador' in df.columns:
            df['resultado'] = (df['ganador'] == 1).astype(int)
        elif 'winner' in df.columns:
            df['resultado'] = (df['winner'] == 1).astype(int)
    
    # Filtrar solo apuestas con valor (EV > 0)
    df_con_valor = df[df['prob_modelo'] * df['cuota'] > 1.0].copy()
    print(f"📊 Apuestas con valor positivo: {len(df_con_valor)} ({len(df_con_valor)/len(df)*100:.1f}%)")
    
    return df_con_valor


def generar_datos_ejemplo():
    """Genera datos de ejemplo si no hay backtesting disponible"""
    
    print("\n🎲 Generando datos de ejemplo...")
    
    np.random.seed(42)
    n_apuestas = 200
    
    # Simular apuestas con modelo calibrado
    # Modelo con ligera ventaja (55% win rate en apuestas con valor)
    df = pd.DataFrame({
        'prob_modelo': np.random.uniform(0.52, 0.68, n_apuestas),
        'cuota': np.random.uniform(1.80, 2.30, n_apuestas),
        'resultado': np.random.binomial(1, 0.55, n_apuestas)
    })
    
    # Filtrar solo apuestas con valor
    df = df[df['prob_modelo'] * df['cuota'] > 1.0].copy()
    
    print(f"✅ Generadas {len(df)} apuestas de ejemplo")
    
    return df


def analizar_distribucion_apuestas(df):
    """Analiza la distribución de las apuestas"""
    
    print("\n" + "="*60)
    print("📊 ANÁLISIS DE DISTRIBUCIÓN DE APUESTAS")
    print("="*60)
    
    # Calcular EV para cada apuesta
    df['ev'] = df['prob_modelo'] * df['cuota'] - 1
    
    print(f"\n📈 Probabilidades del modelo:")
    print(f"   Media:    {df['prob_modelo'].mean():.3f}")
    print(f"   Mediana:  {df['prob_modelo'].median():.3f}")
    print(f"   Min:      {df['prob_modelo'].min():.3f}")
    print(f"   Max:      {df['prob_modelo'].max():.3f}")
    
    print(f"\n💰 Cuotas:")
    print(f"   Media:    {df['cuota'].mean():.2f}")
    print(f"   Mediana:  {df['cuota'].median():.2f}")
    print(f"   Min:      {df['cuota'].min():.2f}")
    print(f"   Max:      {df['cuota'].max():.2f}")
    
    print(f"\n📊 Expected Value (EV):")
    print(f"   Media:    {df['ev'].mean()*100:+.2f}%")
    print(f"   Mediana:  {df['ev'].median()*100:+.2f}%")
    print(f"   Min:      {df['ev'].min()*100:+.2f}%")
    print(f"   Max:      {df['ev'].max()*100:+.2f}%")
    
    print(f"\n🎯 Resultados:")
    win_rate = df['resultado'].mean()
    print(f"   Win Rate: {win_rate*100:.2f}%")
    print(f"   Ganadas:  {df['resultado'].sum()}")
    print(f"   Perdidas: {len(df) - df['resultado'].sum()}")


def ejecutar_simulaciones(df, bankroll_inicial=1000):
    """Ejecuta las simulaciones de bankroll"""
    
    print("\n" + "="*60)
    print("🎲 EJECUTANDO SIMULACIONES")
    print("="*60)
    
    # Crear simulador
    sim = BankrollSimulator(bankroll_inicial=bankroll_inicial)
    
    # Comparar estrategias
    resultados = sim.comparar_estrategias(df, apuesta_flat=10)
    
    return resultados


def probar_diferentes_bankrolls(df):
    """Prueba con diferentes bankrolls iniciales"""
    
    print("\n" + "="*60)
    print("💰 ANÁLISIS CON DIFERENTES BANKROLLS")
    print("="*60)
    
    bankrolls = [500, 1000, 2000, 5000]
    
    resultados_comparacion = []
    
    for bankroll in bankrolls:
        print(f"\n{'─'*60}")
        print(f"💵 Bankroll inicial: {bankroll}€")
        print(f"{'─'*60}")
        
        sim = BankrollSimulator(bankroll_inicial=bankroll)
        
        # Solo simular, no mostrar todo
        hist_flat, df_flat = sim.simular_estrategia(df, 'flat', 10)
        hist_kelly, df_kelly = sim.simular_estrategia(df, 'kelly')
        
        # Calcular ROI
        roi_flat = ((hist_flat[-1] - bankroll) / bankroll) * 100
        roi_kelly = ((hist_kelly[-1] - bankroll) / bankroll) * 100
        
        print(f"   Flat ROI:  {roi_flat:+.2f}%")
        print(f"   Kelly ROI: {roi_kelly:+.2f}%")
        print(f"   Diferencia: {roi_kelly - roi_flat:+.2f}%")
        
        resultados_comparacion.append({
            'bankroll': bankroll,
            'roi_flat': roi_flat,
            'roi_kelly': roi_kelly,
            'diferencia': roi_kelly - roi_flat
        })
    
    return pd.DataFrame(resultados_comparacion)


def main():
    """Función principal"""
    
    print("\n" + "="*70)
    print("💎 VALIDACIÓN KELLY CRITERION - FASE 5")
    print("="*70)
    
    # 1. Cargar datos
    df = cargar_datos_backtesting()
    
    if len(df) == 0:
        print("\n❌ No hay datos para simular")
        return
    
    # 2. Analizar distribución
    analizar_distribucion_apuestas(df)
    
    # 3. Ejecutar simulación principal
    resultados = ejecutar_simulaciones(df, bankroll_inicial=1000)
    
    # 4. Probar con diferentes bankrolls
    df_comparacion = probar_diferentes_bankrolls(df)
    
    print("\n" + "="*60)
    print("📊 RESUMEN DE COMPARACIÓN POR BANKROLL")
    print("="*60)
    print(df_comparacion.to_string(index=False))
    
    # 5. Conclusiones
    print("\n" + "="*60)
    print("✅ CONCLUSIONES")
    print("="*60)
    
    if resultados['kelly']['roi'] > resultados['flat']['roi']:
        mejora = resultados['kelly']['roi'] - resultados['flat']['roi']
        print(f"\n🏆 Kelly Criterion supera a Flat Betting")
        print(f"   Mejora en ROI: {mejora:+.2f}%")
        print(f"   Ganancia adicional: {resultados['kelly']['ganancia'] - resultados['flat']['ganancia']:+.2f}€")
        print(f"\n💡 Recomendación: Usar Kelly Criterion (25%) para optimizar crecimiento")
    else:
        print(f"\n⚠️  En esta muestra, Flat Betting tuvo mejor resultado")
        print(f"   Esto puede deberse a varianza en muestras pequeñas")
        print(f"\n💡 Recomendación: Probar con más datos o ajustar fracción de Kelly")
    
    print(f"\n🔒 Límites de seguridad aplicados:")
    print(f"   - Apuesta mínima: 5€")
    print(f"   - Apuesta máxima: 5% del bankroll")
    print(f"   - Kelly fraccional: 25% (reduce varianza)")
    
    print("\n✅ Validación completada!")
    print(f"📊 Gráficos guardados en: resultados/bankroll_comparison.png")


if __name__ == "__main__":
    main()
