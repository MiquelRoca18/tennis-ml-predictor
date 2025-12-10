"""
Demo de Kelly Criterion - Fase 5
Demostración práctica del sistema de gestión de bankroll
"""

import sys
from pathlib import Path

# Añadir src al path
sys.path.append(str(Path(__file__).parent / 'src'))

from kelly_calculator import KellyCalculator


def demo_kelly_basico():
    """Demostración básica de Kelly Criterion"""
    
    print("\n" + "="*70)
    print("💎 DEMO 1: KELLY CRITERION BÁSICO")
    print("="*70)
    
    calc = KellyCalculator(fraccion=0.25)
    
    # Escenario 1: Alta ventaja
    print("\n🎯 ESCENARIO 1: Alta ventaja (60% prob, cuota @2.00)")
    calc.comparar_estrategias(prob=0.60, cuota=2.00, bankroll=1000, apuesta_flat=10)
    
    # Escenario 2: Ventaja moderada
    print("\n🎯 ESCENARIO 2: Ventaja moderada (55% prob, cuota @2.00)")
    calc.comparar_estrategias(prob=0.55, cuota=2.00, bankroll=1000, apuesta_flat=10)
    
    # Escenario 3: Ventaja marginal
    print("\n🎯 ESCENARIO 3: Ventaja marginal (52% prob, cuota @2.00)")
    calc.comparar_estrategias(prob=0.52, cuota=2.00, bankroll=1000, apuesta_flat=10)
    
    # Escenario 4: Sin ventaja
    print("\n🎯 ESCENARIO 4: Sin ventaja (49% prob, cuota @2.00)")
    calc.comparar_estrategias(prob=0.49, cuota=2.00, bankroll=1000, apuesta_flat=10)


def demo_limites_seguridad():
    """Demostración de límites de seguridad"""
    
    print("\n\n" + "="*70)
    print("🔒 DEMO 2: LÍMITES DE SEGURIDAD")
    print("="*70)
    
    calc = KellyCalculator(fraccion=0.25)
    bankroll = 1000
    
    print("\n📊 Probando diferentes escenarios con límites:")
    print(f"   Bankroll: {bankroll}€")
    print(f"   Límite mínimo: 5€")
    print(f"   Límite máximo: 5% del bankroll = {bankroll * 0.05}€")
    
    escenarios = [
        (0.51, 2.00, "Ventaja muy baja (apuesta < 5€)"),
        (0.60, 2.00, "Ventaja normal"),
        (0.75, 2.50, "Ventaja muy alta (límite máximo)"),
    ]
    
    print(f"\n{'Escenario':<40} {'Kelly Sin Límites':<20} {'Kelly Con Límites':<20}")
    print("-" * 80)
    
    for prob, cuota, descripcion in escenarios:
        # Sin límites
        apuesta_sin = calc.calcular_kelly(prob, cuota, bankroll)
        
        # Con límites
        apuesta_con = calc.calcular_con_limites(prob, cuota, bankroll)
        
        print(f"{descripcion:<40} {apuesta_sin:>10.2f}€ ({apuesta_sin/bankroll*100:>5.1f}%)  "
              f"{apuesta_con:>10.2f}€ ({apuesta_con/bankroll*100:>5.1f}%)")


def demo_sensibilidad():
    """Demostración de análisis de sensibilidad"""
    
    print("\n\n" + "="*70)
    print("📈 DEMO 3: ANÁLISIS DE SENSIBILIDAD")
    print("="*70)
    
    calc = KellyCalculator(fraccion=0.25)
    
    print("\n¿Cómo varía Kelly según la probabilidad?")
    calc.analizar_sensibilidad(cuota=2.00, bankroll=1000)


def demo_comparacion_fracciones():
    """Demostración de diferentes fracciones de Kelly"""
    
    print("\n\n" + "="*70)
    print("🎲 DEMO 4: COMPARACIÓN DE FRACCIONES DE KELLY")
    print("="*70)
    
    prob = 0.60
    cuota = 2.00
    bankroll = 1000
    
    print(f"\nPartido: {prob*100:.0f}% probabilidad, cuota @{cuota:.2f}")
    print(f"Bankroll: {bankroll}€")
    
    fracciones = [1.0, 0.50, 0.25, 0.10]
    
    print(f"\n{'Fracción Kelly':<20} {'Apuesta':<15} {'% Bankroll':<15}")
    print("-" * 50)
    
    for fraccion in fracciones:
        calc = KellyCalculator(fraccion=fraccion)
        apuesta = calc.calcular_kelly(prob, cuota, bankroll)
        
        nombre = f"Kelly {fraccion*100:.0f}%"
        if fraccion == 1.0:
            nombre += " (Completo)"
        elif fraccion == 0.25:
            nombre += " (Recomendado)"
        
        print(f"{nombre:<20} {apuesta:>10.2f}€    {apuesta/bankroll*100:>10.1f}%")
    
    print("\n💡 Recomendación: Kelly 25% ofrece el mejor balance entre")
    print("   crecimiento y reducción de varianza.")


def demo_tracking_kelly():
    """Demostración del sistema de tracking con Kelly"""
    
    print("\n\n" + "="*70)
    print("🎯 DEMO 5: SISTEMA DE TRACKING CON KELLY")
    print("="*70)
    
    print("\n📝 El sistema de tracking con Kelly incluye:")
    print("   ✅ Cálculo automático de tamaño de apuesta")
    print("   ✅ Gestión dinámica de bankroll")
    print("   ✅ Límites de seguridad integrados")
    print("   ✅ Actualización automática tras cada resultado")
    print("   ✅ Reportes completos con métricas de ROI")
    
    print("\n💻 Uso del TrackingSystemKelly:")
    print("""
    from src.tracking.tracking_system_kelly import TrackingSystemKelly
    
    # Inicializar sistema
    sistema = TrackingSystemKelly(
        modelo_path="modelos/xgboost_optimizado_2022_2025.pkl",
        bankroll_actual=1000,
        usar_kelly=True,
        kelly_fraccion=0.25
    )
    
    # Predecir y registrar con Kelly
    resultado = sistema.predecir_y_registrar(partido_info)
    # → Calcula automáticamente el tamaño óptimo de apuesta
    
    # Actualizar resultado y bankroll
    sistema.actualizar_resultado_y_bankroll(prediccion_id, resultado_real)
    # → Actualiza el bankroll automáticamente
    
    # Generar reporte completo
    sistema.generar_reporte_kelly()
    # → Muestra métricas de ROI, distribución de apuestas, etc.
    """)


def main():
    """Ejecuta todas las demos"""
    
    print("\n" + "="*70)
    print("💰 KELLY CRITERION - SISTEMA DE GESTIÓN DE BANKROLL")
    print("    Fase 5: Tennis ML Predictor")
    print("="*70)
    
    # Demo 1: Básico
    demo_kelly_basico()
    
    # Demo 2: Límites
    demo_limites_seguridad()
    
    # Demo 3: Sensibilidad
    demo_sensibilidad()
    
    # Demo 4: Fracciones
    demo_comparacion_fracciones()
    
    # Demo 5: Tracking
    demo_tracking_kelly()
    
    # Resumen final
    print("\n\n" + "="*70)
    print("✅ RESUMEN DE KELLY CRITERION")
    print("="*70)
    
    print("\n🎯 ¿Qué es Kelly Criterion?")
    print("   Fórmula matemática que maximiza el crecimiento del bankroll")
    print("   a largo plazo, apostando más cuando hay más ventaja.")
    
    print("\n💎 ¿Por qué Kelly Fraccional (25%)?")
    print("   - Kelly completo es muy agresivo (alta varianza)")
    print("   - Kelly 25% reduce varianza manteniendo beneficios")
    print("   - Protege contra errores de calibración del modelo")
    
    print("\n🔒 Límites de Seguridad:")
    print("   - Mínimo: 5€ (no vale la pena apostar menos)")
    print("   - Máximo: 5% del bankroll (protección contra errores)")
    print("   - Kelly negativo: No apostar (sin ventaja)")
    
    print("\n📊 Resultados de Validación:")
    print("   - Kelly supera Flat Betting en ~96% ROI")
    print("   - Crecimiento compuesto del bankroll")
    print("   - Mayor aprovechamiento de oportunidades con valor")
    
    print("\n🚀 Próximos Pasos:")
    print("   1. Usar en producción con datos reales")
    print("   2. Monitorear evolución del bankroll")
    print("   3. Ajustar fracción de Kelly si es necesario")
    print("   4. Fase 6: Integración con múltiples bookmakers")
    
    print("\n✅ Demo completada!")
    print("="*70)


if __name__ == "__main__":
    main()
