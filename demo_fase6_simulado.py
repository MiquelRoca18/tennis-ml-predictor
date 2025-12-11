"""
Demo con Datos Simulados - Fase 6

Demuestra el funcionamiento completo del sistema con datos de ejemplo
ya que actualmente no hay torneos ATP activos.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).parent))

from src.bookmakers.odds_comparator import OddsComparator
from src.bookmakers.alert_system import AlertSystem
from src.kelly_calculator import KellyCalculator


def main():
    """
    Demo con datos simulados
    """
    print("\n" + "="*70)
    print(" "*15 + "🎾 TENNIS ML PREDICTOR - FASE 6")
    print(" "*10 + "DEMO CON DATOS SIMULADOS (Sin partidos reales)")
    print("="*70)
    
    # Datos simulados de cuotas
    print("\n📋 PASO 1: Datos Simulados de Cuotas")
    print("-" * 70)
    
    datos_cuotas = [
        # Partido 1: Alcaraz vs Sinner
        {'bookmaker': 'Bet365', 'jugador1': 'Carlos Alcaraz', 'jugador2': 'Jannik Sinner',
         'cuota_jugador1': 2.00, 'cuota_jugador2': 1.85, 'fecha': pd.Timestamp.now()},
        {'bookmaker': 'Pinnacle', 'jugador1': 'Carlos Alcaraz', 'jugador2': 'Jannik Sinner',
         'cuota_jugador1': 2.10, 'cuota_jugador2': 1.80, 'fecha': pd.Timestamp.now()},
        {'bookmaker': 'Betfair', 'jugador1': 'Carlos Alcaraz', 'jugador2': 'Jannik Sinner',
         'cuota_jugador1': 2.05, 'cuota_jugador2': 1.83, 'fecha': pd.Timestamp.now()},
        {'bookmaker': 'Unibet', 'jugador1': 'Carlos Alcaraz', 'jugador2': 'Jannik Sinner',
         'cuota_jugador1': 2.08, 'cuota_jugador2': 1.82, 'fecha': pd.Timestamp.now()},
        
        # Partido 2: Djokovic vs Medvedev
        {'bookmaker': 'Bet365', 'jugador1': 'Novak Djokovic', 'jugador2': 'Daniil Medvedev',
         'cuota_jugador1': 1.75, 'cuota_jugador2': 2.15, 'fecha': pd.Timestamp.now()},
        {'bookmaker': 'Pinnacle', 'jugador1': 'Novak Djokovic', 'jugador2': 'Daniil Medvedev',
         'cuota_jugador1': 1.78, 'cuota_jugador2': 2.20, 'fecha': pd.Timestamp.now()},
        {'bookmaker': 'Betfair', 'jugador1': 'Novak Djokovic', 'jugador2': 'Daniil Medvedev',
         'cuota_jugador1': 1.77, 'cuota_jugador2': 2.18, 'fecha': pd.Timestamp.now()},
        
        # Partido 3: Rune vs Rublev
        {'bookmaker': 'Bet365', 'jugador1': 'Holger Rune', 'jugador2': 'Andrey Rublev',
         'cuota_jugador1': 2.30, 'cuota_jugador2': 1.65, 'fecha': pd.Timestamp.now()},
        {'bookmaker': 'Pinnacle', 'jugador1': 'Holger Rune', 'jugador2': 'Andrey Rublev',
         'cuota_jugador1': 2.40, 'cuota_jugador2': 1.62, 'fecha': pd.Timestamp.now()},
        {'bookmaker': 'Betfair', 'jugador1': 'Holger Rune', 'jugador2': 'Andrey Rublev',
         'cuota_jugador1': 2.35, 'cuota_jugador2': 1.64, 'fecha': pd.Timestamp.now()},
    ]
    
    df_cuotas = pd.DataFrame(datos_cuotas)
    
    print(f"\n✅ {len(df_cuotas)} cuotas simuladas de {df_cuotas['bookmaker'].nunique()} bookmakers")
    print(f"   Partidos: {df_cuotas.groupby(['jugador1', 'jugador2']).ngroups}")
    
    # Crear comparador
    print("\n📋 PASO 2: Comparación de Cuotas")
    print("-" * 70)
    
    comparador = OddsComparator(df_cuotas)
    
    # Analizar cada partido
    partidos_analisis = [
        ('Carlos Alcaraz', 'Jannik Sinner', 0.48),  # 48% Alcaraz
        ('Novak Djokovic', 'Daniil Medvedev', 0.62),  # 62% Djokovic
        ('Holger Rune', 'Andrey Rublev', 0.45),  # 45% Rune
    ]
    
    oportunidades = []
    
    for jugador1, jugador2, prob_j1 in partidos_analisis:
        analisis = comparador.analizar_partido_completo(jugador1, jugador2, prob_j1)
        
        if analisis:
            print(f"\n🎾 {jugador1} vs {jugador2}")
            print(f"   Probabilidad modelo: {jugador1} {prob_j1*100:.0f}% | {jugador2} {(1-prob_j1)*100:.0f}%")
            print(f"   Mejor apuesta: {analisis['jugador']}")
            print(f"   Mejor cuota: @{analisis['cuota']:.2f} ({analisis['bookmaker']})")
            print(f"   Cuota promedio: @{analisis['cuota_promedio']:.2f}")
            print(f"   EV: {analisis['ev']*100:+.2f}%")
            print(f"   Saving vs promedio: {analisis['saving_vs_promedio']*100:+.2f}%")
            
            # Calcular apuesta con Kelly
            kelly = KellyCalculator(fraccion=0.25)
            apuesta = kelly.calcular_con_limites(
                prob=analisis['prob'],
                cuota=analisis['cuota'],
                bankroll=1000
            )
            
            analisis['apuesta_cantidad'] = apuesta
            analisis['pct_bankroll'] = (apuesta / 1000) * 100
            
            # Calcular savings
            savings = comparador.calcular_savings(analisis, apuesta)
            if savings:
                analisis['savings'] = savings
                print(f"   Apuesta Kelly (25%): {apuesta:.2f}€ ({analisis['pct_bankroll']:.1f}% bankroll)")
                print(f"   Ahorro por line shopping: {savings['ahorro_euros']:+.2f}€")
            
            if analisis['ev'] > 0.03:  # EV > 3%
                oportunidades.append(analisis)
    
    # Sistema de alertas
    print("\n📋 PASO 3: Sistema de Alertas")
    print("-" * 70)
    
    alert = AlertSystem(email_enabled=False)  # Solo consola para demo
    detectadas = alert.verificar_oportunidades(oportunidades, umbral_ev=0.05)
    
    # Resumen final
    print("\n📋 PASO 4: Resumen de Line Shopping")
    print("-" * 70)
    
    if len(oportunidades) > 0:
        total_apuesta = sum(op['apuesta_cantidad'] for op in oportunidades)
        total_savings = sum(op.get('savings', {}).get('ahorro_euros', 0) for op in oportunidades)
        ev_promedio = sum(op['ev'] for op in oportunidades) / len(oportunidades)
        
        print(f"\n💰 BENEFICIO DEL LINE SHOPPING:")
        print(f"   Total oportunidades: {len(oportunidades)}")
        print(f"   Total a apostar: {total_apuesta:.2f}€")
        print(f"   EV promedio: {ev_promedio*100:+.2f}%")
        print(f"   Ahorro total por line shopping: {total_savings:+.2f}€")
        print(f"   Ahorro promedio por apuesta: {total_savings/len(oportunidades):+.2f}€")
        print(f"   % Bankroll usado: {(total_apuesta/1000)*100:.1f}%")
        
        print(f"\n💡 CONCLUSIÓN:")
        print(f"   Al comparar cuotas de {df_cuotas['bookmaker'].nunique()} bookmakers,")
        print(f"   mejoras tu EV en promedio {total_savings/len(oportunidades):+.2f}€ por apuesta.")
        print(f"   ¡En 100 apuestas similares ahorrarías ~{(total_savings/len(oportunidades))*100:.0f}€!")
    
    # Comparación visual
    print("\n📋 PASO 5: Comparación Visual de Cuotas")
    print("-" * 70)
    
    comparador.generar_reporte_comparacion()
    
    print("\n" + "="*70)
    print("✅ DEMO COMPLETADO")
    print("="*70)
    
    print("\n💡 NOTA:")
    print("   Este demo usa datos simulados porque actualmente no hay")
    print("   torneos ATP activos (temporada baja en diciembre).")
    print("\n   Cuando haya torneos activos (enero-noviembre), el sistema")
    print("   obtendrá cuotas reales de The Odds API automáticamente.")
    
    print("\n📚 PRÓXIMOS PASOS:")
    print("   1. El sistema está configurado y listo")
    print("   2. Ejecuta el demo regularmente cuando haya torneos")
    print("   3. El sistema te alertará automáticamente de oportunidades")
    print("   4. Usa el caché para optimizar el uso de la API")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error en demo: {e}")
        import traceback
        traceback.print_exc()
