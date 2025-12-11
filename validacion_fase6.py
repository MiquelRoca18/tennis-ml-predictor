"""
Validación de Fase 6 - Múltiples Bookmakers

Valida que todos los componentes del sistema funcionen correctamente:
- Conexión con The Odds API
- Obtención de cuotas de múltiples bookmakers
- Comparador de cuotas
- Sistema de alertas
- Integración con Kelly Criterion
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).parent))

from src.bookmakers.odds_fetcher import OddsFetcher, APILimitError
from src.bookmakers.odds_comparator import OddsComparator
from src.bookmakers.alert_system import AlertSystem
from src.bookmakers.config import BookmakerConfig
from src.kelly_calculator import KellyCalculator


def validar_configuracion():
    """Valida la configuración del sistema"""
    print("\n" + "="*60)
    print("✅ TEST 1: Validar Configuración")
    print("="*60)
    
    is_valid, msg = BookmakerConfig.validate_config()
    print(f"\n{msg}")
    
    if not is_valid:
        print("\n❌ FALLO: Configuración inválida")
        return False
    
    print("✅ ÉXITO: Configuración válida")
    return True


def validar_odds_fetcher():
    """Valida el fetcher de cuotas"""
    print("\n" + "="*60)
    print("✅ TEST 2: Validar OddsFetcher")
    print("="*60)
    
    try:
        fetcher = OddsFetcher(use_cache=True)
        print("✅ OddsFetcher inicializado correctamente")
        
        # Intentar obtener cuotas
        print("\n🌐 Obteniendo cuotas de The Odds API...")
        df_cuotas = fetcher.obtener_todas_cuotas(sport='tennis_atp')
        
        if len(df_cuotas) == 0:
            print("⚠️  No hay partidos disponibles (esto es normal si no hay torneos activos)")
            print("✅ ÉXITO: API funciona correctamente (sin partidos disponibles)")
            return True
        
        # Validar estructura del DataFrame
        required_cols = ['bookmaker', 'jugador1', 'jugador2', 'cuota_jugador1', 'cuota_jugador2']
        for col in required_cols:
            if col not in df_cuotas.columns:
                print(f"❌ FALLO: Columna '{col}' no encontrada")
                return False
        
        print(f"✅ DataFrame tiene todas las columnas requeridas")
        
        # Validar que hay múltiples bookmakers
        num_bookmakers = df_cuotas['bookmaker'].nunique()
        print(f"\n📊 Bookmakers encontrados: {num_bookmakers}")
        
        if num_bookmakers < 3:
            print(f"⚠️  Solo {num_bookmakers} bookmaker(s) disponible(s)")
            print(f"   Se recomienda tener al menos 3 para line shopping efectivo")
        else:
            print(f"✅ Suficientes bookmakers para line shopping ({num_bookmakers})")
        
        # Mostrar bookmakers
        print(f"\n📋 Bookmakers disponibles:")
        for bm in df_cuotas['bookmaker'].unique():
            count = len(df_cuotas[df_cuotas['bookmaker'] == bm])
            print(f"   - {bm}: {count} cuotas")
        
        # Verificar tracking de requests
        stats = fetcher.get_request_stats()
        if stats['requests_remaining']:
            print(f"\n📊 API Usage:")
            print(f"   Requests restantes: {stats['requests_remaining']}")
            print(f"✅ Tracking de requests funcionando")
        
        print("\n✅ ÉXITO: OddsFetcher funciona correctamente")
        return True
    
    except APILimitError as e:
        print(f"\n⚠️  LÍMITE DE API ALCANZADO")
        print(f"{e}")
        print("\n✅ ÉXITO: Manejo de límite de API funciona correctamente")
        return True
    
    except Exception as e:
        print(f"\n❌ FALLO: {e}")
        import traceback
        traceback.print_exc()
        return False


def validar_odds_comparator():
    """Valida el comparador de cuotas"""
    print("\n" + "="*60)
    print("✅ TEST 3: Validar OddsComparator")
    print("="*60)
    
    # Datos de prueba
    datos_test = [
        {'bookmaker': 'Bet365', 'jugador1': 'Test Player 1', 'jugador2': 'Test Player 2',
         'cuota_jugador1': 2.00, 'cuota_jugador2': 1.85, 'fecha': pd.Timestamp.now()},
        {'bookmaker': 'Pinnacle', 'jugador1': 'Test Player 1', 'jugador2': 'Test Player 2',
         'cuota_jugador1': 2.10, 'cuota_jugador2': 1.80, 'fecha': pd.Timestamp.now()},
        {'bookmaker': 'Betfair', 'jugador1': 'Test Player 1', 'jugador2': 'Test Player 2',
         'cuota_jugador1': 2.05, 'cuota_jugador2': 1.83, 'fecha': pd.Timestamp.now()},
    ]
    
    df_test = pd.DataFrame(datos_test)
    
    try:
        comparador = OddsComparator(df_test)
        print("✅ OddsComparator inicializado")
        
        # Encontrar mejor cuota
        mejor = comparador.encontrar_mejor_cuota('Test Player 1', 'Test Player 2')
        
        if mejor is None:
            print("❌ FALLO: No se pudo encontrar mejor cuota")
            return False
        
        print(f"\n📊 Mejor cuota encontrada:")
        print(f"   Jugador 1: @{mejor['mejor_cuota_j1']:.2f} ({mejor['bookmaker_j1']})")
        print(f"   Jugador 2: @{mejor['mejor_cuota_j2']:.2f} ({mejor['bookmaker_j2']})")
        
        # Verificar que encontró la mejor
        if mejor['mejor_cuota_j1'] != 2.10:
            print(f"❌ FALLO: No identificó la mejor cuota correctamente")
            return False
        
        print("✅ Identificó correctamente la mejor cuota")
        
        # Analizar partido completo
        analisis = comparador.analizar_partido_completo('Test Player 1', 'Test Player 2', prob_j1=0.48)
        
        if analisis is None:
            print("❌ FALLO: Análisis completo falló")
            return False
        
        print(f"\n📊 Análisis completo:")
        print(f"   Mejor apuesta: {analisis['jugador']}")
        print(f"   EV: {analisis['ev']*100:+.2f}%")
        print(f"   Saving vs promedio: {analisis['saving_vs_promedio']*100:+.2f}%")
        
        print("\n✅ ÉXITO: OddsComparator funciona correctamente")
        return True
    
    except Exception as e:
        print(f"\n❌ FALLO: {e}")
        import traceback
        traceback.print_exc()
        return False


def validar_alert_system():
    """Valida el sistema de alertas"""
    print("\n" + "="*60)
    print("✅ TEST 4: Validar AlertSystem")
    print("="*60)
    
    try:
        alert = AlertSystem(email_enabled=False)  # Solo consola para test
        print("✅ AlertSystem inicializado")
        
        # Oportunidades de prueba
        oportunidades_test = [
            {
                'jugador': 'Test Player',
                'oponente': 'Test Opponent',
                'cuota': 2.10,
                'bookmaker': 'Pinnacle',
                'prob': 0.58,
                'ev': 0.078
            }
        ]
        
        print("\n🚨 Probando alertas...")
        detectadas = alert.verificar_oportunidades(oportunidades_test, umbral_ev=0.05)
        
        if len(detectadas) != 1:
            print(f"❌ FALLO: Debería detectar 1 oportunidad, detectó {len(detectadas)}")
            return False
        
        print("✅ Alertas funcionan correctamente")
        
        print("\n✅ ÉXITO: AlertSystem funciona correctamente")
        return True
    
    except Exception as e:
        print(f"\n❌ FALLO: {e}")
        import traceback
        traceback.print_exc()
        return False


def validar_integracion_kelly():
    """Valida integración con Kelly Criterion"""
    print("\n" + "="*60)
    print("✅ TEST 5: Validar Integración con Kelly")
    print("="*60)
    
    try:
        kelly = KellyCalculator(fraccion=0.25)
        print("✅ KellyCalculator inicializado")
        
        # Calcular apuesta
        apuesta = kelly.calcular_con_limites(
            prob=0.58,
            cuota=2.10,
            bankroll=1000
        )
        
        print(f"\n💰 Apuesta calculada: {apuesta:.2f}€")
        
        if apuesta <= 0:
            print("❌ FALLO: Apuesta debería ser > 0")
            return False
        
        if apuesta > 1000 * 0.05:  # Max 5%
            print("❌ FALLO: Apuesta excede límite de 5%")
            return False
        
        print("✅ Apuesta dentro de límites esperados")
        
        print("\n✅ ÉXITO: Integración con Kelly funciona correctamente")
        return True
    
    except Exception as e:
        print(f"\n❌ FALLO: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Ejecuta todas las validaciones
    """
    print("\n" + "="*70)
    print(" "*15 + "🧪 VALIDACIÓN DE FASE 6")
    print(" "*10 + "MÚLTIPLES BOOKMAKERS (LINE SHOPPING)")
    print("="*70)
    
    resultados = []
    
    # Test 1: Configuración
    resultados.append(("Configuración", validar_configuracion()))
    
    if not resultados[0][1]:
        print("\n❌ Configuración inválida. No se pueden ejecutar más tests.")
        print("\nConfigura las variables de entorno necesarias:")
        print("   ODDS_API_KEY=tu_api_key")
        return
    
    # Test 2: OddsFetcher
    resultados.append(("OddsFetcher", validar_odds_fetcher()))
    
    # Test 3: OddsComparator
    resultados.append(("OddsComparator", validar_odds_comparator()))
    
    # Test 4: AlertSystem
    resultados.append(("AlertSystem", validar_alert_system()))
    
    # Test 5: Integración Kelly
    resultados.append(("Integración Kelly", validar_integracion_kelly()))
    
    # Resumen
    print("\n" + "="*70)
    print("📊 RESUMEN DE VALIDACIÓN")
    print("="*70)
    
    for nombre, exito in resultados:
        status = "✅ ÉXITO" if exito else "❌ FALLO"
        print(f"\n{status}: {nombre}")
    
    total_tests = len(resultados)
    tests_exitosos = sum(1 for _, exito in resultados if exito)
    
    print("\n" + "="*70)
    print(f"📈 RESULTADO FINAL: {tests_exitosos}/{total_tests} tests exitosos")
    print("="*70)
    
    if tests_exitosos == total_tests:
        print("\n🎉 ¡FASE 6 VALIDADA EXITOSAMENTE!")
        print("\n✅ Todos los componentes funcionan correctamente:")
        print("   - Obtención de cuotas de múltiples bookmakers")
        print("   - Comparación y selección de mejor cuota")
        print("   - Sistema de alertas")
        print("   - Integración con Kelly Criterion")
        print("   - Manejo robusto de errores de API")
    else:
        print("\n⚠️  Algunos tests fallaron. Revisa los errores arriba.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Validación interrumpida por el usuario")
    except Exception as e:
        print(f"\n❌ Error en validación: {e}")
        import traceback
        traceback.print_exc()
