"""
Demo de Sistema Multi-Bookmaker - Fase 6

Demuestra el funcionamiento completo del sistema de line shopping:
- Obtención de cuotas de múltiples bookmakers
- Comparación y selección de mejor cuota
- Cálculo de apuestas con Kelly Criterion
- Sistema de alertas
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.predictor_multibookmaker import PredictorMultiBookmaker
from src.bookmakers.config import BookmakerConfig


def main():
    """
    Demo completo del sistema multi-bookmaker
    """
    print("\n" + "="*70)
    print(" "*20 + "🎾 TENNIS ML PREDICTOR")
    print(" "*15 + "FASE 6: MÚLTIPLES BOOKMAKERS")
    print(" "*20 + "(LINE SHOPPING DEMO)")
    print("="*70)
    
    # Mostrar configuración
    print("\n📋 PASO 1: Verificar Configuración")
    print("-" * 70)
    is_valid = BookmakerConfig.print_config()
    
    if not is_valid:
        print("\n❌ Configuración inválida. Por favor, configura las variables de entorno.")
        print("\nCrea un archivo .env en la raíz del proyecto con:")
        print("-" * 40)
        print("ODDS_API_KEY=tu_api_key_aqui")
        print("EMAIL_ENABLED=true  # opcional")
        print("EMAIL_ADDRESS=tu@email.com  # opcional")
        print("EMAIL_PASSWORD=tu_app_password  # opcional")
        print("EMAIL_RECIPIENT=tu@email.com  # opcional")
        print("-" * 40)
        return
    
    # Crear predictor
    print("\n📋 PASO 2: Inicializar Sistema")
    print("-" * 70)
    
    predictor = PredictorMultiBookmaker(
        bankroll=1000,  # 1000€ de bankroll
        kelly_fraccion=0.25,  # Kelly 25%
        umbral_ev=0.03,  # 3% EV mínimo
        use_cache=True  # Usar caché para optimizar requests
    )
    
    # Obtener oportunidades
    print("\n📋 PASO 3: Buscar Oportunidades")
    print("-" * 70)
    print("\n🔍 Consultando cuotas de bookmakers...")
    print("   (Esto puede tardar unos segundos)")
    
    oportunidades = predictor.analizar_y_alertar(
        sport='tennis_atp',
        force_refresh=False  # Usar caché si está disponible
    )
    
    # Reporte detallado
    if len(oportunidades) > 0:
        print("\n📋 PASO 4: Análisis Detallado")
        print("-" * 70)
        predictor.generar_reporte_detallado(oportunidades)
        
        # Resumen de savings
        print("\n📋 PASO 5: Resumen de Line Shopping")
        print("-" * 70)
        
        total_savings = sum(
            op.get('savings', {}).get('ahorro_euros', 0) 
            for op in oportunidades
        )
        
        print(f"\n💰 BENEFICIO DEL LINE SHOPPING:")
        print(f"   Total de oportunidades: {len(oportunidades)}")
        print(f"   Ahorro total estimado: {total_savings:+.2f}€")
        print(f"   Promedio por apuesta: {total_savings/len(oportunidades):+.2f}€")
        
        print(f"\n💡 CONCLUSIÓN:")
        print(f"   Al usar line shopping (comparar múltiples bookmakers),")
        print(f"   puedes mejorar tu EV en promedio {total_savings/len(oportunidades):+.2f}€ por apuesta.")
        print(f"   ¡Esto se acumula significativamente a largo plazo!")
    
    else:
        print("\n📋 PASO 4: Sin Oportunidades")
        print("-" * 70)
        print(f"\n✅ No hay oportunidades con EV > {predictor.umbral_ev*100:.1f}% en este momento")
        print(f"\n💡 SUGERENCIAS:")
        print(f"   1. Intenta más tarde cuando haya más partidos")
        print(f"   2. Reduce el umbral de EV (actualmente {predictor.umbral_ev*100:.1f}%)")
        print(f"   3. Verifica que haya partidos ATP disponibles")
    
    # Estadísticas de API
    print("\n📋 PASO 6: Estadísticas de API")
    print("-" * 70)
    
    stats = predictor.odds_fetcher.get_request_stats()
    if stats['requests_remaining']:
        print(f"\n📊 Uso de The Odds API:")
        print(f"   Requests restantes: {stats['requests_remaining']}/{stats['max_requests']}")
        print(f"   Requests usados: {stats['requests_used']}")
        
        pct_usado = (int(stats['requests_used']) / stats['max_requests']) * 100
        print(f"   Porcentaje usado: {pct_usado:.1f}%")
        
        if int(stats['requests_remaining']) < 50:
            print(f"\n⚠️  ALERTA: Quedan pocos requests este mes!")
            print(f"   Considera usar el caché para optimizar el uso")
    
    print("\n" + "="*70)
    print("✅ DEMO COMPLETADO")
    print("="*70)
    
    print("\n💡 PRÓXIMOS PASOS:")
    print("   1. Ejecuta este script regularmente para encontrar oportunidades")
    print("   2. Configura alertas por email para recibir notificaciones")
    print("   3. Integra con tu modelo de predicción para probabilidades reales")
    print("   4. Usa el caché para optimizar el uso de la API")
    
    print("\n📚 DOCUMENTACIÓN:")
    print("   - Revisa FASE_6_RESULTADOS.md para más detalles")
    print("   - Consulta README.md para instrucciones de uso")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error en demo: {e}")
        import traceback
        traceback.print_exc()
