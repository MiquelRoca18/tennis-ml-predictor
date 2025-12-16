"""
Predictor Multi-Bookmaker - Sistema integrado de predicción + line shopping + Kelly

Este módulo integra:
- Predictor de tenis existente
- Obtención de cuotas de múltiples bookmakers
- Comparación y selección de mejor cuota
- Kelly Criterion para tamaño de apuesta
- Sistema de alertas automáticas
"""

import sys
from pathlib import Path
import pandas as pd

# Añadir paths
sys.path.append(str(Path(__file__).parent.parent))

from src.bookmakers.odds_fetcher import OddsFetcher, APILimitError
from src.bookmakers.odds_comparator import OddsComparator
from src.bookmakers.alert_system import AlertSystem
from src.config import Config as BookmakerConfig
from src.betting.kelly_calculator import KellyCalculator


class PredictorMultiBookmaker:
    """
    Sistema integrado de predicción con line shopping
    
    Flujo:
    1. Obtener cuotas de múltiples bookmakers
    2. Generar predicción del modelo (probabilidades)
    3. Calcular EV con cada bookmaker
    4. Seleccionar mejor oportunidad
    5. Calcular apuesta con Kelly
    6. Enviar alerta si EV > umbral
    """
    
    def __init__(self, modelo_predictor=None, bankroll=1000, 
                 kelly_fraccion=0.25, umbral_ev=0.03, use_cache=True):
        """
        Inicializa el predictor multi-bookmaker
        
        Args:
            modelo_predictor: Predictor de tenis (si None, solo analiza cuotas)
            bankroll: Bankroll actual para Kelly
            kelly_fraccion: Fracción de Kelly (default: 0.25)
            umbral_ev: EV mínimo para apostar (default: 0.03 = 3%)
            use_cache: Si True, usa caché de cuotas
        """
        print("\n" + "="*60)
        print("🚀 INICIALIZANDO PREDICTOR MULTI-BOOKMAKER")
        print("="*60)
        
        # Validar configuración
        is_valid, msg = BookmakerConfig.validate_config()
        if not is_valid:
            raise ValueError(msg)
        
        # Componentes
        self.odds_fetcher = OddsFetcher(use_cache=use_cache)
        self.alert_system = AlertSystem()
        self.kelly_calc = KellyCalculator(fraccion=kelly_fraccion)
        
        # Predictor (opcional)
        self.modelo_predictor = modelo_predictor
        
        # Parámetros
        self.bankroll = bankroll
        self.umbral_ev = umbral_ev
        
        print(f"\n💰 Bankroll: {bankroll}€")
        print(f"📊 Kelly fraccional: {kelly_fraccion*100:.0f}%")
        print(f"📈 Umbral EV: {umbral_ev*100:.1f}%")
        print(f"✅ Sistema listo!")
    
    def obtener_oportunidades(self, sport='tennis_atp', force_refresh=False):
        """
        Obtiene y analiza todas las oportunidades disponibles
        
        Args:
            sport: Deporte a consultar
            force_refresh: Si True, ignora caché
        
        Returns:
            list: Lista de oportunidades con análisis completo
        """
        try:
            # Obtener cuotas
            print("\n" + "="*60)
            print("🌐 OBTENIENDO CUOTAS DE BOOKMAKERS")
            print("="*60)
            
            df_cuotas = self.odds_fetcher.obtener_todas_cuotas(sport, force_refresh)
            
            if len(df_cuotas) == 0:
                print("\n⚠️  No hay partidos disponibles")
                return []
            
            # Crear comparador
            comparador = OddsComparator(df_cuotas)
            
            # Obtener todos los partidos
            partidos = comparador.obtener_todos_partidos()
            
            print(f"\n📊 Analizando {len(partidos)} partido(s)...")
            
            oportunidades = []
            
            for jugador1, jugador2 in partidos:
                # Si tenemos modelo, predecir probabilidades
                if self.modelo_predictor:
                    # Aquí iría la lógica de predicción con el modelo
                    # Por ahora usamos probabilidades de ejemplo
                    prob_j1 = 0.50  # Placeholder
                else:
                    # Sin modelo, asumimos 50-50
                    prob_j1 = 0.50
                
                # Analizar partido
                analisis = comparador.analizar_partido_completo(jugador1, jugador2, prob_j1)
                
                if analisis and analisis['ev'] > self.umbral_ev:
                    # Calcular tamaño de apuesta con Kelly
                    apuesta = self.kelly_calc.calcular_con_limites(
                        prob=analisis['prob'],
                        cuota=analisis['cuota'],
                        bankroll=self.bankroll
                    )
                    
                    analisis['apuesta_cantidad'] = apuesta
                    analisis['pct_bankroll'] = (apuesta / self.bankroll) * 100
                    
                    # Calcular savings
                    savings = comparador.calcular_savings(analisis, apuesta)
                    if savings:
                        analisis['savings'] = savings
                    
                    oportunidades.append(analisis)
            
            # Ordenar por EV
            oportunidades.sort(key=lambda x: x['ev'], reverse=True)
            
            return oportunidades
        
        except APILimitError as e:
            print(f"\n{e}")
            return []
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return []
    
    def analizar_y_alertar(self, sport='tennis_atp', force_refresh=False):
        """
        Obtiene oportunidades y envía alertas
        
        Args:
            sport: Deporte a consultar
            force_refresh: Si True, ignora caché
        
        Returns:
            list: Oportunidades detectadas
        """
        # Obtener oportunidades
        oportunidades = self.obtener_oportunidades(sport, force_refresh)
        
        if len(oportunidades) == 0:
            print(f"\n✅ No hay oportunidades con EV > {self.umbral_ev*100:.1f}%")
            return []
        
        # Enviar alertas
        self.alert_system.verificar_oportunidades(
            oportunidades, 
            umbral_ev=BookmakerConfig.EV_THRESHOLD_ALERT
        )
        
        return oportunidades
    
    def generar_reporte_detallado(self, oportunidades):
        """
        Genera reporte detallado de oportunidades
        
        Args:
            oportunidades: Lista de oportunidades
        """
        if len(oportunidades) == 0:
            print("\n⚠️  No hay oportunidades para reportar")
            return
        
        print("\n" + "="*60)
        print("📊 REPORTE DETALLADO DE OPORTUNIDADES")
        print("="*60)
        
        for i, op in enumerate(oportunidades, 1):
            print(f"\n{'='*60}")
            print(f"🏆 OPORTUNIDAD #{i}")
            print(f"{'='*60}")
            
            print(f"\n🎾 Partido:")
            print(f"   {op['jugador']} vs {op['oponente']}")
            
            print(f"\n📊 Análisis:")
            print(f"   Apostar a: {op['jugador']}")
            print(f"   Probabilidad modelo: {op['prob']*100:.1f}%")
            print(f"   Expected Value: {op['ev']*100:+.2f}%")
            
            print(f"\n💰 Cuotas:")
            print(f"   Mejor cuota: @{op['cuota']:.2f} ({op['bookmaker']})")
            print(f"   Cuota promedio: @{op['cuota_promedio']:.2f}")
            print(f"   Saving: {op['saving_vs_promedio']*100:+.2f}%")
            
            print(f"\n💎 Apuesta Kelly:")
            print(f"   Cantidad: {op['apuesta_cantidad']:.2f}€")
            print(f"   % Bankroll: {op['pct_bankroll']:.1f}%")
            
            if 'savings' in op:
                s = op['savings']
                print(f"\n💸 Savings por Line Shopping:")
                print(f"   Ganancia esperada (mejor): {s['ganancia_mejor_cuota']:+.2f}€")
                print(f"   Ganancia esperada (promedio): {s['ganancia_cuota_promedio']:+.2f}€")
                print(f"   Ahorro: {s['ahorro_euros']:+.2f}€")
                print(f"   Mejora EV: {s['mejora_ev']:+.2f} puntos %")
            
            print(f"\n📍 Bookmakers disponibles: {op['bookmakers_disponibles']}")
        
        print(f"\n{'='*60}")
        print(f"📈 RESUMEN TOTAL")
        print(f"{'='*60}")
        
        total_apuesta = sum(op['apuesta_cantidad'] for op in oportunidades)
        ev_promedio = sum(op['ev'] for op in oportunidades) / len(oportunidades)
        
        print(f"\n   Total oportunidades: {len(oportunidades)}")
        print(f"   Total a apostar: {total_apuesta:.2f}€")
        print(f"   EV promedio: {ev_promedio*100:+.2f}%")
        print(f"   % Bankroll usado: {(total_apuesta/self.bankroll)*100:.1f}%")


# Ejemplo de uso
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 PREDICTOR MULTI-BOOKMAKER - DEMO")
    print("="*60)
    
    try:
        # Crear predictor
        predictor = PredictorMultiBookmaker(
            bankroll=1000,
            kelly_fraccion=0.25,
            umbral_ev=0.03,
            use_cache=True
        )
        
        # Analizar oportunidades
        oportunidades = predictor.analizar_y_alertar(sport='tennis_atp')
        
        # Reporte detallado
        if len(oportunidades) > 0:
            predictor.generar_reporte_detallado(oportunidades)
        
        print(f"\n✅ Demo completado!")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
