"""
Predictor de partidos de tenis con cálculo de Expected Value
"""
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TennisPredictor:
    """
    Predictor de partidos de tenis con cálculo de EV
    """
    
    def __init__(self, modelo_path="modelos/modelo_rf_v1.pkl"):
        """
        Carga el modelo entrenado
        
        Args:
            modelo_path: Ruta al modelo entrenado
        """
        logger.info("🤖 Cargando modelo...")
        self.modelo = joblib.load(modelo_path)
        logger.info("✅ Modelo cargado!")
        
        self.feature_columns = [
            'jugador_rank', 'oponente_rank', 'rank_diff', 'rank_ratio',
            'jugador_top10', 'oponente_top10', 'jugador_top50', 'oponente_top50',
            'surface_hard', 'surface_clay', 'surface_grass'
        ]
    
    def crear_features_partido(self, jugador_rank, oponente_rank, superficie):
        """
        Crea features para un partido nuevo
        
        Args:
            jugador_rank: int, ranking ATP del jugador
            oponente_rank: int, ranking ATP del oponente
            superficie: str, 'Hard', 'Clay', o 'Grass'
            
        Returns:
            DataFrame con features
        """
        
        features = {
            'jugador_rank': jugador_rank,
            'oponente_rank': oponente_rank,
            'rank_diff': oponente_rank - jugador_rank,
            'rank_ratio': jugador_rank / oponente_rank,
            'jugador_top10': 1 if jugador_rank <= 10 else 0,
            'oponente_top10': 1 if oponente_rank <= 10 else 0,
            'jugador_top50': 1 if jugador_rank <= 50 else 0,
            'oponente_top50': 1 if oponente_rank <= 50 else 0,
            'surface_hard': 1 if superficie == 'Hard' else 0,
            'surface_clay': 1 if superficie == 'Clay' else 0,
            'surface_grass': 1 if superficie == 'Grass' else 0
        }
        
        return pd.DataFrame([features])[self.feature_columns]
    
    def predecir_probabilidad(self, jugador_rank, oponente_rank, superficie):
        """
        Predice la probabilidad de que gane el jugador
        
        Args:
            jugador_rank: int, ranking ATP del jugador
            oponente_rank: int, ranking ATP del oponente
            superficie: str, 'Hard', 'Clay', o 'Grass'
            
        Returns:
            float: Probabilidad entre 0 y 1
        """
        
        X = self.crear_features_partido(jugador_rank, oponente_rank, superficie)
        probabilidad = self.modelo.predict_proba(X)[0][1]
        
        return probabilidad
    
    def calcular_ev(self, probabilidad, cuota):
        """
        Calcula el Expected Value (EV)
        
        Args:
            probabilidad: float, tu probabilidad estimada (0-1)
            cuota: float, cuota del bookmaker (ej: 2.00)
            
        Returns:
            float: EV como decimal
        """
        ev = (probabilidad * cuota) - 1
        return ev
    
    def analizar_partido(self, jugador_nombre, jugador_rank, oponente_nombre, 
                        oponente_rank, superficie, cuota_jugador, umbral_ev=0.03):
        """
        Análisis completo de un partido
        
        Args:
            jugador_nombre: str, nombre del jugador
            jugador_rank: int, ranking ATP
            oponente_nombre: str, nombre del oponente
            oponente_rank: int, ranking ATP
            superficie: str, 'Hard', 'Clay', o 'Grass'
            cuota_jugador: float, cuota del bookmaker para el jugador
            umbral_ev: float, EV mínimo para apostar (default: 3%)
            
        Returns:
            dict con toda la información
        """
        
        # Predecir probabilidad
        prob_jugador = self.predecir_probabilidad(jugador_rank, oponente_rank, superficie)
        prob_oponente = 1 - prob_jugador
        
        # Calcular EV
        ev = self.calcular_ev(prob_jugador, cuota_jugador)
        
        # Probabilidad implícita en la cuota
        prob_implicita = 1 / cuota_jugador
        
        # Decisión
        decision = "APOSTAR ✅" if ev > umbral_ev else "NO APOSTAR ❌"
        
        # Crear resultado
        resultado = {
            'jugador': jugador_nombre,
            'oponente': oponente_nombre,
            'jugador_rank': jugador_rank,
            'oponente_rank': oponente_rank,
            'superficie': superficie,
            'prob_modelo': prob_jugador,
            'prob_implicita': prob_implicita,
            'cuota': cuota_jugador,
            'ev': ev,
            'ev_pct': ev * 100,
            'decision': decision
        }
        
        return resultado
    
    def mostrar_analisis(self, resultado):
        """
        Muestra el análisis de forma bonita
        
        Args:
            resultado: dict con análisis del partido
        """
        
        print("\n" + "=" * 70)
        print("🎾 ANÁLISIS DE PARTIDO")
        print("=" * 70)
        print(f"\n🆚 {resultado['jugador']} (#{resultado['jugador_rank']}) vs {resultado['oponente']} (#{resultado['oponente_rank']})")
        print(f"🏟️  Superficie: {resultado['superficie']}")
        print(f"\n📊 PROBABILIDADES:")
        print(f"   Tu modelo:     {resultado['prob_modelo']*100:.1f}%")
        print(f"   Bookmaker:     {resultado['prob_implicita']*100:.1f}%")
        print(f"   Diferencia:    {(resultado['prob_modelo'] - resultado['prob_implicita'])*100:+.1f}%")
        print(f"\n💰 CUOTA Y EV:")
        print(f"   Cuota:         @{resultado['cuota']:.2f}")
        print(f"   Expected Value: {resultado['ev_pct']:+.2f}%")
        print(f"\n🎯 DECISIÓN: {resultado['decision']}")
        
        if "APOSTAR" in resultado['decision']:
            print(f"\n💵 Ejemplo con 10€:")
            print(f"   Ganancia esperada: {10 * resultado['ev']:.2f}€")
            print(f"   Si ganas: {10 * resultado['cuota']:.2f}€")
            print(f"   Si pierdes: -10.00€")
        
        print("=" * 70)


if __name__ == "__main__":
    # Crear predictor
    predictor = TennisPredictor("modelos/modelo_rf_v1.pkl")
    
    # Ejemplo 1: Alcaraz vs Sinner en Hard
    print("\n📌 EJEMPLO 1:")
    resultado1 = predictor.analizar_partido(
        jugador_nombre="Carlos Alcaraz",
        jugador_rank=3,
        oponente_nombre="Jannik Sinner",
        oponente_rank=1,
        superficie="Hard",
        cuota_jugador=2.10,
        umbral_ev=0.03
    )
    predictor.mostrar_analisis(resultado1)
    
    # Ejemplo 2: Djokovic vs Rune en Clay
    print("\n📌 EJEMPLO 2:")
    resultado2 = predictor.analizar_partido(
        jugador_nombre="Novak Djokovic",
        jugador_rank=7,
        oponente_nombre="Holger Rune",
        oponente_rank=13,
        superficie="Clay",
        cuota_jugador=1.55,
        umbral_ev=0.03
    )
    predictor.mostrar_analisis(resultado2)
    
    # Ejemplo 3: Partido con underdog
    print("\n📌 EJEMPLO 3:")
    resultado3 = predictor.analizar_partido(
        jugador_nombre="Lorenzo Musetti",
        jugador_rank=17,
        oponente_nombre="Andrey Rublev",
        oponente_rank=9,
        superficie="Clay",
        cuota_jugador=2.80,
        umbral_ev=0.03
    )
    predictor.mostrar_analisis(resultado3)
