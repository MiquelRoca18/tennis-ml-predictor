"""
Predictor Calibrado con Análisis de Confianza - Fase 2
=======================================================

Clase predictor que usa modelos calibrados y proporciona:
- Predicciones con probabilidades calibradas
- Cálculo de EV y recomendaciones de apuesta
- Análisis de sensibilidad
- Intervalos de confianza
"""

import numpy as np
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictorCalibrado:
    """
    Predictor que usa modelos calibrados para predicciones confiables
    """
    
    def __init__(self, modelo_path):
        """
        Inicializa el predictor
        
        Args:
            modelo_path: Path al modelo calibrado (.pkl)
        """
        self.modelo_path = Path(modelo_path)
        self.modelo = None
        self.nombre_modelo = self.modelo_path.stem.replace("_calibrado", "").replace("_", " ").title()
        
        self.cargar_modelo()
    
    def cargar_modelo(self):
        """Carga el modelo calibrado"""
        try:
            self.modelo = joblib.load(self.modelo_path)
            logger.info(f"✅ Modelo cargado: {self.nombre_modelo}")
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")
            raise
    
    def predecir(self, features):
        """
        Realiza predicción con el modelo calibrado
        
        Args:
            features: Array o dict con features del partido
            
        Returns:
            dict con predicción completa
        """
        # Convertir a array si es necesario
        if isinstance(features, dict):
            features = np.array(list(features.values())).reshape(1, -1)
        elif len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Predecir
        prob = self.modelo.predict_proba(features)[0, 1]
        pred_class = self.modelo.predict(features)[0]
        
        return {
            'probabilidad': prob,
            'prediccion': int(pred_class),
            'confianza': max(prob, 1 - prob)
        }
    
    
    def calcular_ev(self, prob, cuota):
        """
        Calcula Expected Value
        
        EV = (probabilidad * cuota) - 1
        
        Args:
            prob: Probabilidad del modelo
            cuota: Cuota de la casa de apuestas
            
        Returns:
            EV (Expected Value)
        """
        return (prob * cuota) - 1
    
    
    def predecir_partido(self, jugador1, jugador1_rank, jugador2, jugador2_rank, superficie, cuota):
        """
        Predice un partido usando predicción bidireccional (igual que backtesting)
        
        Este método replica EXACTAMENTE el proceso del backtesting:
        1. Genera features completas para j1 vs j2
        2. Genera features completas para j2 vs j1
        3. Combina con prefijos j1_ y j2_
        4. Hace UNA predicción: ¿Ganará j1?
        
        Args:
            jugador1: Nombre del jugador 1
            jugador1_rank: Ranking del jugador 1 (no usado, se obtiene del histórico)
            jugador2: Nombre del jugador 2  
            jugador2_rank: Ranking del jugador 2 (no usado, se obtiene del histórico)
            superficie: Superficie (Hard/Clay/Grass)
            cuota: Cuota para jugador 1
            
        Returns:
            dict con predicción y análisis
        """
        from src.prediction.feature_generator_service import FeatureGeneratorService
        from datetime import datetime
        
        # Obtener servicio de generación de features (singleton)
        feature_service = FeatureGeneratorService()
        
        # PREDICCIÓN BIDIRECCIONAL (igual que backtesting)
        # Generar features para jugador1 (como 'jugador')
        features_j1 = feature_service.generar_features(
            jugador=jugador1,
            oponente=jugador2,
            superficie=superficie,
            fecha=datetime.now()
        )
        
        # Generar features para jugador2 (como 'jugador')
        features_j2 = feature_service.generar_features(
            jugador=jugador2,
            oponente=jugador1,
            superficie=superficie,
            fecha=datetime.now()
        )
        
        # Combinar con prefijos j1_ y j2_
        features_combined = {}
        for key, value in features_j1.items():
            features_combined[f'j1_{key}'] = value
        for key, value in features_j2.items():
            features_combined[f'j2_{key}'] = value
        
        # Convertir a array en el orden correcto de features
        features_array = []
        for feat_name in feature_service.feature_cols:
            if feat_name in features_combined:
                features_array.append(features_combined[feat_name])
            else:
                # Si falta alguna feature, usar 0
                features_array.append(0.0)
        
        features_array = np.array(features_array).reshape(1, -1)
        
        # Predecir probabilidad de que jugador1 gane
        prob_j1_gana = self.modelo.predict_proba(features_array)[0, 1]
        
        # Calcular EV
        ev = self.calcular_ev(prob_j1_gana, cuota)
        
        # Decisión
        umbral_ev = 0.03
        decision = "APOSTAR" if ev > umbral_ev else "NO APOSTAR"
        
        # Probabilidad implícita de la cuota
        prob_implicita = 1 / cuota
        
        # Edge (ventaja sobre la casa)
        edge = prob_j1_gana - prob_implicita
        
        # Kelly stake
        stake_recomendado = 0
        if ev > 0:
            kelly_pct = (prob_j1_gana * cuota - 1) / (cuota - 1)
            kelly_pct = kelly_pct * 0.25  # Kelly fraction conservador
            stake_recomendado = max(kelly_pct * 100, 0)  # Stake en €
        
        # Formatear respuesta para la API
        return {
            'probabilidad': prob_j1_gana,
            'expected_value': ev,
            'decision': decision,
            'stake_recomendado': stake_recomendado,
            'confianza': max(prob_j1_gana, 1 - prob_j1_gana),
            'edge': edge
        }
    
    def recomendar_apuesta(self, features, cuota, umbral_ev=0.03, stake=10.0):
        """
        Analiza un partido y recomienda si apostar
        
        Args:
            features: Features del partido
            cuota: Cuota disponible
            umbral_ev: EV mínimo para recomendar apuesta
            stake: Cantidad a apostar
            
        Returns:
            dict con análisis completo
        """
        # Predecir
        prediccion = self.predecir(features)
        prob = prediccion['probabilidad']
        
        # Calcular EV
        ev = self.calcular_ev(prob, cuota)
        
        # Decisión
        decision = "APOSTAR" if ev > umbral_ev else "NO APOSTAR"
        
        # Ganancia esperada
        ganancia_esperada = stake * ev
        
        # Probabilidad implícita de la cuota
        prob_implicita = 1 / cuota
        
        # Edge (ventaja sobre la casa)
        edge = prob - prob_implicita
        
        analisis = {
            'probabilidad_modelo': prob,
            'cuota': cuota,
            'ev': ev,
            'ev_porcentaje': ev * 100,
            'decision': decision,
            'stake_recomendado': stake if decision == "APOSTAR" else 0,
            'ganancia_esperada': ganancia_esperada,
            'prob_implicita_cuota': prob_implicita,
            'edge': edge,
            'edge_porcentaje': edge * 100,
            'confianza': prediccion['confianza']
        }
        
        return analisis
    
    def analisis_sensibilidad(self, features, cuotas_rango=None):
        """
        Analiza cómo cambia EV con diferentes cuotas
        
        Args:
            features: Features del partido
            cuotas_rango: Lista de cuotas a analizar (default: 1.5 a 4.0)
            
        Returns:
            dict con análisis de sensibilidad
        """
        # Predecir probabilidad
        prediccion = self.predecir(features)
        prob = prediccion['probabilidad']
        
        # Rango de cuotas
        if cuotas_rango is None:
            cuotas_rango = np.arange(1.5, 4.1, 0.1)
        
        # Calcular EV para cada cuota
        resultados = []
        for cuota in cuotas_rango:
            ev = self.calcular_ev(prob, cuota)
            resultados.append({
                'cuota': cuota,
                'ev': ev,
                'ev_porcentaje': ev * 100
            })
        
        # Encontrar cuota mínima para EV positivo
        cuota_minima_ev_positivo = 1 / prob if prob > 0 else np.inf
        
        return {
            'probabilidad': prob,
            'cuota_minima_ev_positivo': cuota_minima_ev_positivo,
            'sensibilidad': resultados
        }
    
    def explicar_prediccion(self, features, cuota, umbral_ev=0.03):
        """
        Genera explicación detallada de la predicción
        
        Args:
            features: Features del partido
            cuota: Cuota disponible
            umbral_ev: Umbral de EV
            
        Returns:
            str con explicación
        """
        analisis = self.recomendar_apuesta(features, cuota, umbral_ev)
        
        explicacion = f"""
{'='*70}
ANÁLISIS DE PREDICCIÓN - {self.nombre_modelo}
{'='*70}

📊 PROBABILIDADES:
  Probabilidad modelo:     {analisis['probabilidad_modelo']*100:.2f}%
  Prob. implícita cuota:   {analisis['prob_implicita_cuota']*100:.2f}%
  Edge sobre la casa:      {analisis['edge_porcentaje']:+.2f}%

💰 VALOR ESPERADO:
  Cuota disponible:        {analisis['cuota']:.2f}
  EV:                      {analisis['ev_porcentaje']:+.2f}%
  Ganancia esperada:       {analisis['ganancia_esperada']:+.2f}€

🎯 RECOMENDACIÓN:
  Decisión:                {analisis['decision']}
  Stake recomendado:       {analisis['stake_recomendado']:.2f}€
  Confianza:               {analisis['confianza']*100:.1f}%

{'='*70}
"""
        
        if analisis['decision'] == "APOSTAR":
            explicacion += f"""
✅ APOSTAR RECOMENDADO
   Razón: EV ({analisis['ev_porcentaje']:+.2f}%) > Umbral ({umbral_ev*100:.2f}%)
   El modelo estima una ventaja de {analisis['edge_porcentaje']:+.2f}% sobre la cuota.
"""
        else:
            explicacion += f"""
❌ NO APOSTAR
   Razón: EV ({analisis['ev_porcentaje']:+.2f}%) ≤ Umbral ({umbral_ev*100:.2f}%)
   No hay suficiente valor esperado para justificar la apuesta.
"""
        
        explicacion += f"\n{'='*70}\n"
        
        return explicacion


def ejemplo_uso():
    """
    Ejemplo de uso del predictor calibrado
    """
    logger.info("="*70)
    logger.info("EJEMPLO DE USO - PREDICTOR CALIBRADO")
    logger.info("="*70)
    
    # Cargar predictor
    predictor = PredictorCalibrado("modelos/random_forest_calibrado.pkl")
    
    # Ejemplo: Partido entre dos jugadores
    # Features de ejemplo (deberían ser las mismas que usaste en entrenamiento)
    features_ejemplo = np.array([
        50,    # jugador_rank
        20,    # oponente_rank
        # ... resto de features
    ])
    
    # Cuota disponible
    cuota = 2.10
    
    # Análisis completo
    logger.info("\n" + predictor.explicar_prediccion(features_ejemplo, cuota, umbral_ev=0.03))
    
    # Análisis de sensibilidad
    sensibilidad = predictor.analisis_sensibilidad(features_ejemplo)
    logger.info(f"\n📊 ANÁLISIS DE SENSIBILIDAD:")
    logger.info(f"  Probabilidad modelo: {sensibilidad['probabilidad']*100:.2f}%")
    logger.info(f"  Cuota mínima para EV+: {sensibilidad['cuota_minima_ev_positivo']:.2f}")
    logger.info(f"\n  Ejemplos de EV por cuota:")
    for item in sensibilidad['sensibilidad'][::5]:  # Mostrar cada 5
        logger.info(f"    Cuota {item['cuota']:.2f} → EV: {item['ev_porcentaje']:+.2f}%")


if __name__ == "__main__":
    ejemplo_uso()
