"""
Predictor Baseline ELO + Mercado
=================================

Predicciones con probabilidad = BASELINE_ELO_PESO * prob_elo + (1 - BASELINE_ELO_PESO) * prob_mercado.
Incluye cálculo de EV, decisión y stake (Kelly) para la API.
"""

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictorCalibrado:
    """
    Predictor baseline: ELO + probabilidad implícita del mercado (cuota).
    No usa modelo ML.
    """

    def __init__(self, modelo_path=None):
        """
        Inicializa el predictor (siempre en modo baseline).

        Args:
            modelo_path: Ignorado; se mantiene por compatibilidad con get_predictor(Config.MODEL_PATH).
        """
        from src.config.settings import Config

        self._baseline_elo_peso = Config.BASELINE_ELO_PESO
        self.nombre_modelo = "Baseline ELO + Mercado"
        logger.info(
            f"✅ Predictor baseline: {self._baseline_elo_peso*100:.0f}% ELO + {(1-self._baseline_elo_peso)*100:.0f}% mercado"
        )

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

    def predecir_partido(self, jugador1: str, jugador2: str, superficie: str, cuota: float):
        """
        Predice el resultado de un partido y calcula métricas de apuesta

        Args:
            jugador1: Nombre del jugador 1
            jugador2: Nombre del jugador 2
            superficie: Superficie (Hard/Clay/Grass)
            cuota: Cuota del jugador 1

        Returns:
            dict con predicción y análisis
        """
        from src.prediction.feature_generator_service import get_instance as get_feature_service
        from src.config.settings import Config
        from datetime import datetime

        # Obtener servicio de generación de features (singleton, igual que backtesting)
        feature_service = get_feature_service()

        features_j1 = feature_service.generar_features(
            jugador=jugador1, oponente=jugador2, superficie=superficie, fecha=datetime.now()
        )
        confidence_metadata = {k: v for k, v in features_j1.items() if k.startswith("_")}

        prob_elo = features_j1.get("elo_expected_prob", 0.5)
        prob_mercado = 1.0 / cuota if cuota and cuota > 0 else 0.5
        prob_j1_gana = self._baseline_elo_peso * prob_elo + (1.0 - self._baseline_elo_peso) * prob_mercado

        # Calcular EV
        ev = self.calcular_ev(prob_j1_gana, cuota)

        # Decisión (usar mismo umbral que backtesting)
        decision = "APOSTAR" if ev > Config.EV_THRESHOLD else "NO APOSTAR"

        # Probabilidad implícita de la cuota
        prob_implicita = (1.0 / cuota) if cuota and cuota > 0 else 0.0

        # Edge (ventaja sobre la casa)
        edge = prob_j1_gana - prob_implicita

        # Kelly stake igual que backtesting (bankroll, min/max €, max %)
        from src.utils.common import compute_kelly_stake_backtesting
        stake_recomendado = compute_kelly_stake_backtesting(
            prob=prob_j1_gana,
            cuota=cuota,
            bankroll=Config.BANKROLL_INICIAL,
            kelly_fraction=Config.KELLY_FRACTION,
            min_stake_eur=Config.MIN_STAKE_EUR,
            max_stake_pct=Config.MAX_STAKE_PCT,
            max_stake_eur=getattr(Config, "MAX_STAKE_EUR", None),
        )

        # Formatear respuesta para la API
        return {
            "probabilidad": prob_j1_gana,
            "expected_value": ev,
            "decision": decision,
            "stake_recomendado": stake_recomendado,
            "confianza": max(prob_j1_gana, 1 - prob_j1_gana),
            "edge": edge,
            # Agregar metadata de confianza
            "confidence_level": confidence_metadata.get("_confidence_level", "UNKNOWN"),
            "confidence_score": confidence_metadata.get("_confidence_score", 0.0),
            "player1_known": confidence_metadata.get("_player1_known", False),
            "player2_known": confidence_metadata.get("_player2_known", False),
        }


if __name__ == "__main__":
    from src.config.settings import Config
    pred = PredictorCalibrado(Config.MODEL_PATH)
    r = pred.predecir_partido("Alcaraz", "Sinner", "Hard", 2.10)
    print("Probabilidad J1:", r["probabilidad"], "EV:", r["expected_value"], "Decision:", r["decision"])
