"""
Predictor de Producción — ELO Baseline + LightGBM (opcional)
=============================================================

Modos:
  - Baseline (default): prob = 0.6 * elo_prob + 0.4 * market_prob
  - LightGBM (si existe modelos/lgbm_tennis.pkl): prob = lgbm.predict_proba(features)

EV siempre: ev = model_prob * odds - 1
"""
import logging
import pickle
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictorCalibrado:
    """Predictor con soporte baseline ELO + LightGBM opcional."""

    def __init__(self, modelo_path=None):
        from src.config.settings import Config
        self._baseline_elo_peso = Config.BASELINE_ELO_PESO
        self._lgbm_model = None
        self._use_lgbm = False

        use_lgbm_config = getattr(Config, "USE_LGBM_MODEL", "auto").lower()
        lgbm_path = getattr(Config, "LGBM_MODEL_PATH", "modelos/lgbm_tennis.pkl")

        if use_lgbm_config == "false":
            self.nombre_modelo = "Baseline ELO + Mercado"
            logger.info("Predictor: baseline (USE_LGBM_MODEL=false)")
        else:
            loaded = self._try_load_lgbm(lgbm_path)
            if loaded:
                self._lgbm_model = loaded
                self._use_lgbm = True
                self.nombre_modelo = "LightGBM + Features"
                logger.info(f"Predictor: LightGBM desde {lgbm_path}")
            elif use_lgbm_config == "true":
                raise FileNotFoundError(
                    f"USE_LGBM_MODEL=true pero no existe: {lgbm_path}"
                )
            else:
                self.nombre_modelo = "Baseline ELO + Mercado (fallback)"
                logger.info(f"Predictor: baseline (modelo no encontrado en {lgbm_path})")

    def _try_load_lgbm(self, path: str):
        try:
            if Path(path).exists():
                with open(path, "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"No se pudo cargar LightGBM ({path}): {e}")
        return None

    def calcular_ev(self, prob: float, cuota: float) -> float:
        """EV = prob * cuota - 1."""
        return prob * cuota - 1

    def predecir_partido(self, jugador1: str, jugador2: str, superficie: str, cuota: float):
        """
        Predice resultado y calcula métricas de apuesta.
        Returns: dict con probabilidad, EV, decision, stake, edge, modelo, confianza.
        """
        from src.prediction.feature_generator_service import get_instance as get_feature_service
        from src.config.settings import Config
        from src.utils.common import compute_kelly_stake_backtesting
        from datetime import datetime

        feature_service = get_feature_service()
        features_j1 = feature_service.generar_features(
            jugador=jugador1, oponente=jugador2, superficie=superficie, fecha=datetime.now()
        )
        confidence_metadata = {k: v for k, v in features_j1.items() if k.startswith("_")}

        if self._use_lgbm and self._lgbm_model is not None:
            prob_j1_gana = self._predict_lgbm(features_j1)
        else:
            prob_elo = features_j1.get("elo_expected_prob", 0.5)
            prob_mercado = 1.0 / cuota if cuota and cuota > 0 else 0.5
            prob_j1_gana = (
                self._baseline_elo_peso * prob_elo
                + (1.0 - self._baseline_elo_peso) * prob_mercado
            )

        ev = self.calcular_ev(prob_j1_gana, cuota)
        decision = "APOSTAR" if ev > Config.EV_THRESHOLD else "NO APOSTAR"
        prob_implicita = 1.0 / cuota if cuota and cuota > 0 else 0.0
        edge = prob_j1_gana - prob_implicita

        stake_recomendado = compute_kelly_stake_backtesting(
            prob=prob_j1_gana,
            cuota=cuota,
            bankroll=Config.BANKROLL_INICIAL,
            kelly_fraction=Config.KELLY_FRACTION,
            min_stake_eur=Config.MIN_STAKE_EUR,
            max_stake_pct=Config.MAX_STAKE_PCT,
            max_stake_eur=getattr(Config, "MAX_STAKE_EUR", None),
        )

        return {
            "probabilidad": prob_j1_gana,
            "expected_value": ev,
            "decision": decision,
            "stake_recomendado": stake_recomendado,
            "confianza": max(prob_j1_gana, 1 - prob_j1_gana),
            "edge": edge,
            "modelo": self.nombre_modelo,
            "confidence_level": confidence_metadata.get("_confidence_level", "UNKNOWN"),
            "confidence_score": confidence_metadata.get("_confidence_score", 0.0),
            "player1_known": confidence_metadata.get("_player1_known", False),
            "player2_known": confidence_metadata.get("_player2_known", False),
        }

    def _predict_lgbm(self, features: dict) -> float:
        """Predice probabilidad usando LightGBM."""
        from src.config.settings import Config
        import numpy as np

        lgbm_cols = getattr(Config, "LGBM_FEATURE_COLS", [
            "elo_diff_surface", "h2h_reciente_rate", "diff_win_rate_60d",
            "diff_carga_score", "diff_dias_descanso", "ventaja_superficie",
            "rank_diff", "elo_diff",
        ])

        X = np.array([[
            float(features.get(col, 0.0) or 0.0)
            for col in lgbm_cols
        ]])

        try:
            prob = self._lgbm_model.predict_proba(X)[0, 1]
            return float(np.clip(prob, 0.01, 0.99))
        except Exception as e:
            logger.warning(f"Error en predicción LightGBM, fallback a ELO: {e}")
            return float(features.get("elo_expected_prob", 0.5))


if __name__ == "__main__":
    from src.config.settings import Config
    pred = PredictorCalibrado(Config.MODEL_PATH)
    r = pred.predecir_partido("Alcaraz", "Sinner", "Hard", 2.10)
    print(f"Modelo: {r['modelo']}")
    print(f"Probabilidad J1: {r['probabilidad']:.3f}")
    print(f"EV: {r['expected_value']:.3f}")
    print(f"Decisión: {r['decision']}")
