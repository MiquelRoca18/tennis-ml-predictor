"""
Prediction Runner - Lógica compartida para generar y guardar predicciones
==========================================================================

Un solo lugar para: predecir partido, calcular recomendación/Kelly y guardar en BD.
Usado por DailyMatchFetcher (partidos nuevos) y OddsUpdateService (sync de cuotas).
"""

import logging
from typing import Optional

from src.config.settings import Config
from src.utils.common import compute_kelly_stake_backtesting

logger = logging.getLogger(__name__)


def run_prediction_and_save(
    db,
    predictor,
    match_id: int,
    player1_name: str,
    player2_name: str,
    surface: str,
    player1_odds: float,
    player2_odds: float,
) -> bool:
    """
    Genera predicción para un partido y la guarda en BD (misma lógica que backtesting).

    Usado cuando:
    - Se crea un partido nuevo con cuotas (DailyMatchFetcher).
    - Se detectan cuotas nuevas o actualizadas para un partido existente (OddsUpdateService).

    Args:
        db: MatchDatabase instance
        predictor: PredictorCalibrado instance (puede ser None; entonces no se genera predicción)
        match_id: ID del partido en BD
        player1_name: Nombre jugador 1
        player2_name: Nombre jugador 2
        surface: Superficie (Hard/Clay/Grass)
        player1_odds: Cuota jugador 1
        player2_odds: Cuota jugador 2

    Returns:
        True si se generó y guardó la predicción; False si no hay predictor o error
    """
    if predictor is None:
        logger.debug(f"ℹ️  Skipping prediction for match {match_id} (no predictor)")
        return False

    try:
        resultado_pred = predictor.predecir_partido(
            jugador1=player1_name,
            jugador2=player2_name,
            superficie=surface,
            cuota=player1_odds,
        )

        prob_j1 = resultado_pred["probabilidad"]
        prob_j2 = 1 - prob_j1
        ev_j1 = resultado_pred["expected_value"]
        ev_j2 = (prob_j2 * player2_odds) - 1
        edge_j1 = resultado_pred.get("edge", 0)
        edge_j2 = prob_j2 - (1 / player2_odds)

        bankroll = (db.get_bankroll() if db else None) or Config.BANKROLL_INICIAL
        umbral_ev = Config.EV_THRESHOLD
        min_prob = Config.MIN_PROBABILIDAD
        max_cuota = Config.MAX_CUOTA
        max_stake_eur = getattr(Config, "MAX_STAKE_EUR", None)
        # Misma regla que backtesting: si ambos pasan filtros, recomendar el de mayor EV (ev_j1 > ev_j2 para J1)
        if (
            ev_j1 > umbral_ev
            and ev_j1 > ev_j2
            and prob_j1 >= min_prob
            and player1_odds < max_cuota
        ):
            recomendacion = f"APOSTAR a {player1_name}"
            mejor_opcion = player1_name
            prob_recomendado = prob_j1
            kelly_j1 = (
                compute_kelly_stake_backtesting(
                    prob=prob_j1,
                    cuota=player1_odds,
                    bankroll=bankroll,
                    kelly_fraction=Config.KELLY_FRACTION,
                    min_stake_eur=Config.MIN_STAKE_EUR,
                    max_stake_pct=Config.MAX_STAKE_PCT,
                    max_stake_eur=max_stake_eur,
                )
                or None
            )
            kelly_j2 = None
        elif (
            ev_j2 > umbral_ev
            and prob_j2 >= min_prob
            and player2_odds < max_cuota
        ):
            recomendacion = f"APOSTAR a {player2_name}"
            mejor_opcion = player2_name
            prob_recomendado = prob_j2
            kelly_j1 = None
            kelly_j2 = (
                compute_kelly_stake_backtesting(
                    prob=prob_j2,
                    cuota=player2_odds,
                    bankroll=bankroll,
                    kelly_fraction=Config.KELLY_FRACTION,
                    min_stake_eur=Config.MIN_STAKE_EUR,
                    max_stake_pct=Config.MAX_STAKE_PCT,
                    max_stake_eur=max_stake_eur,
                )
                or None
            )
        else:
            recomendacion = "NO APOSTAR"
            mejor_opcion = None
            prob_recomendado = max(prob_j1, prob_j2)
            kelly_j1 = None
            kelly_j2 = None

        # Confianza = probabilidad del jugador recomendado (no "conocimiento ELO")
        resultado_pred = dict(resultado_pred)
        if mejor_opcion:
            if prob_recomendado >= 0.70:
                confianza = "Alta"
                resultado_pred["confidence_level"] = "HIGH"
                resultado_pred["confidence_score"] = prob_recomendado
            elif prob_recomendado >= 0.55:
                confianza = "Media"
                resultado_pred["confidence_level"] = "MEDIUM"
                resultado_pred["confidence_score"] = prob_recomendado
            else:
                confianza = "Baja"
                resultado_pred["confidence_level"] = "LOW"
                resultado_pred["confidence_score"] = prob_recomendado
        else:
            confianza = "Baja"
            resultado_pred["confidence_level"] = "LOW"
            resultado_pred["confidence_score"] = prob_recomendado if prob_recomendado else 0.5

        db.add_prediction(
            match_id=match_id,
            jugador1_cuota=player1_odds,
            jugador2_cuota=player2_odds,
            jugador1_probabilidad=prob_j1,
            jugador2_probabilidad=prob_j2,
            jugador1_ev=ev_j1,
            jugador2_ev=ev_j2,
            jugador1_edge=edge_j1,
            jugador2_edge=edge_j2,
            recomendacion=recomendacion,
            mejor_opcion=mejor_opcion,
            confianza=confianza,
            kelly_stake_jugador1=kelly_j1,
            kelly_stake_jugador2=kelly_j2,
            confidence_level=resultado_pred.get("confidence_level"),
            confidence_score=resultado_pred.get("confidence_score"),
            player1_known=resultado_pred.get("player1_known"),
            player2_known=resultado_pred.get("player2_known"),
        )

        logger.info(f"✅ Prediction saved for match {match_id}: {recomendacion}")
        return True

    except Exception as e:
        logger.error(f"❌ Error generating prediction for match {match_id}: {e}", exc_info=True)
        return False
