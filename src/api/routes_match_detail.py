"""
Routes Match Detail - Endpoints del Detalle de Partido
======================================================

Endpoints optimizados para el marcador deportivo profesional.
Proporcionan todos los datos necesarios de forma eficiente.

Endpoints:
- GET /matches/{id}/full - Datos completos del partido
- GET /matches/{id}/timeline - Timeline de juegos
- GET /matches/{id}/pbp - Punto por punto
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from src.api.models_match_detail import (
    BookmakerOdds,
    H2HData,
    MatchFullResponse,
    MatchInfo,
    MatchOdds,
    MatchPrediction,
    MatchScores,
    MatchStats,
    MatchStatus,
    MatchTimeline,
    PlayerInfo,
    PointByPointData,
    PreviousMatch,
    Surface,
)
from src.services.match_stats_calculator import MatchStatsCalculator

logger = logging.getLogger(__name__)

# Router para endpoints de detalle
router = APIRouter(prefix="/v2/matches", tags=["Match Detail v2"])

# Instancia del calculador de estadísticas
stats_calculator = MatchStatsCalculator()


def get_db():
    """Obtiene la instancia de la base de datos"""
    from src.api.api_v2 import db
    return db


def get_api_client():
    """Obtiene el cliente de API Tennis"""
    from src.api.api_v2 import api_client
    return api_client


# ============================================================
# ENDPOINT PRINCIPAL: /matches/{id}/full
# ============================================================

@router.get("/{match_id}/full", response_model=MatchFullResponse)
async def get_match_full(match_id: int):
    """
    Obtiene todos los datos de un partido en una sola llamada.
    
    Incluye:
    - Información básica del partido
    - Jugadores con ranking y foto
    - Scores por set con tiebreaks
    - Estadísticas completas (si disponibles)
    - Timeline de juegos (si disponible)
    - Head to Head resumido
    - Cuotas de apuestas
    - Predicción ML (si existe)
    
    Este endpoint está optimizado para cargar toda la información
    necesaria para mostrar el detalle de un partido.
    """
    db = get_db()
    api_client = get_api_client()
    
    try:
        # 1. Obtener partido de la BD
        match = db.get_match(match_id)
        if not match:
            raise HTTPException(status_code=404, detail="Partido no encontrado")
        
        # 2. Obtener datos adicionales de la API Tennis si hay event_key
        api_data = None
        event_key = match.get("event_key")
        
        if event_key:
            try:
                # Intentar obtener datos actualizados de la API
                params = {"match_key": event_key}
                response = api_client._make_request("get_fixtures", params)
                
                if response and response.get("result"):
                    results = response["result"]
                    if isinstance(results, list) and results:
                        api_data = results[0]
                    elif isinstance(results, dict):
                        api_data = results
                        
            except Exception as e:
                logger.warning(f"No se pudo obtener datos de API Tennis: {e}")
        
        # 3. Construir información del partido
        match_info = MatchInfo(
            id=match_id,
            status=MatchStatus(match.get("estado", "pendiente")),
            date=match.get("fecha_partido"),
            time=match.get("hora_inicio"),
            tournament=match.get("torneo", "Unknown"),
            round=match.get("ronda"),
            surface=Surface(match.get("superficie", "Hard")),
        )
        
        # 4. Construir información de jugadores
        player1 = PlayerInfo(
            name=match.get("jugador1", "Player 1"),
            country=match.get("jugador1_pais"),
            ranking=match.get("jugador1_ranking"),
            logo_url=match.get("jugador1_logo"),
        )
        
        player2 = PlayerInfo(
            name=match.get("jugador2", "Player 2"),
            country=match.get("jugador2_pais"),
            ranking=match.get("jugador2_ranking"),
            logo_url=match.get("jugador2_logo"),
        )
        
        # 5. Calcular scores
        scores = None
        if api_data and api_data.get("scores"):
            scores = stats_calculator.calculate_scores(
                api_data["scores"],
                api_data
            )
        elif match.get("resultado_marcador"):
            scores = stats_calculator.parse_score_string(match["resultado_marcador"])
        
        # Intentar obtener de match_sets si no hay scores
        if not scores or not scores.sets:
            try:
                sets_db = db.get_match_sets(match_id)
                if sets_db:
                    from src.api.models_match_detail import SetScore
                    sets = []
                    p1_sets = 0
                    p2_sets = 0
                    for s in sets_db:
                        p1 = s.get("player1_score", 0)
                        p2 = s.get("player2_score", 0)
                        winner = 1 if p1 > p2 else 2 if p2 > p1 else None
                        if winner == 1:
                            p1_sets += 1
                        elif winner == 2:
                            p2_sets += 1
                        sets.append(SetScore(
                            set_number=s.get("set_number", len(sets) + 1),
                            player1_games=p1,
                            player2_games=p2,
                            tiebreak_score=s.get("tiebreak_score"),
                            winner=winner
                        ))
                    if sets:
                        scores = MatchScores(sets_won=[p1_sets, p2_sets], sets=sets)
            except Exception as e:
                logger.warning(f"Error obteniendo sets de BD: {e}")
        
        # 6. Calcular estadísticas
        stats = None
        timeline = None
        
        if api_data and api_data.get("pointbypoint"):
            pointbypoint = api_data["pointbypoint"]
            stats = stats_calculator.calculate_stats(pointbypoint, scores)
            timeline = stats_calculator.calculate_timeline(pointbypoint)
            
            # Guardar datos para caché (si no existen)
            _save_match_data_to_db(db, match_id, api_data)
        else:
            # Intentar cargar de BD
            stats, timeline = _load_stats_from_db(db, match_id)
        
        # 7. Obtener H2H resumido
        h2h = await _get_h2h_summary(db, api_client, match)
        
        # 8. Obtener cuotas
        odds = _get_match_odds(db, match_id, match)
        
        # 9. Obtener predicción
        prediction = _get_prediction(match)
        
        # 10. Determinar ganador
        winner = None
        ganador = match.get("resultado_ganador")
        if ganador:
            if ganador == match.get("jugador1"):
                winner = 1
            elif ganador == match.get("jugador2"):
                winner = 2
        
        # 11. Determinar calidad de datos
        data_quality = "basic"
        if stats and stats.has_detailed_stats:
            data_quality = "full"
        elif scores and scores.sets:
            data_quality = "partial"
        
        return MatchFullResponse(
            match=match_info,
            player1=player1,
            player2=player2,
            winner=winner,
            scores=scores,
            stats=stats,
            timeline=timeline,
            h2h=h2h,
            odds=odds,
            prediction=prediction,
            last_updated=datetime.now().isoformat(),
            data_quality=data_quality,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo datos del partido {match_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# ENDPOINT: TIMELINE
# ============================================================

@router.get("/{match_id}/timeline", response_model=MatchTimeline)
async def get_match_timeline(match_id: int):
    """
    Obtiene el timeline de juegos del partido.
    
    Incluye:
    - Juegos agrupados por set
    - Indicadores de break
    - Score progresivo
    - Estadísticas de breaks por set
    """
    db = get_db()
    api_client = get_api_client()
    
    try:
        match = db.get_match(match_id)
        if not match:
            raise HTTPException(status_code=404, detail="Partido no encontrado")
        
        # Intentar obtener de API
        event_key = match.get("event_key")
        if event_key:
            try:
                params = {"match_key": event_key}
                response = api_client._make_request("get_fixtures", params)
                
                if response and response.get("result"):
                    results = response["result"]
                    api_data = results[0] if isinstance(results, list) else results
                    
                    if api_data.get("pointbypoint"):
                        return stats_calculator.calculate_timeline(api_data["pointbypoint"])
            except Exception as e:
                logger.warning(f"Error obteniendo timeline de API: {e}")
        
        # Intentar cargar de BD
        _, timeline = _load_stats_from_db(db, match_id)
        if timeline and timeline.total_games > 0:
            return timeline
        
        return MatchTimeline()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo timeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# ENDPOINT: POINT BY POINT
# ============================================================

@router.get("/{match_id}/pbp", response_model=PointByPointData)
async def get_point_by_point(
    match_id: int,
    set_number: Optional[int] = Query(None, description="Filtrar por número de set")
):
    """
    Obtiene los datos punto por punto del partido.
    
    Args:
        match_id: ID del partido
        set_number: Filtrar por set específico (opcional)
    
    Returns:
        Puntos con indicadores de BP/SP/MP
    """
    db = get_db()
    api_client = get_api_client()
    
    try:
        match = db.get_match(match_id)
        if not match:
            raise HTTPException(status_code=404, detail="Partido no encontrado")
        
        # Intentar obtener de API
        event_key = match.get("event_key")
        if event_key:
            try:
                params = {"match_key": event_key}
                response = api_client._make_request("get_fixtures", params)
                
                if response and response.get("result"):
                    results = response["result"]
                    api_data = results[0] if isinstance(results, list) else results
                    
                    if api_data.get("pointbypoint"):
                        return stats_calculator.extract_point_by_point(
                            api_data["pointbypoint"],
                            set_filter=set_number
                        )
            except Exception as e:
                logger.warning(f"Error obteniendo PBP de API: {e}")
        
        return PointByPointData()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo PBP: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def _save_match_data_to_db(db, match_id: int, api_data: dict):
    """Guarda datos del partido en la BD para caché"""
    try:
        # Guardar scores
        scores = api_data.get("scores", [])
        if scores:
            sets_data = []
            for score in scores:
                sets_data.append({
                    "set_number": int(score.get("score_set", 0)),
                    "player1_score": int(score.get("score_first", 0)),
                    "player2_score": int(score.get("score_second", 0)),
                })
            db.save_match_sets(match_id, sets_data)
        
        # TODO: Guardar games y points si es necesario
        
    except Exception as e:
        logger.warning(f"Error guardando datos en BD: {e}")


def _load_stats_from_db(db, match_id: int):
    """Carga estadísticas desde la BD"""
    stats = None
    timeline = None
    
    try:
        # Cargar games
        from src.api.api_v2 import pbp_service
        if pbp_service:
            games = pbp_service.get_games(match_id)
            points = pbp_service.get_point_by_point(match_id)
            
            if games:
                # Construir timeline desde games de BD
                # (simplificado - se podría mejorar)
                pass
                
    except Exception as e:
        logger.warning(f"Error cargando stats de BD: {e}")
    
    return stats, timeline


async def _get_h2h_summary(db, api_client, match: dict) -> Optional[H2HData]:
    """Obtiene resumen de H2H"""
    try:
        p1_key = match.get("jugador1_key")
        p2_key = match.get("jugador2_key")
        
        if not p1_key or not p2_key:
            return None
        
        # Llamar a API Tennis para H2H
        params = {
            "first_player_key": p1_key,
            "second_player_key": p2_key
        }
        response = api_client._make_request("get_H2H", params)
        
        if not response or not response.get("result"):
            return None
        
        result = response["result"]
        h2h_matches = result.get("H2H", [])
        
        p1_wins = 0
        p2_wins = 0
        matches = []
        
        for m in h2h_matches[:5]:  # Últimos 5
            winner_str = m.get("event_winner", "")
            if "First" in winner_str:
                p1_wins += 1
                winner = 1
            else:
                p2_wins += 1
                winner = 2
            
            matches.append(PreviousMatch(
                date=m.get("event_date", ""),
                tournament=m.get("tournament_name", ""),
                surface=m.get("tournament_surface", "Hard"),
                winner=winner,
                score=m.get("event_final_result", "")
            ))
        
        return H2HData(
            total_matches=len(h2h_matches),
            player1_wins=p1_wins,
            player2_wins=p2_wins,
            matches=matches
        )
        
    except Exception as e:
        logger.warning(f"Error obteniendo H2H: {e}")
        return None


def _get_match_odds(db, match_id: int, match: dict) -> Optional[MatchOdds]:
    """Obtiene cuotas del partido"""
    try:
        cuota1 = match.get("cuota_jugador1")
        cuota2 = match.get("cuota_jugador2")
        
        if not cuota1 and not cuota2:
            return None
        
        # Determinar favorito
        market_consensus = None
        if cuota1 and cuota2:
            market_consensus = 1 if cuota1 < cuota2 else 2
        
        bookmakers = []
        
        # Si hay cuotas básicas, añadir como "General"
        if cuota1 or cuota2:
            bookmakers.append(BookmakerOdds(
                bookmaker="General",
                player1_odds=float(cuota1) if cuota1 else 0,
                player2_odds=float(cuota2) if cuota2 else 0,
            ))
        
        # TODO: Cargar cuotas de múltiples casas desde odds_history
        
        return MatchOdds(
            best_odds_player1=float(cuota1) if cuota1 else None,
            best_odds_player2=float(cuota2) if cuota2 else None,
            bookmakers=bookmakers,
            market_consensus=market_consensus
        )
        
    except Exception as e:
        logger.warning(f"Error obteniendo odds: {e}")
        return None


def _get_prediction(match: dict) -> Optional[MatchPrediction]:
    """Obtiene predicción del partido"""
    try:
        prob1 = match.get("probabilidad_jugador1")
        prob2 = match.get("probabilidad_jugador2")
        confianza = match.get("confianza")
        
        if not prob1 and not prob2:
            return None
        
        prob1 = float(prob1) if prob1 else 0.5
        prob2 = float(prob2) if prob2 else 0.5
        
        predicted_winner = 1 if prob1 > prob2 else 2
        confidence = float(confianza) if confianza else max(prob1, prob2) * 100
        
        # Calcular value bet
        value_bet = None
        cuota1 = match.get("cuota_jugador1")
        cuota2 = match.get("cuota_jugador2")
        
        if cuota1 and cuota2:
            ev1 = prob1 * float(cuota1) - 1
            ev2 = prob2 * float(cuota2) - 1
            if ev1 > 0.05:  # 5% de edge mínimo
                value_bet = 1
            elif ev2 > 0.05:
                value_bet = 2
        
        return MatchPrediction(
            predicted_winner=predicted_winner,
            confidence=confidence,
            probability_player1=prob1 * 100,
            probability_player2=prob2 * 100,
            value_bet=value_bet,
        )
        
    except Exception as e:
        logger.warning(f"Error obteniendo predicción: {e}")
        return None
