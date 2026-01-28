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
from typing import Any, Optional

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
    SetScore,
    Surface,
)
from src.services.match_stats_calculator import MatchStatsCalculator

logger = logging.getLogger(__name__)

# Router para endpoints de detalle
router = APIRouter(prefix="/v2/matches", tags=["Match Detail v2"])

# Instancia del calculador de estadísticas
stats_calculator = MatchStatsCalculator()

# Referencias a db y api_client (se configuran al registrar el router)
_db: Any = None
_api_client: Any = None


def configure_dependencies(db, api_client):
    """Configura las dependencias necesarias para los endpoints"""
    global _db, _api_client
    _db = db
    _api_client = api_client
    logger.info("✅ Dependencias de routes_match_detail configuradas")


def get_db():
    """Obtiene la instancia de la base de datos"""
    if _db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    return _db


def get_api_client():
    """Obtiene el cliente de API Tennis"""
    if _api_client is None:
        raise HTTPException(status_code=500, detail="API client not configured")
    return _api_client


# ============================================================
# ENDPOINT PRINCIPAL: /matches/{id}/full
# ============================================================

@router.get("/{match_id}/full", response_model=MatchFullResponse, response_model_by_alias=True)
async def get_match_full(match_id: int):
    """
    Obtiene todos los datos de un partido en una sola llamada.
    """
    logger.info(f"🎾 GET /v2/matches/{match_id}/full - Iniciando")
    
    try:
        db = get_db()
        logger.info(f"✅ DB obtenida: {type(db)}")
    except Exception as e:
        logger.error(f"❌ Error obteniendo DB: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error DB: {str(e)}")
    
    try:
        api_client = get_api_client()
        logger.info(f"✅ API Client obtenido: {type(api_client)}")
    except Exception as e:
        logger.error(f"❌ Error obteniendo API Client: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error API Client: {str(e)}")
    
    try:
        # 1. Obtener partido de la BD (usando vista con predicciones)
        logger.info(f"📥 Obteniendo partido {match_id} de BD...")
        # Usar la vista que incluye predicciones
        match = db._fetchone(
            "SELECT * FROM matches_with_latest_prediction WHERE id = :id",
            {"id": match_id}
        )
        if not match:
            # Fallback a tabla matches si no está en la vista
            match = db.get_match(match_id)
        if not match:
            logger.warning(f"⚠️ Partido {match_id} no encontrado")
            raise HTTPException(status_code=404, detail="Partido no encontrado")
        
        # Log de campos disponibles para debug
        logger.info(f"✅ Partido encontrado: {match.get('jugador1_nombre')} vs {match.get('jugador2_nombre')}")
        logger.info(f"📋 Campos disponibles: {list(match.keys())[:15]}...")
        
        # 2. Obtener datos adicionales de la API Tennis si hay event_key
        api_data = None
        event_key = match.get("event_key")
        logger.info(f"📡 Event key: {event_key}")
        
        if event_key:
            try:
                params = {"match_key": event_key}
                response = api_client._make_request("get_fixtures", params)
                
                if response and response.get("result"):
                    results = response["result"]
                    if isinstance(results, list) and results:
                        api_data = results[0]
                    elif isinstance(results, dict):
                        api_data = results
                    logger.info(f"✅ Datos API Tennis obtenidos")
            except Exception as e:
                logger.warning(f"⚠️ No se pudo obtener datos de API Tennis: {e}")
        
        # 3. Construir información del partido
        logger.info(f"🔨 Construyendo MatchInfo...")
        estado = match.get("estado", "pendiente")
        if estado not in ["pendiente", "en_juego", "completado", "suspendido", "cancelado"]:
            estado = "pendiente"
        
        superficie_raw = match.get("superficie", "Hard")
        superficie_map = {
            "hard": "Hard", "Hard": "Hard", "dura": "Hard",
            "clay": "Clay", "Clay": "Clay", "tierra": "Clay", "arcilla": "Clay",
            "grass": "Grass", "Grass": "Grass", "hierba": "Grass",
            "carpet": "Carpet", "Carpet": "Carpet", "moqueta": "Carpet",
            "indoor": "Indoor", "Indoor": "Indoor",
        }
        superficie = superficie_map.get(superficie_raw, "Hard")
        logger.info(f"📊 Estado: {estado}, Superficie: {superficie}")
        
        # Obtener fecha de forma segura
        fecha_partido = match.get("fecha_partido")
        logger.info(f"📅 Fecha partido (raw): {fecha_partido}, type: {type(fecha_partido)}")
        
        match_info = MatchInfo(
            id=match_id,
            status=MatchStatus(estado),
            date=fecha_partido,
            time=match.get("hora_inicio"),
            tournament=match.get("torneo", "Unknown"),
            round=match.get("ronda"),
            surface=Surface(superficie),
        )
        logger.info(f"✅ MatchInfo creado")
        
        # 4. Construir información de jugadores
        logger.info(f"👤 Construyendo PlayerInfo...")
        
        # Obtener nombres
        j1_nombre = match.get("jugador1_nombre") or match.get("jugador1") or "Player 1"
        j2_nombre = match.get("jugador2_nombre") or match.get("jugador2") or "Player 2"
        
        # Intentar obtener rankings actualizados de la tabla players
        j1_ranking = match.get("jugador1_ranking")
        j2_ranking = match.get("jugador2_ranking")
        j1_key = match.get("jugador1_key")
        j2_key = match.get("jugador2_key")
        
        # Buscar ranking actual en tabla players si tenemos player_key
        try:
            if j1_key:
                player1_data = db._fetchone(
                    "SELECT atp_ranking, country, player_logo FROM players WHERE player_key = :key",
                    {"key": j1_key}
                )
                if player1_data and player1_data.get("atp_ranking"):
                    j1_ranking = player1_data.get("atp_ranking")
                    logger.info(f"📊 Ranking actualizado J1: {j1_ranking}")
            
            if j2_key:
                player2_data = db._fetchone(
                    "SELECT atp_ranking, country, player_logo FROM players WHERE player_key = :key",
                    {"key": j2_key}
                )
                if player2_data and player2_data.get("atp_ranking"):
                    j2_ranking = player2_data.get("atp_ranking")
                    logger.info(f"📊 Ranking actualizado J2: {j2_ranking}")
        except Exception as e:
            logger.warning(f"⚠️ Error obteniendo rankings de players: {e}")
        
        player1 = PlayerInfo(
            name=j1_nombre,
            country=match.get("jugador1_pais"),
            ranking=j1_ranking,
            logo_url=match.get("jugador1_logo"),
        )
        
        player2 = PlayerInfo(
            name=j2_nombre,
            country=match.get("jugador2_pais"),
            ranking=j2_ranking,
            logo_url=match.get("jugador2_logo"),
        )
        logger.info(f"✅ PlayerInfo creados: {player1.name} (#{j1_ranking}) vs {player2.name} (#{j2_ranking})")
        
        # 5. Calcular scores
        logger.info(f"🎯 Calculando scores...")
        scores = None
        try:
            if api_data and api_data.get("scores"):
                scores = stats_calculator.calculate_scores(api_data["scores"], api_data)
                logger.info(f"✅ Scores de API: {scores}")
            elif match.get("resultado_marcador"):
                scores = stats_calculator.parse_score_string(match["resultado_marcador"])
                logger.info(f"✅ Scores parseados: {scores}")
        except Exception as e:
            logger.warning(f"⚠️ Error calculando scores: {e}")
        
        # Intentar obtener de match_sets si no hay scores
        if not scores or not scores.sets:
            try:
                if hasattr(db, 'get_match_sets'):
                    sets_db = db.get_match_sets(match_id)
                    if sets_db:
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
                            logger.info(f"✅ Scores de BD: {scores}")
            except Exception as e:
                logger.warning(f"⚠️ Error obteniendo sets de BD: {e}")
        
        # 6. Calcular estadísticas y timeline desde pointbypoint
        logger.info(f"📊 Calculando estadísticas...")
        stats = None
        timeline = None
        
        if api_data and api_data.get("pointbypoint"):
            pbp_data = api_data["pointbypoint"]
            logger.info(f"📊 Datos pointbypoint encontrados: {len(pbp_data)} juegos")
            try:
                stats = stats_calculator.calculate_stats(pbp_data, scores)
                logger.info(f"✅ Stats calculadas: has_detailed_stats={stats.has_detailed_stats if stats else False}")
            except Exception as e:
                logger.warning(f"⚠️ Error calculando stats: {e}")
            
            try:
                timeline = stats_calculator.calculate_timeline(pbp_data)
                logger.info(f"✅ Timeline calculado: {timeline.total_games if timeline else 0} juegos")
            except Exception as e:
                logger.warning(f"⚠️ Error calculando timeline: {e}")
        else:
            logger.info(f"ℹ️ No hay datos pointbypoint disponibles")
        
        # 7. Obtener cuotas (simple, desde la BD)
        logger.info(f"💰 Obteniendo cuotas...")
        odds = _get_match_odds(db, match_id, match)
        
        # 8. Obtener predicción (desde la BD)
        logger.info(f"🤖 Obteniendo predicción...")
        prediction = _get_prediction(match)
        
        # 9. Obtener H2H si tenemos las keys de los jugadores
        logger.info(f"🤝 Obteniendo H2H...")
        h2h = None
        try:
            h2h = await _get_h2h_summary(db, api_client, match)
            if h2h:
                logger.info(f"✅ H2H obtenido: {h2h.total_matches} partidos")
            else:
                logger.info(f"ℹ️ No hay datos H2H")
        except Exception as e:
            logger.warning(f"⚠️ Error obteniendo H2H: {e}")
        
        # 10. Determinar ganador
        winner = None
        ganador = match.get("resultado_ganador")
        j1_nombre = match.get("jugador1_nombre") or match.get("jugador1")
        j2_nombre = match.get("jugador2_nombre") or match.get("jugador2")
        if ganador:
            if ganador == j1_nombre:
                winner = 1
            elif ganador == j2_nombre:
                winner = 2
        logger.info(f"🏆 Winner: {winner} (ganador: {ganador})")
        
        # 11. Determinar calidad de datos
        data_quality = "basic"
        if stats and stats.has_detailed_stats:
            data_quality = "full"
        elif scores and scores.sets:
            data_quality = "partial"
        
        logger.info(f"📦 Construyendo respuesta final...")
        response = MatchFullResponse(
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
        logger.info(f"✅ Respuesta construida exitosamente")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ ERROR en get_match_full({match_id}): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")
        
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
            logger.info(f"⚠️ No hay player keys para H2H: p1={p1_key}, p2={p2_key}")
            return None
        
        # Llamar a API Tennis para H2H
        params = {
            "first_player_key": p1_key,
            "second_player_key": p2_key
        }
        response = api_client._make_request("get_H2H", params)
        
        if not response or not response.get("result"):
            logger.info(f"ℹ️ No hay datos H2H en la API")
            return None
        
        result = response["result"]
        h2h_matches = result.get("H2H", [])
        
        if not h2h_matches:
            logger.info(f"ℹ️ No hay enfrentamientos previos")
            return None
        
        p1_wins = 0
        p2_wins = 0
        matches = []
        
        # Records por superficie
        hard_p1, hard_p2 = 0, 0
        clay_p1, clay_p2 = 0, 0
        grass_p1, grass_p2 = 0, 0
        
        for m in h2h_matches:
            winner_str = m.get("event_winner", "")
            surface = (m.get("tournament_surface") or "Hard").lower()
            
            if "First" in winner_str:
                p1_wins += 1
                winner = 1
                if "hard" in surface:
                    hard_p1 += 1
                elif "clay" in surface:
                    clay_p1 += 1
                elif "grass" in surface:
                    grass_p1 += 1
            else:
                p2_wins += 1
                winner = 2
                if "hard" in surface:
                    hard_p2 += 1
                elif "clay" in surface:
                    clay_p2 += 1
                elif "grass" in surface:
                    grass_p2 += 1
            
            # Solo los últimos 5 para mostrar
            if len(matches) < 5:
                matches.append(PreviousMatch(
                    date=m.get("event_date", ""),
                    tournament=m.get("tournament_name", "Unknown"),
                    surface=m.get("tournament_surface", "Hard"),
                    winner=winner,
                    score=m.get("event_final_result", "")
                ))
        
        logger.info(f"✅ H2H encontrado: {p1_wins}-{p2_wins} ({len(h2h_matches)} partidos)")
        
        return H2HData(
            total_matches=len(h2h_matches),
            player1_wins=p1_wins,
            player2_wins=p2_wins,
            matches=matches,
            hard_record=[hard_p1, hard_p2],
            clay_record=[clay_p1, clay_p2],
            grass_record=[grass_p1, grass_p2]
        )
        
    except Exception as e:
        logger.warning(f"Error obteniendo H2H: {e}")
        return None


def _get_match_odds(db, match_id: int, match: dict) -> Optional[MatchOdds]:
    """Obtiene cuotas del partido"""
    try:
        # Campos pueden venir de la vista (jugador1_cuota) o de otro lugar
        cuota1 = match.get("jugador1_cuota") or match.get("cuota_jugador1")
        cuota2 = match.get("jugador2_cuota") or match.get("cuota_jugador2")
        
        if not cuota1 and not cuota2:
            logger.info(f"💰 No hay cuotas para partido {match_id}")
            return None
        
        # Determinar favorito
        market_consensus = None
        if cuota1 and cuota2:
            market_consensus = 1 if float(cuota1) < float(cuota2) else 2
        
        bookmakers = []
        
        # Si hay cuotas básicas, añadir como "General"
        if cuota1 or cuota2:
            bookmakers.append(BookmakerOdds(
                bookmaker="General",
                player1_odds=float(cuota1) if cuota1 else 0,
                player2_odds=float(cuota2) if cuota2 else 0,
            ))
        
        logger.info(f"💰 Cuotas: {cuota1} vs {cuota2}")
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
        # Campos pueden venir con diferentes nombres
        prob1 = match.get("jugador1_probabilidad") or match.get("probabilidad_jugador1")
        prob2 = match.get("jugador2_probabilidad") or match.get("probabilidad_jugador2")
        confianza = match.get("confidence_score") or match.get("confianza")
        
        if not prob1 and not prob2:
            logger.info(f"🤖 No hay predicción para este partido")
            return None
        
        prob1 = float(prob1) if prob1 else 0.5
        prob2 = float(prob2) if prob2 else 0.5
        
        predicted_winner = 1 if prob1 > prob2 else 2
        confidence = float(confianza) if confianza else max(prob1, prob2) * 100
        
        # Calcular value bet
        value_bet = None
        cuota1 = match.get("jugador1_cuota") or match.get("cuota_jugador1")
        cuota2 = match.get("jugador2_cuota") or match.get("cuota_jugador2")
        
        if cuota1 and cuota2:
            ev1 = prob1 * float(cuota1) - 1
            ev2 = prob2 * float(cuota2) - 1
            if ev1 > 0.05:  # 5% de edge mínimo
                value_bet = 1
            elif ev2 > 0.05:
                value_bet = 2
        
        logger.info(f"🤖 Predicción: {prob1:.0%} vs {prob2:.0%}, confianza: {confidence:.0f}%")
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
