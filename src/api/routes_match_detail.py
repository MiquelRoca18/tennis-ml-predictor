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
    
    OPTIMIZADO: Solo lee de la BD para respuesta rápida (<100ms).
    Los datos se sincronizan en background por los schedulers.
    """
    db = get_db()
    
    try:
        # 1. Obtener partido de la BD (usando vista con predicciones)
        match = db._fetchone(
            "SELECT * FROM matches_with_latest_prediction WHERE id = :id",
            {"id": match_id}
        )
        if not match:
            match = db.get_match(match_id)
        if not match:
            raise HTTPException(status_code=404, detail="Partido no encontrado")
        
        # 2. Construir información del partido
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
        
        match_info = MatchInfo(
            id=match_id,
            status=MatchStatus(estado),
            date=match.get("fecha_partido"),
            time=match.get("hora_inicio"),
            tournament=match.get("torneo", "Unknown"),
            round=match.get("ronda"),
            surface=Surface(superficie),
        )
        
        # 3. Construir información de jugadores (de BD, rápido)
        j1_nombre = match.get("jugador1_nombre") or match.get("jugador1") or "Player 1"
        j2_nombre = match.get("jugador2_nombre") or match.get("jugador2") or "Player 2"
        j1_ranking = match.get("jugador1_ranking")
        j2_ranking = match.get("jugador2_ranking")
        j1_key = match.get("jugador1_key")
        j2_key = match.get("jugador2_key")
        
        # Buscar ranking actual en tabla players (consulta rápida a BD local)
        if j1_key:
            player1_data = db._fetchone(
                "SELECT atp_ranking, country, player_logo FROM players WHERE player_key = :key",
                {"key": j1_key}
            )
            if player1_data and player1_data.get("atp_ranking"):
                j1_ranking = player1_data.get("atp_ranking")
        
        if j2_key:
            player2_data = db._fetchone(
                "SELECT atp_ranking, country, player_logo FROM players WHERE player_key = :key",
                {"key": j2_key}
            )
            if player2_data and player2_data.get("atp_ranking"):
                j2_ranking = player2_data.get("atp_ranking")
        
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
        
        # 4. Obtener scores de la BD (rápido)
        scores = None
        
        # PRIORIDAD 1: Intentar obtener de match_sets (más fiable)
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
                        logger.debug(f"✅ Scores de match_sets: {p1_sets}-{p2_sets}")
        except Exception as e:
            logger.debug(f"Error obteniendo match_sets: {e}")
        
        # PRIORIDAD 2: Fallback a resultado_marcador (parsear string)
        if not scores or not scores.sets:
            marcador = match.get("resultado_marcador")
            if marcador and "-" in marcador and any(c.isdigit() for c in marcador):
                # Solo parsear si parece un marcador válido (ej: "6-4, 7-5" no "2 - 1")
                # El formato "2 - 1" es resultado en sets, no juegos
                if "," in marcador or len(marcador.split()) > 2:
                    try:
                        scores = stats_calculator.parse_score_string(marcador)
                        logger.debug(f"✅ Scores parseados de resultado_marcador")
                    except Exception as e:
                        logger.debug(f"Error parseando marcador: {e}")
        
        # PRIORIDAD 3: Usar event_final_result para sets_won si no hay scores detallados
        if not scores or not scores.sets:
            event_final_result = match.get("event_final_result")
            if event_final_result and "-" in event_final_result:
                try:
                    parts = event_final_result.replace(" ", "").split("-")
                    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                        p1_sets = int(parts[0])
                        p2_sets = int(parts[1])
                        scores = MatchScores(sets_won=[p1_sets, p2_sets], sets=[])
                        logger.debug(f"✅ Sets_won de event_final_result: {p1_sets}-{p2_sets}")
                except Exception:
                    pass
        
        # 5. Obtener estadísticas y timeline de la BD (si existen pre-calculadas)
        stats, timeline = _load_stats_from_db(db, match_id)
        
        # 6. Obtener cuotas de la BD (rápido)
        odds = _get_match_odds(db, match_id, match)
        
        # 7. Obtener predicción de la BD (ya está en el match)
        prediction = _get_prediction(match)
        
        # 8. Obtener H2H de la BD (tabla head_to_head si existe)
        h2h = _get_h2h_from_db(db, match)
        
        # 9. Determinar ganador
        winner = None
        ganador = match.get("resultado_ganador")
        if ganador:
            if ganador == j1_nombre:
                winner = 1
            elif ganador == j2_nombre:
                winner = 2
        
        # 10. Determinar calidad de datos
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
        logger.error(f"Error en get_match_full({match_id}): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# ENDPOINT: TIMELINE
# ============================================================

@router.get("/{match_id}/timeline")
async def get_match_timeline(match_id: int):
    """
    Obtiene el timeline de juegos del partido.
    
    Estrategia lazy loading:
    1. Si hay datos en BD → Devuelve instantáneo
    2. Si NO hay datos → Llama a API, guarda en BD, devuelve
    """
    db = get_db()
    api_client = get_api_client()
    
    try:
        match = db.get_match(match_id)
        if not match:
            raise HTTPException(status_code=404, detail="Partido no encontrado")
        
        # 1. Intentar cargar de BD
        _, timeline = _load_stats_from_db(db, match_id)
        if timeline and timeline.total_games > 0:
            return timeline
        
        # 2. No hay datos - lazy loading desde API
        event_key = match.get("event_key")
        if not event_key:
            return MatchTimeline()
        
        try:
            response = api_client._make_request("get_fixtures", {"match_key": event_key})
            if response and response.get("result"):
                results = response["result"]
                api_data = results[0] if isinstance(results, list) else results
                
                if api_data.get("pointbypoint"):
                    # Guardar en BD para caché
                    _save_pointbypoint_to_db(db, match_id, api_data["pointbypoint"])
                    
                    timeline = stats_calculator.calculate_timeline(api_data["pointbypoint"])
                    return timeline
        except Exception as e:
            logger.warning(f"Error obteniendo timeline de API: {e}")
        
        return MatchTimeline()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo timeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# ENDPOINT: STATS (Estadísticas del partido)
# ============================================================

@router.get("/{match_id}/stats")
async def get_match_stats(match_id: int):
    """
    Obtiene las estadísticas detalladas del partido.
    
    Estrategia lazy loading:
    1. Si hay datos en BD → Devuelve instantáneo
    2. Si NO hay datos → Llama a API, guarda en BD, devuelve
    """
    db = get_db()
    api_client = get_api_client()
    
    try:
        match = db.get_match(match_id)
        if not match:
            raise HTTPException(status_code=404, detail="Partido no encontrado")
        
        # 1. Intentar cargar de BD
        stats, _ = _load_stats_from_db(db, match_id)
        if stats and stats.has_detailed_stats:
            return stats
        
        # 2. No hay datos - lazy loading desde API
        event_key = match.get("event_key")
        if not event_key:
            return {"has_detailed_stats": False, "message": "No hay estadísticas disponibles"}
        
        try:
            response = api_client._make_request("get_fixtures", {"match_key": event_key})
            if response and response.get("result"):
                results = response["result"]
                api_data = results[0] if isinstance(results, list) else results
                
                if api_data.get("pointbypoint"):
                    # Guardar en BD para caché
                    _save_pointbypoint_to_db(db, match_id, api_data["pointbypoint"])
                    
                    # Calcular scores primero
                    scores = None
                    if api_data.get("scores"):
                        scores = stats_calculator.calculate_scores(api_data["scores"], api_data)
                    
                    stats = stats_calculator.calculate_stats(api_data["pointbypoint"], scores)
                    if stats:
                        return stats
        except Exception as e:
            logger.warning(f"Error obteniendo stats de API: {e}")
        
        return {"has_detailed_stats": False, "message": "No hay estadísticas disponibles"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo stats: {e}")
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
    Lazy loading con caché.
    """
    db = get_db()
    api_client = get_api_client()
    
    try:
        match = db.get_match(match_id)
        if not match:
            raise HTTPException(status_code=404, detail="Partido no encontrado")
        
        # Intentar obtener de API con lazy loading
        event_key = match.get("event_key")
        if event_key:
            try:
                response = api_client._make_request("get_fixtures", {"match_key": event_key})
                if response and response.get("result"):
                    results = response["result"]
                    api_data = results[0] if isinstance(results, list) else results
                    
                    if api_data.get("pointbypoint"):
                        return stats_calculator.extract_point_by_point(
                            api_data["pointbypoint"],
                            set_filter=set_number
                        )
            except Exception as e:
                logger.warning(f"Error obteniendo PBP: {e}")
        
        return PointByPointData()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo PBP: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# ENDPOINT: ODDS (Cuotas detalladas de bookmakers)
# ============================================================

@router.get("/{match_id}/odds")
async def get_match_odds_detailed(match_id: int):
    """
    Obtiene las cuotas detalladas de todas las casas de apuestas.
    
    Estrategia "lazy loading":
    1. Si hay datos en BD → Devuelve instantáneo
    2. Si NO hay datos en BD → Llama a API, guarda en BD, devuelve
    3. Siguientes peticiones → Instantáneo desde BD
    """
    db = get_db()
    
    try:
        match = db.get_match(match_id)
        if not match:
            raise HTTPException(status_code=404, detail="Partido no encontrado")
        
        p1_name = match.get("jugador1_nombre") or match.get("jugador1")
        p2_name = match.get("jugador2_nombre") or match.get("jugador2")
        event_key = match.get("event_key")
        
        # 1. Intentar obtener de la BD (instantáneo)
        odds_from_db = _get_detailed_odds_from_db(db, match_id, match)
        if odds_from_db and odds_from_db.get("bookmakers") and len(odds_from_db["bookmakers"]) > 1:
            # Ya tenemos datos de múltiples bookmakers en BD
            return odds_from_db
        
        # 2. No hay datos completos en BD - llamar a API (lazy loading)
        if not event_key:
            return odds_from_db or {
                "success": True,
                "message": "No hay cuotas disponibles",
                "player1_name": p1_name,
                "player2_name": p2_name,
                "bookmakers": []
            }
        
        api_client = get_api_client()
        
        try:
            response = api_client._make_request("get_odds", {"match_key": event_key})
        except Exception as e:
            logger.warning(f"Error llamando API odds: {e}")
            return odds_from_db or {
                "success": True,
                "message": "No hay cuotas disponibles",
                "player1_name": p1_name,
                "player2_name": p2_name,
                "bookmakers": []
            }
        
        if not response or not response.get("result"):
            return odds_from_db or {
                "success": True,
                "message": "No hay cuotas disponibles",
                "player1_name": p1_name,
                "player2_name": p2_name,
                "bookmakers": []
            }
        
        result = response["result"]
        match_odds = result.get(str(event_key), {})
        
        # Extraer cuotas Home/Away
        home_away = match_odds.get("Home/Away", {})
        home_odds = home_away.get("Home", {})
        away_odds = home_away.get("Away", {})
        
        bookmakers_list = []
        all_bookmakers = set(home_odds.keys()) | set(away_odds.keys())
        
        for bookmaker in all_bookmakers:
            p1_odds = home_odds.get(bookmaker)
            p2_odds = away_odds.get(bookmaker)
            
            if p1_odds or p2_odds:
                p1_float = float(p1_odds) if p1_odds else None
                p2_float = float(p2_odds) if p2_odds else None
                
                bookmakers_list.append({
                    "bookmaker": bookmaker,
                    "player1_odds": p1_float,
                    "player2_odds": p2_float,
                })
                
                # Guardar en BD para caché (lazy loading)
                try:
                    db._execute(
                        """
                        INSERT INTO odds_history (match_id, bookmaker, odds_player1, odds_player2, created_at)
                        VALUES (:match_id, :bookmaker, :p1, :p2, CURRENT_TIMESTAMP)
                        """,
                        {"match_id": match_id, "bookmaker": bookmaker, "p1": p1_float, "p2": p2_float}
                    )
                except Exception:
                    pass  # Ignorar errores de inserción duplicada
        
        bookmakers_list.sort(
            key=lambda x: (x["player1_odds"] or 0, x["player2_odds"] or 0),
            reverse=True
        )
        
        best_p1 = max([b["player1_odds"] for b in bookmakers_list if b["player1_odds"]], default=None)
        best_p2 = max([b["player2_odds"] for b in bookmakers_list if b["player2_odds"]], default=None)
        
        return {
            "success": True,
            "player1_name": p1_name,
            "player2_name": p2_name,
            "best_odds_player1": best_p1,
            "best_odds_player2": best_p2,
            "bookmakers": bookmakers_list,
            "total_bookmakers": len(bookmakers_list)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo cuotas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# ENDPOINT: H2H (Head to Head)
# ============================================================

@router.get("/{match_id}/h2h")
async def get_match_h2h(match_id: int):
    """
    Obtiene el historial de enfrentamientos entre los jugadores.
    
    Estrategia "lazy loading":
    1. Si hay datos en BD → Devuelve instantáneo
    2. Si NO hay datos en BD → Llama a API, guarda en BD, devuelve
    """
    db = get_db()
    api_client = get_api_client()
    
    try:
        match = db.get_match(match_id)
        if not match:
            raise HTTPException(status_code=404, detail="Partido no encontrado")
        
        p1_key = match.get("jugador1_key")
        p2_key = match.get("jugador2_key")
        p1_name = match.get("jugador1_nombre") or match.get("jugador1")
        p2_name = match.get("jugador2_nombre") or match.get("jugador2")
        
        if not p1_key or not p2_key:
            return {
                "success": True,
                "message": "No hay datos de jugadores",
                "total_matches": 0,
                "player1_wins": 0,
                "player2_wins": 0,
                "recent_matches": []
            }
        
        # 1. Intentar obtener de BD (instantáneo)
        h2h_from_db = _get_h2h_from_db(db, match)
        if h2h_from_db and h2h_from_db.total_matches > 0:
            return {
                "success": True,
                "total_matches": h2h_from_db.total_matches,
                "player1_wins": h2h_from_db.player1_wins,
                "player2_wins": h2h_from_db.player2_wins,
                "surface_records": h2h_from_db.surface_records,
                "recent_matches": [m.model_dump() for m in h2h_from_db.recent_matches] if h2h_from_db.recent_matches else []
            }
        
        # 2. No hay datos en BD - llamar a API (lazy loading)
        try:
            response = api_client._make_request("get_H2H", {
                "first_player_key": p1_key,
                "second_player_key": p2_key
            })
        except Exception as e:
            logger.warning(f"Error llamando API H2H: {e}")
            return {
                "success": True,
                "message": "No hay datos de H2H disponibles",
                "total_matches": 0,
                "player1_wins": 0,
                "player2_wins": 0,
                "recent_matches": []
            }
        
        if not response or not response.get("result"):
            return {
                "success": True,
                "message": "No hay enfrentamientos previos",
                "total_matches": 0,
                "player1_wins": 0,
                "player2_wins": 0,
                "recent_matches": []
            }
        
        result = response["result"]
        h2h_matches = result.get("H2H", [])
        
        if not h2h_matches:
            return {
                "success": True,
                "message": "No hay enfrentamientos previos",
                "total_matches": 0,
                "player1_wins": 0,
                "player2_wins": 0,
                "recent_matches": []
            }
        
        # Procesar datos
        p1_wins = 0
        p2_wins = 0
        hard_p1, hard_p2 = 0, 0
        clay_p1, clay_p2 = 0, 0
        grass_p1, grass_p2 = 0, 0
        recent_matches = []
        
        for m in h2h_matches:
            winner_str = m.get("event_winner", "")
            surface = _detect_surface_from_match(m)
            
            if "First" in winner_str:
                p1_wins += 1
                winner = 1
                if "hard" in surface.lower(): hard_p1 += 1
                elif "clay" in surface.lower(): clay_p1 += 1
                elif "grass" in surface.lower(): grass_p1 += 1
            else:
                p2_wins += 1
                winner = 2
                if "hard" in surface.lower(): hard_p2 += 1
                elif "clay" in surface.lower(): clay_p2 += 1
                elif "grass" in surface.lower(): grass_p2 += 1
            
            if len(recent_matches) < 5:
                recent_matches.append({
                    "date": m.get("event_date", ""),
                    "tournament": m.get("tournament_name", "Unknown"),
                    "surface": surface,
                    "winner": winner,
                    "score": m.get("event_final_result", "")
                })
        
        # Guardar en BD para caché
        try:
            db._execute(
                """
                INSERT INTO head_to_head (player1_key, player2_key, player1_wins, player2_wins,
                    hard_p1_wins, hard_p2_wins, clay_p1_wins, clay_p2_wins, grass_p1_wins, grass_p2_wins,
                    updated_at)
                VALUES (:p1_key, :p2_key, :p1_wins, :p2_wins, :hard_p1, :hard_p2, :clay_p1, :clay_p2, 
                    :grass_p1, :grass_p2, CURRENT_TIMESTAMP)
                ON CONFLICT (player1_key, player2_key) DO UPDATE SET
                    player1_wins = :p1_wins, player2_wins = :p2_wins,
                    hard_p1_wins = :hard_p1, hard_p2_wins = :hard_p2,
                    clay_p1_wins = :clay_p1, clay_p2_wins = :clay_p2,
                    grass_p1_wins = :grass_p1, grass_p2_wins = :grass_p2,
                    updated_at = CURRENT_TIMESTAMP
                """,
                {
                    "p1_key": p1_key, "p2_key": p2_key,
                    "p1_wins": p1_wins, "p2_wins": p2_wins,
                    "hard_p1": hard_p1, "hard_p2": hard_p2,
                    "clay_p1": clay_p1, "clay_p2": clay_p2,
                    "grass_p1": grass_p1, "grass_p2": grass_p2
                }
            )
        except Exception as e:
            logger.warning(f"Error guardando H2H en BD: {e}")
        
        return {
            "success": True,
            "total_matches": p1_wins + p2_wins,
            "player1_wins": p1_wins,
            "player2_wins": p2_wins,
            "surface_records": {
                "Hard": [hard_p1, hard_p2],
                "Clay": [clay_p1, clay_p2],
                "Grass": [grass_p1, grass_p2]
            },
            "recent_matches": recent_matches
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo H2H: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def _save_pointbypoint_to_db(db, match_id: int, pointbypoint_data: list):
    """
    Guarda datos pointbypoint en la BD para caché.
    Esto permite que futuras requests sean instantáneas.
    """
    try:
        import json
        
        # Guardar como JSON en la tabla match_pointbypoint
        db._execute(
            """
            INSERT INTO match_pointbypoint (match_id, data, created_at)
            VALUES (:match_id, :data, CURRENT_TIMESTAMP)
            ON CONFLICT (match_id) DO UPDATE SET 
                data = :data,
                created_at = CURRENT_TIMESTAMP
            """,
            {"match_id": match_id, "data": json.dumps(pointbypoint_data)}
        )
        logger.info(f"✅ Pointbypoint guardado en caché para match {match_id}")
    except Exception as e:
        logger.warning(f"Error guardando pointbypoint en BD: {e}")


def _load_pointbypoint_from_db(db, match_id: int) -> Optional[list]:
    """Carga datos pointbypoint de la BD si existen"""
    try:
        import json
        
        result = db._fetchone(
            "SELECT data FROM match_pointbypoint WHERE match_id = :match_id",
            {"match_id": match_id}
        )
        
        if result and result.get("data"):
            return json.loads(result["data"])
    except Exception as e:
        logger.warning(f"Error cargando pointbypoint de BD: {e}")
    
    return None


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
    """
    Carga estadísticas y timeline desde la BD (caché de pointbypoint).
    
    Returns:
        Tuple[MatchStats, MatchTimeline] o (None, None) si no hay datos
    """
    stats = None
    timeline = None
    
    try:
        # Intentar cargar pointbypoint de la caché
        pbp_data = _load_pointbypoint_from_db(db, match_id)
        
        if pbp_data:
            logger.info(f"✅ Pointbypoint encontrado en caché para match {match_id}")
            
            # Calcular stats
            try:
                # Obtener scores del match para calcular stats
                match = db.get_match(match_id)
                scores = None
                if match and match.get("resultado_marcador"):
                    scores = stats_calculator.parse_score_string(match["resultado_marcador"])
                
                stats = stats_calculator.calculate_stats(pbp_data, scores)
            except Exception as e:
                logger.warning(f"Error calculando stats desde caché: {e}")
            
            # Calcular timeline
            try:
                timeline = stats_calculator.calculate_timeline(pbp_data)
            except Exception as e:
                logger.warning(f"Error calculando timeline desde caché: {e}")
                
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
        
        # Log para debug: ver qué campos trae la API
        if h2h_matches:
            sample = h2h_matches[0]
            logger.info(f"🔍 H2H sample keys: {list(sample.keys())}")
            logger.info(f"🔍 H2H sample tournament: {sample.get('tournament_name')}, surface field: {sample.get('tournament_surface')}")
        
        p1_wins = 0
        p2_wins = 0
        matches = []
        
        # Records por superficie
        hard_p1, hard_p2 = 0, 0
        clay_p1, clay_p2 = 0, 0
        grass_p1, grass_p2 = 0, 0
        
        for m in h2h_matches:
            winner_str = m.get("event_winner", "")
            
            # Intentar obtener superficie de varios campos
            surface = _detect_surface_from_match(m)
            
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
                    surface=surface,  # Usa la superficie detectada
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


def _get_detailed_odds_from_db(db, match_id: int, match: dict) -> Optional[dict]:
    """
    Obtiene cuotas detalladas de la BD (tabla odds_history).
    """
    try:
        p1_name = match.get("jugador1_nombre") or match.get("jugador1")
        p2_name = match.get("jugador2_nombre") or match.get("jugador2")
        
        # Buscar en odds_history
        odds_rows = db._fetchall(
            """
            SELECT bookmaker, odds_player1, odds_player2, created_at
            FROM odds_history 
            WHERE match_id = :match_id
            ORDER BY created_at DESC
            """,
            {"match_id": match_id}
        )
        
        if not odds_rows:
            # Fallback a cuotas simples del match
            cuota1 = match.get("jugador1_cuota") or match.get("cuota_jugador1")
            cuota2 = match.get("jugador2_cuota") or match.get("cuota_jugador2")
            
            if cuota1 or cuota2:
                return {
                    "success": True,
                    "player1_name": p1_name,
                    "player2_name": p2_name,
                    "best_odds_player1": float(cuota1) if cuota1 else None,
                    "best_odds_player2": float(cuota2) if cuota2 else None,
                    "bookmakers": [{
                        "bookmaker": "Default",
                        "player1_odds": float(cuota1) if cuota1 else None,
                        "player2_odds": float(cuota2) if cuota2 else None,
                    }] if (cuota1 or cuota2) else [],
                    "total_bookmakers": 1 if (cuota1 or cuota2) else 0
                }
            return None
        
        # Construir lista de bookmakers (último registro de cada bookmaker)
        bookmakers_dict = {}
        for row in odds_rows:
            bm = row.get("bookmaker", "Unknown")
            if bm not in bookmakers_dict:
                bookmakers_dict[bm] = {
                    "bookmaker": bm,
                    "player1_odds": row.get("odds_player1"),
                    "player2_odds": row.get("odds_player2"),
                }
        
        bookmakers_list = list(bookmakers_dict.values())
        
        # Ordenar por mejor cuota
        bookmakers_list.sort(
            key=lambda x: (x["player1_odds"] or 0, x["player2_odds"] or 0),
            reverse=True
        )
        
        best_p1 = max([b["player1_odds"] for b in bookmakers_list if b["player1_odds"]], default=None)
        best_p2 = max([b["player2_odds"] for b in bookmakers_list if b["player2_odds"]], default=None)
        
        return {
            "success": True,
            "player1_name": p1_name,
            "player2_name": p2_name,
            "best_odds_player1": best_p1,
            "best_odds_player2": best_p2,
            "bookmakers": bookmakers_list,
            "total_bookmakers": len(bookmakers_list)
        }
    except Exception as e:
        logger.warning(f"Error obteniendo odds de BD: {e}")
        return None


def _get_h2h_from_db(db, match: dict) -> Optional[H2HData]:
    """
    Obtiene datos de H2H de la base de datos local.
    Los datos se pre-cargan por el scheduler de H2H.
    """
    try:
        p1_key = match.get("jugador1_key")
        p2_key = match.get("jugador2_key")
        
        if not p1_key or not p2_key:
            return None
        
        # Buscar en tabla head_to_head
        h2h_record = db._fetchone(
            """
            SELECT * FROM head_to_head 
            WHERE (player1_key = :p1 AND player2_key = :p2)
               OR (player1_key = :p2 AND player2_key = :p1)
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            {"p1": p1_key, "p2": p2_key}
        )
        
        if not h2h_record:
            return None
        
        # Construir H2HData desde el registro
        p1_wins = h2h_record.get("player1_wins", 0)
        p2_wins = h2h_record.get("player2_wins", 0)
        
        # Si las keys están invertidas en el registro, invertir los wins
        if h2h_record.get("player1_key") != p1_key:
            p1_wins, p2_wins = p2_wins, p1_wins
        
        return H2HData(
            total_matches=p1_wins + p2_wins,
            player1_wins=p1_wins,
            player2_wins=p2_wins,
            # Records por superficie si están disponibles
            surface_records={
                "Hard": [h2h_record.get("hard_p1_wins", 0), h2h_record.get("hard_p2_wins", 0)],
                "Clay": [h2h_record.get("clay_p1_wins", 0), h2h_record.get("clay_p2_wins", 0)],
                "Grass": [h2h_record.get("grass_p1_wins", 0), h2h_record.get("grass_p2_wins", 0)],
            },
            recent_matches=[]  # Los partidos recientes se pueden cargar por separado si necesario
        )
    except Exception as e:
        logger.warning(f"Error obteniendo H2H de BD: {e}")
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


def _detect_surface_from_match(match_data: dict) -> str:
    """
    Detecta la superficie de un partido H2H.
    Intenta varios métodos ya que la API no siempre devuelve tournament_surface.
    """
    # 1. Intentar campo directo
    surface = match_data.get("tournament_surface")
    if surface:
        return surface
    
    # 2. Intentar detectar por nombre del torneo
    tournament_name = (match_data.get("tournament_name") or "").lower()
    
    # Torneos conocidos de clay
    clay_keywords = [
        "roland garros", "french open", "rome", "roma", "madrid", 
        "monte carlo", "monte-carlo", "barcelona", "buenos aires",
        "rio", "sao paulo", "estoril", "geneva", "lyon", "hamburg",
        "bastad", "gstaad", "kitzbuhel", "umag", "stuttgart"
    ]
    
    # Torneos conocidos de grass
    grass_keywords = [
        "wimbledon", "queens", "queen's", "halle", "stuttgart grass",
        "eastbourne", "s-hertogenbosch", "'s-hertogenbosch", "mallorca",
        "newport", "london"  # Queen's Club London (grass) - ATP Finals usa otro nombre
    ]
    
    for keyword in clay_keywords:
        if keyword in tournament_name:
            return "Clay"
    
    for keyword in grass_keywords:
        if keyword in tournament_name:
            return "Grass"
    
    # 3. Por defecto Hard (más común)
    return "Hard"


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
