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
from datetime import datetime, date, time as dt_time
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

from src.api.models_match_detail import (
    BookmakerOdds,
    H2HData,
    LiveScore,
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


def _is_set_completed(p1: int, p2: int) -> bool:
    """True si el set está terminado (6-4, 7-5, 7-6, etc.)."""
    lo, hi = min(p1, p2), max(p1, p2)
    if hi < 6:
        return False
    return (hi - lo >= 2) or (lo >= 6)


def _is_match_future(match: dict) -> bool:
    """True si el partido aún no ha empezado (fecha o hora_inicio en el futuro)."""
    match_date = match.get("fecha_partido")
    if not match_date:
        return False
    try:
        if isinstance(match_date, str):
            match_date_val = datetime.strptime(str(match_date)[:10], "%Y-%m-%d").date()
        elif hasattr(match_date, "date"):
            match_date_val = match_date.date() if callable(match_date.date) else match_date
        else:
            return False
        if match_date_val > date.today():
            return True
        if match_date_val == date.today():
            hora = match.get("hora_inicio")
            if hora:
                if isinstance(hora, str):
                    parts = str(hora).strip().split(":")
                    h = int(parts[0]) if len(parts) > 0 else 0
                    m = int(parts[1]) if len(parts) > 1 else 0
                    start_dt = datetime.combine(match_date_val, dt_time(h, m, 0))
                else:
                    start_dt = datetime.combine(match_date_val, hora)
                return start_dt > datetime.now()
        return False
    except (ValueError, TypeError):
        return False


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
        # 1. Obtener partido de la BD (usando vista con predicciones; fallback si no existe aún)
        match = db._fetchone_with_view_fallback(
            "SELECT * FROM matches_with_latest_prediction WHERE id = :id",
            """
            SELECT m.*, p.version as prediction_version, p.timestamp as prediction_timestamp,
                p.jugador1_cuota, p.jugador2_cuota, p.jugador1_probabilidad, p.jugador2_probabilidad,
                p.jugador1_ev, p.jugador2_ev, p.recomendacion, p.mejor_opcion, p.confianza,
                p.kelly_stake_jugador1, p.kelly_stake_jugador2,
                p.confidence_level, p.confidence_score,
                b.id as bet_id, b.jugador_apostado, b.cuota_apostada, b.stake,
                b.resultado as bet_resultado, b.ganancia
            FROM matches m
            LEFT JOIN predictions p ON m.id = p.match_id AND p.version = (
                SELECT MAX(version) FROM predictions WHERE match_id = m.id
            )
            LEFT JOIN bets b ON m.id = b.match_id AND b.estado = 'activa'
            WHERE m.id = :id
            """,
            {"id": match_id},
        )
        if not match:
            match = db.get_match(match_id)
        if not match:
            raise HTTPException(status_code=404, detail="Partido no encontrado")
        
        # 2. Construir información del partido
        db_estado = match.get("estado", "pendiente")
        if db_estado not in ["pendiente", "en_juego", "completado", "suspendido", "cancelado"]:
            db_estado = "pendiente"
        # Si el partido es futuro, no mostrar en_juego (API-Tennis puede tener bugs).
        # NUNCA sobrescribir "completado": partidos terminados siempre muestran stats/timeline.
        if db_estado == "completado":
            estado = "completado"
        elif _is_match_future(match) and db_estado == "en_juego":
            estado = "pendiente"
        else:
            estado = db_estado
        
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
            event_status=match.get("event_status"),
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
                    # En partidos en vivo solo contar sets COMPLETADOS para sets_won (no el set en curso)
                    count_all = estado != "en_juego"
                    for s in sets_db:
                        p1 = s.get("player1_score", 0)
                        p2 = s.get("player2_score", 0)
                        winner = 1 if p1 > p2 else 2 if p2 > p1 else None
                        completed = _is_set_completed(p1, p2)
                        if count_all or completed:
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
        
        # PRIORIDAD 3: Usar event_final_result para sets_won SOLO si partido completado
        # En vivo NO usar event_final_result: la API puede enviar "2-0" como sets actuales
        # aunque el set 2 siga en curso (ej. 3-1 en juegos).
        if (not scores or not scores.sets) and estado != "en_juego":
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
        
        # Datos en vivo: set actual, puntos del juego, quién saca (solo si partido en juego)
        if scores and estado == "en_juego":
            serve_raw = (match.get("event_serve") or "").strip().lower()
            current_server = 1 if serve_raw == "first player" else 2
            current_set = len(scores.sets) if scores.sets else 1
            is_tiebreak = "tiebreak" in (match.get("event_status") or "").lower()
            live = LiveScore(
                current_game=match.get("event_game_result") or "0-0",
                current_server=current_server,
                current_set=current_set,
                is_tiebreak=is_tiebreak,
            )
            scores = MatchScores(sets_won=scores.sets_won, sets=scores.sets, live=live)
        
        # 5. Obtener estadísticas y timeline de la BD (si existen pre-calculadas)
        stats, timeline = _load_stats_from_db(db, match_id)
        
        # 5b. Fallback: si no hay timeline de pointbypoint pero sí scores, generar desde scores
        # (La API Tennis no siempre proporciona pointbypoint - ej. Australian Open)
        if (not timeline or timeline.total_games == 0) and scores and scores.sets:
            timeline = stats_calculator.calculate_timeline_from_scores(scores)
            if timeline.total_games > 0:
                logger.debug(f"📈 Timeline fallback desde scores en /full para match {match_id}")
        
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
        
        # 1. Intentar cargar de BD (pointbypoint cache)
        _, timeline = _load_stats_from_db(db, match_id)
        if timeline and timeline.total_games > 0:
            return timeline
        
        # 2. Fallback: timeline desde scores (match_sets o resultado_marcador)
        # La API Tennis no siempre proporciona pointbypoint (ej. Grand Slams)
        scores = _get_scores_for_match(db, match_id, match)
        if scores and scores.sets:
            timeline = stats_calculator.calculate_timeline_from_scores(scores)
            if timeline.total_games > 0:
                logger.info(f"📈 Timeline fallback desde scores para match {match_id} ({timeline.total_games} juegos)")
                return timeline
        
        # 3. Lazy loading desde API (get_fixtures con match_key)
        event_key = match.get("event_key")
        if not event_key:
            return MatchTimeline()
        
        try:
            fecha = match.get("fecha_partido")
            date_str = (fecha.strftime("%Y-%m-%d") if hasattr(fecha, "strftime") else str(fecha)[:10]) if fecha else None
            if not date_str:
                return MatchTimeline()
            params = {"date_start": date_str, "date_stop": date_str, "match_key": event_key}
            response = api_client._make_request("get_fixtures", params)
            if response and response.get("result"):
                results = response["result"]
                api_data = None
                if isinstance(results, list):
                    for m in results:
                        if str(m.get("event_key")) == str(event_key):
                            api_data = m
                            break
                    if not api_data and results:
                        api_data = results[0]
                else:
                    api_data = results
                
                if api_data and api_data.get("pointbypoint"):
                    _save_pointbypoint_to_db(db, match_id, api_data["pointbypoint"])
                    timeline = stats_calculator.calculate_timeline(api_data["pointbypoint"])
                    return timeline
                # API devolvió partido pero sin pointbypoint - intentar desde scores de API
                if api_data and api_data.get("scores"):
                    _save_scores_to_match_sets(db, match_id, api_data["scores"], match)
                    scores_api = stats_calculator.calculate_scores(api_data["scores"], api_data)
                    if scores_api and scores_api.sets:
                        return stats_calculator.calculate_timeline_from_scores(scores_api)
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
        
        # 1. Intentar cargar de BD (caché match_pointbypoint_cache)
        stats, _ = _load_stats_from_db(db, match_id)
        if stats and stats.has_detailed_stats:
            logger.debug(f"Stats match {match_id}: desde caché")
            return stats
        
        # 2. No hay datos - lazy loading desde API (get_fixtures requiere date_start/date_stop)
        event_key = match.get("event_key")
        if not event_key:
            return {"has_detailed_stats": False, "message": "No hay estadísticas disponibles"}
        
        try:
            fecha = match.get("fecha_partido")
            date_str = (fecha.strftime("%Y-%m-%d") if hasattr(fecha, "strftime") else str(fecha)[:10]) if fecha else None
            if not date_str:
                return {"has_detailed_stats": False, "message": "No hay estadísticas disponibles"}
            params = {"date_start": date_str, "date_stop": date_str, "match_key": event_key}
            response = api_client._make_request("get_fixtures", params)
            if response and response.get("result"):
                results = response["result"]
                api_data = None
                if isinstance(results, list):
                    for m in results:
                        if str(m.get("event_key")) == str(event_key):
                            api_data = m
                            break
                    if not api_data and results:
                        api_data = results[0]
                else:
                    api_data = results
                
                pbp_val = api_data.get("pointbypoint") if api_data else None
                if not api_data or not pbp_val:
                    keys = list(api_data.keys()) if api_data else []
                    pbp_type = type(pbp_val).__name__ if pbp_val is not None else "None"
                    pbp_len = len(pbp_val) if hasattr(pbp_val, "__len__") and pbp_val else 0
                    logger.warning(
                        f"📊 Stats match {match_id} (event_key={event_key}): pointbypoint vacío o ausente. "
                        f"Keys: {keys}, pointbypoint type={pbp_type}, len={pbp_len}"
                    )
                if api_data and api_data.get("pointbypoint"):
                    pbp = api_data["pointbypoint"]
                    logger.info(f"📊 Stats match {match_id}: API devolvió {len(pbp)} juegos pointbypoint")
                    _save_pointbypoint_to_db(db, match_id, pbp)
                    scores = None
                    if api_data.get("scores"):
                        scores = stats_calculator.calculate_scores(api_data["scores"], api_data)
                    stats = stats_calculator.calculate_stats(pbp, scores)
                    if stats:
                        return stats
                
                # Fallback: API tiene "statistics" aunque pointbypoint esté vacío (ej. Australian Open)
                if api_data and api_data.get("statistics"):
                    stats = stats_calculator.parse_statistics_from_api(
                        api_data["statistics"],
                        api_data,
                    )
                    if stats and stats.has_detailed_stats:
                        logger.info(f"📊 Stats match {match_id}: desde API statistics (sin pointbypoint)")
                        # Guardar scores en match_sets si tenemos, para timeline/fallback
                        if api_data.get("scores"):
                            _save_scores_to_match_sets(db, match_id, api_data["scores"])
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
        
        # Intentar obtener de API con lazy loading (get_fixtures requiere date_start/date_stop)
        event_key = match.get("event_key")
        if event_key:
            try:
                fecha = match.get("fecha_partido")
                date_str = (fecha.strftime("%Y-%m-%d") if hasattr(fecha, "strftime") else str(fecha)[:10]) if fecha else None
                if date_str:
                    params = {"date_start": date_str, "date_stop": date_str, "match_key": event_key}
                    response = api_client._make_request("get_fixtures", params)
                else:
                    response = None
                if response and response.get("result"):
                    results = response["result"]
                    api_data = None
                    if isinstance(results, list):
                        for m in results:
                            if str(m.get("event_key")) == str(event_key):
                                api_data = m
                                break
                        if not api_data and results:
                            api_data = results[0]
                    else:
                        api_data = results
                    if api_data and api_data.get("pointbypoint"):
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
                
                # Guardar en BD para caché (lazy loading). Schema: jugador1_cuota, jugador2_cuota, timestamp
                try:
                    db._execute(
                        """
                        INSERT INTO odds_history (match_id, bookmaker, jugador1_cuota, jugador2_cuota, timestamp)
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
            # H2HData usa hard_record, clay_record, grass_record (no surface_records)
            surface_records = {
                "Hard": h2h_from_db.hard_record,
                "Clay": h2h_from_db.clay_record,
                "Grass": h2h_from_db.grass_record,
            }
            return {
                "success": True,
                "total_matches": h2h_from_db.total_matches,
                "player1_wins": h2h_from_db.player1_wins,
                "player2_wins": h2h_from_db.player2_wins,
                "surface_records": surface_records,
                "recent_matches": [m.model_dump() for m in h2h_from_db.matches] if h2h_from_db.matches else []
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
                INSERT INTO h2h_cache (player1_key, player2_key, player1_wins, player2_wins,
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

def _get_scores_for_match(db, match_id: int, match: dict):
    """
    Obtiene MatchScores para un partido desde match_sets o resultado_marcador.
    Usado para fallback de timeline cuando no hay pointbypoint.
    """
    scores = None
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
    except Exception as e:
        logger.debug(f"Error obteniendo match_sets para scores: {e}")
    
    if not scores or not scores.sets:
        marcador = match.get("resultado_marcador")
        if marcador and "-" in marcador and "," in marcador:
            try:
                scores = stats_calculator.parse_score_string(marcador)
            except Exception:
                pass
    
    return scores


def _save_scores_to_match_sets(db, match_id: int, api_scores: list, match: dict = None):
    """
    Guarda scores de la API en match_sets.
    Convierte formato API (score_first, score_second, score_set) a nuestro formato.
    """
    if not api_scores:
        return
    try:
        match = match or db.get_match(match_id)
        swap = False
        if match:
            api_first = (match.get("jugador1_nombre") or "").lower()
            j1 = (match.get("jugador1_nombre") or "").lower()
            # La API devuelve first_player = event_first_player. Si nuestro jugador1 != API first, swap
            # Simplificado: asumir que si tenemos match, comparar nombres
            pass  # Por ahora no swap - la API first/second suele coincidir con nuestro orden
        sets_data = []
        for score in api_scores:
            p_first = int(score.get("score_first", 0))
            p_second = int(score.get("score_second", 0))
            set_num = int(score.get("score_set", len(sets_data) + 1))
            player1_score = p_second if swap else p_first
            player2_score = p_first if swap else p_second
            sets_data.append({
                "set_number": set_num,
                "player1_score": player1_score,
                "player2_score": player2_score,
                "tiebreak_score": None,
            })
        if sets_data and hasattr(db, "save_match_sets"):
            db.save_match_sets(match_id, sets_data)
            logger.debug(f"✅ Scores guardados en match_sets para match {match_id}")
    except Exception as e:
        logger.warning(f"Error guardando scores en match_sets: {e}")


def _save_pointbypoint_to_db(db, match_id: int, pointbypoint_data: list):
    """
    Guarda datos pointbypoint en la BD para caché (tabla match_pointbypoint_cache).
    Esto permite que futuras requests sean instantáneas.
    """
    try:
        import json
        data_json = json.dumps(pointbypoint_data)
        params = {"match_id": match_id, "data": data_json}
        db._execute(
            """
            INSERT INTO match_pointbypoint_cache (match_id, data, created_at)
            VALUES (:match_id, :data, CURRENT_TIMESTAMP)
            ON CONFLICT (match_id) DO UPDATE SET data = :data, created_at = CURRENT_TIMESTAMP
            """,
            params
        )
        logger.info(f"✅ Pointbypoint guardado en caché para match {match_id} ({len(pointbypoint_data)} juegos)")
    except Exception as e:
        logger.warning(f"Error guardando pointbypoint en BD: {e}")


def _load_pointbypoint_from_db(db, match_id: int) -> Optional[list]:
    """Carga datos pointbypoint de la BD (tabla match_pointbypoint_cache) si existen"""
    try:
        import json
        result = db._fetchone(
            "SELECT data FROM match_pointbypoint_cache WHERE match_id = :match_id",
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
        
        # Buscar en odds_history (schema: jugador1_cuota, jugador2_cuota, timestamp)
        odds_rows = db._fetchall(
            """
            SELECT bookmaker, jugador1_cuota, jugador2_cuota, timestamp
            FROM odds_history
            WHERE match_id = :match_id
            ORDER BY timestamp DESC
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
                    "player1_odds": row.get("jugador1_cuota"),
                    "player2_odds": row.get("jugador2_cuota"),
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
        
        # Buscar en tabla h2h_cache (por player keys API)
        h2h_record = db._fetchone(
            """
            SELECT * FROM h2h_cache
            WHERE (player1_key = :p1 AND player2_key = :p2)
               OR (player1_key = :p2 AND player2_key = :p1)
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            {"p1": str(p1_key), "p2": str(p2_key)}
        )
        
        if not h2h_record:
            return None
        
        # Construir H2HData desde el registro (H2HData usa hard_record, clay_record, grass_record)
        p1_wins = h2h_record.get("player1_wins", 0)
        p2_wins = h2h_record.get("player2_wins", 0)
        hard_p1 = h2h_record.get("hard_p1_wins", 0)
        hard_p2 = h2h_record.get("hard_p2_wins", 0)
        clay_p1 = h2h_record.get("clay_p1_wins", 0)
        clay_p2 = h2h_record.get("clay_p2_wins", 0)
        grass_p1 = h2h_record.get("grass_p1_wins", 0)
        grass_p2 = h2h_record.get("grass_p2_wins", 0)
        
        # Si las keys están invertidas en el registro, invertir wins y records
        if h2h_record.get("player1_key") != str(p1_key):
            p1_wins, p2_wins = p2_wins, p1_wins
            hard_p1, hard_p2 = hard_p2, hard_p1
            clay_p1, clay_p2 = clay_p2, clay_p1
            grass_p1, grass_p2 = grass_p2, grass_p1
        
        return H2HData(
            total_matches=p1_wins + p2_wins,
            player1_wins=p1_wins,
            player2_wins=p2_wins,
            hard_record=[hard_p1, hard_p2],
            clay_record=[clay_p1, clay_p2],
            grass_record=[grass_p1, grass_p2],
            matches=[],
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
        confidence_score = match.get("confidence_score")
        confidence_level = match.get("confidence_level")
        confianza_legacy = match.get("confianza")
        
        if not prob1 and not prob2:
            logger.info(f"🤖 No hay predicción para este partido")
            return None
        
        prob1 = float(prob1) if prob1 else 0.5
        prob2 = float(prob2) if prob2 else 0.5
        
        predicted_winner = 1 if prob1 > prob2 else 2
        
        # Confianza en % (0-100) - coherente con MatchCard que usa confidence_level
        # confidence_score viene en 0-1 (HIGH=1.0, MEDIUM=0.5, LOW=0.0)
        # confidence_level: HIGH/MEDIUM/LOW
        if confidence_level:
            level_to_pct = {"HIGH": 100, "MEDIUM": 50, "LOW": 0, "UNKNOWN": 25}
            confidence = level_to_pct.get(str(confidence_level).upper(), 50)
        elif confidence_score is not None:
            val = float(confidence_score)
            confidence = val * 100 if val <= 1 else val  # 0-1 → 0-100
        elif confianza_legacy:
            # Legacy: "Alta"/"Media"/"Baja" o número
            if isinstance(confianza_legacy, str):
                legacy_map = {"alta": 100, "media": 50, "baja": 0}
                confidence = legacy_map.get(str(confianza_legacy).lower(), 50)
            else:
                val = float(confianza_legacy)
                confidence = val * 100 if val <= 1 else val
        else:
            confidence = max(prob1, prob2) * 100  # Fallback: certeza del modelo
        
        # Calcular value bet
        value_bet = None
        cuota1_raw = match.get("jugador1_cuota") or match.get("cuota_jugador1")
        cuota2_raw = match.get("jugador2_cuota") or match.get("cuota_jugador2")
        cuota1 = float(cuota1_raw) if cuota1_raw else None
        cuota2 = float(cuota2_raw) if cuota2_raw else None

        if cuota1 and cuota2:
            ev1 = prob1 * cuota1 - 1
            ev2 = prob2 * cuota2 - 1
            if ev1 > 0.05:  # 5% de edge mínimo
                value_bet = 1
            elif ev2 > 0.05:
                value_bet = 2

        recomendacion = match.get("recomendacion") or ""
        mejor_opcion = match.get("mejor_opcion") or ""
        j1_name = (match.get("jugador1_nombre") or "").strip()
        j2_name = (match.get("jugador2_nombre") or "").strip()

        # Stake sugerido (Kelly) con bankroll actual
        kelly_j1 = None
        kelly_j2 = None
        bankroll_used = None
        if cuota1 and cuota2 and cuota1 > 0 and cuota2 > 0:
            try:
                db = get_db()
                bankroll_used = db.get_bankroll() if db else None
                if bankroll_used is None:
                    from src.config.settings import Config
                    bankroll_used = float(Config.BANKROLL_INICIAL)
                from src.config.settings import Config
                from src.utils.common import compute_kelly_stake_backtesting
                max_stake_eur = getattr(Config, "MAX_STAKE_EUR", None)
                rec_lower = recomendacion.lower()
                if "apostar" in rec_lower and "no" not in rec_lower[:10]:
                    if mejor_opcion == j1_name:
                        kelly_j1 = compute_kelly_stake_backtesting(
                            prob1, cuota1, bankroll_used,
                            kelly_fraction=Config.KELLY_FRACTION,
                            min_stake_eur=Config.MIN_STAKE_EUR,
                            max_stake_pct=Config.MAX_STAKE_PCT,
                            max_stake_eur=max_stake_eur,
                        ) or None
                    elif mejor_opcion == j2_name:
                        kelly_j2 = compute_kelly_stake_backtesting(
                            prob2, cuota2, bankroll_used,
                            kelly_fraction=Config.KELLY_FRACTION,
                            min_stake_eur=Config.MIN_STAKE_EUR,
                            max_stake_pct=Config.MAX_STAKE_PCT,
                            max_stake_eur=max_stake_eur,
                        ) or None
            except Exception as e:
                logger.debug("Stake ELO en detalle: %s", e)

        logger.info(f"🤖 Predicción: {prob1:.0%} vs {prob2:.0%}, confianza: {confidence:.0f}%")
        return MatchPrediction(
            predicted_winner=predicted_winner,
            confidence=confidence,
            probability_player1=prob1 * 100,
            probability_player2=prob2 * 100,
            value_bet=value_bet,
            recommendation=recomendacion or None,
            kelly_stake_jugador1=kelly_j1,
            kelly_stake_jugador2=kelly_j2,
            bankroll_used=bankroll_used,
        )
        
    except Exception as e:
        logger.warning(f"Error obteniendo predicción: {e}")
        return None
