"""
Tennis ML Predictor API v2.0
=============================

API REST rediseñada con sistema de gestión de partidos por fecha,
predicciones versionadas y tracking de apuestas.
"""

import sys
from pathlib import Path
from datetime import datetime, date, time as dt_time, timedelta
from typing import Optional
import logging
import asyncio
import json
import threading
import traceback


from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Añadir src al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.models_v2 import (
    MatchCreateRequest,
    MatchResultRequest,
    MatchResponse,
    MatchesDateResponse,
    Superficie,
    EstadoPartido,
    JugadorInfo,
    PredictionVersion,
    MatchResult,
    MatchDetails,
    MatchAnalysis,
    MatchStatsBasic,
    MatchStatsAdvanced,
    SetScore,
    # Nuevos modelos para scores detallados
    SetScoreSimple,
    LiveData,
    MatchScores,
)
from src.database.match_database import MatchDatabase
from src.prediction.predictor_calibrado import PredictorCalibrado
from src.config.settings import Config
from src.services.odds_update_service import OddsUpdateService
from src.services.api_tennis_client import APITennisClient
from src.services.match_stats_service import MatchStatsService

# APScheduler para actualizaciones automáticas
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

# Configurar logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Crear aplicación FastAPI
app = FastAPI(
    title="Tennis ML Predictor API v2.0",
    description="API REST para predicciones de tenis con ML - Sistema completo de gestión de partidos",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar componentes
# Usar variable de entorno para DB path (permite volumen persistente en Railway)
import os
DB_PATH = os.getenv("DB_PATH", "matches_v2.db")
logger.info(f"📁 Database path: {DB_PATH}")
db = MatchDatabase(DB_PATH)
predictor = None

# Inicializar APITennisClient y servicios
api_client = APITennisClient()

# Registrar routers de endpoints v2 (después de inicializar db y api_client)
from src.api.routes_match_detail import router as match_detail_router, configure_dependencies
configure_dependencies(db, api_client)
app.include_router(match_detail_router)
logger.info("✅ Router de detalle de partidos v2 registrado")

# Servicios Day 1
odds_service = OddsUpdateService(db, api_client)
logger.info("✅ OddsUpdateService inicializado con API-Tennis")

from src.services.match_update_service import MatchUpdateService

try:
    match_update_service = MatchUpdateService(db, api_client)
    logger.info("✅ MatchUpdateService inicializado")
except Exception as e:
    logger.error(f"❌ Error inicializando MatchUpdateService: {e}")
    match_update_service = None

# Elite Services (Day 2)
from src.services.player_service import PlayerService
from src.services.h2h_service import H2HService
from src.services.ranking_service_elite import RankingServiceElite
from src.services.tournament_service import TournamentService
from src.services.multi_odds_service import MultiBookmakerOddsService
from src.services.pointbypoint_service import PointByPointService

try:
    player_service = PlayerService(db)
    
    # Para servicios que necesitan conexión directa:
    # - En SQLite: usar db.conn
    # - En PostgreSQL: usar db (MatchDatabase) que tiene métodos de acceso
    def get_db_connection():
        """Obtiene la conexión adecuada según el tipo de DB"""
        if db.is_postgres:
            return db  # MatchDatabase maneja las conexiones internamente
        else:
            return db.conn  # SQLite usa conexión directa
    
    db_connection = get_db_connection()
    
    h2h_service = H2HService(db_connection, api_client) if db_connection else None
    ranking_service = RankingServiceElite(db_connection, api_client, player_service) if db_connection else None
    tournament_service = TournamentService(db_connection, api_client) if db_connection else None
    multi_odds_service = MultiBookmakerOddsService(db_connection, api_client) if db_connection else None
    
    # PointByPointService puede recibir db o db.conn - prefiere db para PostgreSQL
    pbp_service = PointByPointService(db)
    
    logger.info(f"✅ Elite Services inicializados (PostgreSQL: {db.is_postgres})")
except Exception as e:
    logger.warning(f"⚠️  Elite Services parcialmente inicializados: {e}")
    player_service = None
    h2h_service = None
    ranking_service = None
    tournament_service = None
    multi_odds_service = None
    pbp_service = None

# Variable global para LiveEventsService
live_events_service = None

# Inicializar ModelRetrainingExecutor
from src.services.model_retraining_executor import ModelRetrainingExecutor

retraining_executor = ModelRetrainingExecutor()

# Inicializar GitHub Polling Monitor (sin necesidad de webhooks)
from src.services.github_polling_monitor import GitHubPollingMonitor

github_monitor = GitHubPollingMonitor(repo_owner="Tennismylife", repo_name="TML-Database")

scheduler = BackgroundScheduler()


def get_predictor(raise_on_error: bool = True):
    """Lazy loading del predictor
    
    Args:
        raise_on_error: Si True, lanza excepción si no se puede cargar. Si False, retorna None.
    """
    global predictor
    if predictor is None:
        try:
            predictor = PredictorCalibrado(Config.MODEL_PATH)
            logger.info(f"✅ Predictor cargado desde {Config.MODEL_PATH}")
        except Exception as e:
            logger.error(f"❌ Error cargando predictor: {e}")
            if raise_on_error:
                raise HTTPException(status_code=500, detail=f"Error cargando modelo: {e}")
            return None
    return predictor


# ============================================================
# ENDPOINTS PRINCIPALES
# ============================================================


@app.get("/", tags=["Info"])
async def root():
    """Información de la API"""
    return {
        "name": "Tennis ML Predictor API",
        "version": "2.0.0",
        "description": "Sistema completo de gestión de partidos con predicciones ML",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "matches": "/matches?date=YYYY-MM-DD",
            "predict": "/matches/predict",
            "update_result": "/matches/{id}/result",
            "refresh": "/matches/{id}/refresh",
            "stats_summary": "/stats/summary",
            "stats_daily": "/stats/daily",
            "config": "/config",
        },
    }


@app.get("/health", tags=["Info"])
async def health_check():
    """Health check"""
    model_loaded = predictor is not None
    db_connected = db.conn is not None

    return {
        "status": "ok",
        "timestamp": datetime.now(),
        "model_loaded": model_loaded,
        "database_connected": db_connected,
        "version": "2.0.0",
    }


@app.get("/keepalive", tags=["Info"])
async def keepalive():
    """Keepalive endpoint for external cron jobs to prevent Railway from sleeping"""
    return {
        "status": "alive",
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================
# FUNCIONES AUXILIARES PARA SCORES
# ============================================================

def _is_set_completed(p1: int, p2: int) -> bool:
    """True si el set está terminado (alguien llegó a 6 con ventaja de 2, o 7-6/6-7)."""
    lo, hi = min(p1, p2), max(p1, p2)
    if hi < 6:
        return False
    return (hi - lo >= 2) or (lo >= 6)


def _build_match_scores(match_data: dict, database) -> Optional[MatchScores]:
    """
    Construye el objeto MatchScores con los datos del partido.
    Incluye scores por set y datos en vivo si está en juego.
    
    Args:
        match_data: Datos del partido de la BD
        database: Instancia de MatchDatabase
    
    Returns:
        MatchScores o None si no hay datos
    """
    try:
        match_id = match_data.get("id")
        estado = match_data.get("estado", "pendiente")
        
        # Obtener sets de la tabla match_sets
        sets_data = []
        try:
            sets_raw = database.get_match_sets(match_id)
            for s in sets_raw:
                sets_data.append(SetScoreSimple(
                    set_number=s.get("set_number", 0),
                    player1_score=s.get("player1_score", 0),
                    player2_score=s.get("player2_score", 0),
                    tiebreak_score=s.get("tiebreak_score")
                ))
        except Exception as e:
            logger.debug(f"No se pudieron obtener sets para match {match_id}: {e}")
        
        # Si no hay sets en la tabla, intentar parsear de resultado_marcador
        if not sets_data and match_data.get("resultado_marcador"):
            sets_data = _parse_marcador_to_sets(match_data["resultado_marcador"])
        
        # Si no hay sets en la tabla, intentar parsear de event_final_result
        if not sets_data and match_data.get("event_final_result"):
            # event_final_result es "2-0", "2-1" - no es suficiente para scores por set
            pass
        
        # Calcular resultado en sets (2-0, 2-1, etc.)
        # En partidos en vivo NO usar event_final_result: la API puede enviar "2-3"
        # contando el set en curso para el que va ganando. Solo contar sets COMPLETADOS.
        sets_result = None
        if estado == "en_juego" and sets_data:
            completed = [s for s in sets_data if _is_set_completed(s.player1_score, s.player2_score)]
            if completed:
                p1_sets = sum(1 for s in completed if s.player1_score > s.player2_score)
                p2_sets = sum(1 for s in completed if s.player2_score > s.player1_score)
                sets_result = f"{p1_sets}-{p2_sets}"
        if sets_result is None and estado != "en_juego":
            # Solo usar event_final_result si NO está en vivo (en vivo puede ser incorrecto)
            sets_result = match_data.get("event_final_result")
        if not sets_result and sets_data:
            # Calcular desde todos los sets (partido completado) o solo completados (en vivo)
            if estado == "en_juego":
                completed = [s for s in sets_data if _is_set_completed(s.player1_score, s.player2_score)]
                p1_sets = sum(1 for s in completed if s.player1_score > s.player2_score)
                p2_sets = sum(1 for s in completed if s.player2_score > s.player1_score)
            else:
                p1_sets = sum(1 for s in sets_data if s.player1_score > s.player2_score)
                p2_sets = sum(1 for s in sets_data if s.player2_score > s.player1_score)
            sets_result = f"{p1_sets}-{p2_sets}"
        if not sets_result and estado == "en_juego":
            sets_result = "0-0"  # En vivo sin datos de sets: mostrar 0-0
        
        # Construir datos en vivo si está en juego
        live_data = None
        if estado == "en_juego":
            live_data = LiveData(
                current_game_score=match_data.get("event_game_result"),
                current_server=match_data.get("event_serve"),
                current_set=len(sets_data) + 1 if sets_data else 1,
                is_tiebreak="tiebreak" in (match_data.get("event_status_detail") or "").lower()
            )
        
        # Si no hay datos, retornar None
        if not sets_data and not sets_result and not live_data:
            return None
        
        return MatchScores(
            sets_result=sets_result,
            sets=sets_data,
            live=live_data
        )
        
    except Exception as e:
        logger.error(f"Error construyendo scores para match: {e}")
        return None


def _parse_marcador_to_sets(marcador: str) -> list:
    """
    Parsea un marcador como "6-4, 7-5, 6-3" a lista de SetScoreSimple
    
    Args:
        marcador: String con el marcador
        
    Returns:
        Lista de SetScoreSimple
    """
    sets = []
    if not marcador:
        return sets
    
    try:
        # Separar por coma o espacio
        partes = [p.strip() for p in marcador.replace(",", " ").split() if "-" in p]
        
        for i, parte in enumerate(partes, 1):
            # Parsear "6-4" o "7-6(5)"
            if "(" in parte:
                # Tiebreak: "7-6(5)"
                score_part = parte.split("(")[0]
                tiebreak = parte.split("(")[1].rstrip(")")
                p1, p2 = score_part.split("-")
                sets.append(SetScoreSimple(
                    set_number=i,
                    player1_score=int(p1),
                    player2_score=int(p2),
                    tiebreak_score=tiebreak
                ))
            else:
                p1, p2 = parte.split("-")
                sets.append(SetScoreSimple(
                    set_number=i,
                    player1_score=int(p1),
                    player2_score=int(p2)
                ))
    except Exception as e:
        logger.debug(f"Error parseando marcador '{marcador}': {e}")
    
    return sets


# ============================================================
# ENDPOINTS DE PARTIDOS
# ============================================================


@app.get("/matches", response_model=MatchesDateResponse, tags=["Matches"])
async def get_matches_by_date(
    date_param: Optional[str] = Query(None, alias="date", description="Fecha en formato YYYY-MM-DD")
):
    """
    Obtiene todos los partidos de una fecha específica.
    OPTIMIZADO: Una sola consulta a BD, sin llamadas adicionales.
    """
    import time
    start_time = time.time()
    
    try:
        # Parsear fecha
        if date_param:
            try:
                fecha = datetime.strptime(date_param, "%Y-%m-%d").date()
            except ValueError:
                raise HTTPException(
                    status_code=400, detail="Formato de fecha inválido. Use YYYY-MM-DD"
                )
        else:
            fecha = date.today()

        # Validar rango (7 días atrás hasta 7 días adelante)
        fecha_minima = date.today() - timedelta(days=7)
        fecha_maxima = date.today() + timedelta(days=7)
        
        if fecha < fecha_minima:
            raise HTTPException(
                status_code=400,
                detail=f"Fecha fuera de rango. Mínimo: {fecha_minima} (7 días atrás)",
            )
        
        if fecha > fecha_maxima:
            raise HTTPException(
                status_code=400,
                detail=f"Fecha fuera de rango. Máximo: {fecha_maxima} (7 días adelante)",
            )

        # Obtener partidos de la BD (una sola consulta)
        db_start = time.time()
        partidos_raw = db.get_matches_by_date(fecha)
        db_time = time.time() - db_start
        logger.info(f"⏱️ DB query took {db_time:.2f}s for {len(partidos_raw)} matches")


        # Convertir a modelos Pydantic
        partidos = []
        today = date.today()
        for p in partidos_raw:
            # Partidos con fecha futura: nunca mostrar como "en directo" ni resultado
            match_date = p.get("fecha_partido")
            if isinstance(match_date, str):
                try:
                    match_date = datetime.strptime(match_date[:10], "%Y-%m-%d").date()
                except (ValueError, TypeError):
                    match_date = today
            elif hasattr(match_date, "date") and callable(getattr(match_date, "date", None)):
                match_date = match_date.date()
            elif not isinstance(match_date, date):
                match_date = today
            # Partido futuro: fecha > hoy, o mismo día pero hora_inicio aún no ha llegado
            is_future = match_date > today
            if not is_future and match_date == today:
                hora_inicio_val = p.get("hora_inicio")
                if hora_inicio_val is not None:
                    try:
                        if isinstance(hora_inicio_val, dt_time):
                            start_dt = datetime.combine(match_date, hora_inicio_val)
                        elif isinstance(hora_inicio_val, str):
                            parts = hora_inicio_val.strip().split(":")
                            h = int(parts[0]) if len(parts) > 0 else 0
                            m = int(parts[1]) if len(parts) > 1 else 0
                            start_dt = datetime.combine(match_date, dt_time(h, m, 0))
                        else:
                            start_dt = None
                        if start_dt is not None and start_dt > datetime.now():
                            is_future = True
                    except (ValueError, TypeError):
                        pass
            # Si el partido es futuro (fecha o hora_inicio), SIEMPRE mostrar pendiente.
            # No confiar en db_estado="en_juego" para partidos futuros (API-Tennis puede tener bugs).
            db_estado = p.get("estado", "pendiente")
            effective_estado = "pendiente" if is_future else db_estado

            # Construir jugadores
            jugador1 = JugadorInfo(
                nombre=p["jugador1_nombre"],
                ranking=p["jugador1_ranking"],
                cuota=p.get("jugador1_cuota", 0) or 2.0,  # Default si no hay predicción
                logo=p.get("jugador1_logo"),  # URL del logo desde API-Tennis
            )
            jugador2 = JugadorInfo(
                nombre=p["jugador2_nombre"],
                ranking=p["jugador2_ranking"],
                cuota=p.get("jugador2_cuota", 0) or 2.0,
                logo=p.get("jugador2_logo"),  # URL del logo desde API-Tennis
            )

            # Construir predicción si existe
            prediccion = None
            if p.get("prediction_version"):
                prediccion = PredictionVersion(
                    version=p["prediction_version"],
                    timestamp=p["prediction_timestamp"],
                    jugador1_cuota=p["jugador1_cuota"],
                    jugador2_cuota=p["jugador2_cuota"],
                    jugador1_probabilidad=p["jugador1_probabilidad"],
                    jugador2_probabilidad=p["jugador2_probabilidad"],
                    jugador1_ev=p["jugador1_ev"],
                    jugador2_ev=p["jugador2_ev"],
                    jugador1_edge=p.get("jugador1_edge"),
                    jugador2_edge=p.get("jugador2_edge"),
                    recomendacion=p["recomendacion"],
                    mejor_opcion=p.get("mejor_opcion"),
                    confianza=p.get("confianza"),
                    kelly_stake_jugador1=p.get("kelly_stake_jugador1"),
                    kelly_stake_jugador2=p.get("kelly_stake_jugador2"),
                )

            # Construir scores: misma lógica que detalle (match_sets primero, luego resultado_marcador).
            # Para en_juego SIEMPRE intentar construir (aunque no haya marcador aún), así la card puede mostrar 0-0 o sets en curso.
            match_scores = None
            if not is_future and (p.get("resultado_marcador") or p.get("event_final_result") or effective_estado == "en_juego"):
                try:
                    match_scores = _build_match_scores(p, db)
                    if match_scores is None and p.get("event_final_result"):
                        marcador = p.get("resultado_marcador") or p.get("event_final_result")
                        sets_data = _parse_marcador_to_sets(marcador) if marcador else []
                        if sets_data:
                            match_scores = MatchScores(
                                sets_result=p.get("event_final_result"),
                                sets=sets_data,
                                live=LiveData(
                                    current_game_score=p.get("event_game_result"),
                                    current_server=p.get("event_serve"),
                                    current_set=len(sets_data) + 1,
                                    is_tiebreak=False
                                ) if effective_estado == "en_juego" else None
                            )
                    # En vivo sin scores aún: devolver al menos sets_result "0-0" y live para que la card no muestre @2.00
                    if match_scores is None and effective_estado == "en_juego":
                        match_scores = MatchScores(
                            sets_result="0-0",
                            sets=[],
                            live=LiveData(
                                current_game_score=p.get("event_game_result"),
                                current_server=p.get("event_serve"),
                                current_set=1,
                                is_tiebreak=False
                            )
                        )
                except Exception:
                    pass
            # Garantía: si es en_juego y no tenemos scores (p. ej. excepción arriba), forzar mínimo para que la card muestre 0-0
            if effective_estado == "en_juego" and match_scores is None and not is_future:
                match_scores = MatchScores(
                    sets_result="0-0",
                    sets=[],
                    live=LiveData(
                        current_game_score=p.get("event_game_result"),
                        current_server=p.get("event_serve"),
                        current_set=1,
                        is_tiebreak=False
                    )
                )
            
            # Construir resultado (solo para completados o en juego, nunca para fecha futura)
            resultado = None
            if not is_future and (p.get("resultado_ganador") or effective_estado in ["completado", "en_juego"]):
                resultado = MatchResult(
                    ganador=p.get("resultado_ganador"),
                    marcador=p.get("resultado_marcador"),
                    scores=match_scores,
                    apostamos=p.get("bet_id") is not None,
                    resultado_apuesta=p.get("bet_resultado"),
                    stake=p.get("stake"),
                    ganancia=p.get("ganancia"),
                    roi=p.get("ganancia") / p.get("stake") if p.get("stake") else None,
                )

            # Construir partido completo (estado efectivo: pendiente si fecha futura)
            partido = MatchResponse(
                id=p["id"],
                estado=EstadoPartido(effective_estado),
                fecha_partido=p["fecha_partido"],
                hora_inicio=p.get("hora_inicio"),
                torneo=p.get("torneo"),
                ronda=p.get("ronda"),
                superficie=Superficie(p["superficie"]),
                jugador1=jugador1,
                jugador2=jugador2,
                prediccion=prediccion,
                resultado=resultado,
                event_status=p.get("event_status"),
            )

            partidos.append(partido)

        # Calcular resumen
        total = len(partidos)
        completados = sum(1 for p in partidos if p.estado == EstadoPartido.COMPLETADO)
        en_juego = sum(1 for p in partidos if p.estado == EstadoPartido.EN_JUEGO)
        pendientes = sum(1 for p in partidos if p.estado == EstadoPartido.PENDIENTE)

        total_time = time.time() - start_time
        logger.info(f"⏱️ /matches endpoint took {total_time:.2f}s total ({len(partidos)} matches)")
        
        return MatchesDateResponse(
            fecha=fecha,
            es_hoy=(fecha == date.today()),
            resumen={
                "total_partidos": total,
                "completados": completados,
                "en_juego": en_juego,
                "pendientes": pendientes,
            },
            partidos=partidos,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo partidos: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/matches/{match_id}/details", response_model=MatchDetails, tags=["Matches"])
async def get_match_details(match_id: int):
    """
    Vista Detallada - Obtiene estadísticas detalladas de un partido
    
    Incluye:
    - Estadísticas básicas (sets, juegos)
    - Estadísticas avanzadas (% saque, break points) - si disponible
    - Duración estimada - si disponible
    """
    try:
        # Obtener partido de la DB
        match = db.get_match(match_id)
        if not match:
            raise HTTPException(status_code=404, detail="Partido no encontrado")
        
        # Inicializar variables
        stats_basicas = None
        stats_avanzadas = None
        duracion = None
        
        # Intentar obtener datos de la API si hay event_key
        event_key = match.get("event_key")
        if event_key:
            try:
                # Consultar API
                params = {"event_key": event_key}
                api_data = api_client._make_request("get_fixtures", params)
                
                if api_data and api_data.get("result"):
                    api_match = api_data["result"][0] if isinstance(api_data["result"], list) else api_data["result"]
                    
                    # Extraer datos
                    scores = api_match.get("scores", [])
                    pointbypoint = api_match.get("pointbypoint", [])
                    
                    # Inicializar servicio de estadísticas
                    stats_service = MatchStatsService()
                    
                    # Calcular estadísticas básicas si hay scores
                    if scores:
                        stats_basicas = stats_service.calcular_estadisticas_basicas(scores)
                    
                    # Calcular estadísticas avanzadas si hay datos punto por punto
                    if pointbypoint:
                        stats_avanzadas_raw = stats_service.calcular_estadisticas_avanzadas(pointbypoint, scores)
                        if stats_avanzadas_raw:
                            stats_avanzadas = MatchStatsAdvanced(**stats_avanzadas_raw)
                        
                        # Estimar duración
                        timeline = stats_service.generar_timeline(pointbypoint)
                        duracion = stats_service._estimar_duracion(len(timeline))
            except Exception as e:
                logger.warning(f"No se pudieron obtener datos de la API para partido {match_id}: {e}")
        
        # Si no hay estadísticas de la API, crear básicas desde resultado_marcador
        if not stats_basicas and match.get("resultado_marcador"):
            # Intentar parsear resultado_marcador (ej: "2 - 0", "6-4, 7-5")
            marcador = match.get("resultado_marcador", "")
            try:
                # Si es formato "2 - 0" (sets)
                if " - " in marcador:
                    parts = marcador.split(" - ")
                    if len(parts) == 2:
                        stats_basicas = {
                            "total_sets": int(parts[0]) + int(parts[1]),
                            "sets_ganados_jugador1": int(parts[0]),
                            "sets_ganados_jugador2": int(parts[1]),
                            "total_juegos": 0,
                            "juegos_ganados_jugador1": 0,
                            "juegos_ganados_jugador2": 0,
                            "marcador_por_sets": []
                        }
            except:
                pass
        
        # Si aún no hay estadísticas básicas, retornar error
        if not stats_basicas:
            raise HTTPException(
                status_code=404,
                detail="No hay estadísticas disponibles para este partido"
            )
        
        # Construir respuesta
        return MatchDetails(
            match_id=match_id,
            estado=match.get("estado", "pendiente"),
            ganador=match.get("resultado_ganador"),
            duracion_estimada=duracion,
            estadisticas_basicas=MatchStatsBasic(**stats_basicas),
            estadisticas_avanzadas=stats_avanzadas,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo detalles del partido: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/matches/{match_id}/analysis", response_model=MatchAnalysis, tags=["Matches"])
async def get_match_analysis(match_id: int):
    """
    Análisis Profundo - Obtiene análisis completo del partido
    
    Incluye:
    - Todas las estadísticas de /details
    - Timeline juego por juego
    - Análisis de momentum
    - Puntos clave (break points, set points, match points)
    """
    try:
        # Obtener partido de la DB
        match = db.get_match(match_id)
        if not match:
            raise HTTPException(status_code=404, detail="Partido no encontrado")
        
        # Obtener datos detallados de la API
        event_key = match.get("event_key")
        if not event_key:
            raise HTTPException(
                status_code=404, 
                detail="Partido no tiene event_key, no se pueden obtener detalles"
            )
        
        # Consultar API
        params = {"event_key": event_key}
        api_data = api_client._make_request("get_fixtures", params)
        
        if not api_data or not api_data.get("result"):
            raise HTTPException(
                status_code=404, 
                detail="No se encontraron datos en la API"
            )
        
        api_match = api_data["result"][0] if isinstance(api_data["result"], list) else api_data["result"]
        
        # Extraer datos
        scores = api_match.get("scores", [])
        pointbypoint = api_match.get("pointbypoint", [])
        
        if not pointbypoint:
            raise HTTPException(
                status_code=404, 
                detail="No hay datos punto por punto disponibles para este partido"
            )
        
        # Inicializar servicio de estadísticas
        stats_service = MatchStatsService()
        
        # Generar resumen completo
        resumen = stats_service.generar_resumen_completo(scores, pointbypoint, api_match)
        
        # Construir respuesta
        return MatchAnalysis(
            match_id=match_id,
            estado=match.get("estado", "pendiente"),
            ganador=match.get("resultado_ganador"),
            duracion_estimada=resumen.get("duracion_estimada"),
            estadisticas_basicas=MatchStatsBasic(**resumen["basicas"]) if resumen.get("basicas") else None,
            estadisticas_avanzadas=MatchStatsAdvanced(**resumen["avanzadas"]) if resumen.get("avanzadas") else None,
            timeline=resumen.get("timeline", []),
            momentum=resumen.get("momentum", []),
            puntos_clave=resumen.get("puntos_clave", []),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo análisis del partido: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/matches/predict", response_model=MatchResponse, tags=["Matches"])
async def create_match_and_predict(request: MatchCreateRequest):
    """
    Crea un partido y genera predicción automáticamente

    Args:
        request: Datos del partido con cuotas

    Returns:
        Partido creado con predicción

    Example:
        ```json
        {
            "fecha_partido": "2026-01-09",
            "hora_inicio": "15:00",
            "superficie": "Hard",
            "jugador1_nombre": "Alcaraz",
            "jugador1_cuota": 2.10,
            "jugador2_nombre": "Sinner",
            "jugador2_cuota": 1.75
        }
        ```
    """
    try:
        pred = get_predictor()

        # 1. Crear partido en base de datos
        match_id = db.create_match(
            fecha_partido=request.fecha_partido,
            superficie=request.superficie.value,
            jugador1_nombre=request.jugador1_nombre,
            jugador1_cuota=request.jugador1_cuota,
            jugador2_nombre=request.jugador2_nombre,
            jugador2_cuota=request.jugador2_cuota,
            hora_inicio=request.hora_inicio.strftime("%H:%M") if request.hora_inicio else None,
            torneo=request.torneo,
            ronda=request.ronda,
            jugador1_ranking=request.jugador1_ranking,
            jugador2_ranking=request.jugador2_ranking,
        )

        # 2. Generar predicción
        resultado_pred = pred.predecir_partido(
            jugador1=request.jugador1_nombre,
            jugador1_rank=request.jugador1_ranking or 999,
            jugador2=request.jugador2_nombre,
            jugador2_rank=request.jugador2_ranking or 999,
            superficie=request.superficie.value,
            cuota=request.jugador1_cuota,
        )

        # Calcular para jugador2
        prob_j1 = resultado_pred["probabilidad"]
        prob_j2 = 1 - prob_j1
        ev_j1 = resultado_pred["expected_value"]
        ev_j2 = (prob_j2 * request.jugador2_cuota) - 1
        edge_j1 = resultado_pred.get("edge", 0)
        edge_j2 = prob_j2 - (1 / request.jugador2_cuota)

        # Determinar recomendación (FILTROS CONSERVADORES del backtesting)
        umbral_ev = Config.EV_THRESHOLD  # 10%
        max_cuota = Config.MAX_CUOTA  # 2.0
        min_prob = Config.MIN_PROBABILIDAD  # 0.60
        
        # Aplicar TODOS los filtros (igual que backtesting)
        if (ev_j1 > umbral_ev and 
            ev_j1 > ev_j2 and 
            request.jugador1_cuota < max_cuota and 
            prob_j1 > min_prob):
            # Apostar a jugador 1
            recomendacion = f"APOSTAR a {request.jugador1_nombre}"
            mejor_opcion = request.jugador1_nombre
            kelly_j1 = resultado_pred.get("stake_recomendado", 0)
            kelly_j2 = None
        elif (ev_j2 > umbral_ev and 
              request.jugador2_cuota < max_cuota and 
              prob_j2 > min_prob):
            # Apostar a jugador 2 (stake igual que backtesting)
            from src.utils.common import compute_kelly_stake_backtesting
            recomendacion = f"APOSTAR a {request.jugador2_nombre}"
            mejor_opcion = request.jugador2_nombre
            kelly_j1 = None
            kelly_j2 = compute_kelly_stake_backtesting(
                prob=prob_j2, cuota=request.jugador2_cuota, bankroll=Config.BANKROLL_INICIAL,
                kelly_fraction=Config.KELLY_FRACTION,
                min_stake_eur=Config.MIN_STAKE_EUR, max_stake_pct=Config.MAX_STAKE_PCT,
            ) or None
        else:
            recomendacion = "NO APOSTAR"
            mejor_opcion = None
            kelly_j1 = None
            kelly_j2 = None

        # Determinar confianza
        if abs(prob_j1 - 0.5) > 0.15:
            confianza = "Alta"
        elif abs(prob_j1 - 0.5) > 0.08:
            confianza = "Media"
        else:
            confianza = "Baja"

        # 3. Guardar predicción en base de datos
        prediction_id = db.add_prediction(
            match_id=match_id,
            jugador1_cuota=request.jugador1_cuota,
            jugador2_cuota=request.jugador2_cuota,
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
            # Agregar metadata de confianza del predictor
            confidence_level=resultado_pred.get("confidence_level"),
            confidence_score=resultado_pred.get("confidence_score"),
            player1_known=resultado_pred.get("player1_known"),
            player2_known=resultado_pred.get("player2_known"),
        )

        # 4. Si cumple umbral EV, registrar apuesta automáticamente
        if mejor_opcion:
            if mejor_opcion == request.jugador1_nombre:
                db.register_bet(
                    match_id=match_id,
                    prediction_id=prediction_id,
                    jugador_apostado=mejor_opcion,
                    cuota_apostada=request.jugador1_cuota,
                    stake=kelly_j1 or 10.0,
                )
            else:
                db.register_bet(
                    match_id=match_id,
                    prediction_id=prediction_id,
                    jugador_apostado=mejor_opcion,
                    cuota_apostada=request.jugador2_cuota,
                    stake=kelly_j2 or 10.0,
                )

        # 5. Obtener partido completo y devolverlo
        partidos = db.get_matches_by_date(request.fecha_partido)
        partido_creado = next((p for p in partidos if p["id"] == match_id), None)

        if not partido_creado:
            raise HTTPException(status_code=500, detail="Error obteniendo partido creado")

        # Convertir a modelo Pydantic (reutilizar lógica de get_matches_by_date)
        # ... (código similar al de get_matches_by_date)

        logger.info(
            f"✅ Partido creado y predicción generada: {request.jugador1_nombre} vs {request.jugador2_nombre}"
        )

        # Por ahora devolver respuesta simplificada
        return JSONResponse(
            content={
                "match_id": match_id,
                "prediction_id": prediction_id,
                "recomendacion": recomendacion,
                "mensaje": "Partido creado y predicción generada exitosamente",
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creando partido: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/matches/{match_id}/result", tags=["Matches"])
async def update_match_result(match_id: int, request: MatchResultRequest):
    """Actualiza el resultado de un partido"""
    try:
        partido = db.get_match(match_id)
        if not partido:
            raise HTTPException(status_code=404, detail=f"Partido {match_id} no encontrado")

        if partido.get("resultado_ganador"):
            raise HTTPException(status_code=400, detail="El partido ya tiene un resultado")

        success = db.update_match_result(match_id, request.ganador, request.marcador)
        if not success:
            raise HTTPException(status_code=500, detail="Error actualizando resultado")

        bet_updated = db.update_bet_result(match_id, request.ganador)

        response = {
            "match_id": match_id,
            "ganador": request.ganador,
            "marcador": request.marcador,
            "actualizado": True,
        }

        if bet_updated:
            bet = db._fetchone(
                "SELECT * FROM bets WHERE match_id = :match_id AND estado = 'completada'",
                {"match_id": match_id}
            )

            if bet:
                response["apuesta"] = {
                    "jugador_apostado": bet["jugador_apostado"],
                    "resultado": bet["resultado"],
                    "ganancia": bet["ganancia"],
                    "roi": bet["roi"],
                }

        logger.info(f"✅ Resultado actualizado para partido {match_id}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/matches/{match_id}/refresh", tags=["Matches"])
async def refresh_match_odds(match_id: int, jugador1_cuota: float, jugador2_cuota: float):
    """
    Actualiza las cuotas de un partido y regenera la predicción

    Proceso:
    1. Verifica que el partido no ha empezado
    2. Genera nueva predicción con cuotas actualizadas
    3. Compara con predicción anterior
    4. Si recomendación cambió:
       - NO APOSTAR → APOSTAR: registra nueva apuesta
       - APOSTAR → NO APOSTAR: cancela apuesta existente
       - APOSTAR a J1 → APOSTAR a J2: cancela y registra nueva
    5. Guarda nueva versión de predicción

    Args:
        match_id: ID del partido
        jugador1_cuota: Nueva cuota para jugador 1
        jugador2_cuota: Nueva cuota para jugador 2

    Returns:
        Nueva predicción con indicador de cambios

    Example:
        POST /matches/2/refresh?jugador1_cuota=1.95&jugador2_cuota=2.00
    """
    try:
        # 1. Obtener partido
        partido = db.get_match(match_id)
        if not partido:
            raise HTTPException(status_code=404, detail=f"Partido {match_id} no encontrado")

        # 2. Validar que no ha empezado (estado pendiente)
        if partido["estado"] != "pendiente":
            raise HTTPException(
                status_code=400,
                detail=f"No se puede actualizar: partido en estado '{partido['estado']}'",
            )

        # 2.5. Actualizar cuotas en la tabla matches
        db.update_match_odds(match_id, jugador1_cuota, jugador2_cuota)

        # 3. Obtener predicción anterior
        prediccion_anterior = db.get_latest_prediction(match_id)

        # 4. Generar nueva predicción
        pred = get_predictor()

        resultado_pred = pred.predecir_partido(
            jugador1=partido["jugador1_nombre"],
            jugador1_rank=partido["jugador1_ranking"] or 999,
            jugador2=partido["jugador2_nombre"],
            jugador2_rank=partido["jugador2_ranking"] or 999,
            superficie=partido["superficie"],
            cuota=jugador1_cuota,
        )

        # Calcular para jugador2
        prob_j1 = resultado_pred["probabilidad"]
        prob_j2 = 1 - prob_j1
        ev_j1 = resultado_pred["expected_value"]
        ev_j2 = (prob_j2 * jugador2_cuota) - 1
        edge_j1 = resultado_pred.get("edge", 0)
        edge_j2 = prob_j2 - (1 / jugador2_cuota)

        # Determinar recomendación (FILTROS CONSERVADORES del backtesting)
        umbral_ev = Config.EV_THRESHOLD  # 10%
        max_cuota = Config.MAX_CUOTA  # 2.0
        min_prob = Config.MIN_PROBABILIDAD  # 0.60
        
        # Aplicar TODOS los filtros (igual que backtesting)
        if (ev_j1 > umbral_ev and 
            ev_j1 > ev_j2 and 
            jugador1_cuota < max_cuota and 
            prob_j1 > min_prob):
            # Apostar a jugador 1
            recomendacion_nueva = f"APOSTAR a {partido['jugador1_nombre']}"
            mejor_opcion_nueva = partido["jugador1_nombre"]
            kelly_j1 = resultado_pred.get("stake_recomendado", 0)
            kelly_j2 = None
        elif (ev_j2 > umbral_ev and 
              jugador2_cuota < max_cuota and 
              prob_j2 > min_prob):
            # Apostar a jugador 2 (stake igual que backtesting)
            from src.utils.common import compute_kelly_stake_backtesting
            recomendacion_nueva = f"APOSTAR a {partido['jugador2_nombre']}"
            mejor_opcion_nueva = partido["jugador2_nombre"]
            kelly_j1 = None
            kelly_j2 = compute_kelly_stake_backtesting(
                prob=prob_j2, cuota=jugador2_cuota, bankroll=Config.BANKROLL_INICIAL,
                kelly_fraction=Config.KELLY_FRACTION,
                min_stake_eur=Config.MIN_STAKE_EUR, max_stake_pct=Config.MAX_STAKE_PCT,
            ) or None
        else:
            recomendacion_nueva = "NO APOSTAR"
            mejor_opcion_nueva = None
            kelly_j1 = None
            kelly_j2 = None

        # Determinar confianza
        if abs(prob_j1 - 0.5) > 0.15:
            confianza = "Alta"
        elif abs(prob_j1 - 0.5) > 0.08:
            confianza = "Media"
        else:
            confianza = "Baja"

        # 5. Guardar nueva versión de predicción
        prediction_id = db.add_prediction(
            match_id=match_id,
            jugador1_cuota=jugador1_cuota,
            jugador2_cuota=jugador2_cuota,
            jugador1_probabilidad=prob_j1,
            jugador2_probabilidad=prob_j2,
            jugador1_ev=ev_j1,
            jugador2_ev=ev_j2,
            jugador1_edge=edge_j1,
            jugador2_edge=edge_j2,
            recomendacion=recomendacion_nueva,
            mejor_opcion=mejor_opcion_nueva,
            confianza=confianza,
            kelly_stake_jugador1=kelly_j1,
            kelly_stake_jugador2=kelly_j2,
            # Agregar metadata de confianza del predictor
            confidence_level=resultado_pred.get("confidence_level"),
            confidence_score=resultado_pred.get("confidence_score"),
            player1_known=resultado_pred.get("player1_known"),
            player2_known=resultado_pred.get("player2_known"),
        )

        # 6. Analizar cambios en predicción (SIN modificar apuestas existentes)
        cambio_recomendacion = False
        apuesta_actual = None
        nueva_recomendacion_info = None

        # Verificar si ya existe una apuesta registrada
        bet_existente = db._fetchone(
            """
            SELECT * FROM bets 
            WHERE match_id = :match_id AND estado = 'activa'
            """,
            {"match_id": match_id},
        )

        if bet_existente:
            # Ya hay una apuesta registrada - NO la modificamos
            apuesta_actual = {
                "jugador_apostado": bet_existente["jugador_apostado"],
                "cuota_apostada": bet_existente["cuota_apostada"],
                "stake": bet_existente["stake"],
                "prediction_version": prediccion_anterior["version"] if prediccion_anterior else 1,
                "nota": "Apuesta ya realizada - no se modifica",
            }
            logger.info(f"ℹ️  Apuesta existente mantenida: {bet_existente['jugador_apostado']}")

        # Detectar cambios en recomendación (solo informativo)
        if prediccion_anterior:
            recomendacion_anterior = prediccion_anterior["recomendacion"]
            prediccion_anterior.get("mejor_opcion")

            if recomendacion_anterior != recomendacion_nueva:
                cambio_recomendacion = True

                # Información sobre la nueva recomendación (solo informativo)
                nueva_recomendacion_info = {
                    "recomendacion_anterior": recomendacion_anterior,
                    "recomendacion_nueva": recomendacion_nueva,
                    "nota": (
                        "Cambio detectado - apuesta original se mantiene"
                        if bet_existente
                        else "Nueva oportunidad de apuesta"
                    ),
                }

                # Si NO hay apuesta existente y ahora SÍ vale la pena apostar
                if not bet_existente and mejor_opcion_nueva is not None:
                    # Registrar nueva apuesta (solo si es la primera vez)
                    if mejor_opcion_nueva == partido["jugador1_nombre"]:
                        db.register_bet(
                            match_id=match_id,
                            prediction_id=prediction_id,
                            jugador_apostado=mejor_opcion_nueva,
                            cuota_apostada=jugador1_cuota,
                            stake=kelly_j1 or 10.0,
                        )
                    else:
                        db.register_bet(
                            match_id=match_id,
                            prediction_id=prediction_id,
                            jugador_apostado=mejor_opcion_nueva,
                            cuota_apostada=jugador2_cuota,
                            stake=kelly_j2 or 10.0,
                        )
                    apuesta_actual = {
                        "jugador_apostado": mejor_opcion_nueva,
                        "cuota_apostada": (
                            jugador1_cuota
                            if mejor_opcion_nueva == partido["jugador1_nombre"]
                            else jugador2_cuota
                        ),
                        "stake": kelly_j1 or kelly_j2 or 10.0,
                        "prediction_version": (
                            prediccion_anterior["version"] + 1 if prediccion_anterior else 2
                        ),
                        "nota": "Nueva apuesta registrada",
                    }
                    logger.info(f"✅ Primera apuesta registrada: {mejor_opcion_nueva}")

        # 7. Preparar respuesta
        response = {
            "match_id": match_id,
            "prediction_id": prediction_id,
            "version": prediccion_anterior["version"] + 1 if prediccion_anterior else 1,
            "cuotas_actualizadas": {"jugador1": jugador1_cuota, "jugador2": jugador2_cuota},
            "prediccion_actual": {
                "recomendacion": recomendacion_nueva,
                "mejor_opcion": mejor_opcion_nueva,
                "jugador1_ev": round(ev_j1, 4),
                "jugador2_ev": round(ev_j2, 4),
                "confianza": confianza,
            },
            "apuesta_registrada": apuesta_actual,
            "cambios": {
                "recomendacion_cambio": cambio_recomendacion,
                "info_cambio": nueva_recomendacion_info,
            },
        }

        logger.info(f"✅ Cuotas actualizadas para partido {match_id} (v{response['version']})")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats/summary", tags=["Stats"])
async def get_stats_summary(period: str = Query("7d")):
    """Obtiene resumen de estadísticas"""
    try:
        if period == "all":
            days = None
        elif period.endswith("d"):
            days = int(period[:-1])
        else:
            raise HTTPException(status_code=400, detail="Formato inválido")

        stats = db.get_stats_summary(days=days)
        return stats

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats/daily", tags=["Stats"])
async def get_daily_stats(days: int = Query(30, ge=1, le=365)):
    """Obtiene estadísticas diarias"""
    try:
        stats_daily = db.get_daily_stats(days=days)
        return {"dias": stats_daily}
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config", tags=["Config"])
async def get_config():
    """Obtiene configuración actual"""
    return {
        "ev_threshold": Config.EV_THRESHOLD,
        "kelly_fraction": Config.KELLY_FRACTION,
        "bankroll_inicial": 1000.0,
        "update_frequency_minutes": 15,
    }


@app.get("/matches/{match_id}/history", tags=["Matches"])
async def get_match_history(match_id: int):
    """
    Obtiene el historial completo de predicciones de un partido

    Muestra la evolución de cuotas, probabilidades, EV y recomendaciones
    a lo largo del tiempo. Útil para análisis post-partido.

    Args:
        match_id: ID del partido

    Returns:
        Historial completo de predicciones con cambios

    Example:
        GET /matches/4/history
    """
    try:
        # Obtener partido
        partido = db.get_match(match_id)
        if not partido:
            raise HTTPException(status_code=404, detail=f"Partido {match_id} no encontrado")

        # Obtener historial de predicciones
        predicciones = db.get_prediction_history(match_id)

        if not predicciones:
            return {
                "match_id": match_id,
                "jugador1": partido["jugador1_nombre"],
                "jugador2": partido["jugador2_nombre"],
                "total_versiones": 0,
                "versiones": [],
                "mensaje": "No hay predicciones para este partido",
            }

        # Formatear historial
        versiones = []
        for pred in predicciones:
            version_data = {
                "version": pred["version"],
                "timestamp": pred["timestamp"],
                "cuotas": {"jugador1": pred["jugador1_cuota"], "jugador2": pred["jugador2_cuota"]},
                "probabilidades": {
                    "jugador1": round(pred["jugador1_probabilidad"], 4),
                    "jugador2": round(pred["jugador2_probabilidad"], 4),
                },
                "expected_values": {
                    "jugador1": round(pred["jugador1_ev"], 4),
                    "jugador2": round(pred["jugador2_ev"], 4),
                },
                "recomendacion": pred["recomendacion"],
                "mejor_opcion": pred.get("mejor_opcion"),
                "confianza": pred.get("confianza"),
            }
            versiones.append(version_data)

        # Detectar cambios significativos
        cambios = []
        for i in range(1, len(versiones)):
            prev = versiones[i - 1]
            curr = versiones[i]

            # Cambio de recomendación
            if prev["recomendacion"] != curr["recomendacion"]:
                cambios.append(
                    {
                        "version": curr["version"],
                        "tipo": "recomendacion",
                        "anterior": prev["recomendacion"],
                        "nueva": curr["recomendacion"],
                    }
                )

            # Cambio significativo de cuotas (>5%)
            cambio_cuota_j1 = (
                abs(curr["cuotas"]["jugador1"] - prev["cuotas"]["jugador1"])
                / prev["cuotas"]["jugador1"]
            )
            cambio_cuota_j2 = (
                abs(curr["cuotas"]["jugador2"] - prev["cuotas"]["jugador2"])
                / prev["cuotas"]["jugador2"]
            )

            if cambio_cuota_j1 > 0.05 or cambio_cuota_j2 > 0.05:
                cambios.append(
                    {
                        "version": curr["version"],
                        "tipo": "cuotas",
                        "cambio_j1": f"{cambio_cuota_j1*100:.1f}%",
                        "cambio_j2": f"{cambio_cuota_j2*100:.1f}%",
                    }
                )

        return {
            "match_id": match_id,
            "jugador1": partido["jugador1_nombre"],
            "jugador2": partido["jugador2_nombre"],
            "estado": partido["estado"],
            "total_versiones": len(versiones),
            "versiones": versiones,
            "cambios_significativos": cambios,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo historial: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# ADMIN ENDPOINTS
# ============================================================


@app.post("/admin/update-odds", tags=["Admin"])
async def manual_update_odds():
    """
    Ejecuta actualización manual de cuotas para todos los partidos pendientes

    Este endpoint permite forzar una actualización sin esperar al scheduler.
    Útil para testing y debugging.

    Returns:
        Resultado de la actualización
    """
    try:
        logger.info("🔧 Actualización manual de cuotas solicitada")
        result = match_update_service.update_recent_matches(days=1)
        return result
    except Exception as e:
        logger.error(f"❌ Error en actualización manual: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/sync-odds-and-predictions", tags=["Admin"])
async def manual_sync_odds_and_predictions():
    """
    Sincroniza cuotas desde la API para partidos pendientes (hoy + 2 días)
    y genera/actualiza predicciones (primera vez o cuando cambian las cuotas).

    Útil para forzar sync sin esperar al job cada 4h.
    """
    try:
        if not odds_service:
            raise HTTPException(status_code=503, detail="OddsUpdateService no disponible")
        logger.info("🔧 Sync manual cuotas y predicciones solicitado")
        result = odds_service.sync_odds_and_predictions_for_pending_matches()
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error en sync cuotas/predicciones: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/check-predictions", tags=["Admin"])
async def admin_check_predictions():
    """
    Verifica el estado de las predicciones en la base de datos.
    
    Útil para diagnosticar por qué no se muestran predicciones en la app.
    Devuelve conteos y ejemplos de partidos con/sin predicción.
    """
    from datetime import date, timedelta
    try:
        today = date.today()
        end_date = today + timedelta(days=7)

        # Conteos básicos
        total_matches = db._fetchone("SELECT COUNT(*) as c FROM matches", {})
        total_matches = total_matches["c"] if total_matches else 0

        total_predictions = db._fetchone("SELECT COUNT(*) as c FROM predictions", {})
        total_predictions = total_predictions["c"] if total_predictions else 0

        matches_with_pred = db._fetchone(
            "SELECT COUNT(DISTINCT match_id) as c FROM predictions", {}
        )
        matches_with_pred = matches_with_pred["c"] if matches_with_pred else 0
        matches_without_pred = total_matches - matches_with_pred

        # Pendientes próximos 7 días
        pending = db._fetchall(
            """
            SELECT m.id, m.jugador1_nombre, m.jugador2_nombre, m.fecha_partido,
                   m.event_key,
                   p.id as pred_id, p.jugador1_probabilidad, p.jugador2_probabilidad
            FROM matches m
            LEFT JOIN predictions p ON m.id = p.match_id AND p.version = (
                SELECT MAX(version) FROM predictions WHERE match_id = m.id
            )
            WHERE m.estado = 'pendiente'
            AND m.fecha_partido >= :today
            AND m.fecha_partido <= :end
            ORDER BY m.fecha_partido ASC, m.id ASC
            LIMIT 30
            """,
            {"today": today, "end": end_date},
        )
        pending_with_pred = [r for r in (pending or []) if r.get("pred_id")]
        pending_without_pred = [r for r in (pending or []) if not r.get("pred_id")]

        # Últimas predicciones
        recent_preds = db._fetchall(
            """
            SELECT p.id, p.match_id, p.timestamp, p.jugador1_probabilidad, p.jugador2_probabilidad,
                   m.jugador1_nombre, m.jugador2_nombre, m.fecha_partido
            FROM predictions p
            JOIN matches m ON m.id = p.match_id
            ORDER BY p.timestamp DESC
            LIMIT 5
            """,
            {},
        )

        return {
            "database_type": "PostgreSQL" if db.is_postgres else "SQLite",
            "summary": {
                "total_matches": total_matches,
                "total_predictions": total_predictions,
                "matches_with_prediction": matches_with_pred,
                "matches_without_prediction": matches_without_pred,
                "pct_with_prediction": round(100 * matches_with_pred / total_matches, 1) if total_matches > 0 else 0,
            },
            "pending_next_7_days": {
                "total_sample": len(pending or []),
                "with_prediction": len(pending_with_pred),
                "without_prediction": len(pending_without_pred),
                "examples_without": [
                    {"id": m["id"], "match": f"{m.get('jugador1_nombre')} vs {m.get('jugador2_nombre')}",
                     "date": str(m.get("fecha_partido")), "event_key": m.get("event_key")}
                    for m in (pending_without_pred or [])[:5]
                ],
            },
            "last_predictions": [
                {"match_id": r["match_id"], "match": f"{r.get('jugador1_nombre')} vs {r.get('jugador2_nombre')}",
                 "timestamp": str(r.get("timestamp")), "prob": f"{r.get('jugador1_probabilidad',0):.0%}/{r.get('jugador2_probabilidad',0):.0%}"}
                for r in (recent_preds or [])
            ],
        }
    except Exception as e:
        logger.error(f"❌ Error en check-predictions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/debug-odds-sync", tags=["Admin"])
async def admin_debug_odds_sync():
    """
    Diagnóstico: por qué sync-odds no encuentra cuotas.
    
    Compara los event_keys de partidos pendientes con las claves que devuelve
    la API get_odds. Si no coinciden, el sync no puede generar predicciones.
    """
    from datetime import date, timedelta
    try:
        if not odds_service or not odds_service.odds_client:
            return {"error": "OddsUpdateService o API client no disponible"}

        today = date.today()
        end_date = today + timedelta(days=2)

        # 1. Partidos pendientes que necesitamos
        pending = db._fetchall(
            """
            SELECT id, event_key, jugador1_nombre, jugador2_nombre, fecha_partido
            FROM matches
            WHERE estado = 'pendiente'
            AND fecha_partido >= :today
            AND fecha_partido <= :end
            AND event_key IS NOT NULL AND TRIM(COALESCE(event_key, '')) != ''
            ORDER BY fecha_partido ASC
            LIMIT 20
            """,
            {"today": today, "end": end_date},
        )
        our_event_keys = [str(m["event_key"]) for m in (pending or []) if m.get("event_key")]

        # 2. Llamar a la API get_odds
        all_odds_by_key = {}
        for d in range((end_date - today).days + 1):
            day = today + timedelta(days=d)
            day_str = day.strftime("%Y-%m-%d")
            batch = odds_service.odds_client.get_all_odds_batch(day_str, day_str)
            if batch:
                all_odds_by_key.update(batch)

        api_keys = list(all_odds_by_key.keys()) if isinstance(all_odds_by_key, dict) else []
        api_keys_str = [str(k) for k in api_keys]
        api_keys_set = set(api_keys_str) | {k for k in api_keys if isinstance(k, (int, str))}

        # 3. Intersección
        def key_in_api(ek):
            return ek in api_keys_set or (ek.isdigit() and int(ek) in api_keys_set)
        matched = [k for k in our_event_keys if key_in_api(k)]
        not_matched = [k for k in our_event_keys if not key_in_api(k)]

        # 4. Probar extract_best_odds para el primero
        first_match_odds = None
        if our_event_keys:
            first_match_odds = odds_service.odds_client.extract_best_odds(
                all_odds_by_key, our_event_keys[0]
            )

        return {
            "today": str(today),
            "our_event_keys_sample": our_event_keys[:10],
            "api_odds_keys_count": len(api_keys),
            "api_odds_keys_sample": api_keys_str[:15] if api_keys_str else [],
            "matched_count": len(matched),
            "not_matched_count": len(not_matched),
            "not_matched_sample": not_matched[:5],
            "first_match_extract_result": "OK" if first_match_odds else "None (no Home/Away)",
            "diagnosis": (
                "API no devuelve cuotas para estos event_keys. "
                "Posibles causas: plan API sin odds, fechas fuera de cobertura, o formato de clave distinto."
                if not_matched and not matched else
                "Algunas claves coinciden. Revisar predictor o update_match_odds."
                if matched else
                "API devolvió 0 partidos con cuotas para el rango de fechas."
            ),
        }
    except Exception as e:
        logger.error(f"❌ Error en debug-odds-sync: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/scheduler-status", tags=["Admin"])
async def get_scheduler_status():
    """
    Obtiene el estado del scheduler de actualizaciones automáticas

    Returns:
        Estado del scheduler y próxima ejecución
    """
    try:
        is_running = scheduler.running
        jobs = scheduler.get_jobs()

        jobs_info = []
        for job in jobs:
            job_detail = scheduler.get_job(job.id)
            jobs_info.append({
                "id": job.id,
                "name": job.name,
                "next_run": job_detail.next_run_time.isoformat() if job_detail and job_detail.next_run_time else None,
                "trigger": str(job.trigger)
            })

        stats = match_update_service.get_update_stats()

        return {
            "scheduler_running": is_running,
            "total_jobs": len(jobs),
            "jobs": jobs_info,
            "update_interval_minutes": 15,
            **stats,
        }
    except Exception as e:
        logger.error(f"❌ Error obteniendo estado del scheduler: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/admin/pending-matches", tags=["Admin"])
async def get_pending_matches():
    """
    Obtiene lista de partidos pendientes

    Returns:
        Lista de partidos pendientes de actualización
    """
    try:
        pending = match_update_service.get_pending_matches()
        return {"total": len(pending), "partidos": pending}
    except Exception as e:
        logger.error(f"❌ Error obteniendo partidos pendientes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/detect-new-matches", tags=["Admin"])
async def detect_new_matches_manual():
    """
    Detecta manualmente partidos nuevos en la API (get_all_matches_with_odds)
    y los crea en la BD con predicción si tienen cuotas.

    Útil para testing. En producción la detección automática la hace el job
    cada 2h (DailyMatchFetcher.fetch_and_store_matches).
    """
    try:
        logger.info("🔧 Detección manual de partidos nuevos solicitada")
        if not odds_service or not odds_service.odds_client:
            return {"success": False, "partidos_nuevos": 0, "mensaje": "API de cuotas no disponible"}
        result = odds_service.detect_new_matches()
        return result
    except Exception as e:
        logger.error(f"❌ Error en detección manual: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/backfill-scores", tags=["Admin"])
async def backfill_match_scores(days: int = 7, max_matches: int = 50):
    """
    Rellena los datos de scores para partidos completados sin datos
    
    Este endpoint busca partidos completados que no tienen:
    - resultado_ganador
    - resultado_marcador
    - scores por set
    
    Y los actualiza consultando la API.
    
    Args:
        days: Número de días hacia atrás (default: 7)
        max_matches: Máximo de partidos a procesar (default: 50)
    
    Returns:
        Estadísticas del backfill
    """
    import threading
    
    def _backfill_task():
        try:
            logger.info(f"🔄 BACKFILL: Procesando partidos de los últimos {days} días...")
            
            # Obtener partidos completados sin datos
            matches = db.get_recent_matches(days=days)
            
            matches_to_fill = []
            for m in matches:
                if m.get("estado") != "completado":
                    continue
                
                # Verificar si le faltan datos
                needs_fill = False
                if not m.get("resultado_ganador"):
                    needs_fill = True
                elif not m.get("resultado_marcador"):
                    needs_fill = True
                else:
                    # Verificar si tiene sets guardados
                    try:
                        sets = db.get_match_sets(m["id"])
                        if not sets:
                            needs_fill = True
                    except:
                        needs_fill = True
                
                if needs_fill and m.get("event_key"):
                    matches_to_fill.append(m)
            
            # Limitar cantidad
            matches_to_fill = matches_to_fill[:max_matches]
            
            logger.info(f"📋 Encontrados {len(matches_to_fill)} partidos para rellenar")
            
            filled = 0
            errors = 0
            
            for match in matches_to_fill:
                try:
                    updated = match_update_service._update_single_match(match)
                    if updated:
                        filled += 1
                except Exception as e:
                    logger.debug(f"Error rellenando partido {match.get('id')}: {e}")
                    errors += 1
            
            logger.info(f"✅ BACKFILL completado: {filled} partidos actualizados, {errors} errores")
            
        except Exception as e:
            logger.error(f"❌ Error en backfill: {e}", exc_info=True)
    
    # Ejecutar en background
    thread = threading.Thread(target=_backfill_task, daemon=True)
    thread.start()
    
    return {
        "status": "accepted",
        "message": f"Backfill iniciado para {days} días, máximo {max_matches} partidos",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/admin/retraining-status", tags=["Admin"])
async def get_retraining_status():
    """
    Obtiene el estado del re-entrenamiento del modelo

    Muestra si hay un re-entrenamiento en progreso, cuándo fue la última
    ejecución y el resultado.

    Returns:
        Estado del re-entrenamiento
    """
    try:
        status = retraining_executor.get_status()
        return status
    except Exception as e:
        logger.error(f"❌ Error obteniendo estado de re-entrenamiento: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# NEW ENDPOINTS - MATCH FETCHING & DATE RANGES
# ============================================================


@app.get("/admin/compare-matches", tags=["Admin"])
async def compare_matches_api_vs_db(
    date_param: Optional[str] = Query(None, alias="date", description="Fecha YYYY-MM-DD (ej: 2026-01-29)")
):
    """
    Compara partidos de una fecha: base de datos vs API api-tennis.
    Ejemplo: GET /admin/compare-matches?date=2026-01-29
    """
    date_str = date_param or date.today().strftime("%Y-%m-%d")
    try:
        target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Fecha inválida. Use YYYY-MM-DD")

    db_matches = db.get_matches_by_date(target_date)
    db_count = len(db_matches)
    db_event_keys = {str(m.get("event_key")) for m in db_matches if m.get("event_key")}
    db_list = [
        {"event_key": m.get("event_key"), "jugador1": m.get("jugador1_nombre"), "jugador2": m.get("jugador2_nombre"), "torneo": m.get("torneo")}
        for m in db_matches
    ]

    data = api_client._make_request("get_fixtures", {"date_start": date_str, "date_stop": date_str})
    if not data or "result" not in data:
        return {
            "date": date_str,
            "db_count": db_count,
            "db_matches": db_list,
            "api_total": 0,
            "api_atp_count": 0,
            "api_error": "API no devolvió datos",
            "match": False,
            "missing_in_db": [],
            "summary": f"BD={db_count}. API sin datos.",
        }

    result = data["result"]
    if isinstance(result, list):
        api_raw = result
    elif isinstance(result, dict):
        if result.get("event_key") is not None or result.get("event_first_player") is not None:
            api_raw = [result]
        else:
            api_raw = list(result.values())
    else:
        api_raw = [result] if result else []
    api_total = len(api_raw)

    # Mismo filtro que daily_match_fetcher: ATP + Challenger + ITF Men Singles; excluir dobles, Boys, Girls, WTA
    atp_singles = []
    for raw in api_raw:
        if not isinstance(raw, dict):
            continue
        event_type = (raw.get("event_type_type") or raw.get("event_type") or "").upper()
        tournament = (raw.get("tournament_name") or "").lower()
        p1 = raw.get("event_first_player") or raw.get("event_home_team") or ""
        p2 = raw.get("event_second_player") or raw.get("event_away_team") or ""
        if "/" in p1 or "/" in p2:
            continue
        if "DOUBLES" in event_type or "doubles" in tournament:
            continue
        if "WTA" in event_type or "WOMEN" in event_type or "women" in tournament or "wta" in tournament:
            continue
        if "GIRLS" in event_type or "girls" in tournament or "BOYS" in event_type or "boys" in tournament:
            continue
        if not ("SINGLES" in event_type or "SINGLE" in event_type):
            continue
        is_atp = "ATP" in event_type
        is_challenger = "CHALLENGER" in event_type and "MEN" in event_type
        is_itf_men = "ITF" in event_type and "MEN" in event_type
        is_men_singles = "MEN" in event_type
        if not (is_atp or is_challenger or is_itf_men or is_men_singles):
            continue
        atp_singles.append(raw)

    api_atp_count = len(atp_singles)
    api_event_keys = {str(m.get("event_key")) for m in atp_singles if m.get("event_key")}
    missing_in_db = api_event_keys - db_event_keys
    missing_list = [
        {"event_key": ek, "player1": next((m.get("event_first_player") for m in atp_singles if str(m.get("event_key")) == ek), ""), "player2": next((m.get("event_second_player") for m in atp_singles if str(m.get("event_key")) == ek), ""), "tournament": next((m.get("tournament_name") for m in atp_singles if str(m.get("event_key")) == ek), "")}
        for ek in sorted(missing_in_db)
    ]
    match_ok = db_count == api_atp_count and not missing_in_db

    summary = f"BD={db_count} partidos, API(total)={api_total}, API(ATP Singles)={api_atp_count}. "
    summary += "✅ Coinciden." if match_ok else f"⚠️ Faltan {len(missing_in_db)} en BD." if missing_in_db else "Diferencia en conteo."

    return {
        "date": date_str,
        "db_count": db_count,
        "db_matches": db_list,
        "api_total": api_total,
        "api_atp_count": api_atp_count,
        "match": match_ok,
        "missing_in_db_count": len(missing_in_db),
        "missing_in_db": missing_list,
        "summary": summary,
    }


@app.get("/admin/debug-postgres", tags=["Admin"])
async def debug_postgres():
    """
    Endpoint de debug para probar PostgreSQL directamente
    
    Prueba:
    1. match_exists() con un partido que NO existe
    2. create_match() para insertar un partido de prueba
    3. match_exists() de nuevo (debería ser True)
    4. get_matches_by_date() para recuperar el partido
    5. Limpieza: eliminar el partido de prueba
    
    Returns:
        Resultados detallados de cada prueba
    """
    from datetime import date
    
    results = {
        "database_type": "PostgreSQL" if db.is_postgres else "SQLite",
        "tests": []
    }
    
    test_date = date.today()
    test_player1 = "DEBUG_TEST_PLAYER_1"
    test_player2 = "DEBUG_TEST_PLAYER_2"
    
    try:
        # TEST 1: Verificar que NO existe
        logger.info("🧪 TEST 1: Verificando match_exists() con partido inexistente...")
        exists_before = db.match_exists(test_player1, test_player2, test_date)
        results["tests"].append({
            "test": "1_match_exists_before",
            "expected": False,
            "actual": exists_before,
            "passed": not exists_before,
            "message": f"Partido debería NO existir. Result: {exists_before}"
        })
        
        # TEST 2: Crear partido
        logger.info("🧪 TEST 2: Creando partido de prueba...")
        try:
            match_id = db.create_match(
                fecha_partido=test_date,
                superficie="Hard",
                jugador1_nombre=test_player1,
                jugador1_cuota=1.5,
                jugador2_nombre=test_player2,
                jugador2_cuota=2.5,
                torneo="DEBUG_TEST_TOURNAMENT",
                hora_inicio="10:00",
                estado="pendiente"
            )
            results["tests"].append({
                "test": "2_create_match",
                "expected": "match_id > 0",
                "actual": match_id,
                "passed": match_id > 0,
                "message": f"Partido creado con ID: {match_id}"
            })
        except Exception as e:
            results["tests"].append({
                "test": "2_create_match",
                "expected": "success",
                "actual": str(e),
                "passed": False,
                "message": f"ERROR creando partido: {e}"
            })
            match_id = None
        
        # TEST 3: Verificar que AHORA existe
        logger.info("🧪 TEST 3: Verificando match_exists() después de crear...")
        exists_after = db.match_exists(test_player1, test_player2, test_date)
        results["tests"].append({
            "test": "3_match_exists_after",
            "expected": True,
            "actual": exists_after,
            "passed": exists_after,
            "message": f"Partido debería existir ahora. Result: {exists_after}"
        })
        
        # TEST 4: Recuperar con get_matches_by_date
        logger.info("🧪 TEST 4: Recuperando con get_matches_by_date()...")
        try:
            matches = db.get_matches_by_date(test_date)
            debug_matches = [m for m in matches if m.get('jugador1_nombre') == test_player1]
            results["tests"].append({
                "test": "4_get_matches_by_date",
                "expected": "1 match found",
                "actual": f"{len(debug_matches)} matches found (total: {len(matches)})",
                "passed": len(debug_matches) > 0,
                "message": f"Encontrados {len(debug_matches)} partidos de debug de {len(matches)} totales"
            })
        except Exception as e:
            results["tests"].append({
                "test": "4_get_matches_by_date",
                "expected": "success",
                "actual": str(e),
                "passed": False,
                "message": f"ERROR recuperando partidos: {e}"
            })
        
        # TEST 5: Contar partidos directamente con SQL
        logger.info("🧪 TEST 5: Contando partidos directamente con SQL...")
        try:
            count_result = db._fetchone(
                "SELECT COUNT(*) as count FROM matches WHERE jugador1_nombre = :player",
                {"player": test_player1}
            )
            count = count_result["count"] if count_result else 0
            results["tests"].append({
                "test": "5_direct_sql_count",
                "expected": "1",
                "actual": count,
                "passed": count > 0,
                "message": f"SQL directo encontró {count} partidos"
            })
        except Exception as e:
            results["tests"].append({
                "test": "5_direct_sql_count",
                "expected": "success",
                "actual": str(e),
                "passed": False,
                "message": f"ERROR en SQL directo: {e}"
            })
        
        # CLEANUP: Eliminar partido de prueba
        if match_id:
            logger.info("🧹 CLEANUP: Eliminando partido de prueba...")
            try:
                db.delete_match(match_id)
                results["cleanup"] = "✅ Partido de prueba eliminado"
            except Exception as e:
                results["cleanup"] = f"⚠️ Error eliminando: {e}"
        
        # Resumen
        passed = sum(1 for t in results["tests"] if t["passed"])
        total = len(results["tests"])
        results["summary"] = {
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": f"{passed}/{total} ({passed*100//total}%)"
        }
        
        logger.info(f"🧪 DEBUG TESTS COMPLETED: {passed}/{total} passed")
        return results
        
    except Exception as e:
        logger.error(f"❌ Error en debug endpoint: {e}", exc_info=True)
        return {
            "error": str(e),
            "database_type": "PostgreSQL" if db.is_postgres else "SQLite",
            "tests": results.get("tests", [])
        }


@app.get("/matches/upcoming", tags=["Matches"])
async def get_upcoming_matches():
    """
    Obtiene todos los partidos de los próximos 7 días

    Returns:
        Partidos agrupados por fecha
    """
    try:
        start_date = date.today()
        end_date = start_date + timedelta(days=7)

        matches = db.get_matches_date_range(start_date, end_date)

        # Agrupar por fecha
        matches_by_date = {}
        for match in matches:
            fecha = match["fecha_partido"]
            if fecha not in matches_by_date:
                matches_by_date[fecha] = []
            matches_by_date[fecha].append(match)

        return {
            "start_date": start_date,
            "end_date": end_date,
            "total_matches": len(matches),
            "dates": len(matches_by_date),
            "matches_by_date": matches_by_date,
        }

    except Exception as e:
        logger.error(f"Error obteniendo partidos próximos: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/matches/range", tags=["Matches"])
async def get_matches_by_date_range(
    start_date: str = Query(..., description="Fecha inicial (YYYY-MM-DD)"),
    end_date: str = Query(..., description="Fecha final (YYYY-MM-DD)"),
):
    """
    Obtiene partidos en un rango de fechas personalizado

    Args:
        start_date: Fecha inicial en formato YYYY-MM-DD
        end_date: Fecha final en formato YYYY-MM-DD

    Returns:
        Partidos en el rango especificado
    """
    try:
        # Parsear fechas
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d").date()
            end = datetime.strptime(end_date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(
                status_code=400, detail="Formato de fecha inválido. Use YYYY-MM-DD"
            )

        # Validar rango
        if start > end:
            raise HTTPException(
                status_code=400, detail="start_date debe ser anterior a end_date"
            )

        max_range = timedelta(days=30)
        if (end - start) > max_range:
            raise HTTPException(
                status_code=400, detail="El rango máximo es de 30 días"
            )

        # Obtener partidos
        matches = db.get_matches_date_range(start, end)

        return {
            "start_date": start,
            "end_date": end,
            "total_matches": len(matches),
            "matches": matches,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo partidos por rango: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/fetch-matches", tags=["Admin"])
async def manual_fetch_matches(days_ahead: int = Query(7, ge=1, le=14)):
    """
    Ejecuta manualmente el fetch diario de partidos

    Este endpoint permite forzar la obtención de partidos sin esperar
    al scheduler. Útil para testing y para obtener partidos inmediatamente
    después de desplegar la aplicación.

    Args:
        days_ahead: Número de días hacia adelante (1-14)

    Returns:
        Estadísticas de la operación
    """
    try:
        logger.info(f"🔧 Fetch manual de partidos solicitado ({days_ahead} días)")

        # Initialize daily match fetcher
        from src.automation.daily_match_fetcher import DailyMatchFetcher

        pred = get_predictor()
        fetcher = DailyMatchFetcher(db, api_client, pred)

        # Fetch matches
        stats = fetcher.fetch_and_store_matches(days_ahead=days_ahead)

        return {
            "success": True,
            "message": "Fetch de partidos completado",
            **stats,
        }

    except Exception as e:
        logger.error(f"❌ Error en fetch manual: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/sync-matches-now", tags=["Admin"])
async def sync_matches_now():
    """
    Sincroniza partidos para HOY inmediatamente
    
    Este endpoint es útil cuando la base de datos está vacía y necesitas
    poblarla rápidamente con los partidos de hoy. A diferencia de fetch-matches,
    este endpoint se enfoca solo en el día actual.
    
    Returns:
        Estadísticas de la sincronización
    """
    try:
        from datetime import date
        from src.automation.daily_match_fetcher import DailyMatchFetcher
        
        logger.info("🔄 Sincronización manual de partidos de HOY solicitada")
        
        today = date.today()
        pred = get_predictor()
        fetcher = DailyMatchFetcher(db, api_client, pred)
        
        # Fetch solo para hoy (1 día adelante)
        result = fetcher.fetch_and_store_matches(days_ahead=1)
        
        # El fetcher devuelve: matches_new, matches_existing, matches_found, etc.
        matches_added = result.get("matches_new", 0)
        matches_found = result.get("matches_found", 0)
        
        logger.info(f"✅ Sincronización completada: {matches_added} partidos añadidos de {matches_found} encontrados para {today}")
        
        return {
            "success": True,
            "date": today.isoformat(),
            "matches_found": matches_found,
            "matches_added": matches_added,
            "matches_existing": result.get("matches_existing", 0),
            "predictions_generated": result.get("predictions_generated", 0),
            "message": f"Sincronizados {matches_added} partidos nuevos para hoy ({today})"
        }
        
    except Exception as e:
        logger.error(f"❌ Error en sincronización manual: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")




# ============================================================
# WEBHOOKS
# ============================================================


@app.post("/webhooks/github", tags=["Webhooks"])
async def github_webhook(request: Request):
    """
    Recibe webhooks de GitHub para auto-actualización del modelo

    Cuando el repositorio tennis_atp tiene un commit nuevo con archivos CSV,
    este endpoint trigger la actualización automática del modelo:
    1. Descarga datos nuevos
    2. Aplica ventana temporal (elimina semana antigua, añade nueva)
    3. Re-entrena modelo

    Configuración en GitHub:
    - Payload URL: https://tu-dominio.com/webhooks/github
    - Content type: application/json
    - Secret: GITHUB_WEBHOOK_SECRET (en .env)
    - Events: Just the push event

    Returns:
        Confirmación de recepción
    """
    try:
        # Leer payload
        payload = await request.body()
        signature = request.headers.get("X-Hub-Signature-256", "")

        # Importar handler
        from src.services.github_webhook_handler import GitHubWebhookHandler

        handler = GitHubWebhookHandler()

        # Verificar firma
        if not handler.verify_signature(payload, signature):
            logger.warning("⚠️  Firma de webhook inválida")
            raise HTTPException(status_code=401, detail="Invalid signature")

        # Parsear evento


        event_data = json.loads(payload)

        # Verificar si debe actualizar
        if not handler.should_trigger_update(event_data):
            return {
                "success": True,
                "action": "ignored",
                "mensaje": "Evento ignorado - no hay archivos CSV modificados",
            }

        # Extraer info del commit
        commit_info = handler.extract_commit_info(event_data)

        logger.info(
            f"🔔 Webhook recibido: Commit {commit_info.get('sha')} - {commit_info.get('message')}"
        )

        # Trigger actualización del modelo en background
        retraining_result = retraining_executor.start_retraining(commit_info)

        if retraining_result["success"]:
            logger.info("✅ Re-entrenamiento del modelo iniciado en background")
        else:
            logger.warning(f"⚠️  {retraining_result['mensaje']}")

        return {
            "success": True,
            "action": "retraining_started" if retraining_result["success"] else "already_running",
            "commit": commit_info,
            "retraining": retraining_result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error procesando webhook: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# STARTUP/SHUTDOWN
# ============================================================


def _startup_sync_background():
    """
    Sincronización inicial en background.
    Se ejecuta en un thread separado para no bloquear el healthcheck.
    """
    import time
    
    # Pequeña pausa para que el servidor esté completamente listo
    time.sleep(5)
    
    logger.info("=" * 70)
    logger.info("🔄 INICIANDO SINCRONIZACIÓN EN BACKGROUND")
    logger.info("=" * 70)
    
    # 0. CHECK INICIAL: Importar datos históricos si la DB está vacía
    try:
        from src.database.match_database import MatchDatabase
        db_bg = MatchDatabase()
        
        result = db_bg._fetchone("SELECT COUNT(*) as count FROM matches", {})
        count = result["count"] if result else 0
        
        if count < 100:
            logger.warning(f"⚠️  Base de datos casi vacía ({count} partidos). Iniciando importación...")
            try:
                from scripts.import_historical_data import import_csv_to_db
                import_csv_to_db()
                logger.info("✅ Importación histórica completada")
            except Exception as e:
                logger.error(f"❌ Error importando históricos: {e}")
        else:
            logger.info(f"✅ Base de datos contiene {count} partidos históricos")
            
    except Exception as e:
        logger.error(f"⚠️  Error verificando estado DB: {e}")

    # 1. Actualizar estados de partidos existentes (limitado a 50 partidos max)
    try:
        from src.services.match_update_service import MatchUpdateService
        from src.database.match_database import MatchDatabase
        from src.services.api_tennis_client import APITennisClient
        
        logger.info("\n🔄 Actualizando estados de partidos existentes...")
        db_update = MatchDatabase()
        api_update = APITennisClient()
        update_service = MatchUpdateService(db_update, api_update)
        
        # Solo 1 día para no bloquear mucho tiempo
        stats = update_service.update_recent_matches(days=1)
        
        logger.info(f"✅ Estados actualizados: {stats['matches_updated']} de {stats['matches_checked']}")
        
    except Exception as e:
        logger.error(f"❌ Error actualizando estados: {e}")

    # 2. Fetch de partidos (solo hoy y mañana para arranque rápido)
    try:
        from src.automation.daily_match_fetcher import DailyMatchFetcher
        from src.database.match_database import MatchDatabase
        from src.services.api_tennis_client import APITennisClient
        
        logger.info("\n📥 Fetching partidos de hoy y mañana...")
        
        db_fetch = MatchDatabase()
        api_fetch = APITennisClient()
        pred = get_predictor(raise_on_error=False)
        fetcher = DailyMatchFetcher(db_fetch, api_fetch, pred)
        
        # Solo hoy + 1 día para arranque rápido
        stats = fetcher.fetch_and_store_matches(days_ahead=1)
        logger.info(f"✅ Fetch completado: {stats.get('matches_found', 0)} encontrados, {stats.get('matches_new', 0)} nuevos")
    
    except Exception as e:
        logger.error(f"❌ Error en fetch: {e}")
    
    logger.info("=" * 70)
    logger.info("✅ SINCRONIZACIÓN BACKGROUND COMPLETADA")
    logger.info("=" * 70)


@app.on_event("startup")
async def startup_event():
    """Evento de inicio del servidor - Rápido para pasar healthcheck"""
    logger.info("=" * 70)
    logger.info("🚀 SERVIDOR INICIANDO...")
    logger.info("=" * 70)
    
    # Iniciar sincronización en background (no bloquea el servidor)
    sync_thread = threading.Thread(target=_startup_sync_background, daemon=True)
    sync_thread.start()
    logger.info("📡 Sincronización iniciada en background")

    # Configurar scheduler para actualizaciones automáticas cada 5 minutos
    try:
        # Job 1: Actualizar cuotas y estados de partidos (cada 5 min)
        def update_matches_job():
            try:
                result = match_update_service.update_recent_matches(days=1)
                logger.info(f"✅ Actualización automática: {result}")
            except Exception as e:
                logger.error(f"❌ Error en actualización automática: {e}")
        
        scheduler.add_job(
            func=update_matches_job,
            trigger=IntervalTrigger(minutes=5),
            id="update_odds_job",
            name="Actualización automática de cuotas",
            replace_existing=True,
        )
        
        # Job 1.5: Actualizar estados de partidos existentes (cada 2 min para menos atraso en vivo)
        if match_update_service:
            scheduler.add_job(
                func=lambda: match_update_service.update_recent_matches(days=7),
                trigger=IntervalTrigger(minutes=2),
                id="update_match_status_job",
                name="Actualización de estados de partidos",
                replace_existing=True,
            )

        # Job 1.5b: Corregir partidos marcados en_juego que aún no han empezado (cada 5 min)
        def correct_future_live_matches():
            try:
                n = db.correct_future_matches_marked_live()
                if n > 0:
                    logger.info(f"🔧 Corregidos {n} partidos en_juego erróneos (aún no empezados)")
            except Exception as e:
                logger.debug("Error corrigiendo partidos futuros: %s", e)
        scheduler.add_job(
            func=correct_future_live_matches,
            trigger=IntervalTrigger(minutes=5),
            id="correct_future_live_job",
            name="Corregir partidos en_juego que aún no empezaron",
            replace_existing=True,
        )
        
        # Job 1.6: Sincronizar cuotas multi-bookmaker (cada 5 min)
        # NOTA: Deshabilitado temporalmente - no compatible con PostgreSQL
        if multi_odds_service and not db.is_postgres:
            def sync_multi_odds():
                """Sincroniza cuotas de múltiples bookmakers"""
                try:
                    result = multi_odds_service.sync_all_pending_matches_odds()
                    if result.get("success"):
                        if result.get("odds_synced", 0) > 0:
                            logger.info(
                                f"✅ Cuotas multi-bookmaker sincronizadas: {result['odds_synced']} cuotas "
                                f"para {result['matches_with_odds']}/{result['matches_found']} partidos"
                            )
                    else:
                        logger.warning(f"⚠️  {result.get('message', 'Error sincronizando cuotas')}")
                except Exception as e:
                    logger.error(f"❌ Error sincronizando cuotas multi-bookmaker: {e}")
            
            scheduler.add_job(
                func=sync_multi_odds,
                trigger=IntervalTrigger(minutes=5),
                id="sync_multi_odds_job",
                name="Sincronización de cuotas multi-bookmaker",
                replace_existing=True,
            )
        
        # WebSocket: datos en vivo en cuanto la API los envía
        try:
            from src.services.live_events_service import LiveEventsService
            global live_events_service
            live_events_service = LiveEventsService(db, api_client.api_key if api_client else None)
            live_events_service.start()
            logger.info("✅ WebSocket live events iniciado (tiempo real)")
        except ImportError as e:
            logger.info("ℹ️  WebSocket no disponible (pip install websocket-client): %s", e)
            live_events_service = None
        except Exception as e:
            logger.warning("⚠️  WebSocket live no iniciado: %s", e)
            live_events_service = None

        # Fallback: get_livescore cada 15 s para reducir retraso respecto al resultado real (WebSocket puede no empujar a tiempo)
        if api_client:
            def sync_live_via_livescore():
                try:
                    from src.services.live_events_service import LiveEventsService
                    live_service = LiveEventsService(db, api_client.api_key)
                    for api_match in api_client.get_livescore():
                        live_service.process_match_update(api_match)
                except Exception as e:
                    logger.debug("Sync live (get_livescore): %s", e)
            scheduler.add_job(
                func=sync_live_via_livescore,
                trigger=IntervalTrigger(seconds=15),
                id="sync_live_livescore_job",
                name="Live: get_livescore fallback cada 15s",
                replace_existing=True,
            )

        # Job 3: Verificar commits en TML-Database (cada hora)
        def check_github_commits():
            """Verifica si hay commits nuevos en TML-Database"""
            try:
                new_commit = github_monitor.check_for_new_commits()
                if new_commit:
                    logger.info(
                        f"🆕 Commit nuevo detectado en TML-Database: {new_commit['short_sha']}"
                    )
                    logger.info(f"📝 Mensaje: {new_commit['message']}")

                    # Trigger re-entrenamiento
                    result = retraining_executor.start_retraining(new_commit)
                    if result["success"]:
                        logger.info("✅ Re-entrenamiento iniciado automáticamente")
                    else:
                        logger.warning(f"⚠️  {result['mensaje']}")
            except Exception as e:
                logger.error(f"❌ Error verificando commits: {e}")

        scheduler.add_job(
            func=check_github_commits,
            trigger=IntervalTrigger(hours=1),
            id="github_polling_job",
            name="Verificación de commits en TML-Database",
            replace_existing=True,
        )

        # Job 3: Fetch diario de partidos (6:00 AM cada día)
        def daily_match_fetch():
            """Fetch diario de partidos desde Tennis API"""
            try:
                from src.automation.daily_match_fetcher import DailyMatchFetcher

                logger.info("🌅 Iniciando fetch diario de partidos...")
                pred = get_predictor()
                fetcher = DailyMatchFetcher(db, api_client, pred)
                stats = fetcher.fetch_and_store_matches(days_ahead=7)

                logger.info(f"✅ Fetch diario completado:")
                logger.info(f"   - Partidos encontrados: {stats['matches_found']}")
                logger.info(f"   - Partidos nuevos: {stats['matches_new']}")
                logger.info(f"   - Predicciones generadas: {stats['predictions_generated']}")

            except Exception as e:
                logger.error(f"❌ Error en fetch diario: {e}", exc_info=True)

        from apscheduler.triggers.cron import CronTrigger

        scheduler.add_job(
            func=daily_match_fetch,
            trigger=CronTrigger(hour=6, minute=0),  # 6:00 AM cada día
            id="daily_fetch_job",
            name="Fetch diario de partidos (6 AM)",
            replace_existing=True,
        )

        # Job 4: Detección de partidos nuevos (cada 6 horas)
        def detect_new_matches():
            """Detecta partidos nuevos cada 6 horas"""
            try:
                from src.automation.daily_match_fetcher import DailyMatchFetcher

                logger.info("🔍 Detectando partidos nuevos (próximos 7 días)...")
                pred = get_predictor()
                fetcher = DailyMatchFetcher(db, api_client, pred)
                stats = fetcher.fetch_and_store_matches(days_ahead=7)

                if stats['matches_new'] > 0:
                    logger.info(f"✅ Partidos nuevos detectados: {stats['matches_new']}")
                    logger.info(f"   - Predicciones generadas: {stats['predictions_generated']}")
                else:
                    logger.info("ℹ️  No hay partidos nuevos")

            except Exception as e:
                logger.error(f"❌ Error detectando partidos nuevos: {e}", exc_info=True)

        scheduler.add_job(
            func=detect_new_matches,
            trigger=IntervalTrigger(hours=2),  # Cada 2 horas
            id="new_matches_detection_job",
            name="Detección de partidos nuevos (cada 2h)",
            replace_existing=True,
        )

        # Job 4b: Sincronizar fixtures hoy y mañana (evitar partidos faltantes por torneo)
        def sync_today_tomorrow_fixtures():
            """Obtiene get_fixtures por fecha para hoy y mañana y crea partidos que falten."""
            try:
                from src.automation.daily_match_fetcher import DailyMatchFetcher
                pred = get_predictor()
                fetcher = DailyMatchFetcher(db, api_client, pred)
                stats = fetcher.sync_fixtures_for_dates()
                if stats.get("matches_new", 0) > 0:
                    logger.info(f"✅ Sync hoy/mañana: {stats['matches_new']} partidos nuevos")
            except Exception as e:
                logger.debug(f"Sync hoy/mañana: {e}")

        scheduler.add_job(
            func=sync_today_tomorrow_fixtures,
            trigger=IntervalTrigger(hours=6),  # Cada 6 horas
            id="sync_today_tomorrow_fixtures_job",
            name="Sincronizar fixtures hoy y mañana (cada 6h)",
            replace_existing=True,
        )

        # Job 4c: Sincronizar cuotas y predicciones (partidos sin cuotas → primera predicción; cuotas cambiadas → nueva versión)
        if odds_service:
            def sync_odds_and_predictions_job():
                """Actualiza cuotas desde API para pendientes y genera/regenera predicciones."""
                try:
                    result = odds_service.sync_odds_and_predictions_for_pending_matches()
                    if result.get("success"):
                        ou = result.get("odds_updated", 0)
                        pg = result.get("predictions_generated", 0)
                        if ou > 0 or pg > 0:
                            logger.info(f"✅ Sync cuotas/predicciones: {ou} cuotas actualizadas, {pg} predicciones")
                    elif result.get("error"):
                        logger.debug(f"Sync cuotas/predicciones: {result['error']}")
                except Exception as e:
                    logger.error(f"❌ Error sync cuotas/predicciones: {e}", exc_info=True)

            scheduler.add_job(
                func=sync_odds_and_predictions_job,
                trigger=IntervalTrigger(hours=4),  # Cada 4 horas
                id="sync_odds_and_predictions_job",
                name="Sincronizar cuotas y predicciones (cada 4h)",
                replace_existing=True,
            )

        # Job 5: Limpieza automática de partidos antiguos (2:00 AM cada día)
        def cleanup_old_matches():
            """Elimina partidos antiguos (>7 días)"""
            try:
                deleted_count = db.cleanup_old_matches(days_old=7)
                if deleted_count > 0:
                    logger.info(f"🧹 Limpieza automática: {deleted_count} partidos antiguos eliminados (>7 días)")
                else:
                    logger.info("🧹 Limpieza automática: No hay partidos antiguos para eliminar")

            except Exception as e:
                logger.error(f"❌ Error en limpieza automática: {e}")

        scheduler.add_job(
            func=cleanup_old_matches,
            trigger=CronTrigger(hour=2, minute=0),  # 2:00 AM cada día
            id="cleanup_job",
            name="Limpieza automática de partidos antiguos (>7 días, 2 AM)",
            replace_existing=True,
        )
        
        # ============================================================
        # ELITE SCHEDULER JOBS
        # ============================================================
        
        # Job Elite 1: Actualizar rankings ATP (diario a las 3 AM)
        if ranking_service:
            def sync_rankings_job():
                """Sincroniza rankings ATP y actualiza partidos"""
                try:
                    # Sincronizar rankings desde API
                    atp_count = ranking_service.sync_atp_rankings(limit=500)
                    logger.info(f"✅ Rankings ATP sincronizados: {atp_count} jugadores")
                    
                    # Actualizar partidos con rankings actualizados
                    db._execute("""
                        UPDATE matches
                        SET jugador1_ranking = (
                            SELECT atp_ranking FROM players 
                            WHERE player_key = matches.jugador1_key
                        )
                        WHERE jugador1_key IS NOT NULL
                    """)
                    db._execute("""
                        UPDATE matches
                        SET jugador2_ranking = (
                            SELECT atp_ranking FROM players 
                            WHERE player_key = matches.jugador2_key
                        )
                        WHERE jugador2_key IS NOT NULL
                    """)
                    logger.info("✅ Rankings de partidos actualizados")
                except Exception as e:
                    logger.error(f"❌ Error sincronizando rankings: {e}")
            
            scheduler.add_job(
                func=sync_rankings_job,
                trigger=CronTrigger(hour=3, minute=0),  # 3:00 AM cada día
                id="sync_rankings_job",
                name="Sincronización de rankings ATP (3 AM)",
                replace_existing=True,
            )
        
        # Job Elite 2: Sincronizar torneos (semanal, domingos a las 4 AM)
        if tournament_service:
            def sync_tournaments():
                """Sincroniza catálogo de torneos"""
                try:
                    count = tournament_service.sync_tournaments()
                    logger.info(f"✅ Torneos sincronizados: {count}")
                except Exception as e:
                    logger.error(f"❌ Error sincronizando torneos: {e}")
            
            scheduler.add_job(
                func=sync_tournaments,
                trigger=CronTrigger(day_of_week='sun', hour=4, minute=0),  # Domingos 4 AM
                id="sync_tournaments_job",
                name="Sincronización de torneos (Domingos 4 AM)",
                replace_existing=True,
            )

        scheduler.start()
        # Corrección inmediata al arrancar: partidos en_juego que aún no empezaron
        try:
            n = db.correct_future_matches_marked_live()
            if n > 0:
                logger.info(f"🔧 Al arrancar: corregidos {n} partidos marcados en_juego erróneamente")
        except Exception as e:
            logger.debug("Error corrección inicial: %s", e)
        logger.info("✅ Scheduler iniciado:")
        logger.info("   - Actualizaciones de cuotas: cada 5 minutos")
        logger.info("   - Actualización de estados: cada 2 minutos")
        if not db.is_postgres:
            logger.info("   - Sincronización de cuotas multi-bookmaker: cada 5 minutos")
        logger.info("   - Live: WebSocket + get_livescore fallback cada 15s")
        logger.info("   - Resultados en vivo: WebSocket (tiempo real)")
        logger.info("   - Detección de partidos nuevos: cada 2 horas")
        logger.info("   - Sincronizar fixtures hoy/mañana: cada 6 horas")
        logger.info("   - Verificación de commits TML: cada hora")
        logger.info("   - Fetch diario de partidos: 6:00 AM")
        logger.info("   - Limpieza de partidos antiguos (>7 días): 2:00 AM")
        logger.info("   - [ELITE] Sincronización de rankings ATP: 3:00 AM")
        logger.info("   - [ELITE] Sincronización de torneos: Domingos 4:00 AM")
    except Exception as e:
        logger.error(f"❌ Error iniciando scheduler: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Limpieza al cerrar la API"""
    global live_events_service
    
    # Detener WebSocket (sync)
    if live_events_service:
        live_events_service.stop()
    
    # Detener scheduler
    if scheduler.running:
        scheduler.shutdown()
        logger.info("🛑 Scheduler detenido")

    # Cerrar base de datos
    db.close()
    logger.info("👋 API cerrada correctamente")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api_v2:app",
        host="0.0.0.0",
        port=8001,  # Puerto diferente para no conflictuar con v1
        reload=True,
    )
"""
Elite API Endpoints - Tennis ML Predictor
Endpoints para jugadores, rankings, H2H y torneos
"""

# Estos endpoints se deben agregar a api_v2.py

# ============================================================
# ENDPOINTS ELITE - PLAYERS
# ============================================================

@app.get("/players/{player_key}", tags=["Elite - Players"])
async def get_player_profile(player_key: int):
    """
    Obtiene perfil completo de un jugador
    
    Args:
        player_key: ID del jugador en API-Tennis
        
    Returns:
        Perfil completo con estadísticas
    """
    if not player_service:
        raise HTTPException(status_code=503, detail="Player service not available")
    
    try:
        profile = player_service.get_player_profile(player_key)
        
        if not profile:
            raise HTTPException(status_code=404, detail="Jugador no encontrado")
        
        return profile
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo perfil de jugador: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/players/{player_key}/matches", tags=["Elite - Players"])
async def get_player_matches(
    player_key: int,
    limit: int = Query(20, ge=1, le=100),
    surface: Optional[str] = None
):
    """
    Obtiene últimos partidos de un jugador
    
    Args:
        player_key: ID del jugador
        limit: Número de partidos (1-100)
        surface: Filtrar por superficie (opcional)
        
    Returns:
        Lista de partidos recientes
    """
    if not player_service:
        raise HTTPException(status_code=503, detail="Player service not available")
    
    try:
        matches = player_service.get_player_form(player_key, limit)
        
        # Filtrar por superficie si se especifica
        if surface:
            matches = [m for m in matches if m.get('superficie') == surface]
        
        return {
            "player_key": player_key,
            "total_matches": len(matches),
            "matches": matches
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo partidos del jugador: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/players/{player_key}/stats", tags=["Elite - Players"])
async def get_player_stats(
    player_key: int,
    surface: Optional[str] = None,
    season: Optional[int] = None
):
    """
    Obtiene estadísticas de un jugador
    
    Args:
        player_key: ID del jugador
        surface: Superficie específica (Hard/Clay/Grass)
        season: Temporada específica
        
    Returns:
        Estadísticas del jugador
    """
    if not player_service:
        raise HTTPException(status_code=503, detail="Player service not available")
    
    try:
        if surface:
            stats = player_service.get_surface_stats(player_key, surface, season)
        else:
            profile = player_service.get_player_profile(player_key)
            stats = profile.get('stats', []) if profile else []
        
        return {
            "player_key": player_key,
            "surface": surface,
            "season": season,
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas del jugador: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# ENDPOINTS ELITE - HEAD TO HEAD
# ============================================================

@app.get("/h2h/{player1_key}/{player2_key}", tags=["Elite - H2H"])
async def get_head_to_head(player1_key: int, player2_key: int):
    """
    Obtiene histórico head-to-head entre dos jugadores
    
    Args:
        player1_key: ID del jugador 1
        player2_key: ID del jugador 2
        
    Returns:
        Datos completos de H2H
    """
    if not h2h_service:
        raise HTTPException(status_code=503, detail="H2H service not available")
    
    try:
        h2h_data = h2h_service.get_h2h(player1_key, player2_key)
        
        # Obtener forma reciente de ambos jugadores
        player1_form = h2h_service.get_recent_form(player1_key, 10)
        player2_form = h2h_service.get_recent_form(player2_key, 10)
        
        return {
            "h2h": h2h_data,
            "player1_recent_form": player1_form,
            "player2_recent_form": player2_form
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo H2H: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/h2h/match/{match_id}", tags=["Elite - H2H"])
async def get_match_h2h(match_id: int):
    """
    Obtiene H2H para un partido específico
    
    Args:
        match_id: ID del partido
        
    Returns:
        H2H entre los jugadores del partido
    """
    if not h2h_service:
        raise HTTPException(status_code=503, detail="H2H service not available")
    
    try:
        # Obtener partido
        match = db.get_match_by_id(match_id)
        if not match:
            raise HTTPException(status_code=404, detail="Partido no encontrado")
        
        player1_key = match.get('first_player_key')
        player2_key = match.get('second_player_key')
        
        if not player1_key or not player2_key:
            raise HTTPException(
                status_code=404, 
                detail="No hay player_keys para este partido"
            )
        
        h2h_data = h2h_service.get_h2h(player1_key, player2_key)
        
        return h2h_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo H2H del partido: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# ENDPOINTS ELITE - RANKINGS
# ============================================================

@app.get("/rankings/{league}", tags=["Elite - Rankings"])
async def get_rankings(
    league: str,
    limit: int = Query(100, ge=1, le=500)
):
    """
    Obtiene rankings ATP o WTA
    
    Args:
        league: 'ATP' o 'WTA'
        limit: Número de jugadores (1-500)
        
    Returns:
        Lista de jugadores ordenados por ranking
    """
    if not ranking_service:
        raise HTTPException(status_code=503, detail="Ranking service not available")
    
    if league.upper() not in ['ATP', 'WTA']:
        raise HTTPException(status_code=400, detail="League debe ser ATP o WTA")
    
    try:
        players = ranking_service.get_top_players(league.upper(), limit)
        
        return {
            "league": league.upper(),
            "total": len(players),
            "players": players
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo rankings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rankings/player/{player_key}", tags=["Elite - Rankings"])
async def get_player_ranking(player_key: int):
    """
    Obtiene información de ranking de un jugador
    
    Args:
        player_key: ID del jugador
        
    Returns:
        Info de ranking del jugador
    """
    if not ranking_service:
        raise HTTPException(status_code=503, detail="Ranking service not available")
    
    try:
        ranking_info = ranking_service.get_player_ranking_info(player_key)
        
        if not ranking_info:
            raise HTTPException(status_code=404, detail="Jugador no encontrado")
        
        return ranking_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo ranking del jugador: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rankings/sync/diagnostic", tags=["Elite - Rankings"])
async def rankings_sync_diagnostic():
    """
    Endpoint de diagnóstico para identificar problemas en el sync de rankings
    
    Returns:
        Información detallada de cada paso del proceso
    """
    diagnostic_info = {
        "step_1_api_client": "pending",
        "step_2_api_call": "pending",
        "step_3_data_received": "pending",
        "step_4_player_creation": "pending",
        "step_5_ranking_update": "pending",
        "errors": [],
        "data_sample": None
    }
    
    try:
        # Paso 1: Verificar API client
        if not api_client:
            diagnostic_info["step_1_api_client"] = "FAILED: API client not initialized"
            diagnostic_info["errors"].append("api_client is None")
            return diagnostic_info
        
        diagnostic_info["step_1_api_client"] = "OK"
        
        # Paso 2: Llamar a la API
        try:
            rankings = api_client.get_rankings(league="ATP")
            diagnostic_info["step_2_api_call"] = "OK"
        except Exception as e:
            diagnostic_info["step_2_api_call"] = f"FAILED: {str(e)}"
            diagnostic_info["errors"].append(f"API call error: {str(e)}")

            diagnostic_info["traceback"] = traceback.format_exc()
            return diagnostic_info
        
        # Paso 3: Verificar datos recibidos
        if not rankings:
            diagnostic_info["step_3_data_received"] = "FAILED: No rankings returned"
            diagnostic_info["errors"].append("API returned empty list")
            return diagnostic_info
        
        diagnostic_info["step_3_data_received"] = f"OK: {len(rankings)} rankings received"
        diagnostic_info["data_sample"] = rankings[0] if rankings else None
        
        # Paso 4: Intentar crear un jugador de prueba
        if not ranking_service:
            diagnostic_info["step_4_player_creation"] = "FAILED: ranking_service not initialized"
            diagnostic_info["errors"].append("ranking_service is None")
            return diagnostic_info
        
        try:
            test_player = rankings[0]
            player_key = test_player.get('player_key')
            player_name = test_player.get('player')
            
            # Use ranking_service's player_service
            ranking_service.player_service.get_or_create_player(
                player_key=player_key,
                player_name=player_name
            )
            diagnostic_info["step_4_player_creation"] = f"OK: Created/found player {player_name}"
        except Exception as e:
            diagnostic_info["step_4_player_creation"] = f"FAILED: {str(e)}"
            diagnostic_info["errors"].append(f"Player creation error: {str(e)}")

            diagnostic_info["traceback"] = traceback.format_exc()
            return diagnostic_info
        
        # Paso 5: Intentar actualizar ranking
        try:
            ranking = int(test_player.get('place', 0))
            points = int(test_player.get('points', 0))
            movement = test_player.get('movement', 'same')
            
            ranking_service.player_service.update_ranking(
                player_key=player_key,
                ranking=ranking,
                points=points,
                movement=movement,
                league='ATP'
            )
            diagnostic_info["step_5_ranking_update"] = f"OK: Updated ranking to {ranking} with {points} points"
        except Exception as e:
            diagnostic_info["step_5_ranking_update"] = f"FAILED: {str(e)}"
            diagnostic_info["errors"].append(f"Ranking update error: {str(e)}")

            diagnostic_info["traceback"] = traceback.format_exc()
            return diagnostic_info
        
        diagnostic_info["overall_status"] = "SUCCESS"
        return diagnostic_info
        
    except Exception as e:
        diagnostic_info["overall_status"] = "FAILED"
        diagnostic_info["errors"].append(f"Unexpected error: {str(e)}")

        diagnostic_info["traceback"] = traceback.format_exc()
        return diagnostic_info


@app.post("/rankings/sync", tags=["Elite - Rankings"])
async def sync_rankings():
    """
    Sincroniza rankings ATP desde API y actualiza partidos
    
    Returns:
        Número de jugadores sincronizados y partidos actualizados
    """
    if not ranking_service:
        raise HTTPException(status_code=503, detail="Ranking service not available")
    
    try:
        # Sincronizar rankings ATP (solo ATP, no WTA)
        atp_count = ranking_service.sync_atp_rankings(limit=500)
        
        # Actualizar partidos con rankings (usando métodos compatibles PostgreSQL/SQLite)
        # Actualizar jugador1_ranking
        db._execute("""
            UPDATE matches
            SET jugador1_ranking = (
                SELECT atp_ranking FROM players 
                WHERE player_key = matches.jugador1_key
            )
            WHERE jugador1_key IS NOT NULL
            AND EXISTS (
                SELECT 1 FROM players 
                WHERE player_key = matches.jugador1_key
                AND atp_ranking IS NOT NULL
            )
        """, {})
        
        # Actualizar jugador2_ranking
        db._execute("""
            UPDATE matches
            SET jugador2_ranking = (
                SELECT atp_ranking FROM players 
                WHERE player_key = matches.jugador2_key
            )
            WHERE jugador2_key IS NOT NULL
            AND EXISTS (
                SELECT 1 FROM players 
                WHERE player_key = matches.jugador2_key
                AND atp_ranking IS NOT NULL
            )
        """, {})
        
        return {
            "success": True,
            "rankings_synced": atp_count,
            "message": f"Sincronizados {atp_count} rankings ATP"
        }
        
    except Exception as e:
        logger.error(f"Error sincronizando rankings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# ENDPOINTS ELITE - TOURNAMENTS
# ============================================================

@app.get("/tournaments", tags=["Elite - Tournaments"])
async def get_tournaments(event_type: Optional[str] = None):
    """
    Obtiene lista de torneos
    
    Args:
        event_type: Filtrar por tipo (opcional)
        
    Returns:
        Lista de torneos
    """
    if not tournament_service:
        raise HTTPException(status_code=503, detail="Tournament service not available")
    
    try:
        tournaments = tournament_service.get_all_tournaments(event_type)
        
        return {
            "total": len(tournaments),
            "tournaments": tournaments
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo torneos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tournaments/{tournament_key}", tags=["Elite - Tournaments"])
async def get_tournament(tournament_key: int):
    """
    Obtiene detalles de un torneo
    
    Args:
        tournament_key: ID del torneo
        
    Returns:
        Detalles del torneo
    """
    if not tournament_service:
        raise HTTPException(status_code=503, detail="Tournament service not available")
    
    try:
        tournament = tournament_service.get_tournament(tournament_key)
        
        if not tournament:
            raise HTTPException(status_code=404, detail="Torneo no encontrado")
        
        return tournament
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo torneo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tournaments/{tournament_key}/matches", tags=["Elite - Tournaments"])
async def get_tournament_matches(
    tournament_key: int,
    season: Optional[int] = None
):
    """
    Obtiene partidos de un torneo
    
    Args:
        tournament_key: ID del torneo
        season: Temporada (opcional)
        
    Returns:
        Lista de partidos del torneo
    """
    if not tournament_service:
        raise HTTPException(status_code=503, detail="Tournament service not available")
    
    try:
        matches = tournament_service.get_tournament_matches(tournament_key, season)
        
        return {
            "tournament_key": tournament_key,
            "season": season,
            "total_matches": len(matches),
            "matches": matches
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo partidos del torneo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tournaments/sync", tags=["Elite - Tournaments"])
async def sync_tournaments():
    """
    Sincroniza catálogo de torneos desde API
    
    Returns:
        Número de torneos sincronizados
    """
    if not tournament_service:
        raise HTTPException(status_code=503, detail="Tournament service not available")
    
    try:
        count = tournament_service.sync_tournaments()
        
        return {
            "success": True,
            "tournaments_synced": count
        }
        
    except Exception as e:
        logger.error(f"Error sincronizando torneos: {e}")
        raise HTTPException(status_code=500, detail=str(e))
"""
Elite API Endpoints - Day 2 Advanced Features
Endpoints para cuotas multi-bookmaker y punto por punto
"""

# Estos endpoints se deben agregar a api_v2.py

# ============================================================
# ENDPOINTS ELITE - MULTI-BOOKMAKER ODDS
# ============================================================

@app.get("/matches/{match_id}/odds/multi", tags=["Elite - Odds"])
async def get_multi_bookmaker_odds(match_id: int, market_type: Optional[str] = None):
    """
    Obtiene cuotas de múltiples bookmakers para un partido
    
    Args:
        match_id: ID del partido
        market_type: Tipo de mercado (opcional)
        
    Returns:
        Cuotas de todos los bookmakers
    """
    if not multi_odds_service:
        raise HTTPException(status_code=503, detail="Multi odds service not available")
    
    try:
        odds = multi_odds_service.get_match_odds(match_id, market_type)
        
        return {
            "match_id": match_id,
            "market_type": market_type,
            "total_odds": len(odds),
            "odds": odds
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo cuotas multi: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/matches/{match_id}/odds/best", tags=["Elite - Odds"])
async def get_best_odds(match_id: int, market_type: str = "Match Winner"):
    """
    Obtiene las mejores cuotas para un partido
    
    Args:
        match_id: ID del partido
        market_type: Tipo de mercado
        
    Returns:
        Mejores cuotas por selección
    """
    if not multi_odds_service:
        raise HTTPException(status_code=503, detail="Multi odds service not available")
    
    try:
        best_odds = multi_odds_service.get_best_odds(match_id, market_type)
        
        return {
            "match_id": match_id,
            "market_type": market_type,
            "best_odds": best_odds
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo mejores cuotas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/odds/sync", tags=["Elite - Odds"])
async def sync_multi_bookmaker_odds():
    """
    Sincroniza cuotas de múltiples bookmakers para todos los partidos pendientes
    
    Este endpoint permite ejecutar manualmente la sincronización de cuotas
    sin esperar al job programado.
    
    Returns:
        Estadísticas de la sincronización
    """
    if not multi_odds_service:
        raise HTTPException(status_code=503, detail="Multi odds service not available")
    
    try:
        result = multi_odds_service.sync_all_pending_matches_odds()
        
        return {
            "success": result.get("success", False),
            "matches_found": result.get("matches_found", 0),
            "matches_with_odds": result.get("matches_with_odds", 0),
            "odds_synced": result.get("odds_synced", 0),
            "message": result.get("message", "Sincronización completada")
        }
        
    except Exception as e:
        logger.error(f"Error sincronizando cuotas: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/matches/{match_id}/odds/comparison", tags=["Elite - Odds"])
async def get_odds_comparison(match_id: int):
    """
    Obtiene comparación de cuotas entre bookmakers
    
    Args:
        match_id: ID del partido
        
    Returns:
        Comparación completa de cuotas
    """
    if not multi_odds_service:
        raise HTTPException(status_code=503, detail="Multi odds service not available")
    
    try:
        comparison = multi_odds_service.get_odds_comparison(match_id)
        
        return {
            "match_id": match_id,
            "comparison": comparison
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo comparación de cuotas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# ENDPOINTS ELITE - POINT BY POINT
# ============================================================

@app.get("/matches/{match_id}/pointbypoint", tags=["Elite - Point by Point"])
async def get_point_by_point(match_id: int, set_number: Optional[str] = None):
    """
    Obtiene datos punto por punto de un partido
    
    Args:
        match_id: ID del partido
        set_number: Número de set (opcional)
        
    Returns:
        Datos punto por punto
    """
    if not pbp_service:
        raise HTTPException(status_code=503, detail="Point by point service not available")
    
    try:
        points = pbp_service.get_point_by_point(match_id, set_number)
        
        # Si no hay datos locales, intentar obtenerlos de la API
        if not points:
            logger.info(f"📡 No hay puntos locales para partido {match_id}, obteniendo de API...")
            if _fetch_and_store_match_stats(match_id):
                points = pbp_service.get_point_by_point(match_id, set_number)
        
        return {
            "match_id": match_id,
            "set_number": set_number,
            "total_points": len(points),
            "points": points,
            "has_data": len(points) > 0
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo punto por punto: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/matches/{match_id}/games", tags=["Elite - Point by Point"])
async def get_match_games(match_id: int):
    """
    Obtiene juegos de un partido.
    Si no hay datos locales, intenta obtenerlos de la API Tennis.
    
    Args:
        match_id: ID del partido
        
    Returns:
        Lista de juegos
    """
    if not pbp_service:
        raise HTTPException(status_code=503, detail="Point by point service not available")
    
    try:
        games = pbp_service.get_games(match_id)
        
        # Si no hay datos locales, intentar obtenerlos de la API
        if not games:
            logger.info(f"📡 No hay juegos locales para partido {match_id}, obteniendo de API...")
            if _fetch_and_store_match_stats(match_id):
                games = pbp_service.get_games(match_id)
        
        return {
            "match_id": match_id,
            "total_games": len(games),
            "games": games,
            "has_data": len(games) > 0
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo juegos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/matches/{match_id}/breakpoints", tags=["Elite - Point by Point"])
async def get_break_points_stats(match_id: int):
    """
    Obtiene estadísticas de break points
    
    Args:
        match_id: ID del partido
        
    Returns:
        Estadísticas de break points
    """
    if not pbp_service:
        raise HTTPException(status_code=503, detail="Point by point service not available")
    
    try:
        stats = pbp_service.get_break_points_stats(match_id)
        
        return {
            "match_id": match_id,
            "break_points_stats": stats
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo stats de break points: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# ENDPOINTS ELITE - ESTADÍSTICAS AVANZADAS
# ============================================================

@app.get("/matches/{match_id}/stats/detailed", tags=["Elite - Match Statistics"])
async def get_match_detailed_stats(match_id: int):
    """
    Obtiene estadísticas detalladas completas de un partido
    Incluye: juegos, puntos, breaks, y análisis
    """
    if not pbp_service:
        raise HTTPException(status_code=503, detail="PointByPoint service not available")
    
    try:
        # Obtener juegos
        games = pbp_service.get_games(match_id)
        
        # Obtener puntos
        points = pbp_service.get_point_by_point(match_id)
        
        # Calcular estadísticas de breaks
        break_stats = pbp_service.get_break_points_stats(match_id)
        
        # Calcular estadísticas adicionales
        total_games = len(games) if games else 0
        total_points = len(points) if points else 0
        
        # Contar breaks
        breaks_player1 = sum(1 for g in games if g.get('was_break') and g.get('winner') == 'First Player') if games else 0
        breaks_player2 = sum(1 for g in games if g.get('was_break') and g.get('winner') == 'Second Player') if games else 0
        
        # Organizar por sets
        sets_data = {}
        if games:
            for game in games:
                set_num = game.get('set_number', 'Unknown')
                if set_num not in sets_data:
                    sets_data[set_num] = {
                        'games': [],
                        'games_player1': 0,
                        'games_player2': 0
                    }
                sets_data[set_num]['games'].append(game)
                if game.get('winner') == 'First Player':
                    sets_data[set_num]['games_player1'] += 1
                elif game.get('winner') == 'Second Player':
                    sets_data[set_num]['games_player2'] += 1
        
        return {
            "match_id": match_id,
            "summary": {
                "total_games": total_games,
                "total_points": total_points,
                "breaks_player1": breaks_player1,
                "breaks_player2": breaks_player2,
                "total_sets": len(sets_data)
            },
            "sets": sets_data,
            "break_points": break_stats,
            "has_data": total_games > 0 or total_points > 0
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas detalladas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _fetch_and_store_match_stats(match_id: int) -> bool:
    """
    Obtiene datos de juegos y puntos de la API Tennis si no existen localmente.
    
    Args:
        match_id: ID del partido
        
    Returns:
        True si se obtuvieron/guardaron datos, False si no
    """
    try:
        # Obtener el partido para conseguir event_key
        match = db.get_match(match_id)
        if not match or not match.get("event_key"):
            logger.debug(f"Partido {match_id} no tiene event_key")
            return False
        
        event_key = match["event_key"]
        match_date = match.get("fecha_partido")
        
        if not match_date:
            return False
        
        # Formatear fecha
        if hasattr(match_date, 'strftime'):
            date_str = match_date.strftime('%Y-%m-%d')
        else:
            date_str = str(match_date)
        
        # Obtener datos de la API
        logger.info(f"🔍 Obteniendo datos de API Tennis para partido {match_id} (event_key: {event_key})")
        
        params = {
            "date_start": date_str,
            "date_stop": date_str,
            "match_key": event_key
        }
        data = api_client._make_request("get_fixtures", params)
        
        if not data or not data.get("result"):
            logger.debug(f"No se encontraron datos en API para partido {match_id}")
            return False
        
        # Buscar el partido específico
        results = data["result"]
        api_match = None
        
        if isinstance(results, list):
            for m in results:
                if str(m.get("event_key")) == str(event_key):
                    api_match = m
                    break
        else:
            api_match = results
        
        if not api_match:
            return False
        
        # Guardar scores por set
        scores = api_match.get("scores", [])
        if scores:
            sets_data = []
            for score in scores:
                sets_data.append({
                    "set_number": int(score.get("score_set", 0)),
                    "player1_score": int(score.get("score_first", 0)),
                    "player2_score": int(score.get("score_second", 0)),
                    "tiebreak_score": None
                })
            db.save_match_sets(match_id, sets_data)
            logger.info(f"✅ Guardados {len(sets_data)} sets para partido {match_id}")
        
        # Guardar juegos y puntos desde pointbypoint
        pointbypoint = api_match.get("pointbypoint", [])
        if pointbypoint and pbp_service:
            games_saved = 0
            points_saved = 0
            
            for game in pointbypoint:
                # Guardar juego
                game_data = {
                    "set_number": game.get("set_number", "Set 1"),
                    "game_number": int(game.get("number_game", 0)),
                    "server": game.get("player_served", "First Player"),
                    "winner": game.get("serve_winner", "First Player"),
                    "score_games": game.get("score", "0-0"),
                    "score_sets": "0-0",
                    "was_break": bool(game.get("serve_lost"))
                }
                
                try:
                    pbp_service.store_games(match_id, [game_data])
                    games_saved += 1
                except:
                    pass
                
                # Guardar puntos del juego
                points = game.get("points", [])
                for point in points:
                    point_data = {
                        "set_number": game.get("set_number", "Set 1"),
                        "game_number": int(game.get("number_game", 0)),
                        "point_number": int(point.get("number_point", 0)),
                        "server": game.get("player_served", "First Player"),
                        "score": point.get("score", "0-0"),
                        "is_break_point": bool(point.get("break_point")),
                        "is_set_point": bool(point.get("set_point")),
                        "is_match_point": bool(point.get("match_point"))
                    }
                    
                    try:
                        pbp_service.store_point_by_point(match_id, [point_data])
                        points_saved += 1
                    except:
                        pass
            
            if games_saved > 0 or points_saved > 0:
                logger.info(f"✅ Guardados {games_saved} juegos y {points_saved} puntos para partido {match_id}")
                return True
        
        return bool(scores)
        
    except Exception as e:
        logger.error(f"Error obteniendo datos de API para partido {match_id}: {e}")
        return False


@app.get("/matches/{match_id}/stats/summary", tags=["Elite - Match Statistics"])
async def get_match_stats_summary(match_id: int):
    """
    Obtiene un resumen de estadísticas calculadas del partido.
    Si no hay datos locales, intenta obtenerlos de la API Tennis.
    """
    if not pbp_service:
        raise HTTPException(status_code=503, detail="PointByPoint service not available")
    
    try:
        games = pbp_service.get_games(match_id)
        points = pbp_service.get_point_by_point(match_id)
        
        # Si no hay datos locales, intentar obtenerlos de la API
        if not games and not points:
            logger.info(f"📡 No hay datos locales para partido {match_id}, obteniendo de API...")
            if _fetch_and_store_match_stats(match_id):
                # Reintentar obtener los datos
                games = pbp_service.get_games(match_id)
                points = pbp_service.get_point_by_point(match_id)
        
        if not games and not points:
            return {
                "match_id": match_id,
                "has_data": False,
                "message": "Estadísticas no disponibles para este partido"
            }
        
        # Calcular estadísticas
        total_games = len(games)
        total_points = len(points)
        
        # Juegos ganados
        games_won_p1 = sum(1 for g in games if g.get('winner') == 'First Player')
        games_won_p2 = sum(1 for g in games if g.get('winner') == 'Second Player')
        
        # Breaks
        breaks_p1 = sum(1 for g in games if g.get('was_break') and g.get('winner') == 'First Player')
        breaks_p2 = sum(1 for g in games if g.get('was_break') and g.get('winner') == 'Second Player')
        
        # Break points
        bp_total = sum(1 for p in points if p.get('is_break_point'))
        bp_p1 = sum(1 for p in points if p.get('is_break_point') and p.get('server') == 'Second Player')
        bp_p2 = sum(1 for p in points if p.get('is_break_point') and p.get('server') == 'First Player')
        
        # Puntos al saque
        serve_points_p1 = sum(1 for p in points if p.get('server') == 'First Player')
        serve_points_p2 = sum(1 for p in points if p.get('server') == 'Second Player')
        
        return {
            "match_id": match_id,
            "has_data": True,
            "summary": {
                "total_games": total_games,
                "total_points": total_points,
                "player1": {
                    "games_won": games_won_p1,
                    "breaks": breaks_p1,
                    "break_points_faced": bp_p2,
                    "break_points_won": breaks_p1,
                    "serve_points": serve_points_p1,
                    "break_conversion": f"{(breaks_p1/bp_p1*100):.1f}%" if bp_p1 > 0 else "N/A"
                },
                "player2": {
                    "games_won": games_won_p2,
                    "breaks": breaks_p2,
                    "break_points_faced": bp_p1,
                    "break_points_won": breaks_p2,
                    "serve_points": serve_points_p2,
                    "break_conversion": f"{(breaks_p2/bp_p2*100):.1f}%" if bp_p2 > 0 else "N/A"
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error calculando resumen de estadísticas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


