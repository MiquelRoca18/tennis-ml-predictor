"""
Tennis ML Predictor API v2.0
=============================

API REST rediseñada con sistema de gestión de partidos por fecha,
predicciones versionadas y tracking de apuestas.
"""

import sys
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Optional
import logging

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
)
from src.database.match_database import MatchDatabase
from src.prediction.predictor_calibrado import PredictorCalibrado
from src.config.settings import Config
from src.services.odds_update_service import OddsUpdateService
from src.services.api_tennis_client import APITennisClient

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
db = MatchDatabase("matches_v2.db")
predictor = None

# Inicializar APITennisClient y OddsUpdateService
try:
    odds_client = APITennisClient()
    update_service = OddsUpdateService(db, odds_client)
    logger.info("✅ OddsUpdateService inicializado con API-Tennis")
except Exception as e:
    logger.warning(f"⚠️  OddsUpdateService inicializado SIN API-Tennis: {e}")
    update_service = OddsUpdateService(db, None)

# Inicializar ModelRetrainingExecutor
from src.services.model_retraining_executor import ModelRetrainingExecutor

retraining_executor = ModelRetrainingExecutor()

# Inicializar GitHub Polling Monitor (sin necesidad de webhooks)
from src.services.github_polling_monitor import GitHubPollingMonitor

github_monitor = GitHubPollingMonitor(repo_owner="Tennismylife", repo_name="TML-Database")

scheduler = BackgroundScheduler()


def get_predictor():
    """Lazy loading del predictor"""
    global predictor
    if predictor is None:
        try:
            predictor = PredictorCalibrado(Config.MODEL_PATH)
            logger.info(f"✅ Predictor cargado desde {Config.MODEL_PATH}")
        except Exception as e:
            logger.error(f"❌ Error cargando predictor: {e}")
            raise HTTPException(status_code=500, detail=f"Error cargando modelo: {e}")
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


@app.get("/matches", response_model=MatchesDateResponse, tags=["Matches"])
async def get_matches_by_date(
    date_param: Optional[str] = Query(None, alias="date", description="Fecha en formato YYYY-MM-DD")
):
    """
    Obtiene todos los partidos de una fecha específica

    Args:
        date: Fecha en formato YYYY-MM-DD (default: hoy)

    Returns:
        Partidos con predicciones y resultados

    Example:
        GET /matches?date=2026-01-08
    """
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

        # Validar rango (máximo 7 días atrás)
        fecha_minima = date.today() - timedelta(days=7)
        if fecha < fecha_minima:
            raise HTTPException(
                status_code=400,
                detail=f"Fecha fuera de rango. Máximo 7 días atrás ({fecha_minima})",
            )

        # Obtener partidos
        partidos_raw = db.get_matches_by_date(fecha)

        # Convertir a modelos Pydantic
        partidos = []
        for p in partidos_raw:
            # Construir jugadores
            jugador1 = JugadorInfo(
                nombre=p["jugador1_nombre"],
                ranking=p["jugador1_ranking"],
                cuota=p.get("jugador1_cuota", 0) or 2.0,  # Default si no hay predicción
            )
            jugador2 = JugadorInfo(
                nombre=p["jugador2_nombre"],
                ranking=p["jugador2_ranking"],
                cuota=p.get("jugador2_cuota", 0) or 2.0,
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

            # Construir resultado si existe
            resultado = None
            if p.get("resultado_ganador"):
                resultado = MatchResult(
                    ganador=p["resultado_ganador"],
                    marcador=p.get("resultado_marcador"),
                    apostamos=p.get("bet_id") is not None,
                    resultado_apuesta=p.get("bet_resultado"),
                    stake=p.get("stake"),
                    ganancia=p.get("ganancia"),
                    roi=p.get("ganancia") / p.get("stake") if p.get("stake") else None,
                )

            # Construir partido completo
            partido = MatchResponse(
                id=p["id"],
                estado=EstadoPartido(p["estado"]),
                fecha_partido=p["fecha_partido"],
                hora_inicio=p.get("hora_inicio"),
                torneo=p.get("torneo"),
                ronda=p.get("ronda"),
                superficie=Superficie(p["superficie"]),
                jugador1=jugador1,
                jugador2=jugador2,
                prediccion=prediccion,
                resultado=resultado,
            )

            partidos.append(partido)

        # Calcular resumen
        total = len(partidos)
        completados = sum(1 for p in partidos if p.estado == EstadoPartido.COMPLETADO)
        en_juego = sum(1 for p in partidos if p.estado == EstadoPartido.EN_JUEGO)
        pendientes = sum(1 for p in partidos if p.estado == EstadoPartido.PENDIENTE)

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

        # Determinar recomendación
        umbral_ev = Config.EV_THRESHOLD
        if ev_j1 > umbral_ev:
            recomendacion = f"APOSTAR a {request.jugador1_nombre}"
            mejor_opcion = request.jugador1_nombre
            kelly_j1 = resultado_pred.get("stake_recomendado", 0)
            kelly_j2 = None
        elif ev_j2 > umbral_ev:
            recomendacion = f"APOSTAR a {request.jugador2_nombre}"
            mejor_opcion = request.jugador2_nombre
            kelly_j1 = None
            kelly_pct = (prob_j2 * request.jugador2_cuota - 1) / (request.jugador2_cuota - 1)
            kelly_j2 = round(kelly_pct * 0.25 * 100, 2)
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
            cursor = db.conn.cursor()
            cursor.execute(
                "SELECT * FROM bets WHERE match_id = ? AND estado = 'completada'", (match_id,)
            )
            bet = cursor.fetchone()

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

        # Determinar recomendación
        umbral_ev = Config.EV_THRESHOLD
        if ev_j1 > umbral_ev:
            recomendacion_nueva = f"APOSTAR a {partido['jugador1_nombre']}"
            mejor_opcion_nueva = partido["jugador1_nombre"]
            kelly_j1 = resultado_pred.get("stake_recomendado", 0)
            kelly_j2 = None
        elif ev_j2 > umbral_ev:
            recomendacion_nueva = f"APOSTAR a {partido['jugador2_nombre']}"
            mejor_opcion_nueva = partido["jugador2_nombre"]
            kelly_j1 = None
            kelly_pct = (prob_j2 * jugador2_cuota - 1) / (jugador2_cuota - 1)
            kelly_j2 = round(kelly_pct * 0.25 * 100, 2)
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
        )

        # 6. Analizar cambios en predicción (SIN modificar apuestas existentes)
        cambio_recomendacion = False
        apuesta_actual = None
        nueva_recomendacion_info = None

        # Verificar si ya existe una apuesta registrada
        cursor = db.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM bets 
            WHERE match_id = ? AND estado = 'activa'
        """,
            (match_id,),
        )
        bet_existente = cursor.fetchone()

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
        result = update_service.update_all_pending_matches()
        return result
    except Exception as e:
        logger.error(f"❌ Error en actualización manual: {e}", exc_info=True)
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

        next_run = None
        if jobs:
            next_run = jobs[0].next_run_time.isoformat() if jobs[0].next_run_time else None

        stats = update_service.get_update_stats()

        return {
            "scheduler_running": is_running,
            "total_jobs": len(jobs),
            "next_run": next_run,
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
        pending = update_service.get_pending_matches()
        return {"total": len(pending), "partidos": pending}
    except Exception as e:
        logger.error(f"❌ Error obteniendo partidos pendientes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/detect-new-matches", tags=["Admin"])
async def detect_new_matches_manual():
    """
    Detecta manualmente partidos nuevos en The Odds API

    Útil para testing y debugging. Busca partidos nuevos en The Odds API
    y los crea automáticamente en la base de datos.

    Returns:
        Resultado de la detección
    """
    try:
        logger.info("🔧 Detección manual de partidos nuevos solicitada")
        result = update_service.detect_new_matches()
        return result
    except Exception as e:
        logger.error(f"❌ Error en detección manual: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


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
        import json

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


@app.on_event("startup")
async def startup_event():
    """Inicialización al arrancar la API"""
    logger.info("=" * 70)
    logger.info("🚀 INICIANDO API v2.0")
    logger.info("=" * 70)
    logger.info(f"📡 Servidor: http://localhost:8000")
    logger.info(f"📚 Documentación: http://localhost:8000/docs")
    logger.info("=" * 70)

    # Configurar scheduler para actualizaciones automáticas cada 15 minutos
    try:
        # Job 1: Actualizar cuotas y detectar partidos nuevos (cada 15 min)
        scheduler.add_job(
            func=update_service.update_all_pending_matches,
            trigger=IntervalTrigger(minutes=15),
            id="update_odds_job",
            name="Actualización automática de cuotas",
            replace_existing=True,
        )

        # Job 2: Verificar commits en TML-Database (cada hora)
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

        scheduler.start()
        logger.info("✅ Scheduler iniciado:")
        logger.info("   - Actualizaciones de cuotas: cada 15 minutos")
        logger.info("   - Verificación de commits TML: cada hora")
    except Exception as e:
        logger.error(f"❌ Error iniciando scheduler: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Limpieza al cerrar la API"""
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
