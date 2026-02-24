"""
Modelos Pydantic para API Tennis ML Predictor v2.0
===================================================

Modelos de request/response para el nuevo sistema de gestión de partidos
con predicciones versionadas y tracking de apuestas.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, date, time
from enum import Enum


# ============================================================
# ENUMS
# ============================================================


class Superficie(str, Enum):
    """Tipos de superficie"""

    HARD = "Hard"
    CLAY = "Clay"
    GRASS = "Grass"
    CARPET = "Carpet"


class EstadoPartido(str, Enum):
    """Estados posibles de un partido"""

    PENDIENTE = "pendiente"
    EN_JUEGO = "en_juego"
    COMPLETADO = "completado"
    CANCELADO = "cancelado"


class ResultadoApuesta(str, Enum):
    """Resultados posibles de una apuesta"""

    GANADA = "ganada"
    PERDIDA = "perdida"
    CANCELADA = "cancelada"


class Confianza(str, Enum):
    """Niveles de confianza"""

    ALTA = "Alta"
    MEDIA = "Media"
    BAJA = "Baja"


# ============================================================
# MODELOS DE JUGADOR
# ============================================================


class JugadorInfo(BaseModel):
    """Información de un jugador"""

    nombre: str
    ranking: Optional[int] = None
    cuota: float = Field(..., ge=0.0, le=500.0)  # 0 = sin cuota; 1-500 = cuota válida
    logo: Optional[str] = None  # URL del logo del jugador desde API-Tennis


# ============================================================
# MODELOS DE PREDICCIÓN
# ============================================================


class PredictionVersion(BaseModel):
    """Una versión específica de predicción"""

    version: int
    timestamp: datetime

    # Cuotas
    jugador1_cuota: float
    jugador2_cuota: float

    # Probabilidades
    jugador1_probabilidad: float = Field(..., ge=0.0, le=1.0)
    jugador2_probabilidad: float = Field(..., ge=0.0, le=1.0)

    # Expected Value
    jugador1_ev: float
    jugador2_ev: float

    # Edge
    jugador1_edge: Optional[float] = None
    jugador2_edge: Optional[float] = None

    # Recomendación
    recomendacion: str
    mejor_opcion: Optional[str] = None
    confianza: Optional[Confianza] = None

    # Kelly stakes
    kelly_stake_jugador1: Optional[float] = None
    kelly_stake_jugador2: Optional[float] = None

    # Nivel de confianza del modelo (HIGH, MEDIUM, LOW, UNKNOWN)
    confidence_level: Optional[str] = None
    confidence_score: Optional[float] = None

    class Config:
        json_schema_extra = {
            "example": {
                "version": 1,
                "timestamp": "2026-01-08T10:00:00",
                "jugador1_cuota": 2.10,
                "jugador2_cuota": 1.75,
                "jugador1_probabilidad": 0.44,
                "jugador2_probabilidad": 0.56,
                "jugador1_ev": -0.067,
                "jugador2_ev": -0.027,
                "recomendacion": "NO APOSTAR",
                "mejor_opcion": None,
                "confianza": "Baja",
            }
        }


# ============================================================
# MODELOS DE APUESTA
# ============================================================


class BetRecord(BaseModel):
    """Registro de una apuesta"""

    id: Optional[int] = None
    jugador_apostado: str
    cuota_apostada: float
    stake: float
    timestamp_apuesta: datetime

    # Resultado (NULL si no ha terminado)
    resultado: Optional[ResultadoApuesta] = None
    ganancia: Optional[float] = None
    roi: Optional[float] = None

    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "jugador_apostado": "Djokovic",
                "cuota_apostada": 1.85,
                "stake": 12.5,
                "timestamp_apuesta": "2026-01-08T10:00:00",
                "resultado": "ganada",
                "ganancia": 10.62,
                "roi": 0.85,
            }
        }


# ============================================================
# MODELOS DE RESULTADO Y DATOS EN VIVO
# ============================================================


class SetScoreSimple(BaseModel):
    """Score de un set individual"""
    set_number: int
    player1_score: int
    player2_score: int
    tiebreak_score: Optional[str] = None  # Ej: "7-5" si hubo tiebreak


class LiveData(BaseModel):
    """Datos en vivo del partido (solo cuando está en juego)"""
    
    current_game_score: Optional[str] = None  # Ej: "30-15", "40-30", "Deuce"
    current_server: Optional[str] = None  # "First Player" o "Second Player" o nombre
    current_set: Optional[int] = None  # Set actual (1, 2, 3...)
    current_game: Optional[int] = None  # Juego actual del set
    is_tiebreak: bool = False


class MatchScores(BaseModel):
    """Marcador completo del partido"""
    
    # Resultado en sets (ej: "2-0", "2-1")
    sets_result: Optional[str] = None
    
    # Score detallado por set
    sets: List[SetScoreSimple] = []
    
    # Datos en vivo (solo si está en juego)
    live: Optional[LiveData] = None


class MatchResult(BaseModel):
    """Resultado de un partido"""

    ganador: Optional[str] = None
    marcador: Optional[str] = None  # Formato string: "6-4, 7-5, 6-3"
    
    # Scores estructurados
    scores: Optional[MatchScores] = None

    # Si había apuesta
    apostamos: bool = False
    resultado_apuesta: Optional[ResultadoApuesta] = None
    stake: Optional[float] = None
    ganancia: Optional[float] = None
    roi: Optional[float] = None


# ============================================================
# MODELOS DE ESTADÍSTICAS DETALLADAS
# ============================================================


class SetScore(BaseModel):
    """Marcador de un set"""

    set: int
    jugador1: int
    jugador2: int


class MatchStatsBasic(BaseModel):
    """Estadísticas básicas del partido"""

    total_sets: int
    sets_ganados_jugador1: int
    sets_ganados_jugador2: int
    total_juegos: int
    juegos_ganados_jugador1: int
    juegos_ganados_jugador2: int
    marcador_por_sets: List[SetScore]


class PlayerStatsAdvanced(BaseModel):
    """Estadísticas avanzadas de un jugador"""

    juegos_al_saque: int
    juegos_al_resto: int
    juegos_ganados_al_saque: int
    juegos_ganados_al_resto: int
    porcentaje_saque: float
    break_points_enfrentados: int
    break_points_salvados: int
    break_points_a_favor: int
    break_points_convertidos: int
    puntos_totales: int


class MatchStatsAdvanced(BaseModel):
    """Estadísticas avanzadas del partido"""

    jugador1: PlayerStatsAdvanced
    jugador2: PlayerStatsAdvanced


class TimelineEntry(BaseModel):
    """Entrada en el timeline del partido"""

    set: str
    juego: str
    servidor: str
    ganador: str
    marcador_juegos: str
    marcador_sets: str
    fue_break: bool


class MomentumEntry(BaseModel):
    """Entrada de momentum del partido"""

    juego: int
    set: str
    momentum: float  # -100 a +100
    dominando: str  # "jugador1", "jugador2", "equilibrado"


class KeyPoint(BaseModel):
    """Punto clave del partido"""

    tipo: str  # "break_point", "set_point", "match_point"
    set: str
    juego: str
    punto: str
    marcador: str
    descripcion: str


class MatchDetails(BaseModel):
    """Vista Detallada - Estadísticas completas del partido"""

    # Información básica
    match_id: int
    estado: str
    ganador: Optional[str] = None
    duracion_estimada: Optional[str] = None

    # Estadísticas
    estadisticas_basicas: MatchStatsBasic
    estadisticas_avanzadas: Optional[MatchStatsAdvanced] = None


class MatchAnalysis(BaseModel):
    """Análisis Profundo - Análisis completo del partido"""

    # Información básica
    match_id: int
    estado: str
    ganador: Optional[str] = None
    duracion_estimada: Optional[str] = None

    # Estadísticas completas
    estadisticas_basicas: MatchStatsBasic
    estadisticas_avanzadas: Optional[MatchStatsAdvanced] = None

    # Análisis profundo
    timeline: List[TimelineEntry] = []
    momentum: List[MomentumEntry] = []
    puntos_clave: List[KeyPoint] = []


# ============================================================
# REQUEST MODELS
# ============================================================


class MatchCreateRequest(BaseModel):
    """Request para crear un partido y generar predicción"""

    fecha_partido: date
    hora_inicio: Optional[time] = None
    torneo: Optional[str] = None
    ronda: Optional[str] = None
    superficie: Superficie

    jugador1_nombre: str = Field(..., min_length=2)
    jugador1_cuota: float = Field(..., gt=1.0, le=100.0)
    jugador1_ranking: Optional[int] = Field(None, ge=1, le=1000)

    jugador2_nombre: str = Field(..., min_length=2)
    jugador2_cuota: float = Field(..., gt=1.0, le=100.0)
    jugador2_ranking: Optional[int] = Field(None, ge=1, le=1000)

    class Config:
        json_schema_extra = {
            "example": {
                "fecha_partido": "2026-01-09",
                "hora_inicio": "15:00",
                "torneo": "Australian Open",
                "ronda": "Semifinal",
                "superficie": "Hard",
                "jugador1_nombre": "Alcaraz",
                "jugador1_cuota": 2.10,
                "jugador2_nombre": "Sinner",
                "jugador2_cuota": 1.75,
            }
        }


class MatchResultRequest(BaseModel):
    """Request para actualizar resultado de un partido"""

    ganador: str = Field(..., min_length=2)
    marcador: Optional[str] = None

    class Config:
        json_schema_extra = {"example": {"ganador": "Djokovic", "marcador": "6-4, 7-5, 6-3"}}


# ============================================================
# RESPONSE MODELS
# ============================================================


class MatchResponse(BaseModel):
    """Response completa de un partido"""

    id: int
    estado: EstadoPartido
    fecha_partido: date
    hora_inicio: Optional[time] = None
    torneo: Optional[str] = None
    ronda: Optional[str] = None
    superficie: Superficie

    # Jugadores
    jugador1: JugadorInfo
    jugador2: JugadorInfo

    # Predicción (última versión)
    prediccion: Optional[PredictionVersion] = None

    # Resultado (si existe)
    resultado: Optional[MatchResult] = None

    # Motivo de finalización (Retired, Walk Over, etc.) para mostrar en card
    event_status: Optional[str] = None

    # Flag explícito para que la card muestre "En directo" (alineado con estado === en_juego)
    is_live: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "estado": "pendiente",
                "fecha_partido": "2026-01-08",
                "hora_inicio": "14:00",
                "torneo": "Australian Open",
                "ronda": "Cuartos de Final",
                "superficie": "Hard",
                "jugador1": {"nombre": "Alcaraz", "ranking": 3, "cuota": 2.10},
                "jugador2": {"nombre": "Sinner", "ranking": 4, "cuota": 1.75},
                "prediccion": {
                    "version": 1,
                    "timestamp": "2026-01-08T09:00:00",
                    "jugador1_probabilidad": 0.44,
                    "jugador2_probabilidad": 0.56,
                    "recomendacion": "NO APOSTAR",
                },
                "resultado": None,
            }
        }


class MatchesDateResponse(BaseModel):
    """Response de partidos por fecha"""

    fecha: date
    es_hoy: bool
    resumen: dict  # total, completados, pendientes
    partidos: List[MatchResponse]
    # Bankroll usado para los stakes mostrados (permite al frontend mostrar "según tu bankroll de X€")
    betting_config: Optional[dict] = None

    class Config:
        json_schema_extra = {
            "example": {
                "fecha": "2026-01-08",
                "es_hoy": True,
                "resumen": {"total_partidos": 8, "completados": 3, "en_juego": 1, "pendientes": 4},
                "partidos": [],
            }
        }


# ============================================================
# MODELOS DE ESTADÍSTICAS
# ============================================================


class StatsApuestas(BaseModel):
    """Estadísticas de apuestas"""

    total: int
    ganadas: int
    perdidas: int
    win_rate: float = Field(..., ge=0.0, le=1.0)


class StatsFinanciero(BaseModel):
    """Estadísticas financieras"""

    stake_total: float
    ganancia_bruta: float
    ganancia_neta: float
    roi: float


class StatsModelo(BaseModel):
    """Estadísticas del modelo"""

    accuracy: Optional[float] = None
    brier_score: Optional[float] = None
    ev_promedio: Optional[float] = None


class StatsSummaryResponse(BaseModel):
    """Response de resumen de estadísticas"""

    periodo: str
    fecha_inicio: date
    fecha_fin: date
    apuestas: StatsApuestas
    financiero: StatsFinanciero
    modelo: StatsModelo

    class Config:
        json_schema_extra = {
            "example": {
                "periodo": "Últimos 7 días",
                "fecha_inicio": "2026-01-01",
                "fecha_fin": "2026-01-08",
                "apuestas": {"total": 25, "ganadas": 17, "perdidas": 8, "win_rate": 0.68},
                "financiero": {
                    "stake_total": 250.0,
                    "ganancia_bruta": 312.5,
                    "ganancia_neta": 62.5,
                    "roi": 0.25,
                },
                "modelo": {"accuracy": 0.72, "brier_score": 0.18, "ev_promedio": 0.045},
            }
        }


class DailyStat(BaseModel):
    """Estadística de un día"""

    fecha: date
    apuestas: int
    ganadas: int
    win_rate: float
    ganancia: float
    roi: float


class DailyStatsResponse(BaseModel):
    """Response de estadísticas diarias"""

    dias: List[DailyStat]

    class Config:
        json_schema_extra = {
            "example": {
                "dias": [
                    {
                        "fecha": "2026-01-08",
                        "apuestas": 3,
                        "ganadas": 2,
                        "win_rate": 0.67,
                        "ganancia": 15.5,
                        "roi": 0.31,
                    }
                ]
            }
        }


# ============================================================
# MODELOS DE CONFIGURACIÓN
# ============================================================


class ConfigResponse(BaseModel):
    """Response de configuración"""

    ev_threshold: float
    kelly_fraction: float
    bankroll_inicial: float
    update_frequency_minutes: int = 15

    class Config:
        json_schema_extra = {
            "example": {
                "ev_threshold": 0.03,
                "kelly_fraction": 0.25,
                "bankroll_inicial": 1000.0,
                "update_frequency_minutes": 15,
            }
        }
