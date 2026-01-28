"""
Modelos Pydantic para el Detalle de Partido - Marcador Profesional
===================================================================

Estos modelos representan todos los datos necesarios para mostrar
un partido de tenis como un marcador deportivo profesional.

Estructura:
- MatchFullResponse: Respuesta completa con todos los datos
- PlayerInfo: Información del jugador
- MatchScores: Scores del partido (sets, live)
- MatchStats: Estadísticas calculadas
- MatchTimeline: Timeline de juegos
- H2HData: Head to Head
"""

from datetime import date, time
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ============================================================
# ENUMS
# ============================================================

class MatchStatus(str, Enum):
    """Estado del partido"""
    SCHEDULED = "pendiente"
    LIVE = "en_juego"
    FINISHED = "completado"
    SUSPENDED = "suspendido"
    CANCELLED = "cancelado"


class Surface(str, Enum):
    """Superficie de la pista"""
    HARD = "Hard"
    CLAY = "Clay"
    GRASS = "Grass"
    CARPET = "Carpet"
    INDOOR = "Indoor"


# ============================================================
# PLAYER INFO
# ============================================================

class PlayerInfo(BaseModel):
    """Información de un jugador"""
    name: str = Field(..., description="Nombre del jugador")
    country: Optional[str] = Field(None, description="País (código ISO)")
    ranking: Optional[int] = Field(None, description="Ranking ATP/WTA")
    logo_url: Optional[str] = Field(None, description="URL del logo/foto")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "C. Alcaraz",
                "country": "ESP",
                "ranking": 3,
                "logo_url": "https://api.api-tennis.com/logo-tennis/2382_c-alcaraz.jpg"
            }
        }


# ============================================================
# SCORES
# ============================================================

class SetScore(BaseModel):
    """Score de un set individual"""
    set_number: int = Field(..., description="Número del set (1, 2, 3...)")
    player1_games: int = Field(..., description="Juegos ganados por jugador 1")
    player2_games: int = Field(..., description="Juegos ganados por jugador 2")
    tiebreak_score: Optional[str] = Field(None, description="Score del tiebreak (ej: '7-5')")
    winner: Optional[int] = Field(None, description="1 o 2 según quién ganó el set")
    
    class Config:
        json_schema_extra = {
            "example": {
                "set_number": 3,
                "player1_games": 7,
                "player2_games": 6,
                "tiebreak_score": "7-5",
                "winner": 1
            }
        }


class LiveScore(BaseModel):
    """Datos en vivo del partido (solo cuando está en juego)"""
    current_game: str = Field(..., description="Score del juego actual (ej: '30-15')")
    current_server: int = Field(..., description="1 o 2 según quién saca")
    current_set: int = Field(..., description="Set actual")
    is_tiebreak: bool = Field(False, description="Si está en tiebreak")
    
    class Config:
        json_schema_extra = {
            "example": {
                "current_game": "30-15",
                "current_server": 1,
                "current_set": 3,
                "is_tiebreak": False
            }
        }


class MatchScores(BaseModel):
    """Todos los scores del partido"""
    sets_won: List[int] = Field(..., description="[sets_p1, sets_p2]")
    sets: List[SetScore] = Field(default_factory=list, description="Detalle de cada set")
    live: Optional[LiveScore] = Field(None, description="Datos en vivo si está en juego")
    
    class Config:
        json_schema_extra = {
            "example": {
                "sets_won": [2, 1],
                "sets": [
                    {"set_number": 1, "player1_games": 6, "player2_games": 4, "winner": 1},
                    {"set_number": 2, "player1_games": 4, "player2_games": 6, "winner": 2},
                    {"set_number": 3, "player1_games": 7, "player2_games": 6, "tiebreak_score": "7-5", "winner": 1}
                ]
            }
        }


# ============================================================
# STATISTICS
# ============================================================

class ServeStats(BaseModel):
    """Estadísticas de saque de un jugador"""
    aces: int = Field(0, description="Aces")
    double_faults: int = Field(0, description="Dobles faltas")
    first_serve_pct: float = Field(0, description="% primer servicio")
    first_serve_won_pct: float = Field(0, description="% puntos ganados al 1er servicio")
    second_serve_won_pct: float = Field(0, description="% puntos ganados al 2do servicio")
    service_games_won: int = Field(0, description="Juegos de saque ganados")
    service_games_total: int = Field(0, description="Total juegos al saque")


class ReturnStats(BaseModel):
    """Estadísticas de resto de un jugador"""
    return_points_won: int = Field(0, description="Puntos ganados al resto")
    return_points_total: int = Field(0, description="Total puntos al resto")
    return_games_won: int = Field(0, description="Juegos de resto ganados (breaks)")
    return_games_total: int = Field(0, description="Total juegos al resto")


class BreakPointStats(BaseModel):
    """Estadísticas de break points de un jugador"""
    break_points_won: int = Field(0, description="BP convertidos")
    break_points_total: int = Field(0, description="BP totales a favor")
    break_points_saved: int = Field(0, description="BP salvados")
    break_points_faced: int = Field(0, description="BP enfrentados")
    
    @property
    def conversion_pct(self) -> float:
        """Porcentaje de conversión de BP"""
        if self.break_points_total == 0:
            return 0.0
        return round(self.break_points_won / self.break_points_total * 100, 1)
    
    @property
    def save_pct(self) -> float:
        """Porcentaje de BP salvados"""
        if self.break_points_faced == 0:
            return 0.0
        return round(self.break_points_saved / self.break_points_faced * 100, 1)


class PlayerStats(BaseModel):
    """Todas las estadísticas de un jugador"""
    serve: ServeStats = Field(default_factory=ServeStats)
    return_: ReturnStats = Field(default_factory=ReturnStats, alias="return")
    break_points: BreakPointStats = Field(default_factory=BreakPointStats)
    total_points_won: int = Field(0, description="Total puntos ganados")
    total_games_won: int = Field(0, description="Total juegos ganados")
    winners: int = Field(0, description="Winners (si disponible)")
    unforced_errors: int = Field(0, description="Errores no forzados (si disponible)")
    
    class Config:
        populate_by_name = True


class MatchStats(BaseModel):
    """Estadísticas completas del partido"""
    player1: PlayerStats = Field(default_factory=PlayerStats)
    player2: PlayerStats = Field(default_factory=PlayerStats)
    total_games: int = Field(0, description="Total de juegos jugados")
    total_points: int = Field(0, description="Total de puntos jugados")
    duration_minutes: Optional[int] = Field(None, description="Duración en minutos")
    has_detailed_stats: bool = Field(False, description="Si hay stats detalladas disponibles")


# ============================================================
# TIMELINE
# ============================================================

class GameInfo(BaseModel):
    """Información de un juego individual"""
    set_number: int = Field(..., description="Número del set")
    game_number: int = Field(..., description="Número del juego en el set")
    server: int = Field(..., description="1 o 2 según quién sacó")
    winner: int = Field(..., description="1 o 2 según quién ganó")
    is_break: bool = Field(False, description="Si fue break")
    score_after: str = Field(..., description="Score después del juego (ej: '3-2')")


class SetTimeline(BaseModel):
    """Timeline de un set"""
    set_number: int
    games: List[GameInfo]
    final_score: str = Field(..., description="Score final del set (ej: '6-4')")
    has_tiebreak: bool = Field(False)
    tiebreak_score: Optional[str] = Field(None)
    breaks_player1: int = Field(0, description="Breaks de jugador 1 en este set")
    breaks_player2: int = Field(0, description="Breaks de jugador 2 en este set")


class MatchTimeline(BaseModel):
    """Timeline completo del partido"""
    sets: List[SetTimeline] = Field(default_factory=list)
    total_games: int = Field(0)
    total_breaks: int = Field(0)
    momentum_shifts: int = Field(0, description="Cambios de momentum (breaks consecutivos)")


# ============================================================
# POINT BY POINT
# ============================================================

class PointInfo(BaseModel):
    """Información de un punto individual"""
    set_number: int
    game_number: int
    point_number: int
    score: str = Field(..., description="Score después del punto (ej: '30-15')")
    server: int = Field(..., description="1 o 2")
    winner: Optional[int] = Field(None, description="1 o 2")
    is_break_point: bool = Field(False)
    is_set_point: bool = Field(False)
    is_match_point: bool = Field(False)


class PointByPointData(BaseModel):
    """Datos punto por punto"""
    total_points: int = Field(0)
    points: List[PointInfo] = Field(default_factory=list)
    key_points: List[PointInfo] = Field(default_factory=list, description="Puntos clave (BP, SP, MP)")


# ============================================================
# HEAD TO HEAD
# ============================================================

class PreviousMatch(BaseModel):
    """Partido previo entre los jugadores"""
    date: str
    tournament: str
    surface: str
    winner: int = Field(..., description="1 o 2")
    score: str = Field(..., description="Score final")


class H2HData(BaseModel):
    """Datos de Head to Head"""
    total_matches: int = Field(0)
    player1_wins: int = Field(0)
    player2_wins: int = Field(0)
    matches: List[PreviousMatch] = Field(default_factory=list, description="Últimos enfrentamientos")
    
    # Stats por superficie
    hard_record: List[int] = Field(default_factory=lambda: [0, 0], description="[p1_wins, p2_wins] en hard")
    clay_record: List[int] = Field(default_factory=lambda: [0, 0], description="[p1_wins, p2_wins] en clay")
    grass_record: List[int] = Field(default_factory=lambda: [0, 0], description="[p1_wins, p2_wins] en grass")


# ============================================================
# ODDS
# ============================================================

class BookmakerOdds(BaseModel):
    """Cuotas de una casa de apuestas"""
    bookmaker: str
    player1_odds: float
    player2_odds: float
    updated_at: Optional[str] = None


class MatchOdds(BaseModel):
    """Todas las cuotas del partido"""
    best_odds_player1: Optional[float] = None
    best_odds_player2: Optional[float] = None
    bookmakers: List[BookmakerOdds] = Field(default_factory=list)
    market_consensus: Optional[int] = Field(None, description="1 o 2 según el favorito del mercado")


# ============================================================
# PREDICTION (si existe)
# ============================================================

class MatchPrediction(BaseModel):
    """Predicción del modelo ML"""
    predicted_winner: int = Field(..., description="1 o 2")
    confidence: float = Field(..., description="0-100%")
    probability_player1: float
    probability_player2: float
    value_bet: Optional[int] = Field(None, description="1, 2 o None si no hay value")
    recommendation: Optional[str] = None


# ============================================================
# MAIN RESPONSE
# ============================================================

class MatchInfo(BaseModel):
    """Información básica del partido"""
    id: int
    status: MatchStatus
    date: date
    time: Optional[time] = None
    tournament: str
    round: Optional[str] = None
    surface: Surface
    court: Optional[str] = None


class MatchFullResponse(BaseModel):
    """
    Respuesta completa del endpoint /matches/{id}/full
    
    Contiene todos los datos necesarios para mostrar el detalle
    de un partido como un marcador deportivo profesional.
    """
    # Información básica
    match: MatchInfo
    player1: PlayerInfo
    player2: PlayerInfo
    winner: Optional[int] = Field(None, description="1 o 2 si terminó")
    
    # Scores
    scores: Optional[MatchScores] = None
    
    # Estadísticas
    stats: Optional[MatchStats] = None
    
    # Timeline
    timeline: Optional[MatchTimeline] = None
    
    # Head to Head
    h2h: Optional[H2HData] = None
    
    # Cuotas
    odds: Optional[MatchOdds] = None
    
    # Predicción
    prediction: Optional[MatchPrediction] = None
    
    # Metadata
    last_updated: Optional[str] = None
    data_quality: str = Field("basic", description="basic, partial, full")
    
    class Config:
        json_schema_extra = {
            "example": {
                "match": {
                    "id": 123,
                    "status": "completado",
                    "date": "2026-01-27",
                    "time": "10:30:00",
                    "tournament": "ATP Australian Open",
                    "round": "Quarter-finals",
                    "surface": "Hard"
                },
                "player1": {"name": "C. Alcaraz", "country": "ESP", "ranking": 3},
                "player2": {"name": "A. De Minaur", "country": "AUS", "ranking": 8},
                "winner": 1,
                "scores": {
                    "sets_won": [3, 0],
                    "sets": [
                        {"set_number": 1, "player1_games": 6, "player2_games": 4, "winner": 1}
                    ]
                }
            }
        }
