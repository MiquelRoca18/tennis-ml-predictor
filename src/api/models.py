"""
Modelos Pydantic para validación de requests/responses de la API
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
from enum import Enum


class Superficie(str, Enum):
    """Tipos de superficie"""
    HARD = "Hard"
    CLAY = "Clay"
    GRASS = "Grass"
    CARPET = "Carpet"


class MatchPredictionRequest(BaseModel):
    """Request para predicción de partido"""
    jugador_nombre: str = Field(..., description="Nombre del jugador", min_length=2)
    jugador_rank: int = Field(..., description="Ranking ATP del jugador", ge=1, le=1000)
    oponente_nombre: str = Field(..., description="Nombre del oponente", min_length=2)
    oponente_rank: int = Field(..., description="Ranking ATP del oponente", ge=1, le=1000)
    superficie: Superficie = Field(..., description="Superficie del partido")
    cuota: float = Field(..., description="Cuota del bookmaker", gt=1.0, le=100.0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "jugador_nombre": "Alcaraz",
                "jugador_rank": 2,
                "oponente_nombre": "Sinner",
                "oponente_rank": 1,
                "superficie": "Hard",
                "cuota": 2.10
            }
        }


class PredictionResponse(BaseModel):
    """Response de predicción"""
    probabilidad: float = Field(..., description="Probabilidad de victoria (0-1)")
    probabilidad_porcentaje: str = Field(..., description="Probabilidad en formato porcentaje")
    expected_value: float = Field(..., description="Valor esperado de la apuesta")
    decision: str = Field(..., description="APOSTAR o NO APOSTAR")
    confianza: str = Field(..., description="Alta, Media o Baja")
    kelly_stake: Optional[float] = Field(None, description="Stake recomendado (Kelly Criterion)")
    razon: str = Field(..., description="Razón de la decisión")
    
    class Config:
        json_schema_extra = {
            "example": {
                "probabilidad": 0.65,
                "probabilidad_porcentaje": "65.00%",
                "expected_value": 0.365,
                "decision": "APOSTAR ✅",
                "confianza": "Alta",
                "kelly_stake": 12.5,
                "razon": "EV positivo (36.5%) con alta confianza"
            }
        }


class MatchInfo(BaseModel):
    """Información de un partido"""
    id: Optional[int] = None
    fecha_partido: datetime
    jugador1: str
    jugador2: str
    superficie: str
    probabilidad: float
    cuota: float
    expected_value: float
    decision: str
    resultado_real: Optional[int] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": 123,
                "fecha_partido": "2026-01-07T14:00:00",
                "jugador1": "Djokovic",
                "jugador2": "Nadal",
                "superficie": "Clay",
                "probabilidad": 0.58,
                "cuota": 1.85,
                "expected_value": 0.073,
                "decision": "APOSTAR ✅",
                "resultado_real": None
            }
        }


class SystemStats(BaseModel):
    """Estadísticas del sistema"""
    total_predicciones: int
    predicciones_completadas: int
    predicciones_pendientes: int
    win_rate: float
    roi: float
    accuracy: Optional[float] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_predicciones": 150,
                "predicciones_completadas": 120,
                "predicciones_pendientes": 30,
                "win_rate": 0.65,
                "roi": 0.12,
                "accuracy": 0.71
            }
        }


class HealthResponse(BaseModel):
    """Response del health check"""
    status: str
    timestamp: datetime
    model_loaded: bool
    database_connected: bool
    version: str = "2.0.0"
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok",
                "timestamp": "2026-01-07T09:41:00",
                "model_loaded": True,
                "database_connected": True,
                "version": "2.0.0"
            }
        }


class ErrorResponse(BaseModel):
    """Response de error"""
    status: str = "error"
    message: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "error",
                "message": "Campo requerido faltante",
                "detail": "El campo 'jugador_nombre' es obligatorio",
                "timestamp": "2026-01-07T09:41:00"
            }
        }
