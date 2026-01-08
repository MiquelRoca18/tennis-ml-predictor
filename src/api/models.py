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
    jugador1_nombre: str = Field(..., description="Nombre del jugador 1", min_length=2)
    jugador1_cuota: float = Field(..., description="Cuota para jugador 1", gt=1.0, le=100.0)
    jugador2_nombre: str = Field(..., description="Nombre del jugador 2", min_length=2)
    jugador2_cuota: float = Field(..., description="Cuota para jugador 2", gt=1.0, le=100.0)
    superficie: Superficie = Field(..., description="Superficie del partido")
    
    # Campos opcionales para compatibilidad con versión anterior
    jugador_nombre: Optional[str] = Field(None, description="(Deprecated) Usar jugador1_nombre")
    oponente_nombre: Optional[str] = Field(None, description="(Deprecated) Usar jugador2_nombre")
    cuota: Optional[float] = Field(None, description="(Deprecated) Usar jugador1_cuota")
    jugador_rank: Optional[int] = Field(None, description="Ranking ATP del jugador (opcional, se obtiene del histórico)", ge=1, le=1000)
    oponente_rank: Optional[int] = Field(None, description="Ranking ATP del oponente (opcional, se obtiene del histórico)", ge=1, le=1000)
    
    class Config:
        json_schema_extra = {
            "example": {
                "jugador1_nombre": "Alcaraz",
                "jugador1_cuota": 2.10,
                "jugador2_nombre": "Sinner",
                "jugador2_cuota": 1.75,
                "superficie": "Hard"
            }
        }


class PlayerPrediction(BaseModel):
    """Predicción para un jugador específico"""
    nombre: str
    probabilidad: float
    probabilidad_porcentaje: str
    cuota: float
    expected_value: float
    decision: str
    kelly_stake: Optional[float] = None
    edge: float


class DualPredictionResponse(BaseModel):
    """Response con análisis de ambas opciones de apuesta"""
    jugador1: PlayerPrediction
    jugador2: PlayerPrediction
    recomendacion_final: str
    mejor_opcion: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "jugador1": {
                    "nombre": "Alcaraz",
                    "probabilidad": 0.44,
                    "probabilidad_porcentaje": "44.42%",
                    "cuota": 2.10,
                    "expected_value": -0.067,
                    "decision": "NO APOSTAR ❌",
                    "kelly_stake": None,
                    "edge": -0.032
                },
                "jugador2": {
                    "nombre": "Sinner",
                    "probabilidad": 0.56,
                    "probabilidad_porcentaje": "55.58%",
                    "cuota": 1.75,
                    "expected_value": -0.027,
                    "decision": "NO APOSTAR ❌",
                    "kelly_stake": None,
                    "edge": -0.015
                },
                "recomendacion_final": "Ninguna apuesta recomendada - EV negativo en ambas opciones",
                "mejor_opcion": None
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
