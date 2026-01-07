"""
API REST v2 con FastAPI - Tennis ML Predictor
==============================================

API moderna y rápida para servir predicciones de tenis con:
- Documentación automática (Swagger UI)
- Validación automática de requests
- Mejor rendimiento que Flask
- Async/await support
- CORS configurado

Endpoints:
    GET  /                    - Información de la API
    GET  /health              - Health check
    POST /predict             - Generar predicción para un partido
    GET  /predictions/today   - Predicciones del día
    GET  /predictions/upcoming - Predicciones próximas
    GET  /stats               - Estadísticas del sistema
    GET  /history             - Historial de predicciones
    GET  /docs                - Documentación interactiva (Swagger)

Uso:
    # Desarrollo
    python src/api/api_server_v2.py
    
    # Producción
    uvicorn src.api.api_server_v2:app --host 0.0.0.0 --port 8000 --workers 4
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime, date, timedelta
from pathlib import Path
import sys
import logging
from typing import List, Optional
import pandas as pd

# Añadir src al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.models import (
    MatchPredictionRequest,
    PredictionResponse,
    MatchInfo,
    SystemStats,
    HealthResponse,
    ErrorResponse
)
from src.config.settings import Config
from src.tracking.database_setup import TennisDatabase

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Crear aplicación FastAPI
app = FastAPI(
    title="Tennis ML Predictor API",
    description="API REST para predicciones de tenis con Machine Learning",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar componentes
db = TennisDatabase(Config.DB_PATH)
predictor = None  # Se carga bajo demanda


def get_predictor():
    """Carga el predictor bajo demanda"""
    global predictor
    if predictor is None:
        try:
            # Importar aquí para evitar circular imports
            from src.prediction.predictor_calibrado import PredictorCalibrado
            predictor = PredictorCalibrado(Config.MODEL_PATH)
            logger.info(f"✅ Modelo cargado desde {Config.MODEL_PATH}")
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")
            raise HTTPException(status_code=500, detail=f"Error cargando modelo: {str(e)}")
    return predictor


@app.get("/", tags=["Info"])
async def root():
    """Información básica de la API"""
    return {
        "name": "Tennis ML Predictor API",
        "version": "2.0.0",
        "description": "API REST para predicciones de tenis con Machine Learning",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "today": "/predictions/today",
            "upcoming": "/predictions/upcoming",
            "stats": "/stats",
            "history": "/history"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check del sistema
    
    Verifica que:
    - La API está funcionando
    - El modelo está cargado (opcional)
    - La base de datos está accesible
    
    Retorna 200 OK siempre que la API responda, incluso si el modelo no está cargado.
    """
    # Verificar modelo (no crítico)
    model_loaded = False
    try:
        # Solo verificar si el archivo existe, no intentar cargarlo
        model_path = Path(Config.MODEL_PATH)
        if model_path.exists():
            model_loaded = True
            logger.info(f"✅ Modelo encontrado en {Config.MODEL_PATH}")
        else:
            logger.warning(f"⚠️  Modelo no encontrado en {Config.MODEL_PATH}")
    except Exception as e:
        logger.warning(f"⚠️  Error verificando modelo: {e}")
    
    # Verificar base de datos (no crítico)
    db_connected = False
    try:
        db.obtener_predicciones()
        db_connected = True
        logger.info("✅ Base de datos conectada")
    except Exception as e:
        logger.warning(f"⚠️  Error conectando a base de datos: {e}")
    
    # Siempre retornar 200 OK si la API responde
    return HealthResponse(
        status="ok" if (model_loaded and db_connected) else "degraded",
        timestamp=datetime.now(),
        model_loaded=model_loaded,
        database_connected=db_connected
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_match(request: MatchPredictionRequest):
    """
    Genera una predicción para un partido
    
    Args:
        request: Datos del partido (jugadores, rankings, superficie, cuota)
    
    Returns:
        Predicción con probabilidad, EV, decisión y stake recomendado
    
    Example:
        ```json
        {
            "jugador_nombre": "Alcaraz",
            "jugador_rank": 2,
            "oponente_nombre": "Sinner",
            "oponente_rank": 1,
            "superficie": "Hard",
            "cuota": 2.10
        }
        ```
    """
    try:
        pred = get_predictor()
        
        # Generar predicción
        resultado = pred.predecir_partido(
            jugador1=request.jugador_nombre,
            jugador1_rank=request.jugador_rank,
            jugador2=request.oponente_nombre,
            jugador2_rank=request.oponente_rank,
            superficie=request.superficie.value,
            cuota=request.cuota
        )
        
        # Formatear respuesta
        probabilidad = resultado.get('probabilidad', 0.5)
        ev = resultado.get('expected_value', 0)
        
        # Determinar decisión
        decision = "APOSTAR ✅" if ev > Config.EV_THRESHOLD else "NO APOSTAR ❌"
        
        # Determinar confianza
        if abs(probabilidad - 0.5) > 0.15:
            confianza = "Alta"
        elif abs(probabilidad - 0.5) > 0.08:
            confianza = "Media"
        else:
            confianza = "Baja"
        
        # Kelly stake (si está habilitado)
        kelly_stake = None
        if Config.KELLY_ENABLED and ev > 0:
            kelly_pct = (probabilidad * request.cuota - 1) / (request.cuota - 1)
            kelly_stake = round(kelly_pct * Config.KELLY_FRACTION * 100, 2)
        
        # Razón
        if ev > Config.EV_THRESHOLD:
            razon = f"EV positivo ({ev*100:.1f}%) con confianza {confianza.lower()}"
        else:
            razon = f"EV insuficiente ({ev*100:.1f}%), umbral mínimo {Config.EV_THRESHOLD*100:.1f}%"
        
        return PredictionResponse(
            probabilidad=probabilidad,
            probabilidad_porcentaje=f"{probabilidad*100:.2f}%",
            expected_value=ev,
            decision=decision,
            confianza=confianza,
            kelly_stake=kelly_stake,
            razon=razon
        )
    
    except Exception as e:
        logger.error(f"Error en predicción: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions/today", response_model=List[MatchInfo], tags=["Predictions"])
async def get_today_predictions():
    """
    Obtiene predicciones del día actual
    
    Returns:
        Lista de partidos con predicciones para hoy
    """
    try:
        df = db.obtener_predicciones()
        
        if df.empty:
            return []
        
        # Filtrar por hoy y decisión de apostar
        df['fecha_partido'] = pd.to_datetime(df['fecha_partido'])
        df_hoy = df[
            (df['fecha_partido'].dt.date == date.today()) &
            (df['decision'].str.contains('APOSTAR', na=False))
        ]
        
        # Ordenar por EV descendente
        if 'expected_value' in df_hoy.columns:
            df_hoy = df_hoy.sort_values('expected_value', ascending=False)
        
        # Convertir a lista de MatchInfo
        matches = []
        for _, row in df_hoy.iterrows():
            matches.append(MatchInfo(
                id=row.get('id'),
                fecha_partido=row['fecha_partido'],
                jugador1=row.get('jugador1', ''),
                jugador2=row.get('jugador2', ''),
                superficie=row.get('superficie', ''),
                probabilidad=row.get('probabilidad', 0),
                cuota=row.get('cuota', 0),
                expected_value=row.get('expected_value', 0),
                decision=row.get('decision', ''),
                resultado_real=row.get('resultado_real')
            ))
        
        return matches
    
    except Exception as e:
        logger.error(f"Error obteniendo predicciones de hoy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions/upcoming", response_model=List[MatchInfo], tags=["Predictions"])
async def get_upcoming_predictions(
    days: int = Query(default=7, ge=1, le=30, description="Días hacia adelante")
):
    """
    Obtiene predicciones para los próximos días
    
    Args:
        days: Número de días hacia adelante (1-30)
    
    Returns:
        Lista de partidos con predicciones
    """
    try:
        df = db.obtener_predicciones()
        
        if df.empty:
            return []
        
        # Filtrar por rango de fechas
        df['fecha_partido'] = pd.to_datetime(df['fecha_partido'])
        fecha_limite = date.today() + timedelta(days=days)
        
        df_upcoming = df[
            (df['fecha_partido'].dt.date >= date.today()) &
            (df['fecha_partido'].dt.date <= fecha_limite) &
            (df['decision'].str.contains('APOSTAR', na=False))
        ]
        
        # Ordenar por fecha y EV
        if 'expected_value' in df_upcoming.columns:
            df_upcoming = df_upcoming.sort_values(['fecha_partido', 'expected_value'], 
                                                  ascending=[True, False])
        
        # Convertir a lista
        matches = []
        for _, row in df_upcoming.iterrows():
            matches.append(MatchInfo(
                id=row.get('id'),
                fecha_partido=row['fecha_partido'],
                jugador1=row.get('jugador1', ''),
                jugador2=row.get('jugador2', ''),
                superficie=row.get('superficie', ''),
                probabilidad=row.get('probabilidad', 0),
                cuota=row.get('cuota', 0),
                expected_value=row.get('expected_value', 0),
                decision=row.get('decision', ''),
                resultado_real=row.get('resultado_real')
            ))
        
        return matches
    
    except Exception as e:
        logger.error(f"Error obteniendo predicciones upcoming: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=SystemStats, tags=["Statistics"])
async def get_system_stats():
    """
    Obtiene estadísticas generales del sistema
    
    Returns:
        Estadísticas de predicciones, win rate, ROI, etc.
    """
    try:
        df = db.obtener_predicciones({'decision': 'APOSTAR'})
        
        if df.empty:
            return SystemStats(
                total_predicciones=0,
                predicciones_completadas=0,
                predicciones_pendientes=0,
                win_rate=0.0,
                roi=0.0
            )
        
        # Calcular estadísticas
        completadas = df[df['resultado_real'].notna()]
        
        stats = {
            'total_predicciones': len(df),
            'predicciones_completadas': len(completadas),
            'predicciones_pendientes': len(df) - len(completadas),
            'win_rate': 0.0,
            'roi': 0.0
        }
        
        if len(completadas) > 0:
            ganadas = (completadas['resultado_real'] == 1).sum()
            stats['win_rate'] = ganadas / len(completadas)
            
            # Calcular ROI si hay información de stakes
            if 'stake' in completadas.columns and 'ganancia' in completadas.columns:
                total_apostado = completadas['stake'].sum()
                total_ganancia = completadas['ganancia'].sum()
                stats['roi'] = (total_ganancia / total_apostado) if total_apostado > 0 else 0
        
        # Añadir accuracy del modelo si existe
        import json
        metricas_path = Path(Config.MODEL_PATH).parent / "production_model_metrics.json"
        if metricas_path.exists():
            with open(metricas_path, 'r') as f:
                metricas = json.load(f)
                stats['accuracy'] = metricas.get('accuracy')
        
        return SystemStats(**stats)
    
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history", response_model=List[MatchInfo], tags=["History"])
async def get_history(
    limit: int = Query(default=50, ge=1, le=500, description="Número de resultados"),
    decision: Optional[str] = Query(default=None, description="Filtrar por decisión")
):
    """
    Obtiene historial de predicciones
    
    Args:
        limit: Número máximo de resultados (1-500)
        decision: Filtrar por decisión (APOSTAR, NO APOSTAR)
    
    Returns:
        Lista de predicciones históricas
    """
    try:
        filtros = {}
        if decision:
            filtros['decision'] = decision
        
        df = db.obtener_predicciones(filtros)
        
        if df.empty:
            return []
        
        # Ordenar por fecha descendente
        if 'fecha_prediccion' in df.columns:
            df['fecha_prediccion'] = pd.to_datetime(df['fecha_prediccion'])
            df = df.sort_values('fecha_prediccion', ascending=False)
        
        # Limitar resultados
        df = df.head(limit)
        
        # Convertir a lista
        matches = []
        for _, row in df.iterrows():
            matches.append(MatchInfo(
                id=row.get('id'),
                fecha_partido=pd.to_datetime(row['fecha_partido']),
                jugador1=row.get('jugador1', ''),
                jugador2=row.get('jugador2', ''),
                superficie=row.get('superficie', ''),
                probabilidad=row.get('probabilidad', 0),
                cuota=row.get('cuota', 0),
                expected_value=row.get('expected_value', 0),
                decision=row.get('decision', ''),
                resultado_real=row.get('resultado_real')
            ))
        
        return matches
    
    except Exception as e:
        logger.error(f"Error obteniendo historial: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Manejador de errores global
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Manejador global de excepciones"""
    logger.error(f"Error no manejado: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            message="Error interno del servidor",
            detail=str(exc)
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 70)
    print("🚀 INICIANDO API REST v2 (FastAPI)")
    print("=" * 70)
    print(f"📡 Servidor: http://localhost:8000")
    print(f"📚 Documentación: http://localhost:8000/docs")
    print(f"📖 ReDoc: http://localhost:8000/redoc")
    print("\n📋 Endpoints disponibles:")
    print("  GET  /                    - Información de la API")
    print("  GET  /health              - Health check")
    print("  POST /predict             - Generar predicción")
    print("  GET  /predictions/today   - Predicciones de hoy")
    print("  GET  /predictions/upcoming - Predicciones próximas")
    print("  GET  /stats               - Estadísticas del sistema")
    print("  GET  /history             - Historial de predicciones")
    print("=" * 70)
    
    # Ejecutar servidor
    uvicorn.run(
        "api_server_v2:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload en desarrollo
        log_level="info"
    )
