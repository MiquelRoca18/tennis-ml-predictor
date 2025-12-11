"""
API REST para consultar predicciones
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from datetime import date, datetime
import sys
import os
import logging

# Añadir src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.automation.config import Config
from src.tracking.database_setup import TennisDatabase
from src.predictor_multibookmaker import PredictorMultiBookmaker

app = Flask(__name__)
CORS(app)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Inicializar componentes
db = TennisDatabase(Config.DB_PATH)
predictor = None  # Se carga bajo demanda


def get_predictor():
    """Carga el predictor bajo demanda"""
    global predictor
    if predictor is None:
        predictor = PredictorMultiBookmaker(Config.MODEL_PATH)
    return predictor


@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'model_path': Config.MODEL_PATH,
        'db_path': Config.DB_PATH
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predice un partido
    
    POST /predict
    {
        "jugador_nombre": "Alcaraz",
        "jugador_rank": 3,
        "oponente_nombre": "Sinner",
        "oponente_rank": 1,
        "superficie": "Hard",
        "cuota": 2.10
    }
    """
    try:
        data = request.json
        
        # Validar campos requeridos
        required_fields = ['jugador_nombre', 'jugador_rank', 'oponente_nombre', 
                          'oponente_rank', 'superficie', 'cuota']
        
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'status': 'error',
                    'message': f'Campo requerido faltante: {field}'
                }), 400
        
        # Generar predicción
        pred = get_predictor()
        
        resultado = pred.analizar_partido(
            jugador_nombre=data['jugador_nombre'],
            jugador_rank=data['jugador_rank'],
            oponente_nombre=data['oponente_nombre'],
            oponente_rank=data['oponente_rank'],
            superficie=data['superficie'],
            cuota_jugador=data['cuota'],
            umbral_ev=Config.EV_THRESHOLD
        )
        
        return jsonify({
            'status': 'success',
            'prediccion': resultado
        })
    
    except Exception as e:
        logger.error(f"Error en predicción: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/apuestas/hoy', methods=['GET'])
def apuestas_hoy():
    """
    Devuelve apuestas recomendadas de hoy
    
    GET /apuestas/hoy
    """
    try:
        df = db.obtener_predicciones()
        
        if df.empty:
            return jsonify({
                'status': 'success',
                'total': 0,
                'apuestas': []
            })
        
        # Filtrar por hoy y decisión de apostar
        df['fecha_partido'] = pd.to_datetime(df['fecha_partido'])
        df_hoy = df[
            (df['fecha_partido'].dt.date == date.today()) &
            (df['decision'].str.contains('APOSTAR', na=False))
        ]
        
        # Ordenar por EV descendente
        if 'expected_value' in df_hoy.columns:
            df_hoy = df_hoy.sort_values('expected_value', ascending=False)
        
        return jsonify({
            'status': 'success',
            'total': len(df_hoy),
            'apuestas': df_hoy.to_dict('records')
        })
    
    except Exception as e:
        logger.error(f"Error obteniendo apuestas de hoy: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/metricas', methods=['GET'])
def metricas():
    """
    Devuelve métricas generales del sistema
    
    GET /metricas
    """
    try:
        metricas = db.calcular_metricas()
        
        # Añadir información del modelo
        import json
        from pathlib import Path
        
        metricas_modelo_path = Path(Config.MODEL_PATH).parent / "production_model_metrics.json"
        
        if metricas_modelo_path.exists():
            with open(metricas_modelo_path, 'r') as f:
                metricas_modelo = json.load(f)
                metricas['modelo'] = metricas_modelo
        
        return jsonify({
            'status': 'success',
            'metricas': metricas
        })
    
    except Exception as e:
        logger.error(f"Error obteniendo métricas: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/historial', methods=['GET'])
def historial():
    """
    Devuelve historial de apuestas
    
    GET /historial?limit=50&decision=APOSTAR
    """
    try:
        limit = request.args.get('limit', 50, type=int)
        decision = request.args.get('decision', None)
        
        # Obtener predicciones
        filtros = {}
        if decision:
            filtros['decision'] = decision
        
        df = db.obtener_predicciones(filtros)
        
        if df.empty:
            return jsonify({
                'status': 'success',
                'total': 0,
                'historial': []
            })
        
        # Ordenar por fecha descendente
        if 'fecha_prediccion' in df.columns:
            df['fecha_prediccion'] = pd.to_datetime(df['fecha_prediccion'])
            df = df.sort_values('fecha_prediccion', ascending=False)
        
        # Limitar resultados
        df = df.head(limit)
        
        return jsonify({
            'status': 'success',
            'total': len(df),
            'historial': df.to_dict('records')
        })
    
    except Exception as e:
        logger.error(f"Error obteniendo historial: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/estadisticas', methods=['GET'])
def estadisticas():
    """
    Devuelve estadísticas detalladas
    
    GET /estadisticas
    """
    try:
        df = db.obtener_predicciones({'decision': 'APOSTAR'})
        
        if df.empty:
            return jsonify({
                'status': 'success',
                'estadisticas': {
                    'total_apuestas': 0,
                    'apuestas_completadas': 0,
                    'win_rate': 0,
                    'roi': 0
                }
            })
        
        # Calcular estadísticas
        completadas = df[df['resultado_real'].notna()]
        
        stats = {
            'total_apuestas': len(df),
            'apuestas_completadas': len(completadas),
            'apuestas_pendientes': len(df) - len(completadas),
            'win_rate': 0,
            'roi': 0
        }
        
        if len(completadas) > 0:
            ganadas = (completadas['resultado_real'] == 1).sum()
            stats['win_rate'] = ganadas / len(completadas)
            
            # Calcular ROI si hay información de stakes
            if 'stake' in completadas.columns and 'ganancia' in completadas.columns:
                total_apostado = completadas['stake'].sum()
                total_ganancia = completadas['ganancia'].sum()
                stats['roi'] = (total_ganancia / total_apostado) if total_apostado > 0 else 0
        
        return jsonify({
            'status': 'success',
            'estadisticas': stats
        })
    
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 INICIANDO API REST")
    print("=" * 60)
    print(f"📡 Disponible en: http://localhost:5000")
    print("\n📋 Endpoints disponibles:")
    print("  GET  /health              - Health check")
    print("  POST /predict             - Generar predicción")
    print("  GET  /apuestas/hoy        - Apuestas recomendadas hoy")
    print("  GET  /metricas            - Métricas del sistema")
    print("  GET  /historial           - Historial de apuestas")
    print("  GET  /estadisticas        - Estadísticas detalladas")
    print("=" * 60)
    
    # Ejecutar en modo desarrollo
    # En producción usar: gunicorn -w 4 -b 0.0.0.0:5000 src.api.api_server:app
    app.run(host='0.0.0.0', port=5000, debug=False)
