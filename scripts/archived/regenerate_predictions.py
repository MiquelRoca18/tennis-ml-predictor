"""
Script para regenerar predicciones de partidos existentes
"""
import sys
from pathlib import Path
from datetime import date, timedelta

sys.path.insert(0, str(Path(__file__).parent))

from src.database.match_database import MatchDatabase
from src.prediction.predictor_calibrado import PredictorCalibrado
from src.config import Config

print("\n" + "="*70)
print("REGENERANDO PREDICCIONES")
print("="*70)

# Inicializar
db = MatchDatabase("matches_v2.db")
predictor = PredictorCalibrado("modelos/random_forest_calibrado.pkl")

# Obtener partidos de hoy sin predicción
today = date.today()
partidos = db.get_matches_by_date(today)

print(f"\n📊 Partidos encontrados para {today}: {len(partidos)}")

regenerados = 0
for partido in partidos:
    # Solo regenerar si tiene cuotas
    if not partido.get("jugador1_cuota") or not partido.get("jugador2_cuota"):
        continue
    
    try:
        # Generar predicción
        resultado_pred = predictor.predecir_partido(
            jugador1=partido["jugador1_nombre"],
            jugador1_rank=999,
            jugador2=partido["jugador2_nombre"],
            jugador2_rank=999,
            superficie=partido["superficie"],
            cuota=partido["jugador1_cuota"],
        )
        
        # Calcular probabilidades
        prob_j1 = resultado_pred["probabilidad"]
        prob_j2 = 1 - prob_j1
        ev_j1 = resultado_pred["expected_value"]
        ev_j2 = (prob_j2 * partido["jugador2_cuota"]) - 1
        
        # Determinar recomendación
        umbral_ev = Config.EV_THRESHOLD
        if ev_j1 > umbral_ev:
            recomendacion = f"APOSTAR a {partido['jugador1_nombre']}"
            mejor_opcion = partido["jugador1_nombre"]
        elif ev_j2 > umbral_ev:
            recomendacion = f"APOSTAR a {partido['jugador2_nombre']}"
            mejor_opcion = partido["jugador2_nombre"]
        else:
            recomendacion = "NO APOSTAR"
            mejor_opcion = None
        
        # Guardar predicción
        db.create_prediction(
            match_id=partido["id"],
            version=1,
            jugador1_cuota=partido["jugador1_cuota"],
            jugador2_cuota=partido["jugador2_cuota"],
            jugador1_probabilidad=prob_j1,
            jugador2_probabilidad=prob_j2,
            jugador1_ev=ev_j1,
            jugador2_ev=ev_j2,
            recomendacion=recomendacion,
            mejor_opcion=mejor_opcion,
            confianza="Media"
        )
        
        regenerados += 1
        if regenerados % 50 == 0:
            print(f"  Regeneradas: {regenerados}")
            
    except Exception as e:
        print(f"  ❌ Error en partido {partido['id']}: {e}")

print(f"\n✅ Predicciones regeneradas: {regenerados}/{len(partidos)}")
print("="*70)
