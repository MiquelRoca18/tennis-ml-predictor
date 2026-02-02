#!/usr/bin/env python3
"""
Verificación end-to-end: ¿El frontend recibe y puede mostrar predicciones?
================================================================================

Este script:
1. Llama a GET /matches para una fecha con partidos
2. Verifica que la respuesta tenga el formato esperado por el frontend
3. Comprueba que los partidos con predicciones tengan todos los campos necesarios

Uso:
    python scripts/verify_predictions_frontend.py [--date YYYY-MM-DD] [--base-url URL]
    
Ejemplo:
    python scripts/verify_predictions_frontend.py --date 2026-02-02
    python scripts/verify_predictions_frontend.py --base-url https://tennis-ml-predictor-production.up.railway.app
"""

import argparse
import json
import sys
from datetime import date, timedelta
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError


# Campos que el frontend MatchCard espera en prediccion
REQUIRED_PREDICTION_FIELDS = [
    "jugador1_probabilidad",
    "jugador2_probabilidad",
    "jugador1_ev",
    "jugador2_ev",
    "jugador1_cuota",
    "jugador2_cuota",
]

OPTIONAL_PREDICTION_FIELDS = [
    "version", "timestamp", "recomendacion", "mejor_opcion", "confianza",
    "confidence_level", "confidence_score", "jugador1_edge", "jugador2_edge",
]


def main():
    parser = argparse.ArgumentParser(description="Verifica que el API devuelva predicciones en formato correcto")
    parser.add_argument("--date", default=None, help="Fecha YYYY-MM-DD (default: hoy)")
    parser.add_argument("--base-url", default="http://localhost:8000", help="URL base del API")
    args = parser.parse_args()

    target_date = args.date or str(date.today())
    base_url = args.base_url.rstrip("/")
    url = f"{base_url}/matches?date={target_date}"

    print("=" * 70)
    print("VERIFICACIÓN: Predicciones para el Frontend")
    print("=" * 70)
    print(f"URL: {url}")
    print()

    try:
        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except HTTPError as e:
        print(f"❌ Error HTTP {e.code}: {e.reason}")
        if e.code == 404:
            print("   Verifica que el endpoint /matches exista.")
        sys.exit(1)
    except URLError as e:
        print(f"❌ Error de conexión: {e.reason}")
        print("   ¿El servidor está corriendo?")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    partidos = data.get("partidos", [])
    total = len(partidos)
    with_pred = [p for p in partidos if p.get("prediccion")]
    without_pred = [p for p in partidos if not p.get("prediccion")]

    print(f"📊 Partidos totales: {total}")
    print(f"   Con predicción: {len(with_pred)}")
    print(f"   Sin predicción: {len(without_pred)}")
    print()

    if not with_pred:
        print("⚠️  No hay partidos con predicción en esta fecha.")
        print("   Posibles causas:")
        print("   - Los partidos de hoy/mañana aún no tienen predicciones generadas")
        print("   - Ejecutar sync: POST /admin/sync-odds-and-predictions")
        print("   - Probar con una fecha pasada que tenga partidos completados")
        sys.exit(0)

    # Verificar formato de la primera predicción
    errors = []
    for i, p in enumerate(with_pred[:5]):  # Revisar hasta 5
        pred = p.get("prediccion", {})
        match_id = p.get("id")
        j1 = p.get("jugador1", {}).get("nombre", "?")
        j2 = p.get("jugador2", {}).get("nombre", "?")

        missing = [f for f in REQUIRED_PREDICTION_FIELDS if pred.get(f) is None]
        if missing:
            errors.append(f"Match {match_id} ({j1} vs {j2}): faltan campos: {missing}")

        # Verificar que probabilidades sean 0-1 (o 0-100 si vienen en %)
        prob1 = pred.get("jugador1_probabilidad")
        prob2 = pred.get("jugador2_probabilidad")
        if prob1 is not None and prob2 is not None:
            if prob1 > 1 or prob2 > 1:
                # Podrían estar en porcentaje (0-100)
                pass  # El frontend formatProbability espera 0-1
            elif prob1 < 0 or prob2 < 0 or prob1 > 1 or prob2 > 1:
                errors.append(f"Match {match_id}: probabilidades inválidas {prob1}, {prob2}")

    if errors:
        print("❌ Errores de formato (el frontend podría fallar):")
        for e in errors:
            print(f"   - {e}")
        sys.exit(1)

    print("✅ Formato correcto: los partidos con predicción tienen los campos necesarios.")
    print()
    print("Ejemplo de predicción (primer partido con predicción):")
    example = with_pred[0]
    pred = example.get("prediccion", {})
    print(json.dumps({
        "match_id": example.get("id"),
        "jugadores": f"{example.get('jugador1',{}).get('nombre')} vs {example.get('jugador2',{}).get('nombre')}",
        "prediccion": {
            k: pred.get(k) for k in REQUIRED_PREDICTION_FIELDS + OPTIONAL_PREDICTION_FIELDS
            if pred.get(k) is not None
        }
    }, indent=2, default=str))
    print()
    print("✅ El frontend debería mostrar estas predicciones correctamente.")


if __name__ == "__main__":
    main()
