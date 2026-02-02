#!/usr/bin/env python3
"""
Diagnóstico: por qué sync-odds no encuentra cuotas.

Compara los event_keys de partidos pendientes con las claves que devuelve
la API get_odds. Si no coinciden, el sync no puede generar predicciones.

Uso:
  DATABASE_URL="postgresql://..." python scripts/debug_odds_sync.py

Requiere DATABASE_URL para conectar a Railway (PostgreSQL).
También necesita API_TENNIS_API_KEY en .env para llamar a la API.
"""

import os
import sys
from pathlib import Path
from datetime import date, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()


def main():
    if not os.getenv("DATABASE_URL"):
        print("❌ DATABASE_URL no configurada. Exporta la conexión de Railway.")
        sys.exit(1)
    if not os.getenv("API_TENNIS_API_KEY"):
        print("❌ API_TENNIS_API_KEY no configurada (necesaria para llamar get_odds).")
        sys.exit(1)

    from src.database.match_database import MatchDatabase
    from src.services.api_tennis_client import APITennisClient

    db = MatchDatabase("matches_v2.db")
    if not db.is_postgres:
        print("❌ No se detectó PostgreSQL.")
        sys.exit(1)

    api_client = APITennisClient()

    today = date.today()
    end_date = today + timedelta(days=2)

    print("=" * 70)
    print("DIAGNÓSTICO: ¿Por qué sync-odds no encuentra cuotas?")
    print("=" * 70)

    # 1. Partidos pendientes
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

    print(f"\n📋 Partidos pendientes (muestra 20): {len(our_event_keys)}")
    if our_event_keys:
        for i, k in enumerate(our_event_keys[:5]):
            m = next((p for p in (pending or []) if str(p.get("event_key")) == k), {})
            print(f"   {i+1}. event_key={k} | {m.get('jugador1_nombre','?')} vs {m.get('jugador2_nombre','?')}")

    # 2. Llamar API get_odds
    print(f"\n📡 Llamando API get_odds ({today} a {end_date})...")
    all_odds_by_key = {}
    for d in range((end_date - today).days + 1):
        day = today + timedelta(days=d)
        day_str = day.strftime("%Y-%m-%d")
        batch = api_client.get_all_odds_batch(day_str, day_str)
        if batch:
            all_odds_by_key.update(batch)

    api_keys = list(all_odds_by_key.keys()) if isinstance(all_odds_by_key, dict) else []
    api_keys_str = [str(k) for k in api_keys]

    print(f"   API devolvió {len(api_keys)} partidos con cuotas")
    if api_keys_str:
        print(f"   Muestra de claves API: {api_keys_str[:10]}")

    # 3. Intersección
    api_keys_set = set(api_keys_str) | {k for k in api_keys if isinstance(k, (int, str))}

    def key_in_api(ek):
        return ek in api_keys_set or (ek.isdigit() and int(ek) in api_keys_set)
    matched = [k for k in our_event_keys if key_in_api(k)]
    not_matched = [k for k in our_event_keys if not key_in_api(k)]

    print(f"\n🔍 RESULTADO:")
    print(f"   Coinciden (encontradas en API): {len(matched)}")
    print(f"   NO coinciden (no están en API): {len(not_matched)}")

    if not_matched:
        print(f"\n   Event_keys que buscamos pero la API no tiene:")
        for k in not_matched[:5]:
            print(f"      - {k}")

    # 4. Probar extract_best_odds
    first_odds = None
    if our_event_keys:
        first_odds = api_client.extract_best_odds(all_odds_by_key, our_event_keys[0])
    print(f"\n   extract_best_odds para primer event_key: {'OK' if first_odds else 'None (no Home/Away)'}")

    # 5. Diagnóstico
    print("\n" + "=" * 70)
    if not api_keys:
        print("⚠️  DIAGNÓSTICO: La API no devolvió NINGUNA cuota para estas fechas.")
        print("   Posibles causas: plan API sin odds, fechas fuera de cobertura.")
    elif not matched:
        print("⚠️  DIAGNÓSTICO: Las claves de nuestros partidos NO coinciden con las de la API.")
        print("   La API tiene cuotas para otros partidos, pero no para los que tenemos en BD.")
        print("   Puede ser: torneos distintos, formato de event_key diferente.")
    else:
        print("✅ Algunas claves coinciden. Si aun así no hay predicciones,")
        print("   revisar: predictor/modelo, update_match_odds, o logs del sync.")
    print("=" * 70)


if __name__ == "__main__":
    main()
