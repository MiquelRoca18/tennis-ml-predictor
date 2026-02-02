#!/usr/bin/env python3
"""
Verifica un partido en la base de datos de Railway (PostgreSQL).

Uso:
  DATABASE_URL="postgresql://..." python scripts/verify_railway_match.py
  # O con .env que tenga DATABASE_URL de Railway

Requiere DATABASE_URL para conectar a Railway (no usa SQLite local).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

# event_key del partido Alcaraz vs Djokovic Australian Open Final
EVENT_KEY = "12099767"


def main():
    if not os.getenv("DATABASE_URL"):
        print("❌ DATABASE_URL no configurada.")
        print("   Para consultar Railway, exporta DATABASE_URL con la conexión de PostgreSQL.")
        print("   Ej: DATABASE_URL='postgresql://user:pass@host:5432/db' python scripts/verify_railway_match.py")
        sys.exit(1)

    from src.database.match_database import MatchDatabase
    from src.services.match_stats_calculator import MatchStatsCalculator

    print("=" * 60)
    print("VERIFICACIÓN PARTIDO EN RAILWAY (PostgreSQL)")
    print("event_key:", EVENT_KEY)
    print("=" * 60)

    db = MatchDatabase("matches_v2.db")  # path ignorado si hay DATABASE_URL

    if not db.is_postgres:
        print("❌ No se detectó PostgreSQL. DATABASE_URL puede ser incorrecta.")
        sys.exit(1)

    print("\n✅ Conectado a PostgreSQL (Railway)")

    match = db.get_match_by_event_key(EVENT_KEY)
    if not match:
        print(f"\n❌ Partido no encontrado con event_key={EVENT_KEY}")
        return

    match_id = match["id"]
    print(f"\n✅ Partido encontrado: id={match_id}")
    print(f"   {match.get('jugador1_nombre')} vs {match.get('jugador2_nombre')}")
    print(f"   fecha: {match.get('fecha_partido')}, estado: {match.get('estado')}")
    print(f"   resultado_marcador: {match.get('resultado_marcador')}")

    sets_db = db.get_match_sets(match_id) if hasattr(db, "get_match_sets") else []
    print(f"\n📊 match_sets: {len(sets_db)}")
    for s in sets_db:
        print(f"   Set {s.get('set_number')}: {s.get('player1_score')}-{s.get('player2_score')}")

    pbp = db._fetchone(
        "SELECT match_id, LENGTH(data) as len FROM match_pointbypoint_cache WHERE match_id = :mid",
        {"mid": match_id},
    )
    print(f"\n📦 pointbypoint_cache: {'SÍ' if pbp else 'NO'}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
