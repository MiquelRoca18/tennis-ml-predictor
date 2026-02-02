#!/usr/bin/env python3
"""
Comprueba si los partidos de hoy y mañana tienen predicciones en la base de datos.

Útil para diagnosticar: si hay predicciones en BD pero no se muestran en la app,
el problema es del frontend o del API. Si no hay predicciones en BD, el problema
es del backend (predictor no genera).

Uso:
  # Con Railway (PostgreSQL):
  DATABASE_URL="postgresql://..." python scripts/check_predictions_today_tomorrow.py

  # Local (SQLite):
  python scripts/check_predictions_today_tomorrow.py
"""

import os
import sys
from pathlib import Path
from datetime import date, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()


def main():
    from src.database.match_database import MatchDatabase

    db = MatchDatabase("matches_v2.db")
    db_type = "PostgreSQL (Railway)" if db.is_postgres else "SQLite (local)"

    today = date.today()
    tomorrow = today + timedelta(days=1)

    print("=" * 70)
    print("PREDICCIONES EN PARTIDOS DE HOY Y MAÑANA")
    print(f"BD: {db_type} | Hoy: {today} | Mañana: {tomorrow}")
    print("=" * 70)

    for label, target_date in [("HOY", today), ("MAÑANA", tomorrow)]:
        matches = db._fetchall(
            """
            SELECT m.id, m.jugador1_nombre, m.jugador2_nombre, m.fecha_partido, m.estado,
                   p.id as pred_id, p.jugador1_probabilidad, p.jugador2_probabilidad
            FROM matches m
            LEFT JOIN predictions p ON m.id = p.match_id AND p.version = (
                SELECT MAX(version) FROM predictions WHERE match_id = m.id
            )
            WHERE m.fecha_partido = :fecha
            ORDER BY m.estado, m.hora_inicio ASC, m.id ASC
            """,
            {"fecha": target_date},
        )

        total = len(matches or [])
        with_pred = [m for m in (matches or []) if m.get("pred_id")]
        without_pred = [m for m in (matches or []) if not m.get("pred_id")]
        pending = [m for m in (matches or []) if m.get("estado") == "pendiente"]
        pending_with_pred = [m for m in pending if m.get("pred_id")]
        pending_without_pred = [m for m in pending if not m.get("pred_id")]

        print(f"\n📅 {label} ({target_date})")
        print(f"   Total partidos: {total}")
        print(f"   Con predicción: {len(with_pred)}")
        print(f"   Sin predicción: {len(without_pred)}")
        print(f"   Pendientes: {len(pending)} (con pred: {len(pending_with_pred)}, sin pred: {len(pending_without_pred)})")

        if pending_without_pred:
            print(f"\n   Pendientes SIN predicción (ejemplos):")
            for m in pending_without_pred[:5]:
                print(f"      #{m['id']} {m.get('jugador1_nombre','?')} vs {m.get('jugador2_nombre','?')}")
        if pending_with_pred:
            print(f"\n   Pendientes CON predicción (ejemplos):")
            for m in pending_with_pred[:3]:
                p1 = m.get("jugador1_probabilidad") or 0
                p2 = m.get("jugador2_probabilidad") or 0
                print(f"      #{m['id']} {m.get('jugador1_nombre','?')} vs {m.get('jugador2_nombre','?')} | {p1:.0%}/{p2:.0%}")

    # Conclusión: partidos pendientes hoy+mañana con predicción
    all_pending_with_pred = []
    for target_date in [today, tomorrow]:
        rows = db._fetchall(
            """
            SELECT m.id, m.estado, p.id as pred_id
            FROM matches m
            LEFT JOIN predictions p ON m.id = p.match_id AND p.version = (
                SELECT MAX(version) FROM predictions WHERE match_id = m.id
            )
            WHERE m.fecha_partido = :fecha AND m.estado = 'pendiente'
            """,
            {"fecha": target_date},
        )
        all_pending_with_pred.extend([r for r in (rows or []) if r.get("pred_id")])

    print("\n" + "=" * 70)
    print("CONCLUSIÓN:")
    if all_pending_with_pred:
        print("   ✅ Hay predicciones en BD para pendientes → si no se ven en app, problema FRONTEND/API")
    else:
        print("   ❌ No hay predicciones en BD para pendientes → problema BACKEND (predictor no genera)")
    print("=" * 70)


if __name__ == "__main__":
    main()
