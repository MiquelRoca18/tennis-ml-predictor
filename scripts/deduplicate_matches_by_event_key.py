#!/usr/bin/env python3
"""
Script para eliminar partidos duplicados por event_key.

Problema: El mismo partido puede aparecer con fecha_partido distinta (ej. 2026-02-02 y 2026-02-03)
debido a timezone. Esto causa que el mismo partido aparezca en dos días.

Solución: Por cada event_key duplicado, mantener solo el partido con la fecha MÁS RECIENTE
(la API suele tener la fecha correcta en la última sincronización).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.match_database import MatchDatabase
from src.config.settings import Config


def main():
    db = MatchDatabase(Config.MATCHES_DB_PATH)
    
    # Encontrar event_keys duplicados
    rows = db._fetchall("""
        SELECT event_key, COUNT(*) as cnt
        FROM matches
        WHERE event_key IS NOT NULL AND TRIM(COALESCE(event_key, '')) != ''
        GROUP BY event_key
        HAVING COUNT(*) > 1
    """)
    dupes = []
    for r in rows:
        ids_rows = db._fetchall(
            "SELECT id FROM matches WHERE event_key = :ek ORDER BY fecha_partido DESC",
            {"ek": r["event_key"]}
        )
        ids = [x["id"] for x in ids_rows]
        dupes.append({"event_key": r["event_key"], "cnt": r["cnt"], "ids": ids})
    
    if not dupes:
        print("✅ No hay partidos duplicados por event_key")
        return
    
    print(f"🔍 Encontrados {len(dupes)} event_keys con partidos duplicados")
    
    deleted = 0
    for d in dupes:
        event_key = d["event_key"]
        ids = d["ids"]
        keep_id = ids[0]  # Mantener el de fecha más reciente
        delete_ids = ids[1:]
        
        match_info = db._fetchone("SELECT jugador1_nombre, jugador2_nombre, fecha_partido FROM matches WHERE id = :id", {"id": keep_id})
        print(f"\n  event_key={event_key}: {match_info.get('jugador1_nombre')} vs {match_info.get('jugador2_nombre')}")
        print(f"    Mantener id={keep_id} (fecha={match_info.get('fecha_partido')})")
        
        for did in delete_ids:
            minfo = db._fetchone("SELECT fecha_partido FROM matches WHERE id = :id", {"id": did})
            print(f"    Eliminar id={did} (fecha={minfo.get('fecha_partido')})")
            db.delete_match(did)
            deleted += 1
    
    print(f"\n✅ Eliminados {deleted} partidos duplicados")


if __name__ == "__main__":
    main()
