#!/usr/bin/env python3
"""
Script para hacer backfill de logos de jugadores
Copia logos de partidos recientes del mismo jugador
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.match_database import MatchDatabase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def backfill_logos():
    """Backfill de logos desde partidos recientes"""
    
    logger.info("=" * 80)
    logger.info("🖼️  BACKFILL DE LOGOS DE JUGADORES")
    logger.info("=" * 80)
    
    db = MatchDatabase()
    cursor = db.conn.cursor()
    
    # Paso 1: Backfill jugador1_logo
    logger.info("\n📝 Paso 1: Backfill de logos jugador 1...")
    
    cursor.execute("""
        UPDATE matches
        SET jugador1_logo = (
            SELECT jugador1_logo 
            FROM matches m2
            WHERE m2.jugador1_nombre = matches.jugador1_nombre
            AND m2.jugador1_logo IS NOT NULL
            AND m2.jugador1_logo != ''
            ORDER BY m2.fecha_partido DESC
            LIMIT 1
        )
        WHERE (jugador1_logo IS NULL OR jugador1_logo = '')
        AND EXISTS (
            SELECT 1 FROM matches m3
            WHERE m3.jugador1_nombre = matches.jugador1_nombre
            AND m3.jugador1_logo IS NOT NULL
            AND m3.jugador1_logo != ''
        )
    """)
    
    updated_p1 = cursor.rowcount
    logger.info(f"✅ Actualizados {updated_p1} logos de jugador 1")
    
    # Paso 2: Backfill jugador2_logo
    logger.info("\n📝 Paso 2: Backfill de logos jugador 2...")
    
    cursor.execute("""
        UPDATE matches
        SET jugador2_logo = (
            SELECT jugador2_logo 
            FROM matches m2
            WHERE m2.jugador2_nombre = matches.jugador2_nombre
            AND m2.jugador2_logo IS NOT NULL
            AND m2.jugador2_logo != ''
            ORDER BY m2.fecha_partido DESC
            LIMIT 1
        )
        WHERE (jugador2_logo IS NULL OR jugador2_logo = '')
        AND EXISTS (
            SELECT 1 FROM matches m3
            WHERE m3.jugador2_nombre = matches.jugador2_nombre
            AND m3.jugador2_logo IS NOT NULL
            AND m3.jugador2_logo != ''
        )
    """)
    
    updated_p2 = cursor.rowcount
    logger.info(f"✅ Actualizados {updated_p2} logos de jugador 2")
    
    db.conn.commit()
    
    # Paso 3: Verificar resultados
    logger.info("\n📈 Paso 3: Verificando resultados...")
    
    total_matches = cursor.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    
    with_p1_logo = cursor.execute("""
        SELECT COUNT(*) FROM matches 
        WHERE jugador1_logo IS NOT NULL AND jugador1_logo != ''
    """).fetchone()[0]
    
    with_p2_logo = cursor.execute("""
        SELECT COUNT(*) FROM matches 
        WHERE jugador2_logo IS NOT NULL AND jugador2_logo != ''
    """).fetchone()[0]
    
    with_both_logos = cursor.execute("""
        SELECT COUNT(*) FROM matches 
        WHERE jugador1_logo IS NOT NULL AND jugador1_logo != ''
        AND jugador2_logo IS NOT NULL AND jugador2_logo != ''
    """).fetchone()[0]
    
    logger.info(f"\nResultados:")
    logger.info(f"  Total de partidos: {total_matches}")
    logger.info(f"  Con logo J1: {with_p1_logo}/{total_matches} ({with_p1_logo/total_matches*100:.1f}%)")
    logger.info(f"  Con logo J2: {with_p2_logo}/{total_matches} ({with_p2_logo/total_matches*100:.1f}%)")
    logger.info(f"  Con ambos logos: {with_both_logos}/{total_matches} ({with_both_logos/total_matches*100:.1f}%)")
    
    logger.info("\n✅ Backfill completado!")
    
    db.close()

if __name__ == "__main__":
    backfill_logos()
