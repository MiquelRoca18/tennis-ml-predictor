#!/usr/bin/env python3
"""
Script para sincronizar rankings ATP manualmente
Soluciona el problema de rankings al 0.1%
"""

import sys
from pathlib import Path

# Añadir src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.api_tennis_client import APITennisClient
from src.services.player_service import PlayerService
from src.services.ranking_service_elite import RankingServiceElite
from src.database.match_database import MatchDatabase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sync_rankings():
    """Sincroniza rankings ATP y actualiza partidos"""
    
    logger.info("=" * 80)
    logger.info("🎾 SINCRONIZACIÓN DE RANKINGS ATP")
    logger.info("=" * 80)
    
    # Inicializar servicios
    db = MatchDatabase()
    api_client = APITennisClient()
    player_service = PlayerService(db.conn)
    ranking_service = RankingServiceElite(db.conn, api_client, player_service)
    
    # Paso 1: Sincronizar rankings desde API
    logger.info("\n📊 Paso 1: Sincronizando rankings desde Tennis API...")
    result = ranking_service.sync_atp_rankings(limit=500)
    logger.info(f"✅ Sincronizados {result} rankings ATP")
    
    # Paso 2: Actualizar partidos con rankings
    logger.info("\n📝 Paso 2: Actualizando partidos con rankings...")
    cursor = db.conn.cursor()
    
    # Actualizar jugador1_ranking
    cursor.execute("""
        UPDATE matches
        SET jugador1_ranking = (
            SELECT atp_ranking FROM players 
            WHERE player_key = matches.jugador1_key
        )
        WHERE jugador1_key IS NOT NULL
        AND EXISTS (
            SELECT 1 FROM players 
            WHERE player_key = matches.jugador1_key
            AND atp_ranking IS NOT NULL
        )
    """)
    
    updated_p1 = cursor.rowcount
    logger.info(f"  - Actualizados {updated_p1} rankings de jugador 1")
    
    # Actualizar jugador2_ranking
    cursor.execute("""
        UPDATE matches
        SET jugador2_ranking = (
            SELECT atp_ranking FROM players 
            WHERE player_key = matches.jugador2_key
        )
        WHERE jugador2_key IS NOT NULL
        AND EXISTS (
            SELECT 1 FROM players 
            WHERE player_key = matches.jugador2_key
            AND atp_ranking IS NOT NULL
        )
    """)
    
    updated_p2 = cursor.rowcount
    logger.info(f"  - Actualizados {updated_p2} rankings de jugador 2")
    
    db.conn.commit()
    
    # Paso 3: Verificar resultados
    logger.info("\n📈 Paso 3: Verificando resultados...")
    
    total_matches = cursor.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    
    with_p1_ranking = cursor.execute("""
        SELECT COUNT(*) FROM matches 
        WHERE jugador1_ranking IS NOT NULL
    """).fetchone()[0]
    
    with_p2_ranking = cursor.execute("""
        SELECT COUNT(*) FROM matches 
        WHERE jugador2_ranking IS NOT NULL
    """).fetchone()[0]
    
    with_both_rankings = cursor.execute("""
        SELECT COUNT(*) FROM matches 
        WHERE jugador1_ranking IS NOT NULL 
        AND jugador2_ranking IS NOT NULL
    """).fetchone()[0]
    
    logger.info(f"\nResultados:")
    logger.info(f"  Total de partidos: {total_matches}")
    logger.info(f"  Con ranking J1: {with_p1_ranking}/{total_matches} ({with_p1_ranking/total_matches*100:.1f}%)")
    logger.info(f"  Con ranking J2: {with_p2_ranking}/{total_matches} ({with_p2_ranking/total_matches*100:.1f}%)")
    logger.info(f"  Con ambos rankings: {with_both_rankings}/{total_matches} ({with_both_rankings/total_matches*100:.1f}%)")
    
    logger.info("\n✅ Sincronización completada!")
    
    db.close()

if __name__ == "__main__":
    sync_rankings()
