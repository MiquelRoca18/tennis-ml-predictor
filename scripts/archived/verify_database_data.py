#!/usr/bin/env python3
"""
Verification script to check what data is actually available in the database
"""

import sys
from pathlib import Path
import sqlite3
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)
logger = logging.getLogger(__name__)

def check_database():
    """Check database for available data"""
    
    conn = sqlite3.connect("matches_v2.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    logger.info("="*80)
    logger.info("DATABASE DATA VERIFICATION REPORT")
    logger.info("="*80)
    
    # 1. Check matches
    logger.info("\n📊 MATCHES TABLE")
    logger.info("-"*80)
    cursor.execute("SELECT COUNT(*) as total FROM matches")
    total_matches = cursor.fetchone()['total']
    logger.info(f"Total matches: {total_matches}")
    
    cursor.execute("SELECT COUNT(*) as total FROM matches WHERE estado = 'pendiente'")
    pending = cursor.fetchone()['total']
    logger.info(f"Pending matches: {pending}")
    
    cursor.execute("SELECT COUNT(*) as total FROM matches WHERE estado = 'completado'")
    completed = cursor.fetchone()['total']
    logger.info(f"Completed matches: {completed}")
    
    # 2. Check odds
    logger.info("\n💰 MATCH_ODDS TABLE")
    logger.info("-"*80)
    cursor.execute("SELECT COUNT(*) as total FROM match_odds")
    total_odds = cursor.fetchone()['total']
    logger.info(f"Total odds entries: {total_odds}")
    
    cursor.execute("""
        SELECT COUNT(DISTINCT match_id) as total 
        FROM match_odds
    """)
    matches_with_odds = cursor.fetchone()['total']
    logger.info(f"Matches with odds: {matches_with_odds}")
    
    cursor.execute("""
        SELECT COUNT(DISTINCT bookmaker) as total 
        FROM match_odds
    """)
    bookmakers = cursor.fetchone()['total']
    logger.info(f"Different bookmakers: {bookmakers}")
    
    # Sample match with odds
    cursor.execute("""
        SELECT m.id, m.jugador1_nombre, m.jugador2_nombre, COUNT(mo.id) as odds_count
        FROM matches m
        JOIN match_odds mo ON m.id = mo.match_id
        GROUP BY m.id
        ORDER BY odds_count DESC
        LIMIT 1
    """)
    sample_odds = cursor.fetchone()
    if sample_odds:
        logger.info(f"\nSample match with most odds:")
        logger.info(f"  Match ID {sample_odds['id']}: {sample_odds['jugador1_nombre']} vs {sample_odds['jugador2_nombre']}")
        logger.info(f"  Total odds: {sample_odds['odds_count']}")
    
    # 3. Check point by point
    logger.info("\n🎾 MATCH_POINTBYPOINT TABLE")
    logger.info("-"*80)
    cursor.execute("SELECT COUNT(*) as total FROM match_pointbypoint")
    total_points = cursor.fetchone()['total']
    logger.info(f"Total points: {total_points}")
    
    cursor.execute("""
        SELECT COUNT(DISTINCT match_id) as total 
        FROM match_pointbypoint
    """)
    matches_with_pbp = cursor.fetchone()['total']
    logger.info(f"Matches with point-by-point data: {matches_with_pbp}")
    
    # Sample matches with PBP
    cursor.execute("""
        SELECT m.id, m.jugador1_nombre, m.jugador2_nombre, m.estado, COUNT(mp.id) as points
        FROM matches m
        JOIN match_pointbypoint mp ON m.id = mp.match_id
        GROUP BY m.id
        ORDER BY points DESC
        LIMIT 5
    """)
    logger.info(f"\nTop 5 matches with most point-by-point data:")
    for row in cursor.fetchall():
        logger.info(f"  Match {row['id']}: {row['jugador1_nombre']} vs {row['jugador2_nombre']}")
        logger.info(f"    Estado: {row['estado']}, Points: {row['points']}")
    
    # 4. Check games
    logger.info("\n🏆 MATCH_GAMES TABLE")
    logger.info("-"*80)
    cursor.execute("SELECT COUNT(*) as total FROM match_games")
    total_games = cursor.fetchone()['total']
    logger.info(f"Total games: {total_games}")
    
    cursor.execute("""
        SELECT COUNT(DISTINCT match_id) as total 
        FROM match_games
    """)
    matches_with_games = cursor.fetchone()['total']
    logger.info(f"Matches with games data: {matches_with_games}")
    
    # Sample matches with games
    cursor.execute("""
        SELECT m.id, m.jugador1_nombre, m.jugador2_nombre, m.estado, COUNT(mg.id) as games
        FROM matches m
        JOIN match_games mg ON m.id = mg.match_id
        GROUP BY m.id
        ORDER BY games DESC
        LIMIT 5
    """)
    logger.info(f"\nTop 5 matches with most games data:")
    for row in cursor.fetchall():
        logger.info(f"  Match {row['id']}: {row['jugador1_nombre']} vs {row['jugador2_nombre']}")
        logger.info(f"    Estado: {row['estado']}, Games: {row['games']}")
    
    # 5. Data completeness analysis
    logger.info("\n📈 DATA COMPLETENESS ANALYSIS")
    logger.info("-"*80)
    
    cursor.execute("""
        SELECT 
            m.estado,
            COUNT(DISTINCT m.id) as total_matches,
            COUNT(DISTINCT mo.match_id) as with_odds,
            COUNT(DISTINCT mp.match_id) as with_pbp,
            COUNT(DISTINCT mg.match_id) as with_games
        FROM matches m
        LEFT JOIN match_odds mo ON m.id = mo.match_id
        LEFT JOIN match_pointbypoint mp ON m.id = mp.match_id
        LEFT JOIN match_games mg ON m.id = mg.match_id
        GROUP BY m.estado
    """)
    
    logger.info(f"\n{'Estado':<15} {'Matches':<10} {'With Odds':<12} {'With PBP':<12} {'With Games':<12}")
    logger.info("-"*80)
    for row in cursor.fetchall():
        logger.info(f"{row['estado']:<15} {row['total_matches']:<10} {row['with_odds']:<12} {row['with_pbp']:<12} {row['with_games']:<12}")
    
    # 6. Specific test case
    logger.info("\n🔍 SPECIFIC TEST CASE: Match 757")
    logger.info("-"*80)
    
    cursor.execute("SELECT * FROM matches WHERE id = 757")
    match = cursor.fetchone()
    if match:
        logger.info(f"Match: {match['jugador1_nombre']} vs {match['jugador2_nombre']}")
        logger.info(f"Estado: {match['estado']}")
        logger.info(f"Fecha: {match['fecha_partido']}")
        
        cursor.execute("SELECT COUNT(*) as total FROM match_odds WHERE match_id = 757")
        odds = cursor.fetchone()['total']
        logger.info(f"Odds entries: {odds}")
        
        cursor.execute("SELECT COUNT(*) as total FROM match_pointbypoint WHERE match_id = 757")
        points = cursor.fetchone()['total']
        logger.info(f"Points: {points}")
        
        cursor.execute("SELECT COUNT(*) as total FROM match_games WHERE match_id = 757")
        games = cursor.fetchone()['total']
        logger.info(f"Games: {games}")
    else:
        logger.info("Match 757 not found")
    
    # 7. Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    
    logger.info(f"\n✅ Data Available:")
    logger.info(f"  - {total_matches} total matches")
    logger.info(f"  - {matches_with_odds} matches with odds ({total_odds} total odds)")
    logger.info(f"  - {matches_with_pbp} matches with point-by-point data ({total_points} total points)")
    logger.info(f"  - {matches_with_games} matches with games data ({total_games} total games)")
    
    logger.info(f"\n⚠️  Data Gaps:")
    pending_without_odds = pending - matches_with_odds
    if pending_without_odds > 0:
        logger.info(f"  - {pending_without_odds} pending matches without odds")
    
    completed_without_pbp = completed - matches_with_pbp
    if completed_without_pbp > 0:
        logger.info(f"  - {completed_without_pbp} completed matches without point-by-point data")
    
    logger.info("\n" + "="*80)
    
    conn.close()

if __name__ == "__main__":
    check_database()
