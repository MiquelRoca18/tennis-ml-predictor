#!/usr/bin/env python3
"""
Test script to verify multi-bookmaker odds sync functionality
"""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.match_database import MatchDatabase
from src.services.api_tennis_client import APITennisClient
from src.services.multi_odds_service import MultiBookmakerOddsService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def test_odds_sync():
    """Test the odds sync functionality"""
    
    logger.info("=" * 70)
    logger.info("TESTING MULTI-BOOKMAKER ODDS SYNC")
    logger.info("=" * 70)
    
    # 1. Initialize services
    logger.info("\n1. Initializing services...")
    db = MatchDatabase("matches_v2.db")
    api_client = APITennisClient()
    multi_odds_service = MultiBookmakerOddsService(db.conn, api_client)
    
    # 2. Check current state
    logger.info("\n2. Checking current database state...")
    cursor = db.conn.cursor()
    
    # Count pending matches
    cursor.execute("""
        SELECT COUNT(*) as count FROM matches 
        WHERE estado = 'pendiente' AND event_key IS NOT NULL
    """)
    pending_count = cursor.fetchone()['count']
    logger.info(f"   Pending matches with event_key: {pending_count}")
    
    # Count current odds
    cursor.execute("SELECT COUNT(*) as count FROM match_odds")
    odds_count_before = cursor.fetchone()['count']
    logger.info(f"   Current odds in database: {odds_count_before}")
    
    # 3. Run sync
    logger.info("\n3. Running odds synchronization...")
    result = multi_odds_service.sync_all_pending_matches_odds()
    
    logger.info(f"\n   Sync Result:")
    logger.info(f"   - Success: {result.get('success')}")
    logger.info(f"   - Matches found: {result.get('matches_found', 0)}")
    logger.info(f"   - Matches with odds: {result.get('matches_with_odds', 0)}")
    logger.info(f"   - Total odds synced: {result.get('odds_synced', 0)}")
    logger.info(f"   - Message: {result.get('message', 'N/A')}")
    
    # 4. Check new state
    logger.info("\n4. Checking database state after sync...")
    cursor.execute("SELECT COUNT(*) as count FROM match_odds")
    odds_count_after = cursor.fetchone()['count']
    logger.info(f"   Odds in database after sync: {odds_count_after}")
    logger.info(f"   New odds added: {odds_count_after - odds_count_before}")
    
    # 5. Show sample odds
    if odds_count_after > 0:
        logger.info("\n5. Sample odds from database:")
        cursor.execute("""
            SELECT m.jugador1_nombre, m.jugador2_nombre, 
                   mo.bookmaker, mo.selection, mo.odds
            FROM match_odds mo
            JOIN matches m ON mo.match_id = m.id
            LIMIT 10
        """)
        
        for row in cursor.fetchall():
            logger.info(
                f"   {row['jugador1_nombre']} vs {row['jugador2_nombre']}: "
                f"{row['bookmaker']} - {row['selection']} @ {row['odds']}"
            )
    
    # 6. Test API endpoint format
    logger.info("\n6. Testing get_match_odds() for a sample match...")
    if pending_count > 0:
        cursor.execute("""
            SELECT id FROM matches 
            WHERE estado = 'pendiente' AND event_key IS NOT NULL
            LIMIT 1
        """)
        sample_match = cursor.fetchone()
        if sample_match:
            match_id = sample_match['id']
            odds = multi_odds_service.get_match_odds(match_id)
            logger.info(f"   Match ID {match_id}: {len(odds)} odds entries")
            if odds:
                logger.info(f"   Sample: {odds[0]}")
    
    logger.info("\n" + "=" * 70)
    logger.info("TEST COMPLETED")
    logger.info("=" * 70)
    
    db.close()

if __name__ == "__main__":
    test_odds_sync()
