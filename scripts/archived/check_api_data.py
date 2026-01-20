#!/usr/bin/env python3
"""
Script to verify if API-Tennis has the data we should have
Compares what's in our database vs what's available in the API
"""

import sys
from pathlib import Path
import sqlite3
import logging
import os
from dotenv import load_dotenv
import requests
import json
from datetime import datetime, timedelta

# Load environment
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)
logger = logging.getLogger(__name__)

API_KEY = os.getenv("API_TENNIS_API_KEY", "")
BASE_URL = "https://api.api-tennis.com/tennis/"

def make_api_request(method, params=None):
    """Make request to API-Tennis"""
    try:
        request_params = {"method": method, "APIkey": API_KEY}
        if params:
            request_params.update(params)
        
        response = requests.get(BASE_URL, params=request_params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        if data.get("success") != 1:
            logger.error(f"API error: {data}")
            return None
        
        return data
    except Exception as e:
        logger.error(f"Request error: {e}")
        return None

def check_api_data_availability():
    """Check what data is available in API-Tennis"""
    
    if not API_KEY:
        logger.error("❌ API_TENNIS_API_KEY not found in .env")
        return
    
    logger.info("="*80)
    logger.info("API-TENNIS DATA AVAILABILITY CHECK")
    logger.info("="*80)
    
    # Connect to database
    conn = sqlite3.connect("matches_v2.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get sample matches from different dates
    cursor.execute("""
        SELECT DISTINCT fecha_partido 
        FROM matches 
        WHERE fecha_partido >= DATE('now', '-7 days')
        ORDER BY fecha_partido DESC
        LIMIT 3
    """)
    
    dates = [row['fecha_partido'] for row in cursor.fetchall()]
    
    logger.info(f"\n📅 Checking data for {len(dates)} recent dates:")
    for d in dates:
        logger.info(f"  - {d}")
    
    # Check fixtures for each date
    total_api_matches = 0
    total_db_matches = 0
    matches_with_odds_api = 0
    matches_with_pbp_api = 0
    
    for fecha in dates:
        logger.info(f"\n{'='*80}")
        logger.info(f"DATE: {fecha}")
        logger.info(f"{'='*80}")
        
        # Get matches from DB for this date
        cursor.execute("""
            SELECT id, event_key, jugador1_nombre, jugador2_nombre, estado
            FROM matches
            WHERE fecha_partido = ?
            LIMIT 10
        """, (fecha,))
        
        db_matches = cursor.fetchall()
        total_db_matches += len(db_matches)
        
        logger.info(f"\n📊 Database: {len(db_matches)} matches")
        
        # Get fixtures from API for this date
        logger.info(f"\n🌐 Fetching from API-Tennis...")
        api_data = make_api_request("get_fixtures", {
            "date_start": fecha,
            "date_stop": fecha
        })
        
        if not api_data:
            logger.error("  ❌ Failed to get fixtures from API")
            continue
        
        api_matches = api_data.get("result", [])
        total_api_matches += len(api_matches)
        
        logger.info(f"  ✅ API returned {len(api_matches)} matches")
        
        # Get odds for this date
        logger.info(f"\n💰 Checking odds availability...")
        odds_data = make_api_request("get_odds", {
            "date_start": fecha,
            "date_stop": fecha
        })
        
        if odds_data and odds_data.get("result"):
            odds_result = odds_data.get("result", {})
            matches_with_odds_api += len(odds_result)
            logger.info(f"  ✅ API has odds for {len(odds_result)} matches")
            
            # Sample odds
            if odds_result:
                sample_key = list(odds_result.keys())[0]
                sample_odds = odds_result[sample_key]
                if "Home/Away" in sample_odds:
                    home_odds = sample_odds["Home/Away"].get("Home", {})
                    logger.info(f"  📝 Sample: {len(home_odds)} bookmakers for match {sample_key}")
        else:
            logger.info(f"  ⚠️  No odds available from API")
        
        # Check point-by-point for sample matches
        logger.info(f"\n🎾 Checking point-by-point availability...")
        
        pbp_count = 0
        for match in db_matches[:3]:  # Check first 3 matches
            if not match['event_key']:
                continue
            
            # Get fixture details
            fixture_data = make_api_request("get_fixtures", {
                "event_key": match['event_key']
            })
            
            if fixture_data and fixture_data.get("result"):
                result = fixture_data["result"]
                if isinstance(result, list) and len(result) > 0:
                    result = result[0]
                
                has_pbp = bool(result.get("pointbypoint"))
                has_scores = bool(result.get("scores"))
                
                if has_pbp:
                    pbp_count += 1
                    pbp_len = len(result.get("pointbypoint", []))
                    logger.info(f"  ✅ Match {match['event_key']}: {pbp_len} points")
                elif has_scores:
                    logger.info(f"  ⚠️  Match {match['event_key']}: Has scores but no PBP")
                else:
                    logger.info(f"  ❌ Match {match['event_key']}: No detailed data")
        
        matches_with_pbp_api += pbp_count
        
        # Compare with DB
        cursor.execute("""
            SELECT COUNT(DISTINCT m.id) as with_odds
            FROM matches m
            JOIN match_odds mo ON m.id = mo.match_id
            WHERE m.fecha_partido = ?
        """, (fecha,))
        db_with_odds = cursor.fetchone()['with_odds']
        
        cursor.execute("""
            SELECT COUNT(DISTINCT m.id) as with_pbp
            FROM matches m
            JOIN match_pointbypoint mp ON m.id = mp.match_id
            WHERE m.fecha_partido = ?
        """, (fecha,))
        db_with_pbp = cursor.fetchone()['with_pbp']
        
        logger.info(f"\n📈 Comparison for {fecha}:")
        logger.info(f"  Database: {len(db_matches)} matches, {db_with_odds} with odds, {db_with_pbp} with PBP")
        logger.info(f"  API:      {len(api_matches)} matches, odds for {len(odds_result) if odds_data else 0}, {pbp_count} with PBP (sampled)")
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY")
    logger.info(f"{'='*80}")
    
    logger.info(f"\n📊 Total Matches:")
    logger.info(f"  Database: {total_db_matches}")
    logger.info(f"  API:      {total_api_matches}")
    
    logger.info(f"\n💰 Odds Availability:")
    logger.info(f"  API has odds for: {matches_with_odds_api} matches")
    
    logger.info(f"\n🎾 Point-by-Point Availability:")
    logger.info(f"  API has PBP for: {matches_with_pbp_api} matches (sampled)")
    
    logger.info(f"\n🔍 Conclusions:")
    
    if matches_with_odds_api > 0:
        logger.info(f"  ✅ API-Tennis DOES provide odds data")
        logger.info(f"     → We should be syncing these automatically")
    else:
        logger.info(f"  ⚠️  API-Tennis may not have odds for these dates")
    
    if matches_with_pbp_api > 0:
        logger.info(f"  ✅ API-Tennis DOES provide point-by-point data")
        logger.info(f"     → We need a process to fetch and store this data")
    else:
        logger.info(f"  ⚠️  API-Tennis may not have PBP for recent matches")
        logger.info(f"     → PBP data may only be available for completed matches")
    
    logger.info(f"\n{'='*80}")
    
    conn.close()

if __name__ == "__main__":
    check_api_data_availability()
