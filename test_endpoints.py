#!/usr/bin/env python3
"""
Test script to verify all match data endpoints are working correctly
"""

import sys
from pathlib import Path
import logging
import requests
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8001"

def test_endpoint(endpoint, description):
    """Test a single endpoint"""
    try:
        logger.info(f"\n{'='*70}")
        logger.info(f"Testing: {description}")
        logger.info(f"Endpoint: GET {endpoint}")
        logger.info(f"{'='*70}")
        
        response = requests.get(f"{BASE_URL}{endpoint}", timeout=10)
        
        logger.info(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ SUCCESS")
            logger.info(f"Response keys: {list(data.keys())}")
            
            # Show sample data
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, list):
                        logger.info(f"  - {key}: {len(value)} items")
                        if len(value) > 0:
                            logger.info(f"    Sample: {json.dumps(value[0], indent=2)[:200]}...")
                    elif isinstance(value, dict):
                        logger.info(f"  - {key}: {len(value)} keys")
                    else:
                        logger.info(f"  - {key}: {value}")
            
            return True, data
        else:
            logger.error(f"❌ FAILED: {response.status_code}")
            logger.error(f"Response: {response.text[:500]}")
            return False, None
            
    except requests.exceptions.ConnectionError:
        logger.error(f"❌ FAILED: Server not running on {BASE_URL}")
        return False, None
    except Exception as e:
        logger.error(f"❌ FAILED: {e}")
        return False, None

def main():
    """Test all endpoints"""
    
    logger.info("\n" + "="*70)
    logger.info("TESTING ALL MATCH DATA ENDPOINTS")
    logger.info("="*70)
    
    # Test match with point-by-point data (757)
    match_id = 757
    
    tests = [
        (f"/matches/{match_id}/odds/multi", "Multi-Bookmaker Odds"),
        (f"/matches/{match_id}/pointbypoint", "Point by Point Data"),
        (f"/matches/{match_id}/games", "Games Data"),
        (f"/matches/{match_id}/breakpoints", "Break Points Stats"),
        (f"/matches/{match_id}/stats/detailed", "Detailed Statistics"),
        (f"/matches/{match_id}/stats/summary", "Statistics Summary"),
    ]
    
    results = {}
    
    for endpoint, description in tests:
        success, data = test_endpoint(endpoint, description)
        results[description] = success
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("TEST SUMMARY")
    logger.info("="*70)
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        logger.warning(f"\n⚠️  {total - passed} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
