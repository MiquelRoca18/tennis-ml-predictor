#!/usr/bin/env python3
"""
Script para verificar que los endpoints devuelven datos en el formato correcto
según api_json_examples.md
"""

import requests
import json
import sys

BASE_URL = "http://localhost:8000"

def test_endpoint(endpoint, expected_keys):
    """Test an endpoint and verify response structure"""
    print(f"\n{'='*70}")
    print(f"Testing: {endpoint}")
    print('='*70)
    
    try:
        response = requests.get(f"{BASE_URL}{endpoint}", timeout=10)
        
        if response.status_code != 200:
            print(f"❌ Status: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
        
        data = response.json()
        
        # Check expected keys
        missing_keys = [key for key in expected_keys if key not in data]
        
        if missing_keys:
            print(f"⚠️  Missing keys: {missing_keys}")
        
        print(f"✅ Status: 200")
        print(f"   Keys present: {list(data.keys())}")
        print(f"\n   Sample response:")
        print(json.dumps(data, indent=2)[:500])
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("\n" + "="*70)
    print("VERIFICACIÓN DE FORMATOS DE ENDPOINTS")
    print("="*70)
    
    # Get a match ID with PBP data
    print("\n1. Buscando partido con datos PBP...")
    try:
        import sqlite3
        conn = sqlite3.connect("matches_v2.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT DISTINCT m.id, m.jugador1_nombre, m.jugador2_nombre
            FROM matches m
            JOIN match_pointbypoint mp ON m.id = mp.match_id
            LIMIT 1
        """)
        
        match = cursor.fetchone()
        conn.close()
        
        if not match:
            print("❌ No hay partidos con datos PBP en la base de datos")
            return 1
        
        match_id = match['id']
        print(f"✅ Usando partido {match_id}: {match['jugador1_nombre']} vs {match['jugador2_nombre']}")
        
    except Exception as e:
        print(f"❌ Error accediendo a la base de datos: {e}")
        return 1
    
    # Test endpoints
    tests = [
        {
            "name": "Point by Point",
            "endpoint": f"/matches/{match_id}/pointbypoint",
            "expected_keys": ["match_id", "total_points", "points"]
        },
        {
            "name": "Games",
            "endpoint": f"/matches/{match_id}/games",
            "expected_keys": ["match_id", "total_games", "games"]
        },
        {
            "name": "Match Details",
            "endpoint": f"/matches/{match_id}/details",
            "expected_keys": ["match_id", "estado", "estadisticas_basicas"]
        },
        {
            "name": "Match Analysis",
            "endpoint": f"/matches/{match_id}/analysis",
            "expected_keys": ["match_id", "timeline", "momentum"]
        },
    ]
    
    results = []
    for test in tests:
        success = test_endpoint(test["endpoint"], test["expected_keys"])
        results.append((test["name"], success))
    
    # Summary
    print("\n" + "="*70)
    print("RESUMEN")
    print("="*70)
    
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {name}")
    
    total_success = sum(1 for _, success in results if success)
    print(f"\nTotal: {total_success}/{len(results)} endpoints funcionando correctamente")
    
    return 0 if total_success == len(results) else 1

if __name__ == "__main__":
    sys.exit(main())
