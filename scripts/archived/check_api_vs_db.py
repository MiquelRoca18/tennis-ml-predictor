#!/usr/bin/env python3
"""
Script para comparar datos de API-Tennis con nuestra DB
"""
import sys
sys.path.insert(0, '.')

from datetime import datetime
from src.services.api_tennis_client import APITennisClient
from src.database.match_database import MatchDatabase

def compare_data(date_str=None):
    if not date_str:
        date_str = datetime.now().strftime('%Y-%m-%d')
    
    print(f"\n🔍 Comparando datos del {date_str}\n")
    
    # Obtener de API
    client = APITennisClient()
    api_data = client._make_request('get_fixtures', {
        'date_start': date_str,
        'date_stop': date_str
    })
    
    api_matches = api_data.get('result', []) if api_data else []
    
    # Obtener de DB
    db = MatchDatabase("matches_v2.db")
    db_matches = db.get_matches_by_date(date_str)
    db.close()
    
    # Crear diccionario por event_key
    api_dict = {str(m.get('event_key')): m for m in api_matches}
    db_dict = {str(m.get('event_key')): m for m in db_matches}
    
    print(f"📊 Resumen:")
    print(f"  API-Tennis: {len(api_matches)} partidos")
    print(f"  Nuestra DB: {len(db_matches)} partidos")
    print()
    
    # Encontrar diferencias
    only_in_api = set(api_dict.keys()) - set(db_dict.keys())
    only_in_db = set(db_dict.keys()) - set(api_dict.keys())
    in_both = set(api_dict.keys()) & set(db_dict.keys())
    
    if only_in_api:
        print(f"⚠️  {len(only_in_api)} partidos solo en API (no en DB)")
    
    if only_in_db:
        print(f"⚠️  {len(only_in_db)} partidos solo en DB (no en API)")
    
    print(f"✅ {len(in_both)} partidos en ambos")
    print()
    
    # Verificar estados de los que están en ambos
    discrepancies = []
    for event_key in list(in_both)[:10]:  # Primeros 10
        api_m = api_dict[event_key]
        db_m = db_dict[event_key]
        
        api_live = api_m.get('event_live')
        db_live = db_m.get('event_live')
        api_status = api_m.get('event_status', '')
        db_status = db_m.get('estado', '')
        
        if api_live != db_live or (api_status.lower() == 'finished' and db_status != 'completado'):
            discrepancies.append({
                'event_key': event_key,
                'players': f"{db_m.get('jugador1_nombre')} vs {db_m.get('jugador2_nombre')}",
                'api_live': api_live,
                'db_live': db_live,
                'api_status': api_status,
                'db_status': db_status
            })
    
    if discrepancies:
        print("⚠️  Discrepancias encontradas:")
        print("-" * 80)
        for d in discrepancies:
            print(f"\n{d['players']}")
            print(f"  API: live={d['api_live']}, status={d['api_status']}")
            print(f"  DB:  live={d['db_live']}, estado={d['db_status']}")
    else:
        print("✅ No se encontraron discrepancias en los primeros 10 partidos")

if __name__ == '__main__':
    date = sys.argv[1] if len(sys.argv) > 1 else None
    compare_data(date)
