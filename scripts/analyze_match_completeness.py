#!/usr/bin/env python3
"""
Analiza la completitud de datos por partido en la base de datos
Genera reporte detallado de qué datos tenemos y cuántos partidos están completos
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path

# Ruta a la base de datos
DB_PATH = Path(__file__).parent.parent / "matches_v2.db"

def analyze_match_completeness():
    """Analiza completitud de datos por partido"""
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    print("=" * 80)
    print("📊 ANÁLISIS DE COMPLETITUD DE DATOS POR PARTIDO")
    print("=" * 80)
    print()
    
    # 1. INFORMACIÓN BÁSICA DE PARTIDOS
    print("1️⃣  INFORMACIÓN BÁSICA DE PARTIDOS")
    print("-" * 80)
    
    total_matches = cursor.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    print(f"Total de partidos: {total_matches}")
    
    # Campos básicos
    basic_fields = {
        "event_key": "ID único (event_key)",
        "fecha_partido": "Fecha",
        "hora_inicio": "Hora",
        "torneo": "Torneo",
        "superficie": "Superficie",
        "ronda": "Ronda",
        "jugador1_nombre": "Jugador 1",
        "jugador2_nombre": "Jugador 2",
        "jugador1_key": "ID Jugador 1",
        "jugador2_key": "ID Jugador 2",
        "jugador1_logo": "Logo Jugador 1",
        "jugador2_logo": "Logo Jugador 2",
        "jugador1_ranking": "Ranking Jugador 1",
        "jugador2_ranking": "Ranking Jugador 2",
        "estado": "Estado",
        "event_live": "En vivo",
        "event_game_result": "Score juego actual",
        "event_serve": "Quién saca",
        "resultado_ganador": "Ganador",
        "resultado_marcador": "Marcador",
        "event_final_result": "Resultado final"
    }
    
    print("\nDisponibilidad de campos básicos:")
    for field, label in basic_fields.items():
        count = cursor.execute(f"SELECT COUNT(*) FROM matches WHERE {field} IS NOT NULL AND {field} != ''").fetchone()[0]
        percentage = (count / total_matches * 100) if total_matches > 0 else 0
        status = "✅" if percentage >= 90 else "⚠️" if percentage >= 50 else "❌"
        print(f"  {status} {label:30} {count:4}/{total_matches} ({percentage:5.1f}%)")
    
    print()
    
    # 2. PREDICCIONES
    print("2️⃣  PREDICCIONES")
    print("-" * 80)
    
    matches_with_predictions = cursor.execute("""
        SELECT COUNT(DISTINCT match_id) FROM predictions
    """).fetchone()[0]
    
    total_predictions = cursor.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    
    print(f"Partidos con predicciones: {matches_with_predictions}/{total_matches} ({matches_with_predictions/total_matches*100:.1f}%)")
    print(f"Total de predicciones (versiones): {total_predictions}")
    
    # Predicciones por confianza
    confidence_stats = cursor.execute("""
        SELECT 
            confidence_level,
            COUNT(*) as count
        FROM predictions
        WHERE version = (SELECT MAX(version) FROM predictions p2 WHERE p2.match_id = predictions.match_id)
        GROUP BY confidence_level
    """).fetchall()
    
    print("\nPredicciones por nivel de confianza (última versión):")
    for row in confidence_stats:
        print(f"  - {row['confidence_level'] or 'UNKNOWN':10} {row['count']:4} predicciones")
    
    print()
    
    # 3. CUOTAS
    print("3️⃣  CUOTAS")
    print("-" * 80)
    
    matches_with_odds = cursor.execute("""
        SELECT COUNT(DISTINCT match_id) FROM predictions
        WHERE jugador1_cuota IS NOT NULL AND jugador2_cuota IS NOT NULL
    """).fetchone()[0]
    
    print(f"Partidos con cuotas: {matches_with_odds}/{total_matches} ({matches_with_odds/total_matches*100:.1f}%)")
    
    # Rango de cuotas
    odds_range = cursor.execute("""
        SELECT 
            MIN(jugador1_cuota) as min_odds,
            MAX(jugador1_cuota) as max_odds,
            AVG(jugador1_cuota) as avg_odds
        FROM predictions
        WHERE jugador1_cuota IS NOT NULL
    """).fetchone()
    
    print(f"\nRango de cuotas:")
    print(f"  - Mínima: {odds_range['min_odds']:.2f}")
    print(f"  - Máxima: {odds_range['max_odds']:.2f}")
    print(f"  - Promedio: {odds_range['avg_odds']:.2f}")
    
    print()
    
    # 4. POINT BY POINT
    print("4️⃣  DATOS PUNTO POR PUNTO")
    print("-" * 80)
    
    matches_with_pbp = cursor.execute("""
        SELECT COUNT(DISTINCT match_id) FROM match_pointbypoint
    """).fetchone()[0]
    
    total_points = cursor.execute("SELECT COUNT(*) FROM match_pointbypoint").fetchone()[0]
    
    print(f"Partidos con datos punto por punto: {matches_with_pbp}/{total_matches} ({matches_with_pbp/total_matches*100:.1f}%)")
    print(f"Total de puntos registrados: {total_points:,}")
    
    if matches_with_pbp > 0:
        avg_points = total_points / matches_with_pbp
        print(f"Promedio de puntos por partido: {avg_points:.0f}")
    
    print()
    
    # 5. JUEGOS (GAMES)
    print("5️⃣  JUEGOS")
    print("-" * 80)
    
    matches_with_games = cursor.execute("""
        SELECT COUNT(DISTINCT match_id) FROM match_games
    """).fetchone()[0]
    
    total_games = cursor.execute("SELECT COUNT(*) FROM match_games").fetchone()[0]
    
    print(f"Partidos con datos de juegos: {matches_with_games}/{total_matches} ({matches_with_games/total_matches*100:.1f}%)")
    print(f"Total de juegos registrados: {total_games:,}")
    
    if matches_with_games > 0:
        avg_games = total_games / matches_with_games
        print(f"Promedio de juegos por partido: {avg_games:.0f}")
    
    print()
    
    # 6. SETS (NUEVA TABLA)
    print("6️⃣  SETS ESTRUCTURADOS")
    print("-" * 80)
    
    # Verificar si la tabla existe
    table_exists = cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='match_sets'
    """).fetchone()
    
    if table_exists:
        matches_with_sets = cursor.execute("""
            SELECT COUNT(DISTINCT match_id) FROM match_sets
        """).fetchone()[0]
        
        total_sets = cursor.execute("SELECT COUNT(*) FROM match_sets").fetchone()[0]
        
        print(f"Partidos con sets estructurados: {matches_with_sets}/{total_matches} ({matches_with_sets/total_matches*100:.1f}%)")
        print(f"Total de sets registrados: {total_sets:,}")
        
        if matches_with_sets > 0:
            avg_sets = total_sets / matches_with_sets
            print(f"Promedio de sets por partido: {avg_sets:.1f}")
    else:
        print("⚠️  Tabla match_sets no existe aún (se creará tras próximo deploy)")
    
    print()
    
    # 7. COMPLETITUD POR PARTIDO
    print("7️⃣  ANÁLISIS DE COMPLETITUD")
    print("-" * 80)
    
    # Definir niveles de completitud
    completeness_query = """
        SELECT 
            m.id,
            m.event_key,
            m.jugador1_nombre,
            m.jugador2_nombre,
            m.fecha_partido,
            m.estado,
            -- Campos básicos (10 puntos)
            CASE WHEN m.event_key IS NOT NULL THEN 1 ELSE 0 END +
            CASE WHEN m.fecha_partido IS NOT NULL THEN 1 ELSE 0 END +
            CASE WHEN m.hora_inicio IS NOT NULL THEN 1 ELSE 0 END +
            CASE WHEN m.torneo IS NOT NULL THEN 1 ELSE 0 END +
            CASE WHEN m.superficie IS NOT NULL THEN 1 ELSE 0 END +
            CASE WHEN m.jugador1_nombre IS NOT NULL THEN 1 ELSE 0 END +
            CASE WHEN m.jugador2_nombre IS NOT NULL THEN 1 ELSE 0 END +
            CASE WHEN m.jugador1_key IS NOT NULL THEN 1 ELSE 0 END +
            CASE WHEN m.jugador2_key IS NOT NULL THEN 1 ELSE 0 END +
            CASE WHEN m.estado IS NOT NULL THEN 1 ELSE 0 END as basic_score,
            -- Logos (2 puntos)
            CASE WHEN m.jugador1_logo IS NOT NULL AND m.jugador1_logo != '' THEN 1 ELSE 0 END +
            CASE WHEN m.jugador2_logo IS NOT NULL AND m.jugador2_logo != '' THEN 1 ELSE 0 END as logo_score,
            -- Rankings (2 puntos)
            CASE WHEN m.jugador1_ranking IS NOT NULL THEN 1 ELSE 0 END +
            CASE WHEN m.jugador2_ranking IS NOT NULL THEN 1 ELSE 0 END as ranking_score,
            -- Resultado (3 puntos)
            CASE WHEN m.resultado_ganador IS NOT NULL THEN 1 ELSE 0 END +
            CASE WHEN m.resultado_marcador IS NOT NULL THEN 1 ELSE 0 END +
            CASE WHEN m.event_final_result IS NOT NULL THEN 1 ELSE 0 END as result_score,
            -- Predicción (1 punto)
            CASE WHEN EXISTS(SELECT 1 FROM predictions WHERE match_id = m.id) THEN 1 ELSE 0 END as prediction_score,
            -- Point by point (1 punto)
            CASE WHEN EXISTS(SELECT 1 FROM match_pointbypoint WHERE match_id = m.id) THEN 1 ELSE 0 END as pbp_score,
            -- Games (1 punto)
            CASE WHEN EXISTS(SELECT 1 FROM match_games WHERE match_id = m.id) THEN 1 ELSE 0 END as games_score
        FROM matches m
    """
    
    matches_completeness = cursor.execute(completeness_query).fetchall()
    
    # Calcular estadísticas
    completeness_levels = {
        "Completo (18-20 puntos)": 0,
        "Muy bueno (15-17 puntos)": 0,
        "Bueno (12-14 puntos)": 0,
        "Regular (9-11 puntos)": 0,
        "Básico (6-8 puntos)": 0,
        "Incompleto (0-5 puntos)": 0
    }
    
    total_score = 0
    max_possible = 20  # 10 básicos + 2 logos + 2 rankings + 3 resultado + 1 pred + 1 pbp + 1 games
    
    for match in matches_completeness:
        score = (match['basic_score'] + match['logo_score'] + match['ranking_score'] + 
                match['result_score'] + match['prediction_score'] + match['pbp_score'] + 
                match['games_score'])
        total_score += score
        
        if score >= 18:
            completeness_levels["Completo (18-20 puntos)"] += 1
        elif score >= 15:
            completeness_levels["Muy bueno (15-17 puntos)"] += 1
        elif score >= 12:
            completeness_levels["Bueno (12-14 puntos)"] += 1
        elif score >= 9:
            completeness_levels["Regular (9-11 puntos)"] += 1
        elif score >= 6:
            completeness_levels["Básico (6-8 puntos)"] += 1
        else:
            completeness_levels["Incompleto (0-5 puntos)"] += 1
    
    avg_completeness = (total_score / (total_matches * max_possible) * 100) if total_matches > 0 else 0
    
    print(f"Completitud promedio: {avg_completeness:.1f}%")
    print(f"\nDistribución por nivel de completitud:")
    for level, count in completeness_levels.items():
        percentage = (count / total_matches * 100) if total_matches > 0 else 0
        bar = "█" * int(percentage / 2)
        print(f"  {level:30} {count:4} ({percentage:5.1f}%) {bar}")
    
    print()
    
    # 8. RESUMEN FINAL
    print("8️⃣  RESUMEN FINAL")
    print("-" * 80)
    
    summary = {
        "total_partidos": total_matches,
        "completitud_promedio": f"{avg_completeness:.1f}%",
        "partidos_completos": completeness_levels["Completo (18-20 puntos)"],
        "con_predicciones": matches_with_predictions,
        "con_cuotas": matches_with_odds,
        "con_point_by_point": matches_with_pbp,
        "con_juegos": matches_with_games,
        "con_sets": matches_with_sets if table_exists else 0,
        "total_puntos": total_points,
        "total_juegos": total_games,
        "total_sets": total_sets if table_exists else 0
    }
    
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    
    # Guardar reporte
    report_path = Path(__file__).parent.parent / "match_completeness_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Reporte guardado en: {report_path}")
    
    conn.close()

if __name__ == "__main__":
    analyze_match_completeness()
