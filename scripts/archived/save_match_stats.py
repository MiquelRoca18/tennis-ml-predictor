#!/usr/bin/env python3
"""
Script para guardar estadísticas detalladas de partidos completados
Guarda juegos y puntos en las tablas match_games y match_pointbypoint
"""
import sys
from src.database.match_database import MatchDatabase
from src.services.api_tennis_client import APITennisClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def store_detailed_stats(db, api_client, match_id, event_key, match_date):
    """Guarda estadísticas detalladas del partido"""
    try:
        # Obtener datos detallados de la API usando get_fixtures
        # Este endpoint SÍ incluye pointbypoint y scores
        data = api_client._make_request("get_fixtures", {
            "date_start": match_date,
            "date_stop": match_date,
            "match_key": event_key
        })
        
        if not data or not data.get("result"):
            logger.warning(f"No hay datos para evento {event_key}")
            return False
        
        # Obtener el partido específico
        matches = data["result"]
        match_data = None
        
        if isinstance(matches, list):
            for m in matches:
                if str(m.get("event_key")) == str(event_key):
                    match_data = m
                    break
        else:
            match_data = matches
        
        if not match_data:
            logger.warning(f"No se encontró match {event_key} en respuesta")
            return False
        cursor = db.conn.cursor()
        
        # Verificar si ya tenemos estos datos
        existing = cursor.execute(
            "SELECT COUNT(*) FROM match_games WHERE match_id = ?",
            (match_id,)
        ).fetchone()[0]
        
        if existing > 0:
            logger.info(f"Estadísticas ya guardadas para match {match_id}")
            return True
        
        # Guardar juegos
        games_saved = 0
        if "pointbypoint" in match_data and match_data["pointbypoint"]:
            for game in match_data["pointbypoint"]:
                try:
                    cursor.execute("""
                        INSERT INTO match_games (
                            match_id, set_number, game_number,
                            server, winner, score_games, score_sets, was_break
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        match_id,
                        game.get("set_number", ""),
                        game.get("number_game", ""),
                        game.get("player_served", ""),
                        game.get("serve_winner", ""),
                        game.get("score", ""),
                        "",  # score_sets
                        1 if game.get("serve_lost") else 0
                    ))
                    games_saved += 1
                except Exception as e:
                    logger.debug(f"Error guardando juego: {e}")
            
            logger.info(f"✅ Guardados {games_saved} juegos para match {match_id}")
        
        # Guardar puntos
        points_saved = 0
        if "pointbypoint" in match_data and match_data["pointbypoint"]:
            for game in match_data["pointbypoint"]:
                if "points" in game and game["points"]:
                    for point in game["points"]:
                        try:
                            cursor.execute("""
                                INSERT INTO match_pointbypoint (
                                    match_id, set_number, game_number, point_number,
                                    server, score, is_break_point, is_set_point, is_match_point
                                )
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                match_id,
                                game.get("set_number", ""),
                                game.get("number_game", ""),
                                point.get("number_point", ""),
                                game.get("player_served", ""),
                                point.get("score", ""),
                                1 if point.get("break_point") else 0,
                                1 if point.get("set_point") else 0,
                                1 if point.get("match_point") else 0
                            ))
                            points_saved += 1
                        except Exception as e:
                            logger.debug(f"Error guardando punto: {e}")
            
            logger.info(f"✅ Guardados {points_saved} puntos para match {match_id}")
        
        db.conn.commit()
        return games_saved > 0 or points_saved > 0
        
    except Exception as e:
        logger.error(f"Error guardando estadísticas: {e}")
        return False

def main():
    """Guardar estadísticas para todos los partidos completados de hoy"""
    print("🎾 Guardando estadísticas detalladas de partidos\n")
    
    db = MatchDatabase("matches_v2.db")
    api_client = APITennisClient()
    
    # Obtener partidos completados de hoy
    cursor = db.conn.cursor()
    matches = cursor.execute("""
        SELECT id, event_key, jugador1_nombre, jugador2_nombre, fecha_partido
        FROM matches
        WHERE fecha_partido = date('now')
          AND estado = 'completado'
          AND event_key IS NOT NULL
    """).fetchall()
    
    print(f"Encontrados {len(matches)} partidos completados\n")
    
    success_count = 0
    for match_id, event_key, p1, p2, fecha in matches:
        print(f"Procesando: {p1} vs {p2}...")
        if store_detailed_stats(db, api_client, match_id, event_key, str(fecha)):
            success_count += 1
    
    # Mostrar resumen
    total_games = cursor.execute("SELECT COUNT(*) FROM match_games").fetchone()[0]
    total_points = cursor.execute("SELECT COUNT(*) FROM match_pointbypoint").fetchone()[0]
    
    print(f"\n{'='*60}")
    print(f"📊 RESUMEN:")
    print(f"  Partidos procesados: {len(matches)}")
    print(f"  Partidos con estadísticas: {success_count}")
    print(f"  Total juegos en DB: {total_games}")
    print(f"  Total puntos en DB: {total_points}")
    print(f"{'='*60}\n")
    
    db.close()
    print("✅ Completado")

if __name__ == "__main__":
    main()
