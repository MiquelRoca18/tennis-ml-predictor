"""
Live Match Data Service
Sincroniza datos de partidos en vivo usando get_livescore
Captura punto por punto en tiempo real
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class LiveMatchDataService:
    """Servicio para sincronizar datos de partidos en vivo en tiempo real"""
    
    def __init__(self, db_connection, api_client, pbp_service):
        """
        Args:
            db_connection: Conexión a la base de datos
            api_client: Cliente de API-Tennis
            pbp_service: Servicio de punto por punto
        """
        self.conn = db_connection
        self.api_client = api_client
        self.pbp_service = pbp_service
        logger.info("✅ LiveMatchDataService initialized")
    
    def sync_live_matches(self) -> Dict:
        """
        Sincroniza datos de partidos en vivo
        
        Returns:
            Dict con estadísticas de la sincronización
        """
        try:
            # 1. Obtener partidos en vivo desde la API
            live_data = self.api_client._make_request("get_livescore", {})
            
            if not live_data or not live_data.get("result"):
                return {
                    "success": True,
                    "matches_live": 0,
                    "points_stored": 0,
                    "games_stored": 0,
                    "message": "No hay partidos en vivo"
                }
            
            live_matches = live_data["result"]
            
            # 2. Procesar cada partido en vivo
            matches_processed = 0
            total_points = 0
            total_games = 0
            
            for live_match in live_matches:
                event_key = live_match.get("event_key")
                
                if not event_key:
                    continue
                
                # Buscar partido en nuestra DB
                match = self._find_match_by_event_key(event_key)
                
                if not match:
                    logger.debug(f"Partido en vivo no encontrado en DB: {event_key}")
                    continue
                
                # Procesar datos del partido
                result = self.process_live_match(live_match, match["id"])
                
                if result["points_stored"] > 0 or result["games_stored"] > 0:
                    matches_processed += 1
                    total_points += result["points_stored"]
                    total_games += result["games_stored"]
            
            return {
                "success": True,
                "matches_live": len(live_matches),
                "matches_processed": matches_processed,
                "points_stored": total_points,
                "games_stored": total_games,
                "message": f"{matches_processed} partidos procesados"
            }
            
        except Exception as e:
            logger.error(f"❌ Error sincronizando partidos en vivo: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "message": "Error en sincronización"
            }
    
    def process_live_match(self, live_match: Dict, match_id: int) -> Dict:
        """
        Procesa datos de un partido en vivo
        
        Args:
            live_match: Datos del partido desde get_livescore
            match_id: ID del partido en nuestra DB
            
        Returns:
            Dict con puntos y juegos guardados
        """
        points_stored = 0
        games_stored = 0
        
        try:
            # 1. Actualizar marcador y estado
            self._update_match_status(match_id, live_match)
            
            # 2. Guardar datos punto por punto
            pointbypoint = live_match.get("pointbypoint", [])
            if pointbypoint:
                points_stored = self.save_new_points(match_id, pointbypoint)
            
            # 3. Guardar scores
            scores = live_match.get("scores", [])
            if scores:
                games_stored = self.save_new_games(match_id, scores, pointbypoint)
            
            return {
                "points_stored": points_stored,
                "games_stored": games_stored
            }
            
        except Exception as e:
            logger.error(f"Error procesando partido {match_id}: {e}")
            return {"points_stored": 0, "games_stored": 0}
    
    def save_new_points(self, match_id: int, pointbypoint: List[Dict]) -> int:
        """
        Guarda solo puntos nuevos (evita duplicados)
        
        Args:
            match_id: ID del partido
            pointbypoint: Lista de juegos con puntos desde la API
            
        Returns:
            Número de puntos nuevos guardados
        """
        if not pointbypoint:
            return 0
        
        try:
            cursor = self.conn.cursor()
            
            # Obtener último punto guardado
            last_point = cursor.execute("""
                SELECT MAX(point_number) as last_point, MAX(game_number) as last_game
                FROM match_pointbypoint
                WHERE match_id = ?
            """, (match_id,)).fetchone()
            
            last_game_num = last_point["last_game"] if last_point and last_point["last_game"] else 0
            last_point_num = last_point["last_point"] if last_point and last_point["last_point"] else 0
            
            # Procesar puntos
            points_to_save = []
            
            for game in pointbypoint:
                set_number = game.get("set_number", "Set 1")
                game_number = int(game.get("number_game", 0))
                server = game.get("player_served", "First Player")
                
                points = game.get("points", [])
                
                for point in points:
                    point_number = int(point.get("number_point", 0))
                    
                    # Solo guardar puntos nuevos
                    if game_number > last_game_num or (game_number == last_game_num and point_number > last_point_num):
                        points_to_save.append({
                            'set_number': set_number,
                            'game_number': game_number,
                            'point_number': point_number,
                            'server': server,
                            'score': point.get('score', '0-0'),
                            'is_break_point': bool(point.get('break_point')),
                            'is_set_point': bool(point.get('set_point')),
                            'is_match_point': bool(point.get('match_point'))
                        })
            
            # Guardar usando el servicio PBP
            if points_to_save:
                count = self.pbp_service.store_point_by_point(match_id, points_to_save)
                return count
            
            return 0
            
        except Exception as e:
            logger.error(f"Error guardando puntos: {e}")
            return 0
    
    def save_new_games(self, match_id: int, scores: List[Dict], pointbypoint: List[Dict] = None) -> int:
        """
        Guarda solo juegos nuevos (evita duplicados)
        
        Args:
            match_id: ID del partido
            scores: Lista de scores por set
            pointbypoint: Datos punto por punto (para extraer info de juegos)
            
        Returns:
            Número de juegos nuevos guardados
        """
        if not pointbypoint:
            return 0
        
        try:
            cursor = self.conn.cursor()
            
            # Obtener último juego guardado
            last_game = cursor.execute("""
                SELECT MAX(game_number) as last_game
                FROM match_games
                WHERE match_id = ?
            """, (match_id,)).fetchone()
            
            last_game_num = last_game["last_game"] if last_game and last_game["last_game"] else 0
            
            # Procesar juegos
            games_to_save = []
            
            for game in pointbypoint:
                set_number = game.get("set_number", "Set 1")
                game_number = int(game.get("number_game", 0))
                
                # Solo guardar juegos nuevos
                if game_number > last_game_num:
                    server = game.get("player_served", "First Player")
                    winner = game.get("serve_winner", "First Player")
                    score = game.get("score", "0-0")
                    was_break = bool(game.get("serve_lost"))
                    
                    games_to_save.append({
                        'set_number': set_number,
                        'game_number': game_number,
                        'server': server,
                        'winner': winner,
                        'score_games': score,
                        'score_sets': "0-0",  # Simplificación
                        'was_break': was_break
                    })
            
            # Guardar usando el servicio PBP
            if games_to_save:
                count = self.pbp_service.store_games(match_id, games_to_save)
                return count
            
            return 0
            
        except Exception as e:
            logger.error(f"Error guardando juegos: {e}")
            return 0
    
    def _find_match_by_event_key(self, event_key: str) -> Optional[Dict]:
        """
        Busca un partido en la DB por event_key
        
        Args:
            event_key: Key del evento
            
        Returns:
            Dict con datos del partido o None
        """
        try:
            cursor = self.conn.cursor()
            match = cursor.execute("""
                SELECT id, jugador1_nombre, jugador2_nombre, estado
                FROM matches
                WHERE event_key = ?
            """, (str(event_key),)).fetchone()
            
            return dict(match) if match else None
            
        except Exception as e:
            logger.debug(f"Error buscando partido: {e}")
            return None
    
    def _update_match_status(self, match_id: int, live_match: Dict):
        """
        Actualiza el estado y marcador del partido
        
        Args:
            match_id: ID del partido
            live_match: Datos del partido en vivo
        """
        try:
            cursor = self.conn.cursor()
            
            event_live = live_match.get("event_live", "0")
            event_status = live_match.get("event_status", "")
            event_final_result = live_match.get("event_final_result", "-")
            
            # Construir marcador
            scores = live_match.get("scores", [])
            score_str = " ".join([f"{s.get('score_first', 0)}-{s.get('score_second', 0)}" for s in scores])
            
            cursor.execute("""
                UPDATE matches
                SET event_live = ?,
                    event_status = ?,
                    event_final_result = ?,
                    resultado_marcador = ?
                WHERE id = ?
            """, (event_live, event_status, event_final_result, score_str, match_id))
            
            self.conn.commit()
            
        except Exception as e:
            logger.debug(f"Error actualizando estado: {e}")


if __name__ == "__main__":
    # Test básico
    import sqlite3
    from src.services.api_tennis_client import APITennisClient
    from src.services.pointbypoint_service import PointByPointService
    
    conn = sqlite3.connect("matches_v2.db")
    conn.row_factory = sqlite3.Row
    
    api_client = APITennisClient()
    pbp_service = PointByPointService(conn)
    
    service = LiveMatchDataService(conn, api_client, pbp_service)
    
    # Sincronizar partidos en vivo
    result = service.sync_live_matches()
    print(f"\n✅ Test result: {result}")
    
    conn.close()
