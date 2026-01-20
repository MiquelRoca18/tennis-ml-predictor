"""
Point by Point Service - Elite Tennis Analytics
Gestiona datos punto por punto de partidos
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class PointByPointService:
    """Servicio para gestión de datos punto por punto"""
    
    def __init__(self, db_connection):
        """
        Args:
            db_connection: Conexión a la base de datos
        """
        self.conn = db_connection
        logger.info("✅ PointByPointService initialized")
    
    def store_point_by_point(self, match_id: int, pbp_data: List[Dict]) -> int:
        """
        Guarda datos punto por punto de un partido
        
        Args:
            match_id: ID del partido
            pbp_data: Lista de puntos
            
        Returns:
            Número de puntos guardados
        """
        cursor = self.conn.cursor()
        count = 0
        
        for point in pbp_data:
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO match_pointbypoint (
                        match_id, set_number, game_number, point_number,
                        server, score, is_break_point, is_set_point, is_match_point
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    match_id,
                    point.get('set_number'),
                    point.get('game_number'),
                    point.get('point_number'),
                    point.get('server'),
                    point.get('score'),
                    point.get('is_break_point', False),
                    point.get('is_set_point', False),
                    point.get('is_match_point', False)
                ))
                count += 1
            except Exception as e:
                logger.error(f"Error guardando punto: {e}")
        
        self.conn.commit()
        logger.debug(f"✅ Guardados {count} puntos para match {match_id}")
        
        return count
    
    def get_point_by_point(self, match_id: int, set_number: str = None) -> List[Dict]:
        """
        Obtiene datos punto por punto de un partido
        
        Args:
            match_id: ID del partido
            set_number: Número de set (opcional)
            
        Returns:
            Lista de puntos
        """
        cursor = self.conn.cursor()
        
        if set_number:
            points = cursor.execute("""
                SELECT * FROM match_pointbypoint
                WHERE match_id = ? AND set_number = ?
                ORDER BY set_number, game_number, point_number
            """, (match_id, set_number)).fetchall()
        else:
            points = cursor.execute("""
                SELECT * FROM match_pointbypoint
                WHERE match_id = ?
                ORDER BY set_number, game_number, point_number
            """, (match_id,)).fetchall()
        
        return [dict(p) for p in points]
    
    def store_games(self, match_id: int, games_data: List[Dict]) -> int:
        """
        Guarda datos de juegos de un partido
        
        Args:
            match_id: ID del partido
            games_data: Lista de juegos
            
        Returns:
            Número de juegos guardados
        """
        cursor = self.conn.cursor()
        count = 0
        
        for game in games_data:
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO match_games (
                        match_id, set_number, game_number,
                        server, winner, score_games, score_sets, was_break
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    match_id,
                    game.get('set_number'),
                    game.get('game_number'),
                    game.get('server'),
                    game.get('winner'),
                    game.get('score_games'),
                    game.get('score_sets'),
                    game.get('was_break', False)
                ))
                count += 1
            except Exception as e:
                logger.error(f"Error guardando juego: {e}")
        
        self.conn.commit()
        logger.debug(f"✅ Guardados {count} juegos para match {match_id}")
        
        return count
    
    def get_games(self, match_id: int) -> List[Dict]:
        """
        Obtiene juegos de un partido
        
        Args:
            match_id: ID del partido
            
        Returns:
            Lista de juegos
        """
        cursor = self.conn.cursor()
        
        games = cursor.execute("""
            SELECT * FROM match_games
            WHERE match_id = ?
            ORDER BY set_number, game_number
        """, (match_id,)).fetchall()
        
        return [dict(g) for g in games]
    
    def get_break_points_stats(self, match_id: int) -> Dict:
        """
        Obtiene estadísticas de break points
        
        Args:
            match_id: ID del partido
            
        Returns:
            Dict con estadísticas
        """
        cursor = self.conn.cursor()
        
        # Contar break points
        total_bp = cursor.execute("""
            SELECT COUNT(*) as total
            FROM match_pointbypoint
            WHERE match_id = ? AND is_break_point = 1
        """, (match_id,)).fetchone()
        
        # Contar breaks conseguidos
        breaks = cursor.execute("""
            SELECT COUNT(*) as total
            FROM match_games
            WHERE match_id = ? AND was_break = 1
        """, (match_id,)).fetchone()
        
        return {
            'total_break_points': total_bp['total'] if total_bp else 0,
            'breaks_converted': breaks['total'] if breaks else 0,
            'conversion_rate': (breaks['total'] / total_bp['total'] * 100) if (total_bp and total_bp['total'] > 0) else 0
        }
    
    def parse_api_pointbypoint(self, api_data: List) -> List[Dict]:
        """
        Parsea datos punto por punto de la API
        
        Args:
            api_data: Datos de la API
            
        Returns:
            Lista de puntos parseados
        """
        points = []
        
        for idx, point_data in enumerate(api_data):
            # La API devuelve datos en formato específico
            # Adaptar según estructura real
            points.append({
                'set_number': point_data.get('set', '1'),
                'game_number': point_data.get('game', 1),
                'point_number': idx + 1,
                'server': point_data.get('server', 'First Player'),
                'score': point_data.get('score', '0-0'),
                'is_break_point': point_data.get('break_point', False),
                'is_set_point': point_data.get('set_point', False),
                'is_match_point': point_data.get('match_point', False)
            })
        
        return points


if __name__ == "__main__":
    # Test básico
    import sqlite3
    
    conn = sqlite3.connect("matches_v2.db")
    conn.row_factory = sqlite3.Row
    
    service = PointByPointService(conn)
    print("✅ PointByPointService test completed")
    
    conn.close()
