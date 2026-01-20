"""
Player Service - Elite Tennis Analytics
Gestiona perfiles de jugadores, estadísticas y rankings
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
import sqlite3

logger = logging.getLogger(__name__)


class PlayerService:
    """Servicio para gestión de jugadores"""
    
    def __init__(self, db_connection):
        """
        Args:
            db_connection: Conexión a la base de datos
        """
        self.conn = db_connection
        logger.info("✅ PlayerService initialized")
    
    def get_or_create_player(self, player_key: int, player_name: str, 
                             player_logo: str = None) -> Dict:
        """
        Obtiene un jugador o lo crea si no existe
        
        Args:
            player_key: ID del jugador en API-Tennis
            player_name: Nombre del jugador
            player_logo: URL del logo
            
        Returns:
            Dict con datos del jugador
        """
        cursor = self.conn.cursor()
        
        # Buscar jugador existente
        player = cursor.execute("""
            SELECT * FROM players WHERE player_key = ?
        """, (player_key,)).fetchone()
        
        if player:
            return dict(player)
        
        # Crear nuevo jugador
        cursor.execute("""
            INSERT INTO players (player_key, player_name, player_logo)
            VALUES (?, ?, ?)
        """, (player_key, player_name, player_logo))
        
        self.conn.commit()
        logger.info(f"✅ Created player: {player_name} (key: {player_key})")
        
        return {
            'player_key': player_key,
            'player_name': player_name,
            'player_logo': player_logo
        }
    
    def update_player_profile(self, api_data: Dict) -> bool:
        """
        Actualiza perfil completo de un jugador desde API
        
        Args:
            api_data: Datos de get_players de API-Tennis
            
        Returns:
            True si se actualizó correctamente
        """
        try:
            cursor = self.conn.cursor()
            
            player_key = api_data.get('player_key')
            
            cursor.execute("""
                INSERT OR REPLACE INTO players (
                    player_key, player_name, player_country, 
                    player_birthday, player_logo, last_updated
                )
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                player_key,
                api_data.get('player_name'),
                api_data.get('player_country'),
                api_data.get('player_bday'),
                api_data.get('player_logo')
            ))
            
            # Actualizar estadísticas por temporada
            stats = api_data.get('stats', [])
            for stat in stats:
                self._update_player_stats(player_key, stat)
            
            self.conn.commit()
            logger.info(f"✅ Updated profile for player {player_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating player profile: {e}")
            return False
    
    def _update_player_stats(self, player_key: int, stat_data: Dict):
        """Actualiza estadísticas de una temporada"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO player_stats (
                player_key, season, type, rank, titles,
                matches_won, matches_lost,
                hard_won, hard_lost,
                clay_won, clay_lost,
                grass_won, grass_lost
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            player_key,
            stat_data.get('season'),
            stat_data.get('type', 'singles'),
            stat_data.get('rank'),
            stat_data.get('titles', 0),
            stat_data.get('matches_won', 0),
            stat_data.get('matches_lost', 0),
            stat_data.get('hard_won', 0),
            stat_data.get('hard_lost', 0),
            stat_data.get('clay_won', 0),
            stat_data.get('clay_lost', 0),
            stat_data.get('grass_won', 0),
            stat_data.get('grass_lost', 0)
        ))
    
    def update_ranking(self, player_key: int, ranking: int, 
                      points: int, movement: str, league: str = 'ATP'):
        """
        Actualiza ranking de un jugador
        
        Args:
            player_key: ID del jugador
            ranking: Posición en el ranking
            points: Puntos del ranking
            movement: 'up', 'down', 'same'
            league: 'ATP' o 'WTA'
        """
        cursor = self.conn.cursor()
        
        if league == 'ATP':
            cursor.execute("""
                UPDATE players 
                SET atp_ranking = ?, ranking_points = ?, 
                    ranking_movement = ?, last_updated = CURRENT_TIMESTAMP
                WHERE player_key = ?
            """, (ranking, points, movement, player_key))
        else:
            cursor.execute("""
                UPDATE players 
                SET wta_ranking = ?, ranking_points = ?, 
                    ranking_movement = ?, last_updated = CURRENT_TIMESTAMP
                WHERE player_key = ?
            """, (ranking, points, movement, player_key))
        
        self.conn.commit()
    
    def get_player_profile(self, player_key: int) -> Optional[Dict]:
        """
        Obtiene perfil completo de un jugador
        
        Args:
            player_key: ID del jugador
            
        Returns:
            Dict con perfil completo o None
        """
        cursor = self.conn.cursor()
        
        # Datos básicos del jugador
        player = cursor.execute("""
            SELECT * FROM players WHERE player_key = ?
        """, (player_key,)).fetchone()
        
        if not player:
            return None
        
        player_dict = dict(player)
        
        # Estadísticas por temporada
        stats = cursor.execute("""
            SELECT * FROM player_stats 
            WHERE player_key = ?
            ORDER BY season DESC
        """, (player_key,)).fetchall()
        
        player_dict['stats'] = [dict(s) for s in stats]
        
        return player_dict
    
    def get_player_form(self, player_key: int, last_n: int = 10) -> List[Dict]:
        """
        Obtiene últimos N partidos de un jugador
        
        Args:
            player_key: ID del jugador
            last_n: Número de partidos
            
        Returns:
            Lista de partidos recientes
        """
        cursor = self.conn.cursor()
        
        matches = cursor.execute("""
            SELECT 
                id, fecha_partido, torneo, superficie,
                jugador1_nombre, jugador2_nombre,
                resultado_ganador, resultado_marcador, estado
            FROM matches
            WHERE (first_player_key = ? OR second_player_key = ?)
            AND estado = 'completado'
            ORDER BY fecha_partido DESC
            LIMIT ?
        """, (player_key, player_key, last_n)).fetchall()
        
        return [dict(m) for m in matches]
    
    def get_surface_stats(self, player_key: int, surface: str, 
                         season: int = None) -> Dict:
        """
        Obtiene estadísticas en una superficie específica
        
        Args:
            player_key: ID del jugador
            surface: 'Hard', 'Clay', 'Grass'
            season: Temporada (opcional)
            
        Returns:
            Dict con estadísticas
        """
        cursor = self.conn.cursor()
        
        surface_lower = surface.lower()
        won_col = f"{surface_lower}_won"
        lost_col = f"{surface_lower}_lost"
        
        if season:
            stats = cursor.execute(f"""
                SELECT 
                    SUM({won_col}) as won,
                    SUM({lost_col}) as lost
                FROM player_stats
                WHERE player_key = ? AND season = ?
            """, (player_key, season)).fetchone()
        else:
            stats = cursor.execute(f"""
                SELECT 
                    SUM({won_col}) as won,
                    SUM({lost_col}) as lost
                FROM player_stats
                WHERE player_key = ?
            """, (player_key,)).fetchone()
        
        if stats and stats['won'] is not None:
            total = stats['won'] + stats['lost']
            win_rate = (stats['won'] / total * 100) if total > 0 else 0
            
            return {
                'surface': surface,
                'won': stats['won'],
                'lost': stats['lost'],
                'total': total,
                'win_percentage': round(win_rate, 2)
            }
        
        return {
            'surface': surface,
            'won': 0,
            'lost': 0,
            'total': 0,
            'win_percentage': 0
        }


if __name__ == "__main__":
    # Test básico
    import sqlite3
    conn = sqlite3.connect("matches_v2.db")
    conn.row_factory = sqlite3.Row
    
    service = PlayerService(conn)
    
    # Crear jugador de prueba
    player = service.get_or_create_player(
        player_key=1905,
        player_name="N. Djokovic",
        player_logo="https://api.api-tennis.com/logo-tennis/1905_n-djokovic.jpg"
    )
    
    print(f"Player created: {player}")
    
    conn.close()
