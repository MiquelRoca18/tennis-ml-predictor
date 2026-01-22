"""
Player Service - Elite Tennis Analytics
Gestiona perfiles de jugadores, estadísticas y rankings
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class PlayerService:
    """Servicio para gestión de jugadores"""
    
    def __init__(self, db):
        """
        Args:
            db: Instancia de MatchDatabase
        """
        self.db = db
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
        # Buscar jugador existente
        player = self.db._fetchone(
            "SELECT * FROM players WHERE player_key = :player_key",
            {"player_key": player_key}
        )
        
        if player:
            return dict(player)
        
        # Crear nuevo jugador
        self.db._execute(
            """
            INSERT INTO players (player_key, player_name, player_logo)
            VALUES (:player_key, :player_name, :player_logo)
        """,
            {"player_key": player_key, "player_name": player_name, "player_logo": player_logo}
        )
        
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
            player_key = api_data.get('player_key')
            
            # Para PostgreSQL, usar INSERT ... ON CONFLICT
            # Para SQLite, usar INSERT OR REPLACE
            if self.db.is_postgres:
                self.db._execute(
                    """
                    INSERT INTO players (
                        player_key, player_name, country, 
                        player_logo, updated_at
                    )
                    VALUES (:player_key, :player_name, :country, :player_logo, CURRENT_TIMESTAMP)
                    ON CONFLICT (player_key) DO UPDATE SET
                        player_name = EXCLUDED.player_name,
                        country = EXCLUDED.country,
                        player_logo = EXCLUDED.player_logo,
                        updated_at = CURRENT_TIMESTAMP
                """,
                    {
                        "player_key": player_key,
                        "player_name": api_data.get('player_name'),
                        "country": api_data.get('player_country'),
                        "player_logo": api_data.get('player_logo')
                    }
                )
            else:
                # SQLite branch (mantener compatibilidad)
                cursor = self.db.conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO players (
                        player_key, player_name, country, 
                        player_logo, updated_at
                    )
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                    (
                        player_key,
                        api_data.get('player_name'),
                        api_data.get('player_country'),
                        api_data.get('player_logo')
                    )
                )
                self.db.conn.commit()
            
            logger.info(f"✅ Updated profile for player {player_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating player profile: {e}")
            return False
    
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
        if league == 'ATP':
            self.db._execute(
                """
                UPDATE players 
                SET atp_ranking = :ranking, 
                    atp_points = :points, 
                    ranking_movement = :movement, 
                    last_ranking_update = CURRENT_TIMESTAMP
                WHERE player_key = :player_key
            """,
                {
                    "ranking": ranking,
                    "points": points,
                    "movement": movement,
                    "player_key": player_key
                }
            )
        else:
            self.db._execute(
                """
                UPDATE players
                SET wta_ranking = :ranking,
                    wta_points = :points,
                    ranking_movement = :movement,
                    last_ranking_update = CURRENT_TIMESTAMP
                WHERE player_key = :player_key
            """,
                {
                    "ranking": ranking,
                    "points": points,
                    "movement": movement,
                    "player_key": player_key
                }
            )
    
    def get_player_profile(self, player_key: int) -> Optional[Dict]:
        """
        Obtiene perfil completo de un jugador
        
        Args:
            player_key: ID del jugador
            
        Returns:
            Dict con perfil completo o None
        """
        # Datos básicos del jugador
        player = self.db._fetchone(
            "SELECT * FROM players WHERE player_key = :player_key",
            {"player_key": player_key}
        )
        
        if not player:
            return None
        
        player_dict = dict(player)
        
        # Nota: player_stats table no existe en schema actual
        # Si se necesita, agregar en el futuro
        player_dict['stats'] = []
        
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
        matches = self.db._fetchall(
            """
            SELECT 
                id, fecha_partido, torneo, superficie,
                jugador1_nombre, jugador2_nombre,
                resultado_ganador, resultado_marcador, estado
            FROM matches
            WHERE (jugador1_key = :player_key OR jugador2_key = :player_key)
            AND estado = 'completado'
            ORDER BY fecha_partido DESC
            LIMIT :limit
        """,
            {"player_key": str(player_key), "limit": last_n}
        )
        
        return matches


if __name__ == "__main__":
    # Test básico
    from src.database.match_database import MatchDatabase
    
    db = MatchDatabase("matches_v2.db")
    service = PlayerService(db)
    
    # Crear jugador de prueba
    player = service.get_or_create_player(
        player_key=1905,
        player_name="N. Djokovic",
        player_logo="https://api.api-tennis.com/logo-tennis/1905_n-djokovic.jpg"
    )
    
    print(f"Player created: {player}")
    
    db.close()
