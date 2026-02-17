"""
Ranking Service - Elite Tennis Analytics
Gestiona rankings ATP/WTA y sincronización con API
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class RankingServiceElite:
    """Servicio Elite para gestión de rankings ATP/WTA"""
    
    def __init__(self, db_connection, api_client, player_service):
        """
        Args:
            db_connection: Conexión a la base de datos (MatchDatabase o sqlite3.Connection)
            api_client: Cliente de API-Tennis
            player_service: Servicio de jugadores
        """
        self.conn = db_connection
        self.api_client = api_client
        self.player_service = player_service
        self.db = db_connection if hasattr(db_connection, '_fetchall') else None
        logger.info("✅ RankingServiceElite initialized")
    
    def sync_atp_rankings(self, limit: int = 100) -> int:
        """
        Sincroniza rankings ATP desde API
        
        Args:
            limit: Número máximo de rankings a sincronizar
            
        Returns:
            Número de rankings sincronizados
        """
        try:
            logger.info(f"🔄 Sincronizando rankings ATP (top {limit})...")
            
            # Usar nuevo método get_rankings()
            rankings = self.api_client.get_rankings(league="ATP")
            
            if not rankings:
                logger.warning("No se obtuvieron rankings ATP de la API")
                return 0
            
            rankings = rankings[:limit] # Apply limit after fetching
            count = 0
            
            for entry in rankings:
                player_key = entry.get('player_key')
                player_name = entry.get('player')
                ranking = int(entry.get('place', 0))
                points = int(entry.get('points', 0))
                movement = entry.get('movement', 'same')
                
                # Crear o actualizar jugador
                self.player_service.get_or_create_player(
                    player_key=player_key,
                    player_name=player_name
                )
                
                # Actualizar ranking
                self.player_service.update_ranking(
                    player_key=player_key,
                    ranking=ranking,
                    points=points,
                    movement=movement,
                    league='ATP'
                )
                
                count += 1
            
            logger.info(f"✅ Sincronizados {count} rankings ATP")
            return count
            
        except Exception as e:
            logger.error(f"Error sincronizando rankings ATP: {e}")
            return 0
    
    # WTA SUPPORT REMOVED - ATP Singles only
    # def sync_wta_rankings(self, limit: int = 100) -> int:
    #     """WTA not supported - ATP Singles only"""
    #     logger.warning("⚠️  WTA rankings not supported - ATP Singles only")
    #     return 0
    
    # def sync_all_rankings(self) -> Dict[str, int]:
    #     """Deprecated - use sync_atp_rankings() directly"""
    #     atp_count = self.sync_atp_rankings()
    #     return {'atp': atp_count, 'wta': 0, 'total': atp_count}
    
    def get_top_players(self, league: str = 'ATP', limit: int = 100) -> List[Dict]:
        """
        Obtiene top N jugadores por ranking
        
        Args:
            league: 'ATP' o 'WTA'
            limit: Número de jugadores
            
        Returns:
            Lista de jugadores ordenados por ranking
        """
        if self.db:
            if league == 'ATP':
                players = self.db._fetchall("""
                    SELECT * FROM players
                    WHERE atp_ranking IS NOT NULL
                    ORDER BY atp_ranking ASC
                    LIMIT :limit
                """, {"limit": limit})
            else:
                players = self.db._fetchall("""
                    SELECT * FROM players
                    WHERE wta_ranking IS NOT NULL
                    ORDER BY wta_ranking ASC
                    LIMIT :limit
                """, {"limit": limit})
            return [dict(p) for p in players] if players else []
        cursor = self.conn.cursor()
        if league == 'ATP':
            players = cursor.execute("""
                SELECT * FROM players
                WHERE atp_ranking IS NOT NULL
                ORDER BY atp_ranking ASC
                LIMIT ?
            """, (limit,)).fetchall()
        else:
            players = cursor.execute("""
                SELECT * FROM players
                WHERE wta_ranking IS NOT NULL
                ORDER BY wta_ranking ASC
                LIMIT ?
            """, (limit,)).fetchall()
        return [dict(p) for p in players]
    
    def get_player_ranking_info(self, player_key: int) -> Optional[Dict]:
        """
        Obtiene información de ranking de un jugador
        
        Args:
            player_key: ID del jugador
            
        Returns:
            Dict con info de ranking o None
        """
        if self.db:
            row = self.db._fetchone("""
                SELECT player_key, player_name,
                       atp_ranking, wta_ranking,
                       atp_points, wta_points,
                       ranking_movement
                FROM players
                WHERE player_key = :player_key
            """, {"player_key": player_key})
            return dict(row) if row else None
        cursor = self.conn.cursor()
        player = cursor.execute("""
            SELECT player_key, player_name,
                   atp_ranking, wta_ranking,
                   atp_points, wta_points,
                   ranking_movement
            FROM players
            WHERE player_key = ?
        """, (player_key,)).fetchone()
        return dict(player) if player else None
