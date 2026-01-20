"""
Tournament Service - Elite Tennis Analytics
Gestiona catálogo de torneos y sincronización con API
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class TournamentService:
    """Servicio para gestión de torneos"""
    
    def __init__(self, db_connection, api_client):
        """
        Args:
            db_connection: Conexión a la base de datos
            api_client: Cliente de API-Tennis
        """
        self.conn = db_connection
        self.api_client = api_client
        logger.info("✅ TournamentService initialized")
    
    def sync_tournaments(self) -> int:
        """
        Sincroniza catálogo completo de torneos desde API
        
        Returns:
            Número de torneos sincronizados
        """
        try:
            # Obtener torneos de la API
            data = self.api_client._make_request("get_tournaments")
            
            if not data or not data.get("result"):
                logger.warning("No se obtuvieron torneos de la API")
                return 0
            
            tournaments = data["result"]
            cursor = self.conn.cursor()
            count = 0
            
            for tournament in tournaments:
                cursor.execute("""
                    INSERT OR REPLACE INTO tournaments (
                        tournament_key, tournament_name, 
                        event_type_key, event_type_type
                    )
                    VALUES (?, ?, ?, ?)
                """, (
                    tournament.get('tournament_key'),
                    tournament.get('tournament_name'),
                    tournament.get('event_type_key'),
                    tournament.get('event_type_type')
                ))
                count += 1
            
            self.conn.commit()
            logger.info(f"✅ Sincronizados {count} torneos")
            return count
            
        except Exception as e:
            logger.error(f"Error sincronizando torneos: {e}")
            return 0
    
    def get_tournament(self, tournament_key: int) -> Optional[Dict]:
        """
        Obtiene información de un torneo
        
        Args:
            tournament_key: ID del torneo
            
        Returns:
            Dict con datos del torneo o None
        """
        cursor = self.conn.cursor()
        
        tournament = cursor.execute("""
            SELECT * FROM tournaments WHERE tournament_key = ?
        """, (tournament_key,)).fetchone()
        
        return dict(tournament) if tournament else None
    
    def get_tournament_matches(self, tournament_key: int, 
                               season: int = None) -> List[Dict]:
        """
        Obtiene partidos de un torneo
        
        Args:
            tournament_key: ID del torneo
            season: Temporada (opcional)
            
        Returns:
            Lista de partidos
        """
        cursor = self.conn.cursor()
        
        if season:
            matches = cursor.execute("""
                SELECT * FROM matches 
                WHERE tournament_key = ? AND tournament_season = ?
                ORDER BY fecha_partido DESC
            """, (str(tournament_key), str(season))).fetchall()
        else:
            matches = cursor.execute("""
                SELECT * FROM matches 
                WHERE tournament_key = ?
                ORDER BY fecha_partido DESC
            """, (str(tournament_key),)).fetchall()
        
        return [dict(m) for m in matches]
    
    def get_all_tournaments(self, event_type: str = None) -> List[Dict]:
        """
        Obtiene lista de todos los torneos
        
        Args:
            event_type: Filtrar por tipo (opcional)
            
        Returns:
            Lista de torneos
        """
        cursor = self.conn.cursor()
        
        if event_type:
            tournaments = cursor.execute("""
                SELECT * FROM tournaments 
                WHERE event_type_type = ?
                ORDER BY tournament_name
            """, (event_type,)).fetchall()
        else:
            tournaments = cursor.execute("""
                SELECT * FROM tournaments 
                ORDER BY tournament_name
            """).fetchall()
        
        return [dict(t) for t in tournaments]


if __name__ == "__main__":
    # Test básico
    import sqlite3
    from src.services.api_tennis_client import APITennisClient
    
    conn = sqlite3.connect("matches_v2.db")
    conn.row_factory = sqlite3.Row
    api_client = APITennisClient()
    
    service = TournamentService(conn, api_client)
    
    # Sincronizar torneos
    count = service.sync_tournaments()
    print(f"Torneos sincronizados: {count}")
    
    conn.close()
