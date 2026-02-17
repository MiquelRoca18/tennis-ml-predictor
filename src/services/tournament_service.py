"""
Tournament Service - Elite Tennis Analytics
Gestiona catálogo de torneos y sincronización con API

Soporta SQLite y PostgreSQL.
"""

import logging
import os
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class TournamentService:
    """Servicio para gestión de torneos"""
    
    def __init__(self, db_connection, api_client):
        """
        Args:
            db_connection: Conexión a la base de datos (sqlite3.Connection o MatchDatabase)
            api_client: Cliente de API-Tennis
        """
        self.conn = db_connection
        self.api_client = api_client
        
        # Detectar si es PostgreSQL
        self.is_postgres = os.getenv("DATABASE_URL") is not None
        
        # Si es MatchDatabase, usar su engine
        if hasattr(db_connection, 'is_postgres'):
            self.is_postgres = db_connection.is_postgres
            self.db = db_connection
        else:
            self.db = None
            
        logger.info(f"✅ TournamentService initialized (PostgreSQL: {self.is_postgres})")
    
    def sync_tournaments(self) -> int:
        """
        Sincroniza catálogo completo de torneos desde API
        
        Returns:
            Número de torneos sincronizados
        """
        try:
            # Obtener torneos de la API (get_tournaments ya filtra solo ATP en api_tennis_client)
            tournaments = self.api_client.get_tournaments()
            if not tournaments:
                logger.warning("No se obtuvieron torneos de la API")
                return 0
            count = 0
            
            for tournament in tournaments:
                params = {
                    "tournament_key": tournament.get('tournament_key'),
                    "tournament_name": tournament.get('tournament_name'),
                    "event_type_key": tournament.get('event_type_key'),
                    "event_type_type": tournament.get('event_type_type')
                }
                
                if self.db:
                    # PostgreSQL - usar ON CONFLICT
                    self.db._execute("""
                        INSERT INTO tournaments (
                            tournament_key, tournament_name, 
                            event_type_key, event_type_type
                        )
                        VALUES (:tournament_key, :tournament_name, 
                                :event_type_key, :event_type_type)
                        ON CONFLICT (tournament_key) DO UPDATE SET
                            tournament_name = EXCLUDED.tournament_name,
                            event_type_key = EXCLUDED.event_type_key,
                            event_type_type = EXCLUDED.event_type_type
                    """, params)
                else:
                    # SQLite
                    cursor = self.conn.cursor()
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
                    self.conn.commit()
                count += 1
            
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
        if self.db:
            return self.db._fetchone(
                "SELECT * FROM tournaments WHERE tournament_key = :key",
                {"key": tournament_key}
            )
        else:
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
        if self.db:
            if season:
                return self.db._fetchall("""
                    SELECT * FROM matches 
                    WHERE tournament_key = :key AND tournament_season = :season
                    ORDER BY fecha_partido DESC
                """, {"key": str(tournament_key), "season": str(season)})
            else:
                return self.db._fetchall("""
                    SELECT * FROM matches 
                    WHERE tournament_key = :key
                    ORDER BY fecha_partido DESC
                """, {"key": str(tournament_key)})
        else:
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
        if self.db:
            if event_type:
                return self.db._fetchall("""
                    SELECT * FROM tournaments 
                    WHERE event_type_type = :event_type
                    ORDER BY tournament_name
                """, {"event_type": event_type})
            else:
                return self.db._fetchall("""
                    SELECT * FROM tournaments 
                    ORDER BY tournament_name
                """, {})
        else:
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
    from src.services.api_tennis_client import APITennisClient
    from src.database.match_database import MatchDatabase
    
    # Usar MatchDatabase que detecta automáticamente PostgreSQL o SQLite
    db = MatchDatabase("matches_v2.db")
    api_client = APITennisClient()
    
    service = TournamentService(db, api_client)
    
    # Sincronizar torneos
    count = service.sync_tournaments()
    print(f"Torneos sincronizados: {count}")
    print(f"Using PostgreSQL: {service.is_postgres}")
