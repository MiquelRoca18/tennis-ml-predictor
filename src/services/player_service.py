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
        Obtiene perfil completo de un jugador desde la BD.
        player_key en BD es VARCHAR, por eso se compara como string.
        """
        key_str = str(player_key)
        player = self.db._fetchone(
            "SELECT * FROM players WHERE player_key = :player_key",
            {"player_key": key_str}
        )
        
        if not player:
            return None
        
        player_dict = dict(player)
        player_dict['stats'] = player_dict.get('stats') or []
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
                jugador1_key, jugador2_key,
                resultado_ganador, resultado_marcador, estado
            FROM matches
            WHERE (jugador1_key = :player_key OR jugador2_key = :player_key)
            AND estado = 'completado'
            ORDER BY fecha_partido DESC
            LIMIT :limit
        """,
            {"player_key": str(player_key), "limit": last_n}
        )

        key_str = str(player_key)
        out = []
        for m in matches:
            row = dict(m)
            j1_key = (row.get("jugador1_key") or "")
            j2_key = (row.get("jugador2_key") or "")
            j1_name = (row.get("jugador1_nombre") or "").strip()
            j2_name = (row.get("jugador2_nombre") or "").strip()
            winner = (row.get("resultado_ganador") or "").strip()
            if str(j1_key) == key_str:
                row["opponent_name"] = j2_name
                row["profile_is_jugador1"] = True
                row["is_win"] = self._winner_matches_name(winner, j1_name)
            else:
                row["opponent_name"] = j1_name
                row["profile_is_jugador1"] = False
                row["is_win"] = self._winner_matches_name(winner, j2_name)
            out.append(row)
        return out

    def _winner_matches_name(self, winner_name: str, player_name: str) -> bool:
        """True si winner_name y player_name se consideran la misma persona (coincidencia flexible)."""
        if not winner_name or not player_name:
            return False
        w = winner_name.lower().strip()
        p = player_name.lower().strip()
        if w == p:
            return True
        if w in p or p in w:
            return True
        w_parts = w.split()
        p_parts = p.split()
        if w_parts and p_parts and w_parts[-1] == p_parts[-1]:
            return True
        return False

    def get_player_key_by_name(self, name: str) -> Optional[int]:
        """
        Busca player_key por nombre (para enlace desde la card cuando no viene key en el partido).
        Coincidencia parcial por nombre; devuelve el primero que coincida (priorizando ranking).

        Args:
            name: Nombre del jugador (ej. "Alcaraz", "Rafael Nadal")

        Returns:
            player_key si se encuentra, None si no
        """
        if not name or not name.strip():
            return None
        name_clean = name.strip().replace("%", "").replace("_", "")
        if not name_clean:
            return None
        pattern = f"%{name_clean}%"
        row = self.db._fetchone(
            """
            SELECT player_key FROM players
            WHERE LOWER(player_name) LIKE LOWER(:pattern)
            ORDER BY atp_ranking ASC NULLS LAST
            LIMIT 1
            """,
            {"pattern": pattern},
        )
        if not row:
            return None
        raw = row.get("player_key")
        if raw is None:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None


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
