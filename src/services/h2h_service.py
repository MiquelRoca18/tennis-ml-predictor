"""
H2H Service - Elite Tennis Analytics
Gestiona histórico head-to-head entre jugadores
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class H2HService:
    """Servicio para gestión de Head-to-Head"""
    
    def __init__(self, db_connection, api_client):
        """
        Args:
            db_connection: Conexión a la base de datos
            api_client: Cliente de API-Tennis
        """
        self.conn = db_connection
        self.api_client = api_client
        logger.info("✅ H2HService initialized")
    
    def get_h2h(self, player1_key: int, player2_key: int) -> Dict:
        """
        Obtiene histórico completo entre dos jugadores
        
        Args:
            player1_key: ID del jugador 1
            player2_key: ID del jugador 2
            
        Returns:
            Dict con datos H2H completos
        """
        cursor = self.conn.cursor()
        
        # Obtener partidos históricos
        matches = cursor.execute("""
            SELECT * FROM head_to_head
            WHERE (player1_key = ? AND player2_key = ?)
               OR (player1_key = ? AND player2_key = ?)
            ORDER BY match_date DESC
        """, (player1_key, player2_key, player2_key, player1_key)).fetchall()
        
        # Calcular estadísticas
        total_matches = len(matches)
        player1_wins = sum(1 for m in matches if m['winner_key'] == player1_key)
        player2_wins = sum(1 for m in matches if m['winner_key'] == player2_key)
        
        # Por superficie
        surfaces = {}
        for match in matches:
            surface = match['surface']
            if surface not in surfaces:
                surfaces[surface] = {'total': 0, 'player1_wins': 0, 'player2_wins': 0}
            
            surfaces[surface]['total'] += 1
            if match['winner_key'] == player1_key:
                surfaces[surface]['player1_wins'] += 1
            else:
                surfaces[surface]['player2_wins'] += 1
        
        return {
            'player1_key': player1_key,
            'player2_key': player2_key,
            'total_matches': total_matches,
            'player1_wins': player1_wins,
            'player2_wins': player2_wins,
            'last_matches': [dict(m) for m in matches[:10]],
            'by_surface': surfaces
        }
    
    def sync_h2h_from_api(self, player1_key: int, player2_key: int) -> int:
        """
        Sincroniza H2H desde API y guarda en DB (solo ATP Singles)
        
        Args:
            player1_key: ID del jugador 1
            player2_key: ID del jugador 2
            
        Returns:
            Número de partidos sincronizados
        """
        try:
            logger.info(f"🔄 Sincronizando H2H: {player1_key} vs {player2_key}...")
            
            # Usar nuevo método get_h2h() (ya filtra ATP Singles)
            result = self.api_client.get_h2h(str(player1_key), str(player2_key))
            
            if not result or not result.get("H2H"):
                logger.warning(f"No se obtuvieron datos H2H de la API")
                return 0
            
            h2h_matches = result.get("H2H", [])
            
            cursor = self.conn.cursor()
            count = 0
            
            for match in h2h_matches:
                # Determinar ganador
                winner_key = player1_key if match.get('event_winner') == 'First Player' else player2_key
                
                # Guardar en tabla head_to_head
                cursor.execute("""
                    INSERT OR IGNORE INTO head_to_head (
                        player1_key, player2_key, match_id,
                        match_date, winner_key, tournament_name,
                        surface, final_result
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    player1_key,
                    player2_key,
                    match.get('event_key'),
                    match.get('event_date'),
                    winner_key,
                    match.get('tournament_name'),
                    'Hard',  # La API no siempre devuelve superficie en H2H
                    match.get('event_final_result')
                ))
                count += 1
            
            self.conn.commit()
            logger.info(f"✅ Sincronizados {count} partidos H2H ATP Singles")
            return count
            
        except Exception as e:
            logger.error(f"Error sincronizando H2H: {e}")
            return 0
    
    def get_recent_form(self, player_key: int, last_n: int = 10) -> List[Dict]:
        """
        Obtiene forma reciente de un jugador
        
        Args:
            player_key: ID del jugador
            last_n: Número de partidos
            
        Returns:
            Lista de últimos partidos con resultado
        """
        cursor = self.conn.cursor()
        
        matches = cursor.execute("""
            SELECT 
                m.id, m.fecha_partido, m.torneo, m.superficie,
                m.jugador1_nombre, m.jugador2_nombre,
                m.resultado_ganador, m.resultado_marcador,
                m.first_player_key, m.second_player_key
            FROM matches m
            WHERE (m.first_player_key = ? OR m.second_player_key = ?)
            AND m.estado = 'completado'
            AND m.resultado_ganador IS NOT NULL
            ORDER BY m.fecha_partido DESC
            LIMIT ?
        """, (player_key, player_key, last_n)).fetchall()
        
        form = []
        for match in matches:
            match_dict = dict(match)
            
            # Determinar si ganó
            is_player1 = match['first_player_key'] == player_key
            opponent = match['jugador2_nombre'] if is_player1 else match['jugador1_nombre']
            won = match['resultado_ganador'] == (match['jugador1_nombre'] if is_player1 else match['jugador2_nombre'])
            
            form.append({
                'date': match['fecha_partido'],
                'tournament': match['torneo'],
                'surface': match['superficie'],
                'opponent': opponent,
                'result': 'W' if won else 'L',
                'score': match['resultado_marcador']
            })
        
        return form
    
    def calculate_win_rate(self, player_key: int, surface: str = None, 
                          last_n_months: int = 12) -> float:
        """
        Calcula win rate de un jugador
        
        Args:
            player_key: ID del jugador
            surface: Superficie específica (opcional)
            last_n_months: Últimos N meses
            
        Returns:
            Win rate (0.0 - 1.0)
        """
        cursor = self.conn.cursor()
        
        # Query base
        query = """
            SELECT COUNT(*) as total,
                   SUM(CASE 
                       WHEN (first_player_key = ? AND resultado_ganador = jugador1_nombre)
                         OR (second_player_key = ? AND resultado_ganador = jugador2_nombre)
                       THEN 1 ELSE 0 
                   END) as wins
            FROM matches
            WHERE (first_player_key = ? OR second_player_key = ?)
            AND estado = 'completado'
            AND resultado_ganador IS NOT NULL
            AND fecha_partido >= date('now', '-{} months')
        """.format(last_n_months)
        
        params = [player_key, player_key, player_key, player_key]
        
        if surface:
            query += " AND superficie = ?"
            params.append(surface)
        
        result = cursor.execute(query, params).fetchone()
        
        if result and result['total'] > 0:
            return result['wins'] / result['total']
        
        return 0.0


if __name__ == "__main__":
    # Test básico
    import sqlite3
    from src.services.api_tennis_client import APITennisClient
    
    conn = sqlite3.connect("matches_v2.db")
    conn.row_factory = sqlite3.Row
    api_client = APITennisClient()
    
    service = H2HService(conn, api_client)
    
    # Test con jugadores de ejemplo
    print("H2HService test completed")
    
    conn.close()
