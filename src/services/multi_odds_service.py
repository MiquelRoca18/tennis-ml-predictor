"""
Enhanced Odds Service - Elite Tennis Analytics
Gestiona cuotas de múltiples casas de apuestas
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class MultiBookmakerOddsService:
    """Servicio para gestión de cuotas multi-bookmaker"""
    
    def __init__(self, db_connection, api_client):
        """
        Args:
            db_connection: Conexión a la base de datos
            api_client: Cliente de API-Tennis
        """
        self.conn = db_connection
        self.api_client = api_client
        logger.info("✅ MultiBookmakerOddsService initialized")
    
    def store_match_odds(self, match_id: int, odds_data: List[Dict]) -> int:
        """
        Guarda cuotas de múltiples bookmakers para un partido
        
        Args:
            match_id: ID del partido
            odds_data: Lista de cuotas por bookmaker
            
        Returns:
            Número de cuotas guardadas
        """
        cursor = self.conn.cursor()
        count = 0
        
        for odd in odds_data:
            try:
                cursor.execute("""
                    INSERT INTO match_odds (
                        match_id, bookmaker, market_type, 
                        selection, odds, timestamp
                    )
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    match_id,
                    odd.get('bookmaker', 'Unknown'),
                    odd.get('market_type', 'Match Winner'),
                    odd.get('selection'),
                    odd.get('odds')
                ))
                count += 1
            except Exception as e:
                logger.error(f"Error guardando cuota: {e}")
        
        self.conn.commit()
        logger.debug(f"✅ Guardadas {count} cuotas para match {match_id}")
        
        return count
    
    def get_match_odds(self, match_id: int, market_type: str = None) -> List[Dict]:
        """
        Obtiene cuotas de un partido
        
        Args:
            match_id: ID del partido
            market_type: Tipo de mercado (opcional)
            
        Returns:
            Lista de cuotas
        """
        cursor = self.conn.cursor()
        
        if market_type:
            odds = cursor.execute("""
                SELECT * FROM match_odds
                WHERE match_id = ? AND market_type = ?
                ORDER BY timestamp DESC
            """, (match_id, market_type)).fetchall()
        else:
            odds = cursor.execute("""
                SELECT * FROM match_odds
                WHERE match_id = ?
                ORDER BY timestamp DESC
            """, (match_id,)).fetchall()
        
        return [dict(o) for o in odds]
    
    def get_best_odds(self, match_id: int, market_type: str = 'Match Winner') -> Dict:
        """
        Obtiene las mejores cuotas para un partido
        
        Args:
            match_id: ID del partido
            market_type: Tipo de mercado
            
        Returns:
            Dict con mejores cuotas por selección
        """
        cursor = self.conn.cursor()
        
        # Obtener todas las cuotas del mercado
        odds = cursor.execute("""
            SELECT bookmaker, selection, odds
            FROM match_odds
            WHERE match_id = ? AND market_type = ?
            ORDER BY odds DESC
        """, (match_id, market_type)).fetchall()
        
        # Agrupar por selección y obtener la mejor
        best_odds = {}
        for bookmaker, selection, odd_value in odds:
            if selection not in best_odds or odd_value > best_odds[selection]['odds']:
                best_odds[selection] = {
                    'bookmaker': bookmaker,
                    'odds': odd_value,
                    'selection': selection
                }
        
        return best_odds
    
    def sync_odds_from_api(self, match_id: int, event_key: str) -> int:
        """
        Sincroniza cuotas desde API-Tennis
        
        Args:
            match_id: ID del partido en DB
            event_key: Key del evento en API
            
        Returns:
            Número de cuotas sincronizadas
        """
        try:
            # Obtener cuotas de la API
            data = self.api_client._make_request("get_odds", {"event_key": event_key})
            
            if not data or not data.get("result"):
                return 0
            
            odds_list = []
            result = data["result"]
            
            # Parsear cuotas de diferentes bookmakers
            # La API devuelve cuotas en formato específico
            if isinstance(result, dict):
                # Extraer cuotas de diferentes mercados
                for bookmaker, markets in result.items():
                    if isinstance(markets, dict):
                        for market, selections in markets.items():
                            if isinstance(selections, dict):
                                for selection, odd_value in selections.items():
                                    odds_list.append({
                                        'bookmaker': bookmaker,
                                        'market_type': market,
                                        'selection': selection,
                                        'odds': float(odd_value)
                                    })
            
            # Guardar en DB
            if odds_list:
                return self.store_match_odds(match_id, odds_list)
            
            return 0
            
        except Exception as e:
            logger.error(f"Error sincronizando cuotas: {e}")
            return 0
    
    def get_odds_comparison(self, match_id: int) -> Dict:
        """
        Obtiene comparación de cuotas entre bookmakers
        
        Args:
            match_id: ID del partido
            
        Returns:
            Dict con comparación de cuotas
        """
        cursor = self.conn.cursor()
        
        # Obtener todas las cuotas
        odds = cursor.execute("""
            SELECT bookmaker, market_type, selection, odds
            FROM match_odds
            WHERE match_id = ?
            ORDER BY market_type, selection, odds DESC
        """, (match_id,)).fetchall()
        
        # Organizar por mercado y selección
        comparison = {}
        for bookmaker, market, selection, odd_value in odds:
            if market not in comparison:
                comparison[market] = {}
            if selection not in comparison[market]:
                comparison[market][selection] = []
            
            comparison[market][selection].append({
                'bookmaker': bookmaker,
                'odds': odd_value
            })
        
        return comparison
    
    def sync_all_pending_matches_odds(self) -> Dict:
        """
        Sincroniza cuotas de todos los partidos pendientes desde la API
        
        Este método:
        1. Obtiene todos los partidos pendientes con event_key
        2. Hace una llamada batch a la API para obtener todas las cuotas
        3. Parsea y guarda las cuotas de cada bookmaker en la DB
        
        Returns:
            Dict con estadísticas de la sincronización
        """
        try:
            from datetime import datetime, timedelta
            
            # 1. Obtener partidos pendientes con event_key
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT id, event_key, jugador1_nombre, jugador2_nombre, fecha_partido
                FROM matches
                WHERE estado = 'pendiente'
                AND event_key IS NOT NULL
                AND fecha_partido >= DATE('now')
                ORDER BY fecha_partido ASC
            """)
            
            pending_matches = [dict(row) for row in cursor.fetchall()]
            
            if not pending_matches:
                logger.info("ℹ️  No hay partidos pendientes para sincronizar cuotas")
                return {
                    "success": True,
                    "matches_found": 0,
                    "odds_synced": 0,
                    "message": "No hay partidos pendientes"
                }
            
            logger.info(f"📊 Sincronizando cuotas para {len(pending_matches)} partidos pendientes...")
            
            # 2. Obtener cuotas batch desde la API
            today = datetime.now()
            date_start = today.strftime("%Y-%m-%d")
            date_stop = (today + timedelta(days=7)).strftime("%Y-%m-%d")
            
            all_odds = self.api_client.get_all_odds_batch(date_start, date_stop)
            
            if not all_odds:
                logger.warning("⚠️  No se pudieron obtener cuotas desde la API")
                return {
                    "success": False,
                    "matches_found": len(pending_matches),
                    "odds_synced": 0,
                    "message": "Error obteniendo cuotas desde la API"
                }
            
            # 3. Procesar y guardar cuotas para cada partido
            total_odds_stored = 0
            matches_with_odds = 0
            
            for match in pending_matches:
                match_id = match['id']
                event_key = str(match['event_key'])
                
                # Buscar cuotas de este partido en el batch
                if event_key not in all_odds:
                    logger.debug(f"⚠️  No hay cuotas para partido {match_id} (event_key: {event_key})")
                    continue
                
                match_odds_data = all_odds[event_key]
                
                # Parsear cuotas por mercado y bookmaker
                # Formato API: {event_key: {"Home/Away": {"Home": {bookmaker: odds}, "Away": {bookmaker: odds}}}}
                odds_list = []
                
                # Procesar mercado "Home/Away" (Match Winner)
                home_away = match_odds_data.get("Home/Away", {})
                if home_away:
                    # Cuotas para jugador 1 (Home)
                    home_odds = home_away.get("Home", {})
                    for bookmaker, odd_value in home_odds.items():
                        try:
                            odds_list.append({
                                'bookmaker': bookmaker,
                                'market_type': 'Match Winner',
                                'selection': match['jugador1_nombre'],
                                'odds': float(odd_value)
                            })
                        except (ValueError, TypeError):
                            logger.warning(f"⚠️  Cuota inválida: {bookmaker} = {odd_value}")
                    
                    # Cuotas para jugador 2 (Away)
                    away_odds = home_away.get("Away", {})
                    for bookmaker, odd_value in away_odds.items():
                        try:
                            odds_list.append({
                                'bookmaker': bookmaker,
                                'market_type': 'Match Winner',
                                'selection': match['jugador2_nombre'],
                                'odds': float(odd_value)
                            })
                        except (ValueError, TypeError):
                            logger.warning(f"⚠️  Cuota inválida: {bookmaker} = {odd_value}")
                
                # Guardar cuotas en la DB
                if odds_list:
                    # Primero, eliminar cuotas antiguas de este partido
                    cursor.execute("DELETE FROM match_odds WHERE match_id = ?", (match_id,))
                    
                    # Guardar nuevas cuotas
                    count = self.store_match_odds(match_id, odds_list)
                    total_odds_stored += count
                    matches_with_odds += 1
                    
                    logger.debug(f"✅ {count} cuotas guardadas para partido {match_id}")
            
            logger.info(
                f"✅ Sincronización completada: {total_odds_stored} cuotas guardadas "
                f"para {matches_with_odds}/{len(pending_matches)} partidos"
            )
            
            return {
                "success": True,
                "matches_found": len(pending_matches),
                "matches_with_odds": matches_with_odds,
                "odds_synced": total_odds_stored,
                "message": f"{total_odds_stored} cuotas sincronizadas para {matches_with_odds} partidos"
            }
            
        except Exception as e:
            logger.error(f"❌ Error sincronizando cuotas: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "message": "Error en sincronización de cuotas"
            }


if __name__ == "__main__":
    # Test básico
    from src.services.api_tennis_client import APITennisClient
    from src.database.match_database import MatchDatabase
    
    # Usar MatchDatabase que detecta automáticamente PostgreSQL o SQLite
    db = MatchDatabase("matches_v2.db")
    api_client = APITennisClient()
    
    service = MultiBookmakerOddsService(db, api_client)
    print("✅ MultiBookmakerOddsService test completed")
    print(f"Using PostgreSQL: {db.is_postgres}")
