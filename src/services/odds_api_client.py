"""
Cliente para The Odds API
=========================

Maneja todas las interacciones con The Odds API para obtener
cuotas de partidos de tenis ATP.

Documentación: https://the-odds-api.com/liveapi/guides/v4/
"""

import requests
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

logger = logging.getLogger(__name__)


class OddsAPIClient:
    """
    Cliente para interactuar con The Odds API
    """
    
    def __init__(self, api_key: str = None):
        """
        Inicializa el cliente
        
        Args:
            api_key: API key de The Odds API (si no se provee, se lee de .env)
        """
        self.api_key = api_key or os.getenv("ODDS_API_KEY")
        if not self.api_key:
            raise ValueError("ODDS_API_KEY no encontrada. Añádela al archivo .env")
        
        self.base_url = "https://api.the-odds-api.com/v4"
        self.sport = "tennis_atp"  # Tennis ATP
        
        logger.info("✅ OddsAPIClient inicializado")
    
    def get_upcoming_matches(self, days_ahead: int = 7) -> List[Dict]:
        """
        Obtiene próximos partidos de tenis ATP con cuotas disponibles
        
        Args:
            days_ahead: Días hacia adelante a buscar (default: 7)
        
        Returns:
            Lista de partidos con cuotas
        
        Example response:
        [
            {
                "id": "abc123",
                "sport_key": "tennis_atp",
                "commence_time": "2026-01-10T15:00:00Z",
                "home_team": "Novak Djokovic",
                "away_team": "Rafael Nadal",
                "bookmakers": [
                    {
                        "key": "betfair",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Novak Djokovic", "price": 1.85},
                                    {"name": "Rafael Nadal", "price": 2.10}
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
        """
        try:
            endpoint = f"{self.base_url}/sports/{self.sport}/odds"
            
            params = {
                "apiKey": self.api_key,
                "regions": "eu",  # Cuotas europeas
                "markets": "h2h",  # Head to head (ganador del partido)
                "oddsFormat": "decimal",
                "dateFormat": "iso"
            }
            
            logger.info(f"📥 Consultando partidos próximos de {self.sport}...")
            
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            
            matches = response.json()
            
            # Filtrar por fecha (próximos N días)
            fecha_limite = datetime.now() + timedelta(days=days_ahead)
            matches_filtrados = []
            
            for match in matches:
                commence_time = datetime.fromisoformat(match['commence_time'].replace('Z', '+00:00'))
                if commence_time <= fecha_limite:
                    matches_filtrados.append(match)
            
            logger.info(f"✅ Encontrados {len(matches_filtrados)} partidos en próximos {days_ahead} días")
            
            # Log de requests restantes
            remaining = response.headers.get('x-requests-remaining')
            if remaining:
                logger.info(f"📊 Requests restantes este mes: {remaining}")
            
            return matches_filtrados
        
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error consultando The Odds API: {e}")
            return []
        except Exception as e:
            logger.error(f"❌ Error procesando respuesta: {e}")
            return []
    
    def extract_best_odds(self, match: Dict) -> Optional[Dict]:
        """
        Extrae las mejores cuotas de un partido
        
        Args:
            match: Partido de The Odds API
        
        Returns:
            Dict con mejores cuotas o None si no hay cuotas
        
        Example:
        {
            "match_id": "abc123",
            "commence_time": "2026-01-10T15:00:00Z",
            "player1_name": "Novak Djokovic",
            "player2_name": "Rafael Nadal",
            "player1_odds": 1.85,
            "player2_odds": 2.10,
            "bookmaker": "betfair"
        }
        """
        try:
            if not match.get('bookmakers'):
                logger.warning(f"⚠️  No hay bookmakers para {match.get('home_team')} vs {match.get('away_team')}")
                return None
            
            # Buscar mejores cuotas entre todos los bookmakers
            mejor_cuota_p1 = 0
            mejor_cuota_p2 = 0
            mejor_bookmaker = None
            
            for bookmaker in match['bookmakers']:
                for market in bookmaker.get('markets', []):
                    if market['key'] == 'h2h':
                        outcomes = market.get('outcomes', [])
                        if len(outcomes) >= 2:
                            # Encontrar cuotas para cada jugador
                            for outcome in outcomes:
                                if outcome['name'] == match['home_team']:
                                    if outcome['price'] > mejor_cuota_p1:
                                        mejor_cuota_p1 = outcome['price']
                                        mejor_bookmaker = bookmaker['key']
                                elif outcome['name'] == match['away_team']:
                                    if outcome['price'] > mejor_cuota_p2:
                                        mejor_cuota_p2 = outcome['price']
            
            if mejor_cuota_p1 > 0 and mejor_cuota_p2 > 0:
                return {
                    "match_id": match['id'],
                    "commence_time": match['commence_time'],
                    "player1_name": match['home_team'],
                    "player2_name": match['away_team'],
                    "player1_odds": mejor_cuota_p1,
                    "player2_odds": mejor_cuota_p2,
                    "bookmaker": mejor_bookmaker
                }
            else:
                logger.warning(f"⚠️  Cuotas incompletas para {match.get('home_team')} vs {match.get('away_team')}")
                return None
        
        except Exception as e:
            logger.error(f"❌ Error extrayendo cuotas: {e}")
            return None
    
    def get_all_matches_with_odds(self, days_ahead: int = 7) -> List[Dict]:
        """
        Obtiene todos los partidos con sus mejores cuotas
        
        Args:
            days_ahead: Días hacia adelante a buscar
        
        Returns:
            Lista de partidos con cuotas procesadas
        """
        matches = self.get_upcoming_matches(days_ahead)
        
        matches_con_cuotas = []
        for match in matches:
            odds_data = self.extract_best_odds(match)
            if odds_data:
                matches_con_cuotas.append(odds_data)
        
        logger.info(f"✅ {len(matches_con_cuotas)} partidos con cuotas válidas")
        return matches_con_cuotas


# ============================================================
# EJEMPLO DE USO
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Crear cliente
    client = OddsAPIClient()
    
    # Obtener partidos próximos
    matches = client.get_all_matches_with_odds(days_ahead=7)
    
    print(f"\n📊 Partidos encontrados: {len(matches)}\n")
    
    for i, match in enumerate(matches, 1):
        print(f"{i}. {match['player1_name']} vs {match['player2_name']}")
        print(f"   Fecha: {match['commence_time']}")
        print(f"   Cuotas: {match['player1_odds']} / {match['player2_odds']}")
        print(f"   Bookmaker: {match['bookmaker']}")
        print()
