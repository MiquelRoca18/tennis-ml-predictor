"""
API-Tennis Client
=================

Cliente para api-tennis.com que proporciona datos de tenis ATP/WTA
con cuotas en tiempo real.

Documentación: https://api-tennis.com/documentation
"""

import requests
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class APITennisClient:
    """
    Cliente para API-Tennis (api-tennis.com)
    
    Proporciona:
    - Partidos ATP/WTA próximos (fixtures)
    - Cuotas en tiempo real (odds)
    - Resultados y estadísticas
    """
    
    def __init__(self):
        """Inicializa el cliente"""
        self.api_key = os.getenv("API_TENNIS_API_KEY", "")
        if not self.api_key:
            logger.warning("⚠️  API_TENNIS_API_KEY no configurada")
        
        # URL base correcta según documentación
        self.base_url = "https://api.api-tennis.com/tennis/"
        
        # Rate limit tracking
        self.requests_made = 0
        
        logger.info("✅ APITennisClient inicializado")
    
    def _make_request(self, method: str, params: Dict = None) -> Optional[Dict]:
        """
        Hace una petición a la API
        
        Args:
            method: Método de la API (get_fixtures, get_odds, etc.)
            params: Parámetros adicionales
        
        Returns:
            Respuesta JSON o None si hay error
        """
        try:
            # Construir parámetros
            request_params = {
                "method": method,
                "APIkey": self.api_key
            }
            
            if params:
                request_params.update(params)
            
            response = requests.get(
                self.base_url,
                params=request_params,
                timeout=10
            )
            
            # Actualizar contador de requests
            self.requests_made += 1
            logger.info(f"📊 Requests hechos: {self.requests_made}")
            
            # Manejar errores HTTP
            response.raise_for_status()
            
            data = response.json()
            
            # Verificar si la API devolvió success
            if data.get('success') != 1:
                logger.error(f"❌ API devolvió error: {data}")
                return None
            
            return data
        
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error en petición a API-Tennis: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Error procesando respuesta: {e}")
            return None
    
    def get_upcoming_matches(self, days_ahead: int = 7) -> List[Dict]:
        """
        Obtiene partidos próximos de tenis (fixtures)
        
        Args:
            days_ahead: Días hacia adelante (default: 7)
        
        Returns:
            Lista de partidos
        """
        if not self.api_key:
            logger.error("❌ API_TENNIS_API_KEY no configurada")
            return []
        
        try:
            logger.info(f"📥 Consultando partidos próximos de tenis...")
            
            # Calcular rango de fechas
            today = datetime.now()
            date_to = (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
            
            params = {
                "date_start": today.strftime('%Y-%m-%d'),
                "date_stop": date_to
            }
            
            data = self._make_request("get_fixtures", params)
            
            if not data:
                logger.warning("⚠️  No se pudo obtener datos de la API")
                return []
            
            matches = data.get('result', [])
            logger.info(f"✅ {len(matches)} partidos encontrados")
            
            return matches
        
        except Exception as e:
            logger.error(f"❌ Error obteniendo partidos: {e}")
            return []
    
    def get_match_odds(self, match_key: str) -> Optional[Dict]:
        """
        Obtiene cuotas para un partido específico
        
        Args:
            match_key: ID del partido (event_key)
        
        Returns:
            Dict con cuotas o None
        """
        if not self.api_key:
            logger.error("❌ API_TENNIS_API_KEY no configurada")
            return None
        
        try:
            logger.info(f"📊 Consultando cuotas para partido {match_key}...")
            
            params = {
                "match_key": match_key
            }
            
            data = self._make_request("get_odds", params)
            
            if not data:
                logger.warning(f"⚠️  No hay cuotas disponibles para partido {match_key}")
                return None
            
            return data.get('result', {})
        
        except Exception as e:
            logger.error(f"❌ Error obteniendo cuotas: {e}")
            return None
    
    def extract_match_info(self, match: Dict) -> Optional[Dict]:
        """
        Extrae información relevante de un partido
        
        Args:
            match: Datos del partido de API-Tennis
        
        Returns:
            Dict con información procesada
        """
        try:
            # Extraer información básica del fixture
            match_info = {
                "match_id": match.get('event_key'),
                "date": match.get('event_date'),
                "time": match.get('event_time'),
                "tournament": match.get('tournament_name', 'Unknown'),
                "round": match.get('tournament_round', 'Unknown'),
                "surface": None,  # API-Tennis no proporciona superficie directamente
                "player1_name": match.get('event_first_player', 'Unknown'),
                "player2_name": match.get('event_second_player', 'Unknown'),
                "player1_key": match.get('first_player_key'),
                "player2_key": match.get('second_player_key'),
                "status": match.get('event_status', 'upcoming'),
                "event_type": match.get('event_type_type', 'Unknown')
            }
            
            return match_info
        
        except Exception as e:
            logger.error(f"❌ Error extrayendo info del partido: {e}")
            return None
    
    def extract_best_odds(self, odds_data: Dict, match_key: str) -> Optional[Dict]:
        """
        Extrae las mejores cuotas de los bookmakers
        
        Args:
            odds_data: Datos de cuotas de la API
            match_key: ID del partido
        
        Returns:
            Dict con mejores cuotas o None
        """
        try:
            # Las cuotas vienen en formato: {match_key: {"Home/Away": {"Home": {...}, "Away": {...}}}}
            match_odds = odds_data.get(str(match_key), {})
            home_away = match_odds.get('Home/Away', {})
            
            if not home_away:
                return None
            
            # Obtener cuotas de Home (Player 1) y Away (Player 2)
            home_odds = home_away.get('Home', {})
            away_odds = home_away.get('Away', {})
            
            if not home_odds or not away_odds:
                return None
            
            # Encontrar la mejor cuota (más alta) para cada jugador
            best_player1_odds = max([float(v) for v in home_odds.values()]) if home_odds else None
            best_player2_odds = max([float(v) for v in away_odds.values()]) if away_odds else None
            
            # Encontrar qué bookmaker tiene las mejores cuotas
            bookmaker_p1 = max(home_odds.items(), key=lambda x: float(x[1]))[0] if home_odds else None
            bookmaker_p2 = max(away_odds.items(), key=lambda x: float(x[1]))[0] if away_odds else None
            
            return {
                "player1_odds": best_player1_odds,
                "player2_odds": best_player2_odds,
                "bookmaker_p1": bookmaker_p1,
                "bookmaker_p2": bookmaker_p2
            }
        
        except Exception as e:
            logger.error(f"❌ Error extrayendo cuotas: {e}")
            return None
    
    def get_all_matches_with_odds(self, days_ahead: int = 7) -> List[Dict]:
        """
        Obtiene todos los partidos próximos con sus cuotas
        
        Args:
            days_ahead: Días hacia adelante
        
        Returns:
            Lista de partidos con cuotas procesadas
        """
        matches = self.get_upcoming_matches(days_ahead)
        
        if not matches:
            logger.info("ℹ️  No hay partidos disponibles")
            return []
        
        matches_with_odds = []
        
        for match in matches:
            # Solo procesar partidos que no hayan empezado
            if match.get('event_status') not in ['', 'upcoming', None]:
                continue
            
            match_info = self.extract_match_info(match)
            if not match_info:
                continue
            
            # Obtener cuotas para este partido
            match_key = match_info['match_id']
            odds_data = self.get_match_odds(match_key)
            
            if odds_data:
                best_odds = self.extract_best_odds(odds_data, match_key)
                if best_odds:
                    match_info.update(best_odds)
                    matches_with_odds.append(match_info)
        
        logger.info(f"✅ {len(matches_with_odds)} partidos con cuotas válidas")
        return matches_with_odds
    
    def get_rate_limit_status(self) -> Dict:
        """
        Obtiene el estado actual del rate limit
        
        Returns:
            Dict con información del rate limit
        """
        return {
            "requests_made": self.requests_made
        }


# ============================================================
# EJEMPLO DE USO
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    client = APITennisClient()
    
    # Obtener partidos con cuotas
    matches = client.get_all_matches_with_odds(days_ahead=3)
    
    print(f"\n🎾 Partidos encontrados: {len(matches)}\n")
    
    for match in matches[:5]:  # Mostrar primeros 5
        print(f"📅 {match.get('date', 'N/A')} {match.get('time', 'N/A')}")
        print(f"🏆 {match.get('tournament', 'N/A')} - {match.get('round', 'N/A')}")
        print(f"👤 {match['player1_name']} - {match.get('player1_odds', 'N/A')}")
        print(f"   vs")
        print(f"👤 {match['player2_name']} - {match.get('player2_odds', 'N/A')}")
        print(f"📊 Bookmakers: {match.get('bookmaker_p1', 'N/A')} / {match.get('bookmaker_p2', 'N/A')}")
        print("-" * 50)
    
    # Mostrar rate limit
    rate_limit = client.get_rate_limit_status()
    print(f"\n📊 Rate Limit Status:")
    print(f"   Requests hechos: {rate_limit['requests_made']}")

    """
    Cliente para API-Tennis (api-tennis.com)
    
    Proporciona:
    - Partidos ATP/WTA próximos
    - Cuotas en tiempo real
    - Resultados y estadísticas
    """
    
    def __init__(self):
        """Inicializa el cliente"""
        self.api_key = os.getenv("API_TENNIS_API_KEY", "")
        if not self.api_key:
            logger.warning("⚠️  API_TENNIS_API_KEY no configurada")
        
        self.base_url = "https://api-tennis.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Rate limit tracking
        self.requests_made = 0
        self.requests_remaining = None
        
        logger.info("✅ APITennisClient inicializado")
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """
        Hace una petición a la API
        
        Args:
            endpoint: Endpoint de la API
            params: Parámetros de la petición
        
        Returns:
            Respuesta JSON o None si hay error
        """
        try:
            url = f"{self.base_url}/{endpoint}"
            response = requests.get(
                url,
                headers=self.headers,
                params=params,
                timeout=10
            )
            
            # Actualizar contador de requests
            self.requests_made += 1
            
            # Verificar rate limit en headers si están disponibles
            if 'X-RateLimit-Remaining' in response.headers:
                self.requests_remaining = int(response.headers['X-RateLimit-Remaining'])
                logger.info(f"📊 Requests restantes: {self.requests_remaining}")
                
                if self.requests_remaining < 10:
                    logger.warning(f"⚠️  ADVERTENCIA: Solo quedan {self.requests_remaining} requests")
            
            # Manejar errores
            if response.status_code == 401:
                logger.error("❌ Error 401: API key inválida o no autorizada")
                return None
            
            if response.status_code == 429:
                logger.error("❌ Error 429: Rate limit excedido")
                return None
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error en petición a API-Tennis: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Error procesando respuesta: {e}")
            return None
    
    def get_upcoming_matches(self, days_ahead: int = 7) -> List[Dict]:
        """
        Obtiene partidos próximos de tenis
        
        Args:
            days_ahead: Días hacia adelante (default: 7)
        
        Returns:
            Lista de partidos
        """
        if not self.api_key:
            logger.error("❌ API_TENNIS_API_KEY no configurada")
            return []
        
        try:
            logger.info(f"📥 Consultando partidos próximos de tenis...")
            
            # Calcular rango de fechas
            today = datetime.now()
            date_to = (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
            
            params = {
                "date_from": today.strftime('%Y-%m-%d'),
                "date_to": date_to,
                "status": "upcoming"
            }
            
            data = self._make_request("matches", params)
            
            if not data:
                logger.warning("⚠️  No se pudo obtener datos de la API")
                return []
            
            matches = data.get('data', [])
            logger.info(f"✅ {len(matches)} partidos encontrados")
            
            return matches
        
        except Exception as e:
            logger.error(f"❌ Error obteniendo partidos: {e}")
            return []
    
    def get_match_odds(self, match_id: str) -> Optional[Dict]:
        """
        Obtiene cuotas para un partido específico
        
        Args:
            match_id: ID del partido
        
        Returns:
            Dict con cuotas o None
        """
        if not self.api_key:
            logger.error("❌ API_TENNIS_API_KEY no configurada")
            return None
        
        try:
            logger.info(f"📊 Consultando cuotas para partido {match_id}...")
            
            data = self._make_request(f"matches/{match_id}/odds")
            
            if not data:
                logger.warning(f"⚠️  No hay cuotas disponibles para partido {match_id}")
                return None
            
            return data.get('data', {})
        
        except Exception as e:
            logger.error(f"❌ Error obteniendo cuotas: {e}")
            return None
    
    def extract_match_info(self, match: Dict) -> Optional[Dict]:
        """
        Extrae información relevante de un partido
        
        Args:
            match: Datos del partido de API-Tennis
        
        Returns:
            Dict con información procesada
        """
        try:
            # Extraer información básica
            match_info = {
                "match_id": match.get('id'),
                "date": match.get('start_time'),
                "tournament": match.get('tournament', {}).get('name', 'Unknown'),
                "round": match.get('round', 'Unknown'),
                "surface": match.get('surface', 'Unknown'),
                "player1_name": match.get('player1', {}).get('name', 'Unknown'),
                "player2_name": match.get('player2', {}).get('name', 'Unknown'),
                "player1_ranking": match.get('player1', {}).get('ranking'),
                "player2_ranking": match.get('player2', {}).get('ranking'),
                "status": match.get('status', 'upcoming')
            }
            
            # Extraer cuotas si están disponibles
            odds = match.get('odds', {})
            if odds:
                match_info['player1_odds'] = odds.get('player1')
                match_info['player2_odds'] = odds.get('player2')
                match_info['bookmaker'] = odds.get('bookmaker', 'Unknown')
            
            return match_info
        
        except Exception as e:
            logger.error(f"❌ Error extrayendo info del partido: {e}")
            return None
    
    def get_all_matches_with_odds(self, days_ahead: int = 7) -> List[Dict]:
        """
        Obtiene todos los partidos próximos con sus cuotas
        
        Args:
            days_ahead: Días hacia adelante
        
        Returns:
            Lista de partidos con cuotas procesadas
        """
        matches = self.get_upcoming_matches(days_ahead)
        
        if not matches:
            logger.info("ℹ️  No hay partidos disponibles")
            return []
        
        matches_with_odds = []
        
        for match in matches:
            match_info = self.extract_match_info(match)
            if not match_info:
                continue
            
            # Si el partido no tiene cuotas en la respuesta inicial,
            # intentar obtenerlas por separado
            if 'player1_odds' not in match_info:
                odds = self.get_match_odds(match_info['match_id'])
                if odds:
                    match_info['player1_odds'] = odds.get('player1')
                    match_info['player2_odds'] = odds.get('player2')
                    match_info['bookmaker'] = odds.get('bookmaker', 'Unknown')
            
            # Solo añadir si tiene cuotas
            if 'player1_odds' in match_info and match_info['player1_odds']:
                matches_with_odds.append(match_info)
        
        logger.info(f"✅ {len(matches_with_odds)} partidos con cuotas válidas")
        return matches_with_odds
    
    def get_rate_limit_status(self) -> Dict:
        """
        Obtiene el estado actual del rate limit
        
        Returns:
            Dict con información del rate limit
        """
        return {
            "requests_made": self.requests_made,
            "requests_remaining": self.requests_remaining or "Unknown"
        }


# ============================================================
# EJEMPLO DE USO
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    client = APITennisClient()
    
    # Obtener partidos con cuotas
    matches = client.get_all_matches_with_odds(days_ahead=3)
    
    print(f"\n🎾 Partidos encontrados: {len(matches)}\n")
    
    for match in matches[:5]:  # Mostrar primeros 5
        print(f"📅 {match.get('date', 'N/A')}")
        print(f"🏆 {match.get('tournament', 'N/A')} - {match.get('round', 'N/A')}")
        print(f"🎾 {match.get('surface', 'N/A')}")
        print(f"👤 {match['player1_name']} (#{match.get('player1_ranking', 'N/A')}) - {match.get('player1_odds', 'N/A')}")
        print(f"   vs")
        print(f"👤 {match['player2_name']} (#{match.get('player2_ranking', 'N/A')}) - {match.get('player2_odds', 'N/A')}")
        print(f"📊 Bookmaker: {match.get('bookmaker', 'N/A')}")
        print("-" * 50)
    
    # Mostrar rate limit
    rate_limit = client.get_rate_limit_status()
    print(f"\n📊 Rate Limit Status:")
    print(f"   Requests hechos: {rate_limit['requests_made']}")
    print(f"   Requests restantes: {rate_limit['requests_remaining']}")
