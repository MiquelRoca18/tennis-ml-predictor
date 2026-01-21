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
            request_params = {"method": method, "APIkey": self.api_key}

            if params:
                request_params.update(params)

            # Reintentos con backoff exponencial
            max_retries = 3
            timeout = 30  # Aumentado de 10s a 30s
            
            for attempt in range(max_retries):
                try:
                    response = requests.get(self.base_url, params=request_params, timeout=timeout)
                    
                    # Actualizar contador de requests
                    self.requests_made += 1
                    logger.debug(f"📊 Requests hechos: {self.requests_made}")

                    # Manejar errores HTTP
                    response.raise_for_status()

                    data = response.json()

                    # Verificar si la API devolvió success
                    if data.get("success") != 1:
                        logger.error(f"❌ API devolvió error: {data}")
                        return None

                    return data
                    
                except requests.exceptions.Timeout:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Backoff: 1s, 2s, 4s
                        logger.warning(f"⚠️  Timeout en intento {attempt + 1}/{max_retries}, reintentando en {wait_time}s...")
                        import time
                        time.sleep(wait_time)
                    else:
                        logger.error(f"❌ Timeout después de {max_retries} intentos")
                        return None

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
            date_to = (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

            params = {"date_start": today.strftime("%Y-%m-%d"), "date_stop": date_to}

            data = self._make_request("get_fixtures", params)

            if not data:
                logger.warning("⚠️  No se pudo obtener datos de la API")
                return []

            matches = data.get("result", [])
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

            params = {"match_key": match_key}

            data = self._make_request("get_odds", params)

            if not data:
                logger.warning(f"⚠️  No hay cuotas disponibles para partido {match_key}")
                return None

            return data.get("result", {})

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
                # IDs para tracking
                "match_id": match.get("event_key"),
                "event_key": match.get("event_key"),
                "player1_key": match.get("first_player_key"),
                "player2_key": match.get("second_player_key"),
                "tournament_key": match.get("tournament_key"),
                
                # Información básica
                "date": match.get("event_date"),
                "time": match.get("event_time"),
                "tournament": match.get("tournament_name", "Unknown"),
                "tournament_season": match.get("tournament_season"),
                "round": match.get("tournament_round", "Unknown"),
                "surface": match.get("surface", "Hard"),
                
                # Información de jugadores
                "player1_name": match.get("event_first_player", "Unknown"),
                "player1_logo": match.get("event_first_player_logo"),
                "player2_name": match.get("event_second_player", "Unknown"),
                "player2_logo": match.get("event_second_player_logo"),
                
                # Estado del partido
                "status": match.get("event_status", ""),
                "event_live": match.get("event_live", "0"),
                "event_qualification": match.get("event_qualification", "False"),
                
                # Resultados (si disponibles)
                "event_final_result": match.get("event_final_result", "-"),
                "event_winner": match.get("event_winner"),
                
                # Cuotas (si disponibles)
                "player1_odds": None,  # Se obtienen con get_match_odds
                "player2_odds": None,
            }
            
            return match_info
            
        except Exception as e:
            logger.error(f"❌ Error extrayendo info del partido: {e}")
            return None

    def get_rankings(self, league: str = "ATP") -> List[Dict]:
        """
        Obtiene rankings ATP oficiales
        
        Args:
            league: "ATP" (WTA no soportado)
            
        Returns:
            Lista de jugadores con ranking, puntos, país
        """
        if league.upper() != "ATP":
            logger.warning(f"⚠️  Solo se soporta ATP, ignorando: {league}")
            return []
            
        if not self.api_key:
            logger.error("❌ API_TENNIS_API_KEY no configurada")
            return []
            
        try:
            logger.info("📊 Consultando rankings ATP...")
            
            params = {"event_type": "ATP"}
            data = self._make_request("get_standings", params)
            
            if not data:
                logger.warning("⚠️  No se obtuvieron rankings ATP de la API")
                return []
                
            rankings = data.get("result", [])
            logger.info(f"✅ {len(rankings)} rankings ATP obtenidos")
            
            return rankings
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo rankings: {e}")
            return []

    def get_h2h(self, player1_key: str, player2_key: str) -> Dict:
        """
        Obtiene historial Head to Head entre 2 jugadores
        
        Args:
            player1_key: ID del jugador 1
            player2_key: ID del jugador 2
            
        Returns:
            Dict con:
            - H2H: Partidos entre ellos
            - firstPlayerResults: Últimos partidos del J1
            - secondPlayerResults: Últimos partidos del J2
        """
        if not self.api_key:
            logger.error("❌ API_TENNIS_API_KEY no configurada")
            return {"H2H": [], "firstPlayerResults": [], "secondPlayerResults": []}
            
        try:
            logger.info(f"📊 Consultando H2H: {player1_key} vs {player2_key}...")
            
            params = {
                "first_player_key": player1_key,
                "second_player_key": player2_key
            }
            
            data = self._make_request("get_H2H", params)
            
            if not data:
                logger.warning(f"⚠️  No hay datos H2H para {player1_key} vs {player2_key}")
                return {"H2H": [], "firstPlayerResults": [], "secondPlayerResults": []}
                
            result = data.get("result", {})
            
            # Filtrar solo ATP Singles en todos los resultados
            if "H2H" in result:
                result["H2H"] = [
                    m for m in result["H2H"]
                    if m.get("event_type_type", "").lower() == "atp singles"
                ]
            
            if "firstPlayerResults" in result:
                result["firstPlayerResults"] = [
                    m for m in result["firstPlayerResults"]
                    if m.get("event_type_type", "").lower() == "atp singles"
                ][:10]  # Últimos 10
                
            if "secondPlayerResults" in result:
                result["secondPlayerResults"] = [
                    m for m in result["secondPlayerResults"]
                    if m.get("event_type_type", "").lower() == "atp singles"
                ][:10]  # Últimos 10
            
            h2h_count = len(result.get("H2H", []))
            logger.info(f"✅ H2H obtenido: {h2h_count} enfrentamientos previos")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo H2H: {e}")
            return {"H2H": [], "firstPlayerResults": [], "secondPlayerResults": []}

    def get_livescore(self) -> List[Dict]:
        """
        Obtiene partidos ATP Singles en vivo
        
        Returns:
            Lista de partidos en vivo con pointbypoint
        """
        if not self.api_key:
            logger.error("❌ API_TENNIS_API_KEY no configurada")
            return []
            
        try:
            logger.info("🔴 Consultando partidos en vivo...")
            
            data = self._make_request("get_livescore", {})
            
            if not data:
                logger.warning("⚠️  No se obtuvieron partidos en vivo")
                return []
                
            matches = data.get("result", [])
            
            # Filtrar solo ATP Singles
            atp_matches = [
                m for m in matches
                if (m.get("event_type") or m.get("event_type_type") or "").lower() == "atp singles"
            ]
            
            logger.info(f"✅ {len(atp_matches)} partidos ATP en vivo (de {len(matches)} totales)")
            
            return atp_matches
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo livescore: {e}")
            return []

    def extract_best_odds(self, odds_data: Dict, match_key: str) -> Optional[Dict]:
        """
        Extrae las mejores cuotas de los bookmakers
        
        Devuelve:
        - Mejor cuota para cada jugador
        - Top 3 cuotas para cada jugador (para mostrar en frontend)

        Args:
            odds_data: Datos de cuotas de la API
            match_key: ID del partido

        Returns:
            Dict con mejores cuotas y top 3 o None
        """
        try:
            # Las cuotas vienen en formato: {match_key: {"Home/Away": {"Home": {...}, "Away": {...}}}}
            match_odds = odds_data.get(str(match_key), {})
            home_away = match_odds.get("Home/Away", {})

            if not home_away:
                return None

            # Obtener cuotas de Home (Player 1) y Away (Player 2)
            home_odds = home_away.get("Home", {})
            away_odds = home_away.get("Away", {})

            if not home_odds or not away_odds:
                return None

            # Convertir a lista de tuplas (bookmaker, cuota) y ordenar
            player1_odds_list = [(bm, float(cuota)) for bm, cuota in home_odds.items()]
            player2_odds_list = [(bm, float(cuota)) for bm, cuota in away_odds.items()]
            
            # Ordenar de mayor a menor (mejores cuotas primero)
            player1_odds_list.sort(key=lambda x: x[1], reverse=True)
            player2_odds_list.sort(key=lambda x: x[1], reverse=True)

            # Mejor cuota (la primera después de ordenar)
            best_player1_odds = player1_odds_list[0][1] if player1_odds_list else None
            best_player2_odds = player2_odds_list[0][1] if player2_odds_list else None
            bookmaker_p1 = player1_odds_list[0][0] if player1_odds_list else None
            bookmaker_p2 = player2_odds_list[0][0] if player2_odds_list else None

            # Top 3 cuotas para cada jugador (para mostrar en frontend)
            top3_player1 = [
                {
                    "bookmaker": bm,
                    "odds": cuota,
                    "is_best": i == 0  # La primera es la mejor
                }
                for i, (bm, cuota) in enumerate(player1_odds_list[:3])
            ]
            
            top3_player2 = [
                {
                    "bookmaker": bm,
                    "odds": cuota,
                    "is_best": i == 0  # La primera es la mejor
                }
                for i, (bm, cuota) in enumerate(player2_odds_list[:3])
            ]

            return {
                # Mejores cuotas (para el modelo)
                "player1_odds": best_player1_odds,
                "player2_odds": best_player2_odds,
                "bookmaker_p1": bookmaker_p1,
                "bookmaker_p2": bookmaker_p2,
                
                # Top 3 para frontend
                "top3_player1": top3_player1,
                "top3_player2": top3_player2,
                
                # Total de bookmakers disponibles
                "total_bookmakers": len(player1_odds_list),
            }

        except Exception as e:
            logger.error(f"❌ Error extrayendo cuotas: {e}")
            return None

    def get_all_odds_batch(self, date_start: str, date_stop: str) -> Dict:
        """
        Obtiene TODAS las cuotas de un rango de fechas en una sola llamada
        
        Esta es la optimización clave: en lugar de hacer N llamadas individuales
        para cada partido, hacemos 1 sola llamada que devuelve todas las cuotas.
        
        Args:
            date_start: Fecha inicial (YYYY-MM-DD)
            date_stop: Fecha final (YYYY-MM-DD)
            
        Returns:
            Dict con {match_key: odds_data}
        """
        if not self.api_key:
            logger.error("❌ API_TENNIS_API_KEY no configurada")
            return {}
        
        try:
            logger.info(f"📊 Obteniendo cuotas batch ({date_start} a {date_stop})...")
            
            params = {
                "date_start": date_start,
                "date_stop": date_stop
            }
            
            data = self._make_request("get_odds", params)
            
            if not data:
                logger.warning("⚠️  No se pudieron obtener cuotas batch")
                return {}
            
            odds_result = data.get("result", {})
            logger.info(f"✅ Cuotas batch obtenidas para {len(odds_result)} partidos")
            
            return odds_result
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo cuotas batch: {e}")
            return {}

    def get_live_results_batch(self, event_keys: List[str]) -> Dict:
        """
        Obtiene resultados actualizados para múltiples partidos
        
        Args:
            event_keys: Lista de IDs de partidos
        
        Returns:
            Dict con event_key como clave y datos del partido
        """
        try:
            # API-Tennis permite filtrar por múltiples IDs
            # Convertir lista a string separado por comas
            ids_str = ",".join(str(k) for k in event_keys)
            
            params = {
                "event_key": ids_str
            }
            
            response = self._make_request("get_events", params)
            
            if not response or "result" not in response:
                return {}
            
            # Convertir lista a dict con event_key como clave
            results = {}
            for match in response["result"]:
                event_key = match.get("event_key")
                if event_key:
                    results[event_key] = {
                        "scores": match.get("scores"),
                        "event_live": match.get("event_live"),
                        "event_status": match.get("event_status"),
                        "event_final_result": match.get("event_final_result")
                    }
            
            return results
            
        except Exception as e:
            logger.error(f"Error obteniendo resultados en vivo: {e}")
            return {}

    def get_standings(self, league: str = "ATP") -> List[Dict]:
        """
        Obtiene rankings ATP
        
        Args:
            league: "ATP" (WTA no necesario por ahora)
        
        Returns:
            Lista de jugadores con ranking
        """
        try:
            params = {
                "method": "get_standings",
                "APIkey": self.api_key,
                "event_type": league
            }
            
            response = self._make_request(params)
            
            if not response or "result" not in response:
                return []
            
            return response["result"]
            
        except Exception as e:
            logger.error(f"Error obteniendo rankings {league}: {e}")
            return []

    def get_all_matches_with_odds(self, days_ahead: int = 7) -> List[Dict]:
        """
        Obtiene todos los partidos próximos con sus cuotas
        
        OPTIMIZACIÓN: Usa batch request para obtener todas las cuotas en 1 llamada
        en lugar de N llamadas individuales.

        Args:
            days_ahead: Días hacia adelante

        Returns:
            Lista de partidos con cuotas procesadas
        """
        # 1. Obtener fixtures
        matches = self.get_upcoming_matches(days_ahead)

        if not matches:
            logger.info("ℹ️  No hay partidos disponibles")
            return []

        logger.info(f"📥 {len(matches)} fixtures obtenidos, obteniendo cuotas batch...")

        # 2. Obtener TODAS las cuotas en una sola llamada batch
        today = datetime.now()
        date_start = today.strftime("%Y-%m-%d")
        date_stop = (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        
        all_odds = self.get_all_odds_batch(date_start, date_stop)
        
        if not all_odds:
            logger.warning("⚠️  No se pudieron obtener cuotas, guardando partidos sin cuotas")

        # 3. Combinar fixtures con cuotas
        matches_processed = []

        for match in matches:
            match_info = self.extract_match_info(match)
            if not match_info:
                continue

            match_key = match_info["match_id"]
            
            # Buscar cuotas en el batch
            if match_key and str(match_key) in all_odds:
                best_odds = self.extract_best_odds(all_odds, match_key)
                if best_odds:
                    match_info.update(best_odds)
                else:
                    # Partido sin cuotas válidas
                    match_info.update({
                        "player1_odds": None,
                        "player2_odds": None,
                        "top3_player1": [],
                        "top3_player2": [],
                    })
            else:
                # Partido sin cuotas
                match_info.update({
                    "player1_odds": None,
                    "player2_odds": None,
                    "top3_player1": [],
                    "top3_player2": [],
                })
            
            matches_processed.append(match_info)

        matches_with_odds = [m for m in matches_processed if m.get("player1_odds")]
        matches_without_odds = [m for m in matches_processed if not m.get("player1_odds")]
        
        logger.info(f"✅ {len(matches_with_odds)} partidos con cuotas")
        logger.info(f"ℹ️  {len(matches_without_odds)} partidos sin cuotas (se guardarán igual)")
        
        return matches_processed  # Devolver TODOS los partidos

    def get_rate_limit_status(self) -> Dict:
        """
        Obtiene el estado actual del rate limit

        Returns:
            Dict con información del rate limit
        """
        return {"requests_made": self.requests_made}


# ============================================================
# EJEMPLO DE USO
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
        print(
            f"📊 Bookmakers: {match.get('bookmaker_p1', 'N/A')} / {match.get('bookmaker_p2', 'N/A')}"
        )
        print("-" * 50)

    # Mostrar rate limit
    rate_limit = client.get_rate_limit_status()
    print(f"\n📊 Rate Limit Status:")
    print(f"   Requests hechos: {rate_limit['requests_made']}")

