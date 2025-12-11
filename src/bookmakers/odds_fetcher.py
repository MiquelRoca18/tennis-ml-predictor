"""
OddsFetcher - Obtención de cuotas de The Odds API

Este módulo maneja:
- Conexión con The Odds API
- Obtención de partidos y cuotas
- Tracking de requests restantes
- Manejo robusto de errores y límites de API
- Sistema de caché para optimizar uso
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json
import time

from .config import BookmakerConfig


class APILimitError(Exception):
    """Excepción personalizada para cuando se alcanza el límite de API"""
    pass


class OddsFetcher:
    """
    Obtiene cuotas de múltiples bookmakers usando The Odds API
    
    Características:
    - Tracking de requests restantes
    - Alertas de límite de API
    - Sistema de caché
    - Manejo robusto de errores
    """
    
    def __init__(self, api_key=None, use_cache=True):
        """
        Inicializa el fetcher de cuotas
        
        Args:
            api_key: API key de The Odds API (si None, usa config)
            use_cache: Si True, usa sistema de caché
        """
        self.api_key = api_key or BookmakerConfig.ODDS_API_KEY
        self.base_url = BookmakerConfig.ODDS_API_BASE_URL
        self.use_cache = use_cache
        
        # Tracking de requests
        self.requests_remaining = None
        self.requests_used = None
        
        # Validar API key
        if not self.api_key:
            raise ValueError("❌ API key no configurada. Define ODDS_API_KEY en .env")
        
        # Crear directorio de caché
        if self.use_cache:
            BookmakerConfig.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ OddsFetcher inicializado")
        print(f"   API Key: {'*' * 20}{self.api_key[-4:]}")
        print(f"   Caché: {'✅ Habilitado' if use_cache else '❌ Deshabilitado'}")
    
    def _update_request_stats(self, response):
        """
        Actualiza estadísticas de requests desde los headers de la respuesta
        
        Args:
            response: Response object de requests
        """
        # Extraer información de headers
        self.requests_remaining = response.headers.get('x-requests-remaining')
        self.requests_used = response.headers.get('x-requests-used')
        
        if self.requests_remaining:
            remaining = int(self.requests_remaining)
            
            # Mostrar estadísticas
            print(f"\n📊 API Usage:")
            print(f"   Requests restantes: {remaining}/{BookmakerConfig.MAX_REQUESTS_PER_MONTH}")
            
            # Alertar si quedan pocos requests
            if remaining <= BookmakerConfig.WARNING_THRESHOLD:
                print(f"\n⚠️  ALERTA: Solo quedan {remaining} requests este mes!")
                print(f"   Considera usar el caché para optimizar el uso")
            
            if remaining == 0:
                raise APILimitError(
                    f"❌ LÍMITE DE API ALCANZADO\n"
                    f"   Has usado todos los {BookmakerConfig.MAX_REQUESTS_PER_MONTH} requests del mes.\n"
                    f"   El límite se resetea el primer día del mes.\n"
                    f"   Opciones:\n"
                    f"   1. Esperar al próximo mes\n"
                    f"   2. Actualizar a plan de pago en https://the-odds-api.com\n"
                    f"   3. Usar datos del caché si están disponibles"
                )
    
    def _get_cache_path(self, sport):
        """
        Obtiene el path del archivo de caché para un deporte
        
        Args:
            sport: Código del deporte (ej: 'tennis_atp')
        
        Returns:
            Path al archivo de caché
        """
        return BookmakerConfig.CACHE_DIR / f"{sport}_odds.json"
    
    def _is_cache_valid(self, cache_path):
        """
        Verifica si el caché es válido (existe y no ha expirado)
        
        Args:
            cache_path: Path al archivo de caché
        
        Returns:
            bool: True si el caché es válido
        """
        if not cache_path.exists():
            return False
        
        # Verificar edad del caché
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        max_age = timedelta(minutes=BookmakerConfig.CACHE_DURATION_MINUTES)
        
        return cache_age < max_age
    
    def _load_from_cache(self, cache_path):
        """
        Carga datos desde el caché
        
        Args:
            cache_path: Path al archivo de caché
        
        Returns:
            list: Datos de partidos
        """
        with open(cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        print(f"📦 Usando caché ({cache_age.seconds // 60} minutos de antigüedad)")
        
        return data
    
    def _save_to_cache(self, cache_path, data):
        """
        Guarda datos en el caché
        
        Args:
            cache_path: Path al archivo de caché
            data: Datos a guardar
        """
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Datos guardados en caché")
    
    def obtener_partidos_disponibles(self, sport='tennis_atp', force_refresh=False):
        """
        Obtiene partidos disponibles con cuotas
        
        Args:
            sport: Deporte a consultar ('tennis_atp' o 'tennis_wta')
            force_refresh: Si True, ignora el caché y hace request a la API
        
        Returns:
            list: Lista de partidos con cuotas
        
        Raises:
            APILimitError: Si se alcanza el límite de requests
        """
        # Verificar caché
        cache_path = self._get_cache_path(sport)
        
        if self.use_cache and not force_refresh and self._is_cache_valid(cache_path):
            return self._load_from_cache(cache_path)
        
        # Hacer request a la API
        url = f"{self.base_url}/sports/{sport}/odds/"
        
        params = {
            'apiKey': self.api_key,
            'regions': BookmakerConfig.REGIONS,
            'markets': BookmakerConfig.MARKETS,
            'oddsFormat': BookmakerConfig.ODDS_FORMAT
        }
        
        try:
            print(f"\n🌐 Consultando The Odds API...")
            print(f"   Deporte: {sport}")
            print(f"   Regiones: {BookmakerConfig.REGIONS}")
            
            response = requests.get(url, params=params, timeout=10)
            
            # Actualizar estadísticas
            self._update_request_stats(response)
            
            # Verificar errores HTTP
            response.raise_for_status()
            
            data = response.json()
            
            print(f"✅ {len(data)} partidos obtenidos")
            
            # Guardar en caché
            if self.use_cache:
                self._save_to_cache(cache_path, data)
            
            return data
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise ValueError(
                    f"❌ ERROR DE AUTENTICACIÓN\n"
                    f"   API key inválida o expirada.\n"
                    f"   Verifica tu API key en: https://the-odds-api.com/account/"
                )
            elif e.response.status_code == 429:
                raise APILimitError(
                    f"❌ LÍMITE DE RATE ALCANZADO\n"
                    f"   Has excedido el límite de requests por minuto.\n"
                    f"   Espera unos segundos e intenta de nuevo."
                )
            else:
                raise Exception(f"❌ Error HTTP {e.response.status_code}: {e}")
        
        except requests.exceptions.Timeout:
            raise Exception(
                f"❌ TIMEOUT\n"
                f"   La API no respondió a tiempo.\n"
                f"   Verifica tu conexión a internet e intenta de nuevo."
            )
        
        except requests.exceptions.RequestException as e:
            raise Exception(f"❌ Error de conexión: {e}")
    
    def parsear_partido(self, partido_data):
        """
        Parsea datos de un partido y extrae cuotas de cada bookmaker
        
        Args:
            partido_data: Datos del partido de la API
        
        Returns:
            list: Lista de dicts con cuotas de cada bookmaker
        """
        jugador1 = partido_data['home_team']
        jugador2 = partido_data['away_team']
        
        # Parsear fecha
        try:
            fecha = datetime.fromisoformat(partido_data['commence_time'].replace('Z', '+00:00'))
        except:
            fecha = datetime.now()
        
        # Extraer cuotas de cada bookmaker
        cuotas = []
        
        for bookmaker in partido_data.get('bookmakers', []):
            nombre_casa = bookmaker['title']
            
            for market in bookmaker.get('markets', []):
                if market['key'] == 'h2h':
                    outcomes = market['outcomes']
                    
                    # Buscar cuotas de cada jugador
                    cuota_j1 = next((o['price'] for o in outcomes if o['name'] == jugador1), None)
                    cuota_j2 = next((o['price'] for o in outcomes if o['name'] == jugador2), None)
                    
                    if cuota_j1 and cuota_j2:
                        cuotas.append({
                            'bookmaker': nombre_casa,
                            'jugador1': jugador1,
                            'jugador2': jugador2,
                            'cuota_jugador1': cuota_j1,
                            'cuota_jugador2': cuota_j2,
                            'fecha': fecha,
                            'sport': partido_data.get('sport_key', 'tennis_atp')
                        })
        
        return cuotas
    
    def obtener_todas_cuotas(self, sport='tennis_atp', force_refresh=False):
        """
        Obtiene y parsea todos los partidos disponibles
        
        Args:
            sport: Deporte a consultar
            force_refresh: Si True, ignora el caché
        
        Returns:
            DataFrame: Cuotas de todos los partidos y bookmakers
        """
        partidos = self.obtener_partidos_disponibles(sport, force_refresh)
        
        todas_cuotas = []
        
        for partido in partidos:
            cuotas = self.parsear_partido(partido)
            todas_cuotas.extend(cuotas)
        
        df = pd.DataFrame(todas_cuotas)
        
        if len(df) > 0:
            # Contar bookmakers únicos
            num_bookmakers = df['bookmaker'].nunique()
            num_partidos = df.groupby(['jugador1', 'jugador2']).ngroups
            
            print(f"\n📊 Resumen de Cuotas:")
            print(f"   Partidos: {num_partidos}")
            print(f"   Bookmakers: {num_bookmakers}")
            print(f"   Total de cuotas: {len(df)}")
        else:
            print(f"\n⚠️  No hay partidos disponibles en este momento")
        
        return df
    
    def get_request_stats(self):
        """
        Obtiene estadísticas de uso de la API
        
        Returns:
            dict: Estadísticas de requests
        """
        return {
            'requests_remaining': self.requests_remaining,
            'requests_used': self.requests_used,
            'max_requests': BookmakerConfig.MAX_REQUESTS_PER_MONTH
        }


# Ejemplo de uso
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🌐 ODDS FETCHER - DEMO")
    print("="*60)
    
    try:
        # Crear fetcher
        fetcher = OddsFetcher(use_cache=True)
        
        # Obtener cuotas
        df_cuotas = fetcher.obtener_todas_cuotas(sport='tennis_atp')
        
        if len(df_cuotas) > 0:
            print(f"\n📋 Primeras cuotas:")
            print(df_cuotas.head(10).to_string())
            
            # Mostrar estadísticas
            stats = fetcher.get_request_stats()
            print(f"\n📊 Estadísticas de API:")
            print(f"   Requests restantes: {stats['requests_remaining']}")
            print(f"   Requests usados: {stats['requests_used']}")
        
        print(f"\n✅ Demo completado!")
    
    except APILimitError as e:
        print(f"\n{e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
