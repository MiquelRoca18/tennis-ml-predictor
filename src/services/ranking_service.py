"""
Servicio para obtener y cachear rankings ATP
"""
import logging
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class RankingService:
    """Gestiona rankings ATP con cache de 7 días"""
    
    def __init__(self, api_client, cache_dir: str = "cache"):
        self.api_client = api_client
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_ttl = timedelta(days=7)
    
    def get_player_ranking(self, player_name: str) -> Optional[int]:
        """
        Obtiene el ranking ATP de un jugador
        
        Args:
            player_name: Nombre del jugador
        
        Returns:
            Ranking del jugador o None si no se encuentra
        """
        try:
            # Obtener rankings (con cache)
            rankings = self._get_rankings()
            
            # Buscar jugador (case-insensitive)
            player_name_lower = player_name.lower()
            for player in rankings:
                if player["player"].lower() == player_name_lower:
                    return int(player["place"])
            
            logger.warning(f"Jugador no encontrado en rankings ATP: {player_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error obteniendo ranking de {player_name}: {e}")
            return None
    
    def _get_rankings(self) -> List[Dict]:
        """Obtiene rankings ATP con cache de 7 días"""
        cache_file = self.cache_dir / "rankings_ATP.json"
        
        # Verificar cache
        if cache_file.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age < self.cache_ttl:
                logger.debug(f"Usando cache de rankings ATP ({cache_age.days} días)")
                with open(cache_file, "r") as f:
                    return json.load(f)
        
        # Obtener de API
        logger.info("Obteniendo rankings ATP de API-Tennis...")
        rankings = self.api_client.get_standings("ATP")
        
        if rankings:
            # Guardar en cache
            with open(cache_file, "w") as f:
                json.dump(rankings, f)
            logger.info(f"✅ Rankings ATP cacheados ({len(rankings)} jugadores)")
        
        return rankings or []
    
    def refresh_cache(self):
        """Fuerza actualización de cache de rankings"""
        logger.info("Refrescando cache de rankings ATP...")
        cache_file = self.cache_dir / "rankings_ATP.json"
        if cache_file.exists():
            cache_file.unlink()
        self._get_rankings()
