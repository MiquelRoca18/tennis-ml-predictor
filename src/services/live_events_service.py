"""
Servicio para manejar eventos de partidos en vivo via WebSocket
"""
import logging
import asyncio
from typing import Dict, Optional
from src.services.websocket_client import TennisWebSocketClient

logger = logging.getLogger(__name__)


class LiveEventsService:
    """
    Servicio para manejar eventos de partidos en vivo via WebSocket
    
    Conecta al WebSocket de api-tennis.com y actualiza la BD
    cuando llegan eventos de partidos en vivo.
    """
    
    def __init__(self, db, api_client):
        self.db = db
        self.api_client = api_client
        self.ws_client: Optional[TennisWebSocketClient] = None
        
    async def start(self):
        """Iniciar servicio de eventos en vivo"""
        logger.info("🚀 Iniciando servicio de eventos en vivo via WebSocket...")
        
        self.ws_client = TennisWebSocketClient(
            api_key=self.api_client.api_key,
            on_event=self.handle_event
        )
        
        # Esto bloqueará hasta que se cierre el WebSocket
        await self.ws_client.listen()
    
    async def handle_event(self, event: Dict):
        """
        Procesar evento de partido en vivo
        
        Formato esperado de api-tennis.com:
        {
            "event_key": "12345",
            "event_live": "1",
            "scores": "6-4, 3-2",
            "event_status": "in_progress",
            "event_final_result": null,
            "event_first_player": "Alcaraz",
            "event_second_player": "Sinner"
        }
        """
        try:
            event_key = event.get("event_key")
            if not event_key:
                logger.debug("Evento sin event_key, ignorando")
                return
            
            # Buscar partido en BD por event_key
            match = self.db.get_match_by_event_key(event_key)
            if not match:
                logger.debug(f"Partido no encontrado en BD: {event_key}")
                return
            
            # Actualizar datos en vivo
            self.db.update_match_live_data(
                match_id=match["id"],
                scores=event.get("scores"),
                event_live=event.get("event_live"),
                event_status=event.get("event_status"),
                event_final_result=event.get("event_final_result")
            )
            
            player1 = match.get("jugador1_nombre", "?")
            player2 = match.get("jugador2_nombre", "?")
            scores = event.get("scores", "")
            
            logger.info(f"🔴 LIVE: {player1} vs {player2} - {scores}")
            
        except Exception as e:
            logger.error(f"❌ Error procesando evento: {e}", exc_info=True)
    
    async def stop(self):
        """Detener servicio"""
        logger.info("🛑 Deteniendo servicio de eventos en vivo...")
        if self.ws_client:
            await self.ws_client.close()
