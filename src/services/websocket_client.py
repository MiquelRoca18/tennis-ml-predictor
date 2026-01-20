"""
Cliente WebSocket para api-tennis.com
"""
import asyncio
import websockets
import json
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class TennisWebSocketClient:
    """Cliente WebSocket para recibir eventos en tiempo real de api-tennis.com"""
    
    def __init__(self, api_key: str, on_event: Callable):
        self.api_key = api_key
        self.on_event = on_event
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.running = False
        
    async def connect(self):
        """Conectar al WebSocket de api-tennis.com"""
        uri = f"wss://wss.api-tennis.com/live?APIkey={self.api_key}"
        logger.info(f"🔌 Conectando a WebSocket: wss://wss.api-tennis.com/live")
        
        try:
            self.ws = await websockets.connect(uri)
            logger.info("✅ WebSocket conectado exitosamente")
            return True
        except Exception as e:
            logger.error(f"❌ Error conectando WebSocket: {e}")
            return False
    
    async def listen(self):
        """
        Escuchar eventos en tiempo real
        Reconecta automáticamente si se pierde la conexión
        """
        self.running = True
        
        while self.running:
            try:
                if not self.ws:
                    connected = await self.connect()
                    if not connected:
                        await asyncio.sleep(5)
                        continue
                
                async for message in self.ws:
                    try:
                        # Ignorar mensajes vacíos o pings
                        if not message or message.strip() == "":
                            continue
                            
                        data = json.loads(message)
                        await self.on_event(data)
                    except json.JSONDecodeError as e:
                        logger.debug(f"Mensaje no-JSON ignorado (probablemente ping/pong)")
                        
            except websockets.exceptions.ConnectionClosed:
                logger.warning("⚠️  WebSocket desconectado, reconectando en 5s...")
                self.ws = None
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"❌ Error en WebSocket: {e}")
                self.ws = None
                await asyncio.sleep(5)
    
    async def close(self):
        """Cerrar conexión WebSocket"""
        self.running = False
        if self.ws:
            await self.ws.close()
            logger.info("🔌 WebSocket cerrado")
