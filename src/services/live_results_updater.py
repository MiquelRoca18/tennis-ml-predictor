"""
Servicio para actualizar resultados de partidos en vivo
"""
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class LiveResultsUpdater:
    """Actualiza scores de partidos en vivo desde API-Tennis"""
    
    def __init__(self, db, api_client):
        self.db = db
        self.api_client = api_client
    
    def update_live_matches(self):
        """
        Actualiza scores de todos los partidos en vivo
        Se ejecuta cada 60 segundos por el scheduler
        """
        try:
            # 1. Obtener partidos en vivo de BD
            live_matches = self.db.get_live_matches()
            
            if not live_matches:
                logger.debug("No hay partidos en vivo")
                return
            
            logger.info(f"🔴 Actualizando {len(live_matches)} partidos en vivo...")
            
            # 2. Obtener event_keys para batch request
            event_keys = [m["event_key"] for m in live_matches if m.get("event_key")]
            
            if not event_keys:
                logger.warning("Partidos en vivo sin event_key")
                return
            
            # 3. Obtener datos actualizados de API-Tennis (1 llamada batch)
            updated_data = self.api_client.get_live_results_batch(event_keys)
            
            # 4. Actualizar cada partido en BD
            updated_count = 0
            for match in live_matches:
                event_key = match.get("event_key")
                if not event_key or event_key not in updated_data:
                    continue
                
                data = updated_data[event_key]
                
                # Actualizar score y estado
                self.db.update_match_live_data(
                    match_id=match["id"],
                    scores=data.get("scores"),
                    event_live=data.get("event_live"),
                    event_status=data.get("event_status"),
                    event_final_result=data.get("event_final_result")
                )
                
                updated_count += 1
            
            logger.info(f"✅ Actualizados {updated_count} partidos en vivo")
            
        except Exception as e:
            logger.error(f"❌ Error actualizando partidos en vivo: {e}")
