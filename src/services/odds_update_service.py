"""
Servicio de Actualización Automática de Cuotas
==============================================

Sistema para actualizar automáticamente las cuotas de partidos pendientes,
detectar partidos nuevos y regenerar predicciones cada 15 minutos.
"""

import logging
from datetime import date, datetime
from typing import List, Dict, Optional
from src.database.match_database import MatchDatabase
from src.services.api_tennis_client import APITennisClient

logger = logging.getLogger(__name__)


class OddsUpdateService:
    """
    Servicio para actualizar cuotas de partidos pendientes y detectar nuevos
    """
    
    def __init__(self, db: MatchDatabase, odds_client: APITennisClient = None):
        """
        Inicializa el servicio de actualización
        
        Args:
            db: Instancia de MatchDatabase
            odds_client: Cliente de API-Tennis (opcional, se crea si no se provee)
        """
        self.db = db
        try:
            self.odds_client = odds_client or APITennisClient()
            logger.info("✅ OddsUpdateService inicializado con API-Tennis")
        except Exception as e:
            logger.warning(f"⚠️  OddsUpdateService inicializado SIN API-Tennis: {e}")
            self.odds_client = None
    
    def get_pending_matches(self) -> List[Dict]:
        """
        Obtiene todos los partidos pendientes (no completados)
        
        Returns:
            Lista de partidos pendientes
        """
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT * FROM matches
            WHERE estado = 'pendiente'
            AND fecha_partido >= DATE('now')
            ORDER BY fecha_partido ASC
        """)
        
        matches = [dict(row) for row in cursor.fetchall()]
        logger.info(f"📊 Encontrados {len(matches)} partidos pendientes")
        return matches
    
    def update_match_odds_manual(
        self,
        match_id: int,
        jugador1_cuota: float,
        jugador2_cuota: float
    ) -> Dict:
        """
        Actualiza las cuotas de un partido manualmente
        
        Este método es para testing o actualización manual.
        En producción, se usaría con datos de The Odds API.
        
        Args:
            match_id: ID del partido
            jugador1_cuota: Nueva cuota para jugador 1
            jugador2_cuota: Nueva cuota para jugador 2
        
        Returns:
            Resultado de la actualización
        """
        try:
            # Obtener partido
            partido = self.db.get_match(match_id)
            if not partido:
                return {
                    "success": False,
                    "error": f"Partido {match_id} no encontrado"
                }
            
            # Verificar que está pendiente
            if partido['estado'] != 'pendiente':
                return {
                    "success": False,
                    "error": f"Partido {match_id} no está pendiente (estado: {partido['estado']})"
                }
            
            # Aquí iría la lógica de actualización
            # Por ahora solo registramos el intento
            logger.info(f"🔄 Actualizando cuotas del partido {match_id}: {jugador1_cuota} / {jugador2_cuota}")
            
            return {
                "success": True,
                "match_id": match_id,
                "cuotas_actualizadas": {
                    "jugador1": jugador1_cuota,
                    "jugador2": jugador2_cuota
                }
            }
        
        except Exception as e:
            logger.error(f"❌ Error actualizando cuotas del partido {match_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def detect_new_matches(self) -> Dict:
        """
        Detecta partidos nuevos en The Odds API y los crea automáticamente
        
        Returns:
            Resumen de partidos nuevos detectados y creados
        """
        if not self.odds_client:
            logger.warning("⚠️  The Odds API no disponible - saltando detección de partidos nuevos")
            return {
                "success": False,
                "partidos_nuevos": 0,
                "mensaje": "The Odds API no disponible"
            }
        
        try:
            logger.info("🔍 Detectando partidos nuevos en The Odds API...")
            
            # Obtener partidos de The Odds API
            matches_api = self.odds_client.get_all_matches_with_odds(days_ahead=7)
            
            if not matches_api:
                logger.info("ℹ️  No hay partidos disponibles en The Odds API")
                return {
                    "success": True,
                    "partidos_nuevos": 0,
                    "mensaje": "No hay partidos disponibles"
                }
            
            # Obtener partidos existentes en DB
            cursor = self.db.conn.cursor()
            cursor.execute("""
                SELECT jugador1_nombre, jugador2_nombre, fecha_partido
                FROM matches
                WHERE estado = 'pendiente'
            """)
            partidos_existentes = {
                (row['jugador1_nombre'], row['jugador2_nombre'], row['fecha_partido'])
                for row in cursor.fetchall()
            }
            
            # Detectar partidos nuevos
            partidos_nuevos_creados = 0
            
            for match in matches_api:
                # Extraer fecha del partido (API-Tennis usa event_date y event_time)
                try:
                    fecha_partido = datetime.strptime(match['date'], '%Y-%m-%d').date()
                except:
                    logger.warning(f"⚠️  Fecha inválida para partido: {match}")
                    continue
                
                # Verificar si ya existe
                match_key = (match['player1_name'], match['player2_name'], str(fecha_partido))
                match_key_reverse = (match['player2_name'], match['player1_name'], str(fecha_partido))
                
                if match_key not in partidos_existentes and match_key_reverse not in partidos_existentes:
                    # Partido nuevo - crear en DB
                    logger.info(f"🆕 Partido nuevo detectado: {match['player1_name']} vs {match['player2_name']}")
                    
                    try:
                        match_id = self.db.create_match(
                            fecha_partido=str(fecha_partido),
                            hora_inicio=match.get('time', '00:00'),
                            torneo=match.get('tournament', 'Unknown'),
                            ronda=match.get('round', 'Unknown'),
                            superficie=match.get('surface', 'Hard'),  # Default Hard si no está disponible
                            jugador1_nombre=match['player1_name'],
                            jugador1_ranking=None,  # API-Tennis no proporciona ranking en fixtures
                            jugador1_cuota=match.get('player1_odds'),
                            jugador2_nombre=match['player2_name'],
                            jugador2_ranking=None,
                            jugador2_cuota=match.get('player2_odds')
                        )
                        
                        logger.info(f"✅ Partido {match_id} creado: {match['player1_name']} vs {match['player2_name']}")
                        partidos_nuevos_creados += 1
                        
                        # TODO: Generar predicción automática
                        
                    except Exception as e:
                        logger.error(f"❌ Error creando partido: {e}")
            
            logger.info(f"✅ Detección completada: {partidos_nuevos_creados} partidos nuevos creados")
            
            return {
                "success": True,
                "partidos_api": len(matches_api),
                "partidos_nuevos": partidos_nuevos_creados,
                "mensaje": f"{partidos_nuevos_creados} partidos nuevos detectados y creados"
            }
        
        except Exception as e:
            logger.error(f"❌ Error detectando partidos nuevos: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    def update_all_pending_matches(self) -> Dict:
        """
        Actualiza todos los partidos pendientes
        
        Proceso completo cada 15 minutos:
        1. Detectar partidos nuevos en The Odds API
        2. Actualizar cuotas de partidos existentes
        
        Returns:
            Resumen de la actualización
        """
        try:
            # PASO 1: Detectar partidos nuevos
            resultado_nuevos = self.detect_new_matches()
            
            # PASO 2: Actualizar partidos existentes
            pending_matches = self.get_pending_matches()
            
            if not pending_matches:
                logger.info("ℹ️  No hay partidos pendientes para actualizar")
                return {
                    "success": True,
                    "partidos_nuevos": resultado_nuevos.get("partidos_nuevos", 0),
                    "partidos_actualizados": 0,
                    "mensaje": "Actualización completada - no hay partidos pendientes"
                }
            
            # Por ahora solo registramos que se ejecutó
            logger.info(f"🔄 Ejecutando actualización de {len(pending_matches)} partidos existentes")
            
            # TODO: Aquí iría la integración con The Odds API
            # Por cada partido:
            # 1. Consultar cuotas actuales en The Odds API
            # 2. Si las cuotas cambiaron significativamente (>5%)
            # 3. Llamar al endpoint /matches/{id}/refresh
            
            return {
                "success": True,
                "partidos_nuevos": resultado_nuevos.get("partidos_nuevos", 0),
                "partidos_encontrados": len(pending_matches),
                "partidos_actualizados": 0,
                "mensaje": "Actualización automática ejecutada (modo mock)",
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"❌ Error en actualización automática: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_update_stats(self) -> Dict:
        """
        Obtiene estadísticas del servicio de actualización
        
        Returns:
            Estadísticas del servicio
        """
        pending = self.get_pending_matches()
        
        return {
            "partidos_pendientes": len(pending),
            "proxima_actualizacion": "Cada 15 minutos",
            "estado": "activo"
        }


# ============================================================
# EJEMPLO DE USO
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Crear servicio
    db = MatchDatabase("matches_v2.db")
    service = OddsUpdateService(db)
    
    # Obtener partidos pendientes
    pending = service.get_pending_matches()
    print(f"\n📊 Partidos pendientes: {len(pending)}")
    
    for match in pending:
        print(f"  - {match['jugador1_nombre']} vs {match['jugador2_nombre']} ({match['fecha_partido']})")
    
    # Ejecutar actualización
    result = service.update_all_pending_matches()
    print(f"\n✅ Resultado: {result}")
    
    # Estadísticas
    stats = service.get_update_stats()
    print(f"\n📈 Estadísticas: {stats}")
    
    db.close()
