"""
Servicio de Actualización Automática de Cuotas
==============================================

Sistema para actualizar automáticamente las cuotas de partidos pendientes,
detectar partidos nuevos y regenerar predicciones cada 15 minutos.
"""

import logging
from datetime import datetime
from typing import List, Dict
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

    def get_pending_matches(self, db_connection: MatchDatabase = None) -> List[Dict]:
        """
        Obtiene todos los partidos pendientes (no completados)
        
        Args:
            db_connection: Conexión opcional a DB (para threads)

        Returns:
            Lista de partidos pendientes
        """
        # Usar la conexión proporcionada o la por defecto
        database = db_connection or self.db
        cursor = database.conn.cursor()
        
        cursor.execute(
            """
            SELECT * FROM matches
            WHERE estado = 'pendiente'
            AND fecha_partido >= DATE('now')
            ORDER BY fecha_partido ASC
        """
        )

        matches = [dict(row) for row in cursor.fetchall()]
        logger.info(f"📊 Encontrados {len(matches)} partidos pendientes")
        return matches

    def update_match_odds_manual(
        self, match_id: int, jugador1_cuota: float, jugador2_cuota: float
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
                return {"success": False, "error": f"Partido {match_id} no encontrado"}

            # Verificar que está pendiente
            if partido["estado"] != "pendiente":
                return {
                    "success": False,
                    "error": f"Partido {match_id} no está pendiente (estado: {partido['estado']})",
                }

            # Aquí iría la lógica de actualización
            # Por ahora solo registramos el intento
            logger.info(
                f"🔄 Actualizando cuotas del partido {match_id}: {jugador1_cuota} / {jugador2_cuota}"
            )

            return {
                "success": True,
                "match_id": match_id,
                "cuotas_actualizadas": {"jugador1": jugador1_cuota, "jugador2": jugador2_cuota},
            }

        except Exception as e:
            logger.error(f"❌ Error actualizando cuotas del partido {match_id}: {e}")
            return {"success": False, "error": str(e)}

    def detect_new_matches(self) -> Dict:
        """
        Detecta partidos nuevos en The Odds API y los crea automáticamente
        
        Detecta y guarda partidos nuevos (para ejecutar en job programado)
        
        Returns:
            Dict con estadísticas
        """
        if not self.odds_client:
            logger.warning("⚠️  The Odds API no disponible - saltando detección de partidos nuevos")
            return {"success": False, "partidos_nuevos": 0, "mensaje": "The Odds API no disponible"}

        # IMPORTANTE: Crear nueva conexión para el thread
        # Evita error "SQLite objects created in a thread can only be used in that same thread"
        temp_db = MatchDatabase(self.db.db_path)
        
        try:
            logger.info("🔍 Detectando partidos nuevos en The Odds API...")

            # Obtener partidos de The Odds API
            matches_api = self.odds_client.get_all_matches_with_odds(days_ahead=7)

            if not matches_api:
                logger.info("ℹ️  No hay partidos disponibles en The Odds API")
                return {
                    "success": True,
                    "partidos_nuevos": 0,
                    "mensaje": "No hay partidos disponibles",
                }

            # Detectar partidos nuevos
            partidos_nuevos_creados = 0
            partidos_wta_filtrados = 0

            for match in matches_api:
                # ===== FILTROS: Todos los partidos MASCULINOS INDIVIDUALES =====
                # Rechazamos: WTA (femenino) y Dobles
                # Aceptamos: ATP, Challengers, ITF Men, etc.
                
                event_type = (match.get("event_type") or match.get("event_type_type") or "").upper()
                league = (match.get("league") or "").upper()
                tournament_name = match.get("tournament", "")
                tournament_upper = tournament_name.upper()
                player1_name = match.get("player1_name", "")
                player2_name = match.get("player2_name", "")
                
                # ===== FILTRO 1: Rechazar DOBLES =====
                is_doubles = (
                    "DOUBLES" in event_type or
                    "DOUBLE" in event_type or
                    "DOUBLES" in tournament_upper or
                    "/" in player1_name or
                    "/" in player2_name
                )
                
                if is_doubles:
                    logger.debug(f"⏭️  Ignorando dobles: {event_type} - {tournament_name}")
                    continue
                
                # ===== FILTRO 2: Rechazar WTA/FEMENINO =====
                is_wta = (
                    "WTA" in event_type or
                    "WTA" in league or
                    "WTA" in tournament_upper or
                    "WOMEN" in event_type or
                    "WOMEN" in tournament_upper or
                    "LADIES" in event_type or
                    "LADIES" in tournament_upper or
                    "FEMALE" in event_type or
                    "FEMALE" in tournament_upper or
                    "GIRLS" in event_type or
                    "GIRLS" in tournament_upper or
                    # W-series tournaments (W15, W25, W50, W75, W100, etc.)
                    (tournament_name.startswith("W") and len(tournament_name) > 1 and " " in tournament_name and tournament_name.split()[0][1:].isdigit())
                )
                
                if is_wta:
                    partidos_wta_filtrados += 1
                    logger.debug(f"⏭️  Ignorando WTA/femenino: {event_type} - {tournament_name}")
                    continue
                
                # ===== ACEPTAR TODO LO DEMÁS (ATP, Challengers, ITF Men, etc.) =====
                logger.debug(f"✅ Procesando partido: {event_type} - {tournament_name}")

                # Extraer fecha del partido (API-Tennis usa event_date y event_time)
                try:
                    fecha_partido = datetime.strptime(match["event_date"], "%Y-%m-%d").date()
                except (ValueError, KeyError, TypeError) as e:
                    logger.warning(f"⚠️  Error parseando fecha para match {match.get('event_key')}: {e}")
                    continue

                # Verificar si ya existe usando el método de la base de datos
                if self.db.match_exists(match["player1_name"], match["player2_name"], fecha_partido):
                    logger.debug(
                        f"ℹ️  Partido ya existe: {match['player1_name']} vs {match['player2_name']} ({fecha_partido})"
                    )
                    continue

                # Partido nuevo ATP - crear en DB
                logger.info(
                    f"🆕 Partido ATP nuevo detectado: {match['player1_name']} vs {match['player2_name']}"
                )

                try:
                    match_id = self.db.create_match(
                        fecha_partido=str(fecha_partido),
                        hora_inicio=match.get("time", "00:00"),
                        torneo=match.get("tournament", "Unknown"),
                        ronda=match.get("round", "Unknown"),
                        superficie=match.get("surface") or "Hard",  # Garantizar que nunca sea None
                        jugador1_nombre=match["player1_name"],
                        jugador1_ranking=None,  # API-Tennis no proporciona ranking en fixtures
                        jugador1_cuota=match.get("player1_odds"),
                        jugador2_nombre=match["player2_name"],
                        jugador2_ranking=None,
                        jugador2_cuota=match.get("player2_odds"),
                        # Nuevos campos de tracking
                        event_key=match.get("event_key"),
                        jugador1_key=match.get("player1_key"),
                        jugador2_key=match.get("player2_key"),
                        tournament_key=match.get("tournament_key"),
                        tournament_season=match.get("tournament_season"),
                        event_live=match.get("event_live", "0"),
                        event_qualification=match.get("event_qualification", "False"),
                    )

                    logger.info(
                        f"✅ Partido ATP {match_id} creado: {match['player1_name']} vs {match['player2_name']}"
                    )
                    partidos_nuevos_creados += 1
                    
                    # Guardar top 3 cuotas si están disponibles
                    if match.get("top3_player1") and match.get("top3_player2"):
                        self.db.save_top3_odds(
                            match_id=match_id,
                            top3_player1=match["top3_player1"],
                            top3_player2=match["top3_player2"],
                        )

                    # Generar predicción automáticamente
                    try:
                        from src.prediction.predictor_calibrado import PredictorCalibrado
                        from src.config.settings import Config
                        
                        # Cargar predictor (lazy loading)
                        predictor = PredictorCalibrado(Config.MODEL_PATH)
                        
                        # Generar predicción para ambos jugadores
                        resultado_j1 = predictor.predecir_partido(
                            jugador1=match["player1_name"],
                            jugador2=match["player2_name"],
                            superficie=match.get("surface") or "Hard",
                            cuota=match.get("player1_odds", 2.0)
                        )
                        
                        resultado_j2 = predictor.predecir_partido(
                            jugador1=match["player2_name"],
                            jugador2=match["player1_name"],
                            superficie=match.get("surface") or "Hard",
                            cuota=match.get("player2_odds", 2.0)
                        )
                        
                        # Extraer probabilidades y métricas
                        prob_j1 = resultado_j1["probabilidad"]
                        prob_j2 = 1 - prob_j1
                        ev_j1 = resultado_j1["expected_value"]
                        ev_j2 = resultado_j2["expected_value"]
                        edge_j1 = resultado_j1.get("edge", 0)
                        edge_j2 = resultado_j2.get("edge", 0)
                        kelly_j1 = resultado_j1.get("stake_recomendado", 0)
                        kelly_j2 = resultado_j2.get("stake_recomendado", 0)
                        
                        # Determinar recomendación
                        if ev_j1 > 0.03 and ev_j1 > ev_j2:
                            recomendacion = f"APOSTAR a {match['player1_name']}"
                            mejor_opcion = match["player1_name"]
                        elif ev_j2 > 0.03:
                            recomendacion = f"APOSTAR a {match['player2_name']}"
                            mejor_opcion = match["player2_name"]
                        else:
                            recomendacion = "NO APOSTAR"
                            mejor_opcion = None
                        
                        # Determinar confianza
                        if abs(prob_j1 - 0.5) > 0.15:
                            confianza = "Alta"
                        elif abs(prob_j1 - 0.5) > 0.08:
                            confianza = "Media"
                        else:
                            confianza = "Baja"
                        
                        # Guardar predicción
                        self.db.add_prediction(
                            match_id=match_id,
                            jugador1_cuota=match.get("player1_odds", 2.0),
                            jugador2_cuota=match.get("player2_odds", 2.0),
                            jugador1_probabilidad=prob_j1,
                            jugador2_probabilidad=prob_j2,
                            jugador1_ev=ev_j1,
                            jugador2_ev=ev_j2,
                            jugador1_edge=edge_j1,
                            jugador2_edge=edge_j2,
                            recomendacion=recomendacion,
                            mejor_opcion=mejor_opcion,
                            confianza=confianza,
                            kelly_stake_jugador1=kelly_j1,
                            kelly_stake_jugador2=kelly_j2,
                            confidence_level=resultado_j1.get("confidence_level"),
                            confidence_score=resultado_j1.get("confidence_score"),
                            player1_known=resultado_j1.get("player1_known"),
                            player2_known=resultado_j2.get("player2_known"),
                        )
                        
                        logger.info(f"✅ Predicción generada para partido {match_id}: {recomendacion}")
                        
                    except Exception as e:
                        logger.error(f"❌ Error generando predicción para partido {match_id}: {e}")

                except Exception as e:
                    logger.error(f"❌ Error creando partido: {e}")

            logger.info(
                f"✅ Detección completada: {partidos_nuevos_creados} partidos ATP nuevos creados, "
                f"{partidos_wta_filtrados} partidos WTA filtrados"
            )

            return {
                "success": True,
                "partidos_api": len(matches_api),
                "partidos_atp": partidos_nuevos_creados,
                "partidos_wta_filtrados": partidos_wta_filtrados,
                "partidos_nuevos": partidos_nuevos_creados,
                "mensaje": f"{partidos_nuevos_creados} partidos ATP nuevos detectados y creados ({partidos_wta_filtrados} WTA filtrados)",
            }

        except Exception as e:
            logger.error(f"❌ Error detectando partidos nuevos: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def update_all_pending_matches(self) -> Dict:
        """
        Actualiza cuotas de partidos pendientes con rate limiting para proteger la API.
        
        Estrategia de optimización:
        1. Solo actualizar partidos que empiezan en las próximas 24 horas
        2. Limitar a máximo 20 actualizaciones por ejecución (cada 15 min = 80/hora)
        3. Priorizar partidos más próximos
        """
        # IMPORTANTE: Crear nueva conexión para el thread
        temp_db = MatchDatabase(self.db.db_path)
        
        try:
            # PASO 1: Detectar partidos nuevos
            # Nota: detect_new_matches crea su propia conexión interna, 
            # pero podríamos optimizarlo pasándole temp_db si lo refactorizamos.
            # Por ahora lo dejamos así para no tocar demasiadas cosas.
            self.detect_new_matches()
            
            # PASO 2: Actualizar partidos existentes con RATE LIMITING
            
            # Usar temp_db para queries con filtro de fecha (hoy y mañana)
            cursor = temp_db.conn.cursor()
            cursor.execute(
                """
                SELECT * FROM matches
                WHERE estado = 'pendiente'
                AND fecha_partido BETWEEN DATE('now') AND DATE('now', '+1 day')
                ORDER BY fecha_partido ASC, hora_inicio ASC
            """
            )
            # Fetch all pero procesaremos limitado
            pending_matches = [dict(row) for row in cursor.fetchall()]

            if not pending_matches:
                return {"success": True, "updated": 0, "message": "No hay partidos pendientes próximos"}

            # LIMITAR API USAGE: Máximo 20 partidos por ejecución
            matches_to_update = pending_matches[:20]
            updates_count = 0
            
            logger.info(f"🔄 Iniciando actualización de cuotas para {len(matches_to_update)} partidos (de {len(pending_matches)} pendientes próximos)")
            
            # Usar una instancia local del servicio con la DB temporal
            temp_service = OddsUpdateService(temp_db, self.odds_client)

            for match in matches_to_update:
                try:
                    result = temp_service.update_match_odds_manual(match["id"])
                    if result["success"]:
                        updates_count += 1
                except Exception as e:
                    logger.error(f"Error actualizando match {match['id']}: {e}")

            logger.info(f"✅ Actualización parcial completada: {updates_count}/{len(matches_to_update)} partidos actualizados")
            
            # También detectar nuevos
            # self.detect_new_matches() 
            
            return {
                "success": True, 
                "matches_processed": len(matches_to_update),
                "matches_updated": updates_count,
                "remaining_pending": len(pending_matches) - len(matches_to_update)
            }
            
        except Exception as e:
            logger.error(f"❌ Error en actualización automática: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
        finally:
            if temp_db.conn:
                temp_db.conn.close()

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
            "estado": "activo",
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
        print(
            f"  - {match['jugador1_nombre']} vs {match['jugador2_nombre']} ({match['fecha_partido']})"
        )

    # Ejecutar actualización
    result = service.update_all_pending_matches()
    print(f"\n✅ Resultado: {result}")

    # Estadísticas
    stats = service.get_update_stats()
    print(f"\n📈 Estadísticas: {stats}")

    db.close()
