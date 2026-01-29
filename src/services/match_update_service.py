"""
Match Update Service
====================

Servicio que actualiza automáticamente el estado y resultados de partidos existentes.
Consulta la API periódicamente para detectar cambios en:
- Estado del partido (pendiente → en_juego → completado)
- Resultados finales
- Cuotas actualizadas

Diseñado para ejecutarse cada 5 minutos vía scheduler.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

from src.database.match_database import MatchDatabase
from src.services.api_tennis_client import APITennisClient
from src.services.player_service import PlayerService

logger = logging.getLogger(__name__)


class MatchUpdateService:
    """
    Servicio para actualizar partidos existentes con datos de la API
    """

    def __init__(self, db: MatchDatabase, api_client: APITennisClient):
        """
        Inicializa el servicio de actualización

        Args:
            db: Instancia de base de datos
            api_client: Cliente de API-Tennis
        """
        self.db = db
        self.api_client = api_client
        self.player_service = PlayerService(db)
        logger.info("✅ MatchUpdateService initialized with PlayerService")

    def update_recent_matches(self, days: int = 7) -> Dict:
        """
        Actualiza partidos de los últimos N días

        Args:
            days: Número de días hacia atrás (default: 7)

        Returns:
            Dict con estadísticas de actualización
        """
        logger.info(f"🔄 Actualizando partidos de los últimos {days} días...")

        stats = {
            "timestamp": datetime.now().isoformat(),
            "matches_checked": 0,
            "matches_updated": 0,
            "matches_live": 0,
            "matches_completed": 0,
            "errors": 0,
        }

        try:
            # Obtener partidos recientes de la DB
            matches = self.db.get_recent_matches(days=days)

            # Filtrar partidos que necesitan actualización:
            # - "pendiente": pueden empezar
            # - "en_juego": pueden terminar
            # - "completado" SIN datos completos: necesitan rellenar scores
            def needs_update(m):
                estado = m.get("estado")
                if estado in ["pendiente", "en_juego"]:
                    return True
                # Completados sin ganador o sin marcador detallado
                if estado == "completado":
                    if not m.get("resultado_ganador"):
                        return True
                    if not m.get("resultado_marcador"):
                        return True
                return False
            
            matches_to_check = [m for m in matches if needs_update(m)]

            total_matches = len(matches)
            completed_count = total_matches - len(matches_to_check)
            stats["matches_checked"] = len(matches_to_check)

            if not matches_to_check:
                logger.info(f"ℹ️  Todos los {total_matches} partidos ya están completados")
                return stats

            logger.info(f"📋 Verificando {len(matches_to_check)} partidos (pendientes/en vivo) de {total_matches} total")
            logger.info(f"   ✅ {completed_count} partidos ya completados (no requieren actualización)")

            # Procesar cada partido
            for match in matches_to_check:
                try:
                    updated = self._update_single_match(match)
                    if updated:
                        stats["matches_updated"] += 1
                        
                        # Contar por tipo de actualización
                        if match.get("estado") == "en_juego":
                            stats["matches_live"] += 1
                        elif match.get("estado") == "completado":
                            stats["matches_completed"] += 1
                
                except Exception as e:
                    logger.error(f"❌ Error actualizando partido {match.get('id')}: {e}")
                    stats["errors"] += 1
            
            # Resumen
            if stats["matches_updated"] > 0:
                logger.info(
                    f"✅ Actualización completada: {stats['matches_updated']} partidos actualizados "
                    f"({stats['matches_live']} en vivo, {stats['matches_completed']} completados)"
                )
            else:
                logger.debug("ℹ️  No se detectaron cambios en partidos recientes")
            
            return stats
        
        except Exception as e:
            logger.error(f"❌ Error en actualización de partidos: {e}", exc_info=True)
            stats["errors"] += 1
            return stats

    def _update_single_match(self, match: Dict) -> bool:
        """
        Actualiza un solo partido consultando la API

        Args:
            match: Datos del partido de la DB

        Returns:
            True si se actualizó algo
        """
        match_id = match["id"]
        event_key = match.get("event_key")

        # Si no hay event_key, no podemos actualizar
        if not event_key:
            return False

        # Si ya está completado Y tiene TODOS los datos, no necesita actualización
        if match.get("estado") == "completado" and match.get("resultado_ganador") and match.get("resultado_marcador"):
            # Verificar si tiene scores por set guardados
            try:
                sets_count = len(self.db.get_match_sets(match_id))
                if sets_count > 0:
                    return False  # Ya tiene todo
            except:
                pass  # Continuar para obtener los sets

        try:
            # Consultar API para obtener estado actual
            # get_fixtures requiere date_start y date_stop
            match_date = match.get("fecha_partido")
            if not match_date:
                return False
            
            # Convertir fecha a string si es necesario
            if hasattr(match_date, 'strftime'):
                date_str = match_date.strftime('%Y-%m-%d')
            else:
                date_str = str(match_date)
            
            params = {
                "date_start": date_str,
                "date_stop": date_str,
                "match_key": event_key  # API usa match_key, no event_key
            }
            data = self.api_client._make_request("get_fixtures", params)

            if not data or not data.get("result"):
                return False

            # Buscar el partido específico por event_key
            api_match = None
            results = data["result"]
            if isinstance(results, list):
                for m in results:
                    # Convertir ambos a string para comparar (API devuelve int)
                    if str(m.get("event_key")) == str(event_key):
                        api_match = m
                        break
            else:
                api_match = results
            
            if not api_match:
                return False

            # Extraer datos actualizados
            event_live = api_match.get("event_live", "0")
            event_final_result = api_match.get("event_final_result", "-")
            event_status = api_match.get("event_status", "")
            event_time = api_match.get("event_time")

            # Determinar nuevo estado
            nuevo_estado = self._determine_estado(event_live, event_final_result, event_status)

            # Verificar si hay cambios
            estado_actual = match.get("estado", "pendiente")
            event_live_actual = match.get("event_live", "0")
            hora_actual = match.get("hora_inicio")

            cambios_detectados = False

            # Actualizar si hay cambios O si está completado sin ganador O si cambió la hora
            necesita_actualizacion = (
                nuevo_estado != estado_actual or 
                event_live != event_live_actual or
                (nuevo_estado == "completado" and not match.get("resultado_ganador")) or
                (event_time and event_time != hora_actual)  # ⏰ NUEVO: actualizar si cambió la hora
            )
            
            if necesita_actualizacion:
                # Preparar datos para actualizar
                update_data = {
                    "event_live": event_live,
                    "event_status": event_status,
                }
                
                # Actualizar hora si ha cambiado (por retrasos)
                event_time = api_match.get("event_time")
                if event_time and event_time != match.get("hora_inicio"):
                    update_data["hora_inicio"] = event_time
                    logger.debug(f"⏰ Hora actualizada: {match.get('hora_inicio')} → {event_time}")

                # Extraer marcador detallado desde scores (juegos por set)
                scores = api_match.get("scores", [])
                if scores:
                    # Construir marcador detallado: "6-4, 7-5, 6-3"
                    marcador_detallado = self._build_detailed_score(scores)
                    if marcador_detallado:
                        update_data["resultado_marcador"] = marcador_detallado
                
                # Fallback: usar event_home/away_final_result si no hay scores
                if not update_data.get("resultado_marcador"):
                    event_scores = api_match.get("event_home_final_result", "") + " - " + api_match.get("event_away_final_result", "")
                    if event_scores and event_scores != " - ":
                        update_data["resultado_marcador"] = event_scores

                # Si hay resultado final, extraer ganador y marcador completo
                if event_final_result and event_final_result != "-":
                    update_data["event_final_result"] = event_final_result

                    # Intentar determinar ganador
                    ganador = self._extract_winner(api_match, match)
                    if ganador:
                        update_data["resultado_ganador"] = ganador

                # Para partidos en vivo, actualizar marcador parcial
                elif nuevo_estado == "en_juego":
                    # Obtener marcador en vivo
                    live_score = self._extract_live_score(api_match)
                    if live_score:
                        update_data["resultado_marcador"] = live_score

                # Actualizar en DB
                self.db.update_match_live_data(
                    match_id=match_id,
                    event_live=update_data.get("event_live"),
                    event_status=update_data.get("event_status"),
                    event_final_result=update_data.get("event_final_result"),
                    scores=update_data.get("resultado_marcador"),
                )
                
                # Actualizar hora si cambió
                if "hora_inicio" in update_data:
                    self.db.update_match_hora_inicio(match_id, update_data["hora_inicio"])
                
                # Si hay ganador, actualizar también ese campo
                if update_data.get("resultado_ganador"):
                    self._update_winner(match_id, update_data["resultado_ganador"])
                
                # Auto-poblar player_keys si no existen
                self._auto_populate_player_keys(match, api_match)
                
                # Si el partido está completado, guardar estadísticas detalladas
                if nuevo_estado == "completado":
                    # Guardar scores por set
                    self._save_match_sets_from_api(match_id, api_match)
                    # Guardar estadísticas detalladas (juegos y puntos)
                    self._store_detailed_stats(match_id, event_key)

                # Log del cambio
                self._log_change(match, estado_actual, nuevo_estado, update_data)
                cambios_detectados = True

            return cambios_detectados

        except Exception as e:
            logger.debug(f"Error consultando API para partido {match_id}: {e}")
            return False

    def _build_detailed_score(self, scores: List[Dict]) -> Optional[str]:
        """
        Construye el marcador detallado desde los scores por set.
        
        Args:
            scores: Lista de scores desde API (ej: [{"score_first": "6", "score_second": "4", "score_set": "1"}])
            
        Returns:
            Marcador formateado (ej: "6-4, 7-5, 6-3") o None
        """
        if not scores:
            return None
        
        try:
            # Ordenar por set_number
            sorted_scores = sorted(scores, key=lambda x: int(x.get("score_set", 0)))
            
            set_scores = []
            for score in sorted_scores:
                p1 = score.get("score_first", "0")
                p2 = score.get("score_second", "0")
                if p1 and p2:
                    set_scores.append(f"{p1}-{p2}")
            
            if set_scores:
                return ", ".join(set_scores)
            
            return None
        except Exception as e:
            logger.debug(f"Error construyendo marcador detallado: {e}")
            return None

    def _extract_live_score(self, api_match: Dict) -> Optional[str]:
        """
        Extrae el marcador en vivo de un partido

        Args:
            api_match: Datos del partido de la API

        Returns:
            Marcador formateado o None
        """
        try:
            # PRIORIDAD 1: Usar scores detallados
            scores = api_match.get("scores", [])
            if scores:
                detailed = self._build_detailed_score(scores)
                if detailed:
                    return detailed
            
            # PRIORIDAD 2: Fallback a home/away final result
            home_score = api_match.get("event_home_final_result", "")
            away_score = api_match.get("event_away_final_result", "")
            
            if home_score and away_score:
                return f"{home_score} - {away_score}"
            
            return None
        except Exception as e:
            logger.debug(f"Error extrayendo marcador en vivo: {e}")
            return None

    def _update_winner(self, match_id: int, ganador: str):
        """
        Actualiza el ganador del partido en la base de datos

        Args:
            match_id: ID del partido
            ganador: Nombre del ganador
        """
        try:
            self.db.update_match_ganador(match_id, ganador)
        except Exception as e:
            logger.error(f"Error actualizando ganador: {e}")

    def _save_match_sets_from_api(self, match_id: int, api_match: Dict):
        """
        Extrae y guarda los scores por set desde la respuesta de la API
        
        Args:
            match_id: ID del partido
            api_match: Datos del partido de la API
        """
        try:
            scores = api_match.get("scores", [])
            if not scores:
                # Intentar parsear desde resultado_marcador si existe
                return
            
            sets_data = []
            for score in scores:
                set_number = int(score.get("score_set", 0))
                player1_score = int(score.get("score_first", 0))
                player2_score = int(score.get("score_second", 0))
                
                # Detectar tiebreak
                tiebreak_score = None
                if (player1_score == 7 and player2_score == 6) or \
                   (player1_score == 6 and player2_score == 7):
                    # Podría ser tiebreak, la API no siempre da el detalle
                    tiebreak_score = f"{player1_score}-{player2_score}"
                
                sets_data.append({
                    "set_number": set_number,
                    "player1_score": player1_score,
                    "player2_score": player2_score,
                    "tiebreak_score": tiebreak_score
                })
            
            if sets_data:
                saved = self.db.save_match_sets(match_id, sets_data)
                if saved > 0:
                    logger.debug(f"✅ Guardados {saved} sets para partido {match_id}")
                    
        except Exception as e:
            logger.debug(f"Error guardando sets: {e}")

    def _determine_estado(
        self, event_live: str, event_final_result: str, event_status: str
    ) -> str:
        """
        Determina el estado del partido basándose en los datos de la API
        
        Args:
            event_live: "0" o "1"
            event_final_result: Resultado final (ej: "2-0") o "-"
            event_status: Estado del evento (Finished, Walk Over, Postponed, etc.)
        
        Returns:
            Estado: "pendiente", "en_juego", "completado", "cancelado", "pospuesto"
        """
        # Normalizar valores
        event_status_lower = event_status.lower() if event_status else ""
        
        # PRIORIDAD 1: Estados finalizados explícitos
        # Estos indican que el partido YA terminó de alguna forma
        finished_statuses = [
            "finished", "ended", "completed", "final",
            "walk over", "walkover", "w.o.", "wo", "w/o",
            "retired", "ret", "retirement",
            "defaulted", "def", "default",
            "awarded"
        ]
        
        for status in finished_statuses:
            if status in event_status_lower:
                logger.debug(f"🏁 Partido finalizado - status detectado: '{status}' en '{event_status}'")
                return "completado"
        
        # PRIORIDAD 2: Estados cancelados/pospuestos
        cancelled_statuses = ["cancelled", "canceled", "postponed", "suspended", "interrupted"]
        for status in cancelled_statuses:
            if status in event_status_lower:
                logger.debug(f"⚠️ Partido cancelado/pospuesto - status: '{status}' en '{event_status}'")
                return "completado"
        
        # PRIORIDAD 3: Si está en vivo
        if event_live == "1":
            return "en_juego"
        
        # PRIORIDAD 4: Si tiene resultado final y NO está en vivo
        if event_final_result and event_final_result != "-" and event_final_result.strip():
            # Verificar que parece un resultado válido (no solo "-")
            parts = event_final_result.replace(" ", "").split("-")
            if len(parts) >= 2 and all(p.isdigit() or p == "" for p in parts):
                logger.debug(f"🏆 Partido completado con resultado: {event_final_result}")
                return "completado"
        
        # PRIORIDAD 5: Por defecto, pendiente
        return "pendiente"

    def _extract_winner(self, api_match: Dict, db_match: Dict) -> Optional[str]:
        """
        Extrae el ganador del partido
        
        Args:
            api_match: Datos del partido de la API
            db_match: Datos del partido de la DB
        
        Returns:
            Nombre del ganador o None
        """
        event_winner = api_match.get("event_winner")
        event_status = api_match.get("event_status", "")
        
        if not event_winner or event_winner == "None":
            # Si no hay ganador pero el partido está finalizado, intentar deducir del resultado
            event_final_result = api_match.get("event_final_result", "")
            if event_final_result and event_final_result != "-":
                # Formato típico: "2 - 0" o "2-1"
                try:
                    parts = event_final_result.replace(" ", "").split("-")
                    if len(parts) == 2:
                        score1, score2 = int(parts[0]), int(parts[1])
                        if score1 > score2:
                            return db_match.get("jugador1_nombre")
                        elif score2 > score1:
                            return db_match.get("jugador2_nombre")
                except:
                    pass
            return None
        
        # event_winner puede ser "First Player" o "Second Player"
        winner_lower = event_winner.lower()
        
        if "first" in winner_lower or "home" in winner_lower:
            return db_match.get("jugador1_nombre")
        elif "second" in winner_lower or "away" in winner_lower:
            return db_match.get("jugador2_nombre")
        
        return None

    def _log_change(
        self, match: Dict, estado_anterior: str, estado_nuevo: str, update_data: Dict
    ):
        """
        Registra el cambio detectado en los logs

        Args:
            match: Datos del partido
            estado_anterior: Estado previo
            estado_nuevo: Nuevo estado
            update_data: Datos actualizados
        """
        jugador1 = match.get("jugador1_nombre", "Unknown")
        jugador2 = match.get("jugador2_nombre", "Unknown")

        if estado_nuevo == "en_juego" and estado_anterior != "en_juego":
            logger.info(f"🔴 Partido en vivo: {jugador1} vs {jugador2}")

        elif estado_nuevo == "completado" and estado_anterior != "completado":
            ganador = update_data.get("resultado_ganador")
            marcador = update_data.get("resultado_marcador", "")

            if ganador:
                logger.info(
                    f"✅ Partido completado: {ganador} venció ({marcador}) - "
                    f"{jugador1} vs {jugador2}"
                )
            else:
                logger.info(f"✅ Partido completado: {jugador1} vs {jugador2} ({marcador})")


# ============================================================
# STANDALONE EXECUTION (para testing)
# ============================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.config.settings import Config

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("\n" + "=" * 60)
    print("🔄 Match Update Service - Test Run")
    print("=" * 60 + "\n")

    # Initialize components
    db = MatchDatabase("matches_v2.db")
    api_client = APITennisClient()

    # Create service
    service = MatchUpdateService(db, api_client)

    # Run update
    stats = service.update_recent_matches(days=7)

    # Print results
    print("\n" + "=" * 60)
    print("📊 RESULTS:")
    print("=" * 60)
    print(f"Matches checked: {stats['matches_checked']}")
    print(f"Matches updated: {stats['matches_updated']}")
    print(f"  - Live: {stats['matches_live']}")
    print(f"  - Completed: {stats['matches_completed']}")
    print(f"Errors: {stats['errors']}")
    print("=" * 60 + "\n")

    db.close()
    def _auto_populate_player_keys(self, match: Dict, api_match: Dict):
        """
        Auto-pobla player_keys si no existen en el match
        
        Args:
            match: Datos del partido en DB
            api_match: Datos del partido de la API
        """
        match_id = match.get("id")
        
        # Verificar si ya tiene player_keys
        if match.get("first_player_key") and match.get("second_player_key"):
            return
        
        try:
            # Obtener nombres y logos de jugadores
            player1_name = match.get("jugador1_nombre")
            player2_name = match.get("jugador2_nombre")
            player1_logo = match.get("jugador1_logo")
            player2_logo = match.get("jugador2_logo")
            
            # Intentar obtener player_keys de la API
            player1_key = api_match.get("event_first_player_key")
            player2_key = api_match.get("event_second_player_key")
            
            # Si no están en la API, crear/buscar jugadores
            if not player1_key and player1_name:
                player1_key = self.player_service.get_or_create_player(
                    player_key=abs(hash(player1_name.lower())) % 10000000,
                    player_name=player1_name,
                    player_logo=player1_logo
                )["player_key"]
            
            if not player2_key and player2_name:
                player2_key = self.player_service.get_or_create_player(
                    player_key=abs(hash(player2_name.lower())) % 10000000,
                    player_name=player2_name,
                    player_logo=player2_logo
                )["player_key"]
            
            # Actualizar en DB si tenemos ambos keys
            if player1_key and player2_key:
                self.db.update_match_player_keys(match_id, player1_key, player2_key)
                logger.debug(f"✅ Player keys poblados para match {match_id}")
        
        except Exception as e:
            logger.debug(f"Error auto-poblando player_keys para match {match_id}: {e}")
    def _store_detailed_stats(self, match_id: int, event_key: str):
        """
        Guarda estadísticas detalladas del partido (juegos y puntos)
        
        Args:
            match_id: ID del partido
            event_key: Key del evento en API
        """
        try:
            # Obtener datos detallados de la API
            data = self.api_client._make_request("get_events", {"event_key": event_key})
            
            if not data or not data.get("result"):
                return
            
            result = data["result"]
            
            # Verificar si ya tenemos estos datos guardados
            existing = self.db.check_match_games_exist(match_id)
            
            if existing > 0:
                logger.debug(f"Estadísticas detalladas ya guardadas para match {match_id}")
                return
            
            # Guardar juegos
            if "games" in result and result["games"]:
                games_saved = 0
                for game in result["games"]:
                    if self.db.save_match_game(match_id, game):
                        games_saved += 1
                
                if games_saved > 0:
                    logger.info(f"✅ Guardados {games_saved} juegos para match {match_id}")
            
            # Guardar puntos
            if "games" in result and result["games"]:
                points_saved = 0
                for game in result["games"]:
                    if "points" in game and game["points"]:
                        set_number = game.get("set_number", "")
                        game_number = game.get("number_game", "")
                        for point in game["points"]:
                            if self.db.save_match_point(match_id, set_number, game_number, point):
                                points_saved += 1
                
                if points_saved > 0:
                    logger.info(f"✅ Guardados {points_saved} puntos para match {match_id}")
            
        except Exception as e:
            logger.debug(f"Error guardando estadísticas detalladas: {e}")

