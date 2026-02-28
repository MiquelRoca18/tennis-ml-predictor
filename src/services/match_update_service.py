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
                # Completados sin ganador, sin marcador, o con marcador solo en sets (0-3, 2-1) para intentar rellenar juegos
                if estado == "completado":
                    if not m.get("resultado_ganador"):
                        return True
                    marcador = (m.get("resultado_marcador") or "").strip()
                    if not marcador:
                        return True
                    # Si el marcador parece solo sets (ej. "0-3", "2-1"), intentar get_events para juegos por set
                    if "-" in marcador and "," not in marcador:
                        try:
                            parts = marcador.replace(" ", "").split("-")
                            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                                a, b = int(parts[0]), int(parts[1])
                                if a <= 3 and b <= 3:
                                    return True  # Re-procesar para intentar obtener juegos
                        except (ValueError, TypeError):
                            pass
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

    def update_matches_by_ids(self, match_ids: List[int]) -> Dict:
        """
        Fuerza la actualización de resultados para una lista de partidos (p. ej. los que tienen apuestas).
        Útil cuando el usuario abre Mis apuestas: así el backend consulta la API y actualiza estado/ganador
        antes de que el frontend llame a status-batch para liquidar.

        Returns:
            Dict con updated_count, errors
        """
        if not match_ids:
            return {"updated_count": 0, "errors": 0}
        updated_count = 0
        errors = 0
        for mid in match_ids:
            try:
                match = self.db.get_match(mid)
                if not match:
                    continue
                if self._update_single_match(match):
                    updated_count += 1
            except Exception as e:
                logger.warning("refresh result match %s: %s", mid, e)
                errors += 1
        return {"updated_count": updated_count, "errors": errors}

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

            # Si el partido es en fecha futura o hora_inicio aún no llegó, no confiar en event_live=1 (bug API o timezone)
            from datetime import date as date_type, time as dt_time
            force_estado_pendiente = False
            match_date_val = match_date
            if hasattr(match_date_val, "date"):
                match_date_val = match_date_val.date()
            elif isinstance(match_date_val, str):
                try:
                    match_date_val = datetime.strptime(match_date_val[:10], "%Y-%m-%d").date()
                except (ValueError, TypeError):
                    match_date_val = date_type.today()
            if match_date_val > date_type.today():
                event_live = "0"
                force_estado_pendiente = True
                if not (event_final_result and event_final_result != "-"):
                    event_status = ""  # forzar pendiente si no hay resultado
                logger.debug(f"📅 Partido {match_id} es futuro ({match_date_val}), forzando event_live=0")
            elif match_date_val == date_type.today():
                # Mismo día: verificar si hora_inicio ya pasó
                hora_inicio_val = match.get("hora_inicio") or event_time
                if hora_inicio_val:
                    try:
                        if isinstance(hora_inicio_val, str):
                            parts = str(hora_inicio_val).strip().split(":")
                            h = int(parts[0]) if len(parts) > 0 else 0
                            m = int(parts[1]) if len(parts) > 1 else 0
                            start_dt = datetime.combine(match_date_val, dt_time(h, m, 0))
                        else:
                            start_dt = datetime.combine(match_date_val, hora_inicio_val)
                        if start_dt > datetime.now():
                            event_live = "0"
                            force_estado_pendiente = True
                            if not (event_final_result and event_final_result != "-"):
                                event_status = ""
                            logger.debug(f"📅 Partido {match_id} hora_inicio {hora_inicio_val} aún no llegó, forzando event_live=0")
                    except (ValueError, TypeError):
                        pass

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
                
                # Actualizar hora desde API (corrige horarios y cambios de la organización)
                event_time = api_match.get("event_time")
                if event_time:
                    update_data["hora_inicio"] = event_time
                    if event_time != match.get("hora_inicio"):
                        logger.debug(f"⏰ Hora actualizada: {match.get('hora_inicio')} → {event_time}")

                # Extraer marcador detallado desde scores (juegos por set), orden jugador1-jugador2
                scores = api_match.get("scores", [])
                if scores:
                    swap = not self._api_first_is_our_jugador1(api_match, match)
                    marcador_detallado = self._build_detailed_score(scores, swap_order=swap)
                    if marcador_detallado:
                        update_data["resultado_marcador"] = marcador_detallado
                
                # Fallback: solo si parece JUEGOS por set (no resultado en sets tipo 0-3)
                if not update_data.get("resultado_marcador"):
                    home = api_match.get("event_home_final_result", "").strip()
                    away = api_match.get("event_away_final_result", "").strip()
                    if home and away:
                        try:
                            h, a = int(home), int(away)
                            if max(h, a) >= 4:
                                update_data["resultado_marcador"] = f"{home}-{away}"
                        except (ValueError, TypeError):
                            pass

                # Si completado y aún no tenemos juegos por set, intentar get_events (scores o games)
                if nuevo_estado == "completado" and not update_data.get("resultado_marcador"):
                    event_detail = self._fetch_scores_from_get_events(event_key, match)
                    if event_detail:
                        update_data["resultado_marcador"] = event_detail.get("marcador")
                        if event_detail.get("scores"):
                            api_match = {**api_match, "scores": event_detail["scores"]}

                # Si hay resultado final, extraer ganador y marcador completo
                if event_final_result and event_final_result != "-":
                    update_data["event_final_result"] = event_final_result

                    # Intentar determinar ganador
                    ganador = self._extract_winner(api_match, match)
                    if ganador:
                        update_data["resultado_ganador"] = ganador

                # Para partidos en vivo: solo sets completados (el set en curso no cuenta hasta que termine)
                elif nuevo_estado == "en_juego":
                    swap = not self._api_first_is_our_jugador1(api_match, match)
                    live_score = self._extract_live_score(
                        api_match, event_final_result=event_final_result or "", swap_order=swap
                    )
                    if live_score:
                        update_data["resultado_marcador"] = live_score
                    # Guardar score del juego actual y quién saca para el frontend
                    if api_match.get("event_game_result") is not None:
                        update_data["event_game_result"] = api_match.get("event_game_result")
                    if api_match.get("event_serve") is not None:
                        update_data["event_serve"] = api_match.get("event_serve")

                # Actualizar en DB (force_estado corrige en_juego erróneo en partidos futuros)
                self.db.update_match_live_data(
                    match_id=match_id,
                    event_live=update_data.get("event_live"),
                    event_status=update_data.get("event_status"),
                    event_final_result=update_data.get("event_final_result"),
                    scores=update_data.get("resultado_marcador"),
                    event_game_result=update_data.get("event_game_result"),
                    event_serve=update_data.get("event_serve"),
                    force_estado="pendiente" if force_estado_pendiente else None,
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
                    # Guardar scores por set (orden jugador1-jugador2)
                    self._save_match_sets_from_api(match_id, api_match, match)
                    # Guardar pointbypoint en caché para stats/timeline (Grand Slams, etc.)
                    pbp = api_match.get("pointbypoint", [])
                    if pbp:
                        self.db.save_pointbypoint_cache(match_id, pbp)
                        logger.debug(f"✅ Pointbypoint cache guardado para match {match_id} ({len(pbp)} juegos)")
                    # Guardar estadísticas detalladas (juegos y puntos)
                    self._store_detailed_stats(match_id, event_key)

                # Log del cambio
                self._log_change(match, estado_actual, nuevo_estado, update_data)

                # Actualizar estado del predictor (igual que backtesting) cuando hay partido completado
                if nuevo_estado == "completado" and update_data.get("resultado_ganador"):
                    self._actualizar_feature_generator_completado(match, update_data)

                cambios_detectados = True

            return cambios_detectados

        except Exception as e:
            logger.debug(f"Error consultando API para partido {match_id}: {e}")
            return False

    def _fetch_scores_from_get_events(self, event_key: str, db_match: Dict) -> Optional[Dict]:
        """
        Obtiene juegos por set desde get_events (scores o construidos desde games).
        Returns: {"marcador": "6-4, 6-3, 6-2", "scores": [{score_set, score_first, score_second}, ...]} o None.
        """
        try:
            data = self.api_client._make_request("get_events", {"event_key": event_key})
            if not data or not data.get("result"):
                return None
            result = data["result"]
            if isinstance(result, list):
                result = result[0] if result else {}
            scores = result.get("scores")
            if scores and isinstance(scores, list):
                swap = not self._api_first_is_our_jugador1(result, db_match)
                marcador = self._build_detailed_score(scores, swap_order=swap)
                if marcador:
                    return {"marcador": marcador, "scores": scores}
            games = result.get("games") or []
            if games:
                set_games: Dict[int, List[Dict]] = {}
                for g in games:
                    sn = int(g.get("set_number", 0))
                    if sn not in set_games:
                        set_games[sn] = []
                    set_games[sn].append(g)
                built = []
                for sn in sorted(set_games.keys()):
                    first_won = second_won = 0
                    for g in set_games[sn]:
                        w = g.get("winner")
                        if w is None:
                            continue
                        ws = str(w).lower()
                        if ws in ("first player", "first", "1") or w == 1:
                            first_won += 1
                        elif ws in ("second player", "second", "2") or w == 2:
                            second_won += 1
                    if first_won > 0 or second_won > 0:
                        built.append({"score_set": sn, "score_first": str(first_won), "score_second": str(second_won)})
                if built:
                    swap = not self._api_first_is_our_jugador1(result, db_match)
                    marcador = self._build_detailed_score(built, swap_order=swap)
                    if marcador:
                        return {"marcador": marcador, "scores": built}
        except Exception as e:
            logger.debug(f"Error obteniendo scores desde get_events: {e}")
        return None

    def _api_first_is_our_jugador1(self, api_match: Dict, db_match: Dict) -> bool:
        """True si API first/home corresponde a nuestro jugador1 (para no intercambiar scores)."""
        api_first = (api_match.get("event_first_player") or api_match.get("event_home_team") or "").strip()
        if not api_first:
            return True
        j1 = (db_match.get("jugador1_nombre") or db_match.get("jugador1") or "").strip()
        if not j1:
            return True
        api_first_norm = api_first.lower().replace("-", " ").split()
        j1_norm = j1.lower().replace("-", " ").split()
        if api_first_norm and j1_norm:
            if api_first_norm[-1] == j1_norm[-1]:
                return True
            if any(a in j1_norm for a in api_first_norm) or any(a in api_first_norm for a in j1_norm):
                return True
        return False

    @staticmethod
    def _parse_score_cell(s) -> tuple:
        """Parsea score_first/score_second de la API. Acepta '6', '7.7' (juegos.puntos_tiebreak). Returns (games_int, tb_point_str_or_none)."""
        if s is None or str(s).strip() == "":
            return (0, None)
        s = str(s).strip()
        if "." in s:
            parts = s.split(".", 1)
            try:
                return (int(parts[0]), parts[1] if len(parts) > 1 else None)
            except (ValueError, TypeError):
                return (0, None)
        try:
            return (int(s), None)
        except (ValueError, TypeError):
            return (0, None)

    def _build_detailed_score(self, scores: List[Dict], swap_order: bool = False) -> Optional[str]:
        """
        Construye el marcador detallado desde los scores por set.
        Acepta formato API 7.7 / 6.5 (tiebreak). Siempre jugador1 - jugador2.
        """
        if not scores:
            return None
        try:
            sorted_scores = sorted(scores, key=lambda x: int(x.get("score_set", 0)))
            set_scores = []
            for score in sorted_scores:
                g1, tb1 = self._parse_score_cell(score.get("score_first"))
                g2, tb2 = self._parse_score_cell(score.get("score_second"))
                if swap_order:
                    g1, g2 = g2, g1
                    tb1, tb2 = tb2, tb1
                seg = f"{g1}-{g2}"
                if tb1 is not None or tb2 is not None:
                    seg += f"({tb1 or 0}-{tb2 or 0})"
                set_scores.append(seg)
            if set_scores:
                return ", ".join(set_scores)
            return None
        except Exception as e:
            logger.debug(f"Error construyendo marcador detallado: {e}")
            return None

    def _completed_sets_count(self, event_final_result: str) -> Optional[int]:
        """
        Número de sets ya terminados (ganados por alguien) según event_final_result.
        Ej: "2-1" -> 3, "1-0" -> 1. Así no mezclamos el set en curso con los finales.
        """
        if not event_final_result or event_final_result.strip() == "-":
            return None
        try:
            parts = event_final_result.replace(" ", "").strip().split("-")
            if len(parts) != 2:
                return None
            a, b = int(parts[0]), int(parts[1])
            return a + b
        except (ValueError, TypeError):
            return None

    def _extract_live_score(
        self, api_match: Dict, event_final_result: Optional[str] = None, swap_order: bool = False
    ) -> Optional[str]:
        """
        Extrae el marcador en vivo. Para partidos en directo solo incluye sets COMPLETADOS:
        el set en curso no debe sumar al resultado final hasta que termine.
        """
        try:
            scores = api_match.get("scores", [])
            if scores:
                n_completed = self._completed_sets_count(event_final_result or "")
                if n_completed is not None and n_completed >= 1:
                    sorted_scores = sorted(scores, key=lambda x: int(x.get("score_set", 0)))
                    completed_only = sorted_scores[:n_completed]
                    if completed_only:
                        detailed = self._build_detailed_score(completed_only, swap_order=swap_order)
                        if detailed:
                            return detailed
                # Sin event_final_result o sin coincidencia: usar todos (comportamiento legacy)
                detailed = self._build_detailed_score(scores, swap_order=swap_order)
                if detailed:
                    return detailed

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

    def _save_match_sets_from_api(self, match_id: int, api_match: Dict, db_match: Optional[Dict] = None):
        """
        Extrae y guarda los scores por set desde la respuesta de la API.
        Orden siempre jugador1-jugador2 (nuestro partido).
        """
        try:
            scores = api_match.get("scores", [])
            if not scores:
                return
            swap = bool(db_match) and not self._api_first_is_our_jugador1(api_match, db_match)
            sets_data = []
            for score in scores:
                set_number = int(score.get("score_set", 0))
                p_first, tb_first = self._parse_score_cell(score.get("score_first"))
                p_second, tb_second = self._parse_score_cell(score.get("score_second"))
                player1_score = p_second if swap else p_first
                player2_score = p_first if swap else p_second
                tiebreak_score = None
                if tb_first is not None or tb_second is not None:
                    tiebreak_score = f"{tb_first or 0}-{tb_second or 0}"
                elif (player1_score == 7 and player2_score == 6) or (player1_score == 6 and player2_score == 7):
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

    def _actualizar_feature_generator_completado(self, match: Dict, update_data: Dict) -> None:
        """
        Notifica al FeatureGeneratorService que un partido se completó,
        para actualizar ELO e histórico (igual que backtesting).
        """
        ganador = update_data.get("resultado_ganador")
        if not ganador:
            return
        j1 = match.get("jugador1_nombre") or ""
        j2 = match.get("jugador2_nombre") or ""
        perdedor = j2 if ganador == j1 else j1
        superficie = match.get("superficie") or "Hard"
        r1 = match.get("jugador1_ranking")
        r2 = match.get("jugador2_ranking")
        winner_rank = int(r1) if r1 is not None and ganador == j1 else (int(r2) if r2 is not None else 999)
        loser_rank = int(r2) if r2 is not None and perdedor == j2 else (int(r1) if r1 is not None else 999)
        fecha = match.get("fecha_partido")
        if not fecha:
            return
        try:
            from src.prediction.feature_generator_service import get_instance
            svc = get_instance()
            svc.actualizar_con_partido(
                winner_name=ganador,
                loser_name=perdedor,
                surface=superficie,
                winner_rank=winner_rank,
                loser_rank=loser_rank,
                fecha=fecha,
            )
        except Exception as e:
            logger.debug(f"No se pudo actualizar feature generator: {e}")


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

    def get_pending_matches(self) -> List[Dict]:
        """Partidos pendientes o en juego (para actualización)."""
        try:
            return self.db._fetchall(
                """
                SELECT id, jugador1_nombre, jugador2_nombre, superficie, fecha_partido, estado
                FROM matches
                WHERE estado IN ('pendiente', 'en_juego')
                ORDER BY fecha_partido ASC
                """,
                {},
            ) or []
        except Exception:
            return []

    def get_update_stats(self) -> Dict:
        """Estadísticas para el endpoint scheduler-status."""
        try:
            pending = self.get_pending_matches()
            return {
                "partidos_pendientes": len(pending),
                "proxima_actualizacion": "Cada 5 min (estados), cada 4h (cuotas/predicciones)",
                "estado": "activo",
            }
        except Exception:
            return {"partidos_pendientes": 0, "estado": "desconocido"}
