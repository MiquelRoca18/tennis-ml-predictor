"""
Live Events Service - WebSocket API-Tennis
==========================================
Conexión en tiempo real a wss://wss.api-tennis.com/live.
Recibe actualizaciones cuando hay eventos (punto, cambio de marcador, etc.)
y actualiza la BD sin hacer polling. Más óptimo en peticiones y menor latencia.
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, date, time as dt_time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _has_match_started(match: Dict) -> bool:
    """
    Verifica si el partido ya ha empezado según fecha_partido + hora_inicio.
    Evita marcar como en_juego partidos que API-Tennis indica como live pero aún no han empezado.
    """
    fecha = match.get("fecha_partido")
    hora = match.get("hora_inicio")
    if not fecha or not hora:
        return False
    try:
        if isinstance(fecha, str):
            match_date = datetime.strptime(str(fecha)[:10], "%Y-%m-%d").date()
        elif hasattr(fecha, "date"):
            match_date = fecha.date() if callable(fecha.date) else fecha
        else:
            return False
        if isinstance(hora, str):
            parts = str(hora).strip().split(":")
            h = int(parts[0]) if len(parts) > 0 else 0
            m = int(parts[1]) if len(parts) > 1 else 0
            start_dt = datetime.combine(match_date, dt_time(h, m, 0))
        else:
            start_dt = datetime.combine(match_date, hora if hasattr(hora, "hour") else dt_time(0, 0, 0))
        return start_dt <= datetime.now()
    except (ValueError, TypeError):
        return False

# Optional: only import when available
try:
    import websocket
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False
    websocket = None


def _parse_score_value(s: Optional[str]) -> tuple:
    """
    Parsea score_first/score_second de la API. Acepta "6", "7.7" (juegos.puntos_tiebreak).
    Returns (games_int, tb_point_str_or_none). Ej: "7.7" -> (7, "7"), "6" -> (6, None).
    """
    if s is None or str(s).strip() == "":
        return (0, None)
    s = str(s).strip()
    if "." in s:
        parts = s.split(".", 1)
        try:
            games = int(parts[0])
            tb = parts[1] if len(parts) > 1 else None
            return (games, tb)
        except (ValueError, TypeError):
            return (0, None)
    try:
        return (int(s), None)
    except (ValueError, TypeError):
        return (0, None)


def _normalize_name(name: Optional[str]) -> str:
    if not name:
        return ""
    return (name or "").strip().lower().replace(".", "").replace("  ", " ")


def _api_first_is_our_jugador1(api_match: Dict, db_match: Dict) -> bool:
    """True si el first player de la API es nuestro jugador1 (para no intercambiar scores)."""
    api_first = _normalize_name(api_match.get("event_first_player") or "")
    j1 = _normalize_name(db_match.get("jugador1_nombre") or "")
    if not api_first or not j1:
        return True
    if api_first == j1:
        return True
    # Comparar por apellido o coincidencia parcial
    api_parts = set(api_first.split())
    j1_parts = set(j1.split())
    if api_parts & j1_parts:
        return True
    return False


def _completed_sets_count(event_final_result: Optional[str]) -> Optional[int]:
    """Número de sets ya terminados según event_final_result. Ej: '2 - 1' -> 3."""
    if not event_final_result or str(event_final_result).strip() in ("-", ""):
        return None
    try:
        s = str(event_final_result).replace(" ", "").strip()
        if "-" not in s:
            return None
        parts = s.split("-")
        if len(parts) != 2:
            return None
        a, b = int(parts[0]), int(parts[1])
        return a + b
    except (ValueError, TypeError):
        return None


def _build_marcador_from_scores(scores: List[Dict], swap: bool, only_completed: Optional[int]) -> Optional[str]:
    """Construye string tipo '6-4, 7-6(7)' desde scores. Acepta formato API 7.7 / 6.5 (tiebreak)."""
    if not scores:
        return None
    try:
        sorted_scores = sorted(scores, key=lambda x: int(x.get("score_set", 0)))
        if only_completed is not None and only_completed >= 1:
            sorted_scores = sorted_scores[:only_completed]
        parts = []
        for sc in sorted_scores:
            g1, tb1 = _parse_score_value(sc.get("score_first"))
            g2, tb2 = _parse_score_value(sc.get("score_second"))
            if swap:
                g1, g2 = g2, g1
                tb1, tb2 = tb2, tb1
            seg = f"{g1}-{g2}"
            if tb1 or tb2:
                seg += f"({tb1 or 0}-{tb2 or 0})"
            parts.append(seg)
        return ", ".join(parts) if parts else None
    except Exception:
        return None


def _scores_to_sets_data(scores: List[Dict], swap: bool) -> List[Dict]:
    """Convierte scores API a lista para save_match_sets. Acepta 7.7 / 6.5 (tiebreak)."""
    out = []
    for sc in sorted(scores, key=lambda x: int(x.get("score_set", 0))):
        set_number = int(sc.get("score_set", 0))
        p_first, tb_first = _parse_score_value(sc.get("score_first"))
        p_second, tb_second = _parse_score_value(sc.get("score_second"))
        player1_score = p_second if swap else p_first
        player2_score = p_first if swap else p_second
        tiebreak = None
        if tb_first is not None or tb_second is not None:
            tiebreak = f"{tb_first or 0}-{tb_second or 0}"
        elif (player1_score == 7 and player2_score == 6) or (player1_score == 6 and player2_score == 7):
            tiebreak = f"{player1_score}-{player2_score}"
        out.append({
            "set_number": set_number,
            "player1_score": player1_score,
            "player2_score": player2_score,
            "tiebreak_score": tiebreak,
        })
    return out


class LiveEventsService:
    """
    Mantiene una conexión WebSocket a API-Tennis y actualiza la BD
    cada vez que llega un evento en vivo (sin polling).
    """

    WSS_URL = "wss://wss.api-tennis.com/live"

    def __init__(self, db, api_key: Optional[str] = None):
        self.db = db
        self.api_key = (api_key or os.getenv("API_TENNIS_API_KEY", "")).strip()
        self._ws = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._connected = False

    def start(self):
        """Lanza el WebSocket en un thread daemon. No bloquea."""
        if not HAS_WEBSOCKET:
            logger.warning("⚠️ websocket-client no instalado: pip install websocket-client")
            return
        if not self.api_key:
            logger.warning("⚠️ API_TENNIS_API_KEY no configurada; WebSocket live no iniciado")
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("✅ WebSocket live events iniciado en background")

    def stop(self):
        """Pide parar el loop y cierra la conexión."""
        self._stop.set()
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
        self._connected = False

    def _run_loop(self):
        backoff = 1
        max_backoff = 60
        while not self._stop.is_set():
            try:
                self._connect_and_run()
            except Exception as e:
                logger.debug("WebSocket run error: %s", e)
            if self._stop.is_set():
                break
            time.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)

    def _connect_and_run(self):
        url = f"{self.WSS_URL}?APIkey={self.api_key}&timezone=Europe/Madrid"
        self._ws = websocket.WebSocketApp(
            url,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open,
        )
        self._ws.run_forever(ping_interval=30, ping_timeout=10)

    def _on_open(self, ws):
        self._connected = True
        logger.info("🔴 WebSocket conectado a API-Tennis (live)")

    def _on_close(self, ws, close_status_code, close_msg):
        self._connected = False
        logger.debug("WebSocket cerrado: %s %s", close_status_code, close_msg)

    def _on_error(self, ws, error):
        logger.debug("WebSocket error: %s", error)

    def _on_message(self, ws, message):
        if not message:
            return
        try:
            data = json.loads(message)
            self._process_match(data)
        except json.JSONDecodeError as e:
            logger.debug("WebSocket message no JSON: %s", e)
        except Exception as e:
            logger.debug("Error procesando mensaje WebSocket: %s", e)

    def process_match_update(self, api_match: Dict) -> bool:
        """
        Actualiza la BD con un partido (desde WebSocket o get_livescore).
        Puede ser llamado desde fuera para refresco por polling.
        Returns True si se actualizó un partido nuestro.
        """
        return self._process_match(api_match)

    def _process_match(self, api_match: Dict):
        """Actualiza la BD con un partido recibido por WebSocket."""
        event_key = api_match.get("event_key")
        if not event_key:
            return
        event_key = str(event_key)
        match = self.db.get_match_by_event_key(event_key)
        if not match:
            return
        match_id = match["id"]
        event_final_result = api_match.get("event_final_result") or "-"
        event_status = (api_match.get("event_status") or "").strip()
        event_live = api_match.get("event_live") or "0"
        event_winner = api_match.get("event_winner")
        event_game_result = api_match.get("event_game_result")
        event_serve = api_match.get("event_serve")

        estado = self._determine_estado(event_live, event_final_result, event_status)
        # Si API dice en_juego pero el partido no ha empezado (hora_inicio), no confiar: API-Tennis puede tener bugs
        event_live_to_save = event_live
        force_estado = None
        if estado == "en_juego" and not _has_match_started(match):
            event_live_to_save = "0"
            force_estado = "pendiente"
            logger.debug("WebSocket: partido %s marcado en_juego por API pero no ha empezado (hora_inicio), forzando pendiente", match_id)

        scores = api_match.get("scores") or []
        swap = not _api_first_is_our_jugador1(api_match, match)
        n_completed = _completed_sets_count(event_final_result)
        marcador = _build_marcador_from_scores(scores, swap, n_completed)

        self.db.update_match_live_data(
            match_id=match_id,
            scores=marcador,
            event_live=event_live_to_save,
            event_status=event_status,
            event_final_result=event_final_result,
            event_game_result=event_game_result,
            event_serve=event_serve,
            force_estado=force_estado,
        )
        if event_winner and estado == "completado":
            self.db.update_match_ganador(match_id, event_winner)

        if scores:
            sets_data = _scores_to_sets_data(scores, swap)
            if estado == "completado":
                pass  # guardar todos los sets
            elif n_completed is not None and n_completed >= 1:
                sets_data = sets_data[:n_completed]
            if sets_data:
                self.db.save_match_sets(match_id, sets_data)
        logger.debug("WS actualizado match_id=%s estado=%s", match_id, estado)

    def _determine_estado(self, event_live: str, event_final_result: str, event_status: str) -> str:
        event_status_lower = (event_status or "").lower()
        finished = (
            "finished" in event_status_lower or "ended" in event_status_lower
            or "completed" in event_status_lower or "final" in event_status_lower
            or "walk over" in event_status_lower or "walkover" in event_status_lower
            or "retired" in event_status_lower or "retirement" in event_status_lower
            or "default" in event_status_lower or "cancelled" in event_status_lower
            or "postponed" in event_status_lower or "awarded" in event_status_lower
        )
        if finished:
            return "completado"
        if event_final_result and str(event_final_result).strip() != "-" and " - " in str(event_final_result):
            return "completado"
        if event_live == "1":
            return "en_juego"
        if event_status and event_status_lower not in ("", "not started", "scheduled"):
            return "en_juego"
        return "pendiente"
